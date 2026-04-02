"""Parameter specifications and ParameterSet for openpkpd."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from openpkpd.utils.errors import NumericalError

logger = logging.getLogger(__name__)

_MAX_TRANSFORM_EXP_ARG = 500.0
_MAX_COVARIANCE_LOG_DIAG = min(
    _MAX_TRANSFORM_EXP_ARG, 0.5 * math.log(np.finfo(np.float64).max) - 1.0
)

# Maximum omega/sigma diagonal in the variance (not SD) scale.
# omega_ii = chol_ii^2; log-Cholesky upper = 0.5*log(MAX_OMEGA_DIAG).
# Default 9.0 → CV ≤ 300 % for each random effect. Prevents
# OMEGA explosion in the outer FOCE loop while covering any realistic
# PK/PD model.
_MAX_OMEGA_DIAG = 9.0
_MAX_COVA_LOG_DIAG_BOUND = 0.5 * math.log(_MAX_OMEGA_DIAG)  # ≈ 1.099




def _exp_covariance_diag(raw: float) -> float:
    """Exponentiate a covariance log-Cholesky diagonal without overflowing on squaring."""
    return math.exp(min(raw, _MAX_COVARIANCE_LOG_DIAG))


@dataclass
class ThetaSpec:
    """Specification for a single THETA parameter."""

    init: float
    lower: float = -float("inf")
    upper: float = float("inf")
    fixed: bool = False
    label: str | None = None

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) > upper ({self.upper}) for theta {self.label}")
        if not (self.lower <= self.init <= self.upper):
            raise ValueError(
                f"init ({self.init}) outside bounds [{self.lower}, {self.upper}] "
                f"for theta {self.label}"
            )


@dataclass
class OmegaSpec:
    """Specification for an OMEGA block."""

    block_size: int  # 1 = scalar; >1 = BLOCK(n)
    values: list[float]  # Lower-triangle column-major (NONMEM BLOCK convention)
    fixed: bool = False
    same: bool = False  # SAME = reuse previous block (IOV)
    label: str | None = None

    def __post_init__(self) -> None:
        expected = self.block_size * (self.block_size + 1) // 2
        if not self.same and len(self.values) != expected:
            raise ValueError(
                f"OmegaSpec block_size={self.block_size} requires {expected} values, "
                f"got {len(self.values)}"
            )

    def to_matrix(self) -> np.ndarray:
        """Expand lower-triangle values to full symmetric matrix."""
        n = self.block_size
        mat = np.zeros((n, n))
        idx = 0
        for col in range(n):
            for row in range(col, n):
                mat[row, col] = self.values[idx]
                mat[col, row] = self.values[idx]
                idx += 1
        return mat


@dataclass
class SigmaSpec:
    """Specification for a SIGMA block (residual error variance)."""

    block_size: int
    values: list[float]
    fixed: bool = False
    label: str | None = None

    def __post_init__(self) -> None:
        expected = self.block_size * (self.block_size + 1) // 2
        if len(self.values) != expected:
            raise ValueError(
                f"SigmaSpec block_size={self.block_size} requires {expected} values, "
                f"got {len(self.values)}"
            )

    def to_matrix(self) -> np.ndarray:
        """Expand lower-triangle values to full symmetric matrix."""
        n = self.block_size
        mat = np.zeros((n, n))
        idx = 0
        for col in range(n):
            for row in range(col, n):
                mat[row, col] = self.values[idx]
                mat[col, row] = self.values[idx]
                idx += 1
        return mat


@dataclass
class ParameterSet:
    """
    A concrete parameter point in (THETA, OMEGA, SIGMA) space.

    THETA: 1-D array of population fixed effects.
    OMEGA: full n_eta × n_eta positive-definite matrix.
    SIGMA: full n_eps × n_eps positive-definite matrix.
    """

    theta: np.ndarray  # shape (n_theta,)
    omega: np.ndarray  # shape (n_eta, n_eta)
    sigma: np.ndarray  # shape (n_eps, n_eps)

    # Specs used to identify which parameters are free vs fixed
    theta_specs: list[ThetaSpec] = field(default_factory=list)
    omega_specs: list[OmegaSpec] = field(default_factory=list)
    sigma_specs: list[SigmaSpec] = field(default_factory=list)

    # ── Vector packing/unpacking for the optimizer ────────────────────────────

    def _free_theta_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.theta_specs) if not s.fixed]

    @staticmethod
    def _lower_triangular_indices(n: int) -> list[tuple[int, int]]:
        return [(r, c) for c in range(n) for r in range(c, n)]

    @staticmethod
    def _block_slices(specs: list[OmegaSpec] | list[SigmaSpec]) -> list[tuple[int, int, bool]]:
        offset = 0
        blocks: list[tuple[int, int, bool]] = []
        for spec in specs:
            block_size = spec.block_size
            blocks.append((offset, block_size, spec.fixed))
            offset += block_size
        return blocks

    def _pack_covariance_blocks(
        self,
        matrix: np.ndarray,
        specs: list[OmegaSpec] | list[SigmaSpec],
        name: str,
    ) -> list[float]:
        try:
            if not specs:
                chol = np.linalg.cholesky(matrix)
                return [
                    math.log(max(float(chol[r, c]), 1e-30)) if r == c else float(chol[r, c])
                    for r, c in self._lower_triangular_indices(matrix.shape[0])
                ]

            parts: list[float] = []
            for offset, block_size, fixed in self._block_slices(specs):
                if fixed:
                    continue
                block = matrix[offset : offset + block_size, offset : offset + block_size]
                chol = np.linalg.cholesky(block)
                for r, c in self._lower_triangular_indices(block_size):
                    v = float(chol[r, c])
                    parts.append(math.log(max(v, 1e-30)) if r == c else v)
            return parts
        except np.linalg.LinAlgError as exc:
            raise NumericalError(f"{name} is not positive-definite: {exc}") from exc

    @classmethod
    def _unpack_covariance_blocks(
        cls,
        vec: np.ndarray,
        idx: int,
        template_matrix: np.ndarray,
        specs: list[OmegaSpec] | list[SigmaSpec],
    ) -> tuple[np.ndarray, int]:
        if not specs:
            n = template_matrix.shape[0]
            chol = np.zeros((n, n))
            for r, c in cls._lower_triangular_indices(n):
                raw = float(vec[idx])
                idx += 1
                chol[r, c] = _exp_covariance_diag(raw) if r == c else raw
            result = chol @ chol.T
            try:
                np.linalg.cholesky(result)
            except np.linalg.LinAlgError:
                logger.warning("_unpack_covariance_blocks: reconstructed matrix is not PD")
            return result, idx

        matrix = np.zeros_like(template_matrix)
        for offset, block_size, fixed in cls._block_slices(specs):
            if fixed:
                matrix[offset : offset + block_size, offset : offset + block_size] = (
                    template_matrix[offset : offset + block_size, offset : offset + block_size]
                )
                continue

            chol = np.zeros((block_size, block_size))
            for r, c in cls._lower_triangular_indices(block_size):
                raw = float(vec[idx])
                idx += 1
                chol[r, c] = _exp_covariance_diag(raw) if r == c else raw
            block_result = chol @ chol.T
            try:
                np.linalg.cholesky(block_result)
            except np.linalg.LinAlgError:
                logger.warning(
                    "_unpack_covariance_blocks: reconstructed block at offset %d is not PD",
                    offset,
                )
            matrix[offset : offset + block_size, offset : offset + block_size] = block_result

        return matrix, idx

    def to_vector(self) -> np.ndarray:
        """
        Pack free parameters into a 1-D vector for the optimizer.

        Convention:
          - Free THETA values (log-transformed if lower=0)
          - Free OMEGA lower-Cholesky elements (diagonal log-transformed)
          - Free SIGMA lower-Cholesky elements (diagonal log-transformed)
        """
        parts: list[float] = []

        # Free THETAs (log-transform positive-only parameters)
        for i in self._free_theta_indices():
            spec = self.theta_specs[i]
            val = float(self.theta[i])
            if spec.lower >= 0 and not math.isinf(spec.upper):
                # Logit-transform bounded parameter
                lo, hi = spec.lower + 1e-10, spec.upper - 1e-10
                val = np.clip(val, lo, hi)
                parts.append(math.log((val - spec.lower) / (spec.upper - val)))
            elif spec.lower >= 0:
                # Log-transform lower-bounded parameter
                parts.append(math.log(max(val, 1e-30)))
            else:
                parts.append(val)

        parts.extend(self._pack_covariance_blocks(self.omega, self.omega_specs, "OMEGA"))
        parts.extend(self._pack_covariance_blocks(self.sigma, self.sigma_specs, "SIGMA"))

        return np.array(parts, dtype=np.float64)

    @classmethod
    def from_vector(
        cls,
        vec: np.ndarray,
        template: ParameterSet,
    ) -> ParameterSet:
        """Unpack optimizer vector back to structured ParameterSet."""
        idx = 0
        theta = template.theta.copy()

        # Free THETAs
        free_theta = template._free_theta_indices()
        for i in free_theta:
            spec = template.theta_specs[i]
            raw = float(vec[idx])
            idx += 1
            if spec.lower >= 0 and not math.isinf(spec.upper):
                # Inverse logit — clamp raw to avoid exp overflow
                raw_clamped = max(-_MAX_TRANSFORM_EXP_ARG, min(_MAX_TRANSFORM_EXP_ARG, raw))
                theta[i] = spec.lower + (spec.upper - spec.lower) / (1.0 + math.exp(-raw_clamped))
            elif spec.lower >= 0:
                theta[i] = math.exp(min(raw, _MAX_TRANSFORM_EXP_ARG))
            else:
                theta[i] = raw

        omega, idx = cls._unpack_covariance_blocks(
            vec,
            idx,
            template.omega,
            template.omega_specs,
        )
        sigma, idx = cls._unpack_covariance_blocks(
            vec,
            idx,
            template.sigma,
            template.sigma_specs,
        )

        return cls(
            theta=theta,
            omega=omega,
            sigma=sigma,
            theta_specs=template.theta_specs,
            omega_specs=template.omega_specs,
            sigma_specs=template.sigma_specs,
        )

    def apply_bounds(self) -> ParameterSet:
        """Clamp THETA to bounds; ensure OMEGA/SIGMA are positive-definite."""
        theta = self.theta.copy()
        for i, spec in enumerate(self.theta_specs):
            if not math.isinf(spec.lower):
                theta[i] = max(theta[i], spec.lower + 1e-10)
            if not math.isinf(spec.upper):
                theta[i] = min(theta[i], spec.upper - 1e-10)

        omega = _repair_pd(self.omega)
        sigma = _repair_pd(self.sigma)
        return ParameterSet(
            theta=theta,
            omega=omega,
            sigma=sigma,
            theta_specs=self.theta_specs,
            omega_specs=self.omega_specs,
            sigma_specs=self.sigma_specs,
        )

    @classmethod
    def from_specs(
        cls,
        theta_specs: list[ThetaSpec],
        omega_specs: list[OmegaSpec],
        sigma_specs: list[SigmaSpec],
    ) -> ParameterSet:
        """Build a ParameterSet from specs using initial values."""
        theta = np.array([s.init for s in theta_specs], dtype=np.float64)

        # Assemble OMEGA from blocks
        n_eta = sum(s.block_size for s in omega_specs)
        omega = np.zeros((n_eta, n_eta))
        offset = 0
        for spec in omega_specs:
            blk = spec.to_matrix()
            n = spec.block_size
            omega[offset : offset + n, offset : offset + n] = blk
            offset += n

        # Assemble SIGMA from blocks
        n_eps = sum(s.block_size for s in sigma_specs)
        sigma = np.zeros((n_eps, n_eps))
        offset = 0
        for sspec in sigma_specs:
            blk = sspec.to_matrix()
            n = sspec.block_size
            sigma[offset : offset + n, offset : offset + n] = blk
            offset += n

        return cls(
            theta=theta,
            omega=omega,
            sigma=sigma,
            theta_specs=theta_specs,
            omega_specs=omega_specs,
            sigma_specs=sigma_specs,
        )

    def n_theta(self) -> int:
        return len(self.theta)

    def n_eta(self) -> int:
        return self.omega.shape[0]

    def n_eps(self) -> int:
        return self.sigma.shape[0]

    # ── IOV helpers ───────────────────────────────────────────────────────────

    def n_iov_occasions(self) -> int:
        """
        Number of occasions for IOV (count OMEGA SAME specs).

        An ``OmegaSpec`` with ``same=True`` means this OMEGA block is a copy
        of the preceding block (NONMEM ``OMEGA SAME`` syntax).  Each such
        repetition represents an additional occasion.

        Returns:
            Number of occasions.  Returns 1 when no IOV is present (every
            individual has exactly one occasion).
        """
        if not self.omega_specs:
            return 1
        same_count = sum(1 for s in self.omega_specs if s.same)
        return same_count + 1 if same_count > 0 else 1

    def has_iov(self) -> bool:
        """
        Return True if any OMEGA spec has ``same=True``.

        An ``OmegaSpec`` with ``same=True`` signals that this block is an
        IOV (Inter-Occasion Variability) repetition of the preceding block.
        """
        return any(s.same for s in self.omega_specs)

    def expand_omega_iov(self, n_occasions: int) -> ParameterSet:
        """
        Expand OMEGA to handle IOV by creating per-occasion ETA blocks.

        For each ``OmegaSpec`` that is *not* marked ``same=True``, the block
        is repeated ``n_occasions`` times to form a block-diagonal OMEGA.
        The resulting OMEGA has ``n_eta_base * n_occasions`` random effects.

        This mirrors the NONMEM pattern::

            $OMEGA BLOCK(1) 0.04
            $OMEGA BLOCK(1) SAME   ; occasion 2
            $OMEGA BLOCK(1) SAME   ; occasion 3

        which openpkpd stores as one BLOCK + two SAME entries.

        Args:
            n_occasions: Number of occasions to expand to.  Must be >= 1.

        Returns:
            A new ParameterSet with expanded OMEGA (block-diagonal over
            occasions) and OMEGA specs updated accordingly.  THETA and SIGMA
            are copied unchanged.

        Raises:
            ValueError: If ``n_occasions < 1``.
        """
        if n_occasions < 1:
            raise ValueError(f"n_occasions must be >= 1, got {n_occasions}")

        if n_occasions == 1:
            return ParameterSet(
                theta=self.theta.copy(),
                omega=self.omega.copy(),
                sigma=self.sigma.copy(),
                theta_specs=list(self.theta_specs),
                omega_specs=list(self.omega_specs),
                sigma_specs=list(self.sigma_specs),
            )

        # Identify "base" OMEGA blocks (those not marked same=True)
        base_specs: list[OmegaSpec] = []
        base_matrices: list[np.ndarray] = []

        for spec in self.omega_specs:
            if not spec.same:
                base_specs.append(spec)
                base_matrices.append(spec.to_matrix())

        # Build expanded block-diagonal OMEGA
        expanded_blocks: list[np.ndarray] = []
        expanded_specs: list[OmegaSpec] = []

        for occ in range(n_occasions):
            for spec, mat in zip(base_specs, base_matrices, strict=False):
                expanded_blocks.append(mat)
                is_same = occ > 0  # first occasion = base; subsequent = SAME
                expanded_specs.append(
                    OmegaSpec(
                        block_size=spec.block_size,
                        values=list(spec.values),
                        fixed=spec.fixed,
                        same=is_same,
                        label=(f"{spec.label}_occ{occ + 1}" if spec.label else f"iov_occ{occ + 1}"),
                    )
                )

        # Assemble block-diagonal OMEGA matrix
        total_eta = sum(b.shape[0] for b in expanded_blocks)
        new_omega = np.zeros((total_eta, total_eta))
        offset = 0
        for blk in expanded_blocks:
            n = blk.shape[0]
            new_omega[offset : offset + n, offset : offset + n] = blk
            offset += n

        return ParameterSet(
            theta=self.theta.copy(),
            omega=new_omega,
            sigma=self.sigma.copy(),
            theta_specs=list(self.theta_specs),
            omega_specs=expanded_specs,
            sigma_specs=list(self.sigma_specs),
        )

    def n_free(self) -> int:
        """Number of free (non-fixed) parameters in the optimization vector."""
        n_free_theta = sum(1 for s in self.theta_specs if not s.fixed)
        if self.omega_specs:
            n_free_omega = sum(
                s.block_size * (s.block_size + 1) // 2 for s in self.omega_specs if not s.fixed
            )
        else:
            n_omega = self.omega.shape[0]
            n_free_omega = n_omega * (n_omega + 1) // 2
        if self.sigma_specs:
            n_free_sigma = sum(
                s.block_size * (s.block_size + 1) // 2 for s in self.sigma_specs if not s.fixed
            )
        else:
            n_eps = self.sigma.shape[0]
            n_free_sigma = n_eps * (n_eps + 1) // 2
        return n_free_theta + n_free_omega + n_free_sigma

    def get_optimizer_bounds(
        self,
    ) -> list[tuple[float | None, float | None]]:
        """
        Return L-BFGS-B bounds for every element of ``to_vector()``.

        - Free THETA: (None, None) — the logit/log transform already
          enforces spec.lower/upper in ``from_vector``/``apply_bounds``.
        - OMEGA/SIGMA log-Cholesky diagonal: (None, _MAX_COVA_LOG_DIAG_BOUND)
          so omega_ii ≤ _MAX_OMEGA_DIAG.  Prevents the outer loop from
          driving IIV to infinity (a degenerate but numerically cheap solution).
          No lower bound is set: if the optimizer drives omega toward zero,
          the ``objective()`` closure detects the resulting 1e10 penalty and
          retries with a cold η-hat reset (see foce.py) to recover clean OFV.
        - OMEGA/SIGMA off-diagonal Cholesky elements: (None, None).
        """
        bounds: list[tuple[float | None, float | None]] = []

        for _ in self._free_theta_indices():
            bounds.append((None, None))

        for specs, matrix in (
            (self.omega_specs, self.omega),
            (self.sigma_specs, self.sigma),
        ):
            if not specs:
                n = matrix.shape[0]
                for r, c in self._lower_triangular_indices(n):
                    if r == c:
                        bounds.append((None, _MAX_COVA_LOG_DIAG_BOUND))
                    else:
                        bounds.append((None, None))
            else:
                for _offset, block_size, fixed in self._block_slices(specs):
                    if fixed:
                        continue
                    for r, c in self._lower_triangular_indices(block_size):
                        if r == c:
                            bounds.append((None, _MAX_COVA_LOG_DIAG_BOUND))
                        else:
                            bounds.append((None, None))

        return bounds

    def __repr__(self) -> str:
        return (
            f"ParameterSet(theta={self.theta}, "
            f"omega_diag={np.diag(self.omega)}, "
            f"sigma_diag={np.diag(self.sigma)})"
        )


def _repair_pd(mat: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Ensure matrix is positive-definite by eigenvalue clipping.

    Replaces any eigenvalue < epsilon with epsilon, then reconstructs.
    """
    mat = (mat + mat.T) / 2  # Ensure symmetry
    if mat.size == 0:
        return mat
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    # PA2: use a scale-relative floor so small matrices are not over-regularised
    max_abs = float(np.max(np.abs(eigenvalues))) if eigenvalues.size > 0 else 0.0
    epsilon = max(1e-10, 1e-7 * max_abs)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
