"""
Prior information for population PK/PD analysis.

Implements the $PRIOR record functionality from NONMEM: adds a quadratic
(Gaussian) penalty to the OFV based on prior estimates and their covariance.

The augmented objective function is::

    OFV_augmented = OFV_data + penalty_theta [ + penalty_omega ]

where::

    penalty_theta = (theta - theta_prior)^T * Sigma_theta^{-1} * (theta - theta_prior)
    penalty_omega = (omega_vec - omega_prior_vec)^T * Sigma_omega^{-1}
                    * (omega_vec - omega_prior_vec)

This is equivalent to a MAP (maximum a posteriori) estimator under Gaussian
priors, and matches NONMEM's $PRIOR NWPRI implementation.

Usage::

    from openpkpd.prior.prior import PriorSpec, PriorAugmentedModel

    prior = PriorSpec(
        theta_prior=np.array([1.5, 0.08, 30.0]),
        theta_prior_cov=np.diag([0.1, 0.01, 25.0]),
    )
    aug_model = PriorAugmentedModel(population_model=pop_model, prior=prior)

    # Use aug_model anywhere a PopulationModel is expected
    result = estimation_method.estimate(aug_model, params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from openpkpd.math.matrix import omega_from_lower_triangle

if TYPE_CHECKING:
    from openpkpd.model.parameters import ParameterSet
    from openpkpd.parser.control_stream import ControlStream


# ── Prior specification ────────────────────────────────────────────────────────


@dataclass
class PriorSpec:
    """
    Specification of prior information for THETA and/or OMEGA.

    The prior is a multivariate Gaussian (normal) distribution.  Only
    the mean and covariance are required; the precision (inverse covariance)
    is computed lazily and cached.

    Attributes:
        theta_prior:      Prior mean for THETA, shape ``(n_theta,)``. Optional.
        theta_prior_cov:  Prior covariance matrix for THETA, shape
                          ``(n_theta, n_theta)``.  Must be positive-definite.
                          Required only when ``theta_prior`` is provided.
        omega_prior:      Prior mean for OMEGA lower-triangle elements,
                          shape ``(n_omega_elements,)``.  Optional.
        omega_prior_cov:  Prior covariance for OMEGA elements, shape
                          ``(n_omega_elements, n_omega_elements)``.  Optional.
        nwpri:            Number of prior records (NONMEM NWPRI parameter).
                          Informational; not used in penalty computation here.
    """

    theta_prior: np.ndarray | None = None
    theta_prior_cov: np.ndarray | None = None
    omega_prior: np.ndarray | None = None
    omega_prior_cov: np.ndarray | None = None
    nwpri: int = 0

    def __post_init__(self) -> None:
        if self.theta_prior is not None and self.theta_prior_cov is None:
            raise ValueError("theta_prior_cov must be provided when theta_prior is given.")
        if self.theta_prior_cov is not None and self.theta_prior is None:
            raise ValueError("theta_prior must be provided when theta_prior_cov is given.")
        if self.theta_prior is not None:
            n = len(self.theta_prior)
            if self.theta_prior_cov is None or self.theta_prior_cov.shape != (n, n):
                got = None if self.theta_prior_cov is None else self.theta_prior_cov.shape
                raise ValueError(f"theta_prior_cov must have shape ({n}, {n}), got {got}")
        if self.omega_prior is not None and self.omega_prior_cov is None:
            raise ValueError("omega_prior_cov must be provided when omega_prior is given.")
        if self.omega_prior_cov is not None and self.omega_prior is None:
            raise ValueError("omega_prior must be provided when omega_prior_cov is given.")
        if self.theta_prior is None and self.omega_prior is None:
            raise ValueError("At least one of theta_prior or omega_prior must be provided.")

    # ── Precision matrix (cached inverse of covariance) ────────────────────────

    @property
    def theta_precision(self) -> np.ndarray | None:
        """
        Inverse of the THETA prior covariance (precision matrix).

        Cached after first computation.
        """
        if self.theta_prior_cov is None:
            return None
        if not hasattr(self, "_theta_precision"):
            try:
                object.__setattr__(
                    self,
                    "_theta_precision",
                    np.linalg.inv(self.theta_prior_cov),
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "theta_prior_cov is singular — cannot invert to get precision."
                ) from exc
        return self._theta_precision  # type: ignore[attr-defined]

    @property
    def omega_precision(self) -> np.ndarray | None:
        """Inverse of the OMEGA prior covariance, or None if not specified."""
        if self.omega_prior_cov is None:
            return None
        if not hasattr(self, "_omega_precision"):
            try:
                object.__setattr__(
                    self,
                    "_omega_precision",
                    np.linalg.inv(self.omega_prior_cov),
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "omega_prior_cov is singular — cannot invert to get precision."
                ) from exc
        return self._omega_precision  # type: ignore[attr-defined]

    # ── Penalty computation ────────────────────────────────────────────────────

    def penalty(
        self,
        theta: np.ndarray,
        omega: np.ndarray | None = None,
    ) -> float:
        """
        Compute the prior penalty (to be added to the data OFV).

        Returns::

            (theta - theta_prior)^T * Sigma_theta^{-1} * (theta - theta_prior)
            [+ (omega_vec - omega_prior)^T * Sigma_omega^{-1} * (omega_vec - omega_prior)]

        Args:
            theta: Current THETA vector, shape ``(n_theta,)``.
            omega: Current OMEGA matrix, shape ``(n_eta, n_eta)``, optional.
                   Required only when ``omega_prior`` is specified.

        Returns:
            Non-negative scalar penalty value.
        """
        pen = 0.0

        if self.theta_prior is not None:
            theta = np.asarray(theta, dtype=np.float64)
            theta_prec = self.theta_precision
            if theta_prec is not None:
                delta_theta = theta - self.theta_prior
                pen += float(delta_theta @ theta_prec @ delta_theta)

        if self.omega_prior is not None and omega is not None:
            omega_prec = self.omega_precision
            if omega_prec is not None:
                # Flatten the lower triangle of OMEGA to match omega_prior shape
                omega_vec = _lower_triangle_vec(np.asarray(omega, dtype=np.float64))
                delta_omega = omega_vec - self.omega_prior
                pen += float(delta_omega @ omega_prec @ delta_omega)

        return pen

    def log_prior(
        self,
        theta: np.ndarray,
        omega: np.ndarray | None = None,
    ) -> float:
        """
        Compute the log prior probability (up to a normalising constant).

        This equals ``-0.5 * penalty(theta, omega)``, consistent with a
        Gaussian prior.

        Args:
            theta: Current THETA vector.
            omega: Current OMEGA matrix (optional).

        Returns:
            Log prior (scalar, <= 0).
        """
        return -0.5 * self.penalty(theta, omega)

    def __repr__(self) -> str:
        return (
            f"PriorSpec("
            f"n_theta={0 if self.theta_prior is None else len(self.theta_prior)}, "
            f"has_omega_prior={self.omega_prior is not None}, "
            f"nwpri={self.nwpri})"
        )


# ── Augmented model wrapper ────────────────────────────────────────────────────


class PriorAugmentedModel:
    """
    Wraps a PopulationModel to add prior penalty to OFV computation.

    Used when $PRIOR is specified.  The prior penalty is added to the data OFV
    without modifying the underlying model, making this a lightweight decorator.

    All attribute access (other than ``ofv`` and ``prior``) is transparently
    delegated to the wrapped ``population_model``.

    Args:
        population_model: The base PopulationModel to wrap.
        prior:            PriorSpec defining the prior distribution.

    Example::

        aug = PriorAugmentedModel(pop_model, prior_spec)
        result = foce_method.estimate(aug, init_params)
    """

    def __init__(self, population_model: Any, prior: PriorSpec) -> None:
        # Store as private to avoid infinite recursion in __getattr__
        object.__setattr__(self, "_population_model", population_model)
        object.__setattr__(self, "_prior", prior)

    @property
    def population_model(self) -> Any:
        """The wrapped base PopulationModel."""
        return object.__getattribute__(self, "_population_model")

    @property
    def prior(self) -> PriorSpec:
        """The prior specification."""
        return object.__getattribute__(self, "_prior")

    def ofv(self, params: Any) -> float:
        """
        Compute the augmented OFV = data OFV + prior penalty.

        Args:
            params: A ParameterSet with ``.theta``, ``.omega``, ``.sigma``.

        Returns:
            Augmented scalar OFV.
        """
        pm = object.__getattribute__(self, "_population_model")
        prior = object.__getattribute__(self, "_prior")

        # Delegate to the underlying model's OFV computation
        data_ofv = pm.ofv_fo(params)
        penalty = prior.penalty(params.theta, params.omega)
        return data_ofv + penalty

    def ofv_fo(self, params: Any) -> float:
        """
        First-order OFV augmented with the prior penalty.

        Equivalent to :meth:`ofv`.
        """
        return self.ofv(params)

    def ofv_foce(self, params: Any, eta_hat: dict[int, Any]) -> float:
        """
        FOCE OFV augmented with the prior penalty.

        Args:
            params:   Current ParameterSet.
            eta_hat:  Dict of post-hoc ETA vectors keyed by subject ID.

        Returns:
            Augmented FOCE OFV.
        """
        pm = object.__getattribute__(self, "_population_model")
        prior = object.__getattribute__(self, "_prior")

        data_ofv = pm.ofv_foce(params, eta_hat)
        penalty = prior.penalty(params.theta, params.omega)
        return data_ofv + penalty

    def __getattr__(self, name: str) -> Any:
        """
        Delegate all other attribute access to the wrapped PopulationModel.

        This allows :class:`PriorAugmentedModel` to be used anywhere a
        ``PopulationModel`` is expected without subclassing.
        """
        pm = object.__getattribute__(self, "_population_model")
        return getattr(pm, name)

    def __repr__(self) -> str:
        pm = object.__getattribute__(self, "_population_model")
        prior = object.__getattribute__(self, "_prior")
        return f"PriorAugmentedModel(model={pm!r}, prior={prior!r})"


# ── Utility helpers ────────────────────────────────────────────────────────────


def _lower_triangle_vec(mat: np.ndarray) -> np.ndarray:
    """
    Extract the lower-triangle elements of a symmetric matrix as a 1-D vector.

    Follows the NONMEM column-major convention (column then row within block).

    Args:
        mat: Square symmetric matrix.

    Returns:
        1-D array of lower-triangle elements (including diagonal).
    """
    n = mat.shape[0]
    elements: list[float] = []
    for col in range(n):
        for row in range(col, n):
            elements.append(float(mat[row, col]))
    return np.array(elements, dtype=np.float64)


def make_theta_prior(
    theta_mean: np.ndarray | list[float],
    theta_cv: np.ndarray | list[float] | float,
) -> PriorSpec:
    """
    Convenience constructor: build a ``PriorSpec`` from a THETA mean and
    coefficient of variation (CV).

    The prior covariance for THETA *i* is ``(theta_mean[i] * cv[i])^2``.

    Args:
        theta_mean: Prior mean vector.
        theta_cv:   CV (as a fraction, e.g., 0.3 for 30%) per THETA, or a
                    single scalar applied to all.

    Returns:
        PriorSpec with a diagonal covariance matrix.
    """
    mu = np.asarray(theta_mean, dtype=np.float64)
    n = len(mu)
    if np.isscalar(theta_cv):
        cv = np.full(n, float(np.asarray(theta_cv, dtype=float)))
    else:
        cv = np.asarray(theta_cv, dtype=np.float64)
        if len(cv) != n:
            raise ValueError(f"theta_cv length ({len(cv)}) must match theta_mean ({n}).")
    variances = (mu * cv) ** 2
    cov = np.diag(variances)
    return PriorSpec(theta_prior=mu, theta_prior_cov=cov)


def prior_from_control_stream(
    cs: ControlStream,
    params: ParameterSet,
) -> PriorSpec | None:
    """
    Build a supported runtime prior from parsed control-stream records.

    Currently supported subset:

    - ``$THETAP`` + ``$THETAPV`` for Gaussian THETA priors
    - ``$OMEGAP`` + ``$OMEGAPD`` for Gaussian penalties on OMEGA lower-triangle
      elements, with ``$OMEGAPD`` treated as precision-like weights

    Returns ``None`` when no supported prior blocks are present.
    """
    theta_prior = _theta_prior_from_control_stream(cs, params)
    omega_prior = _omega_prior_from_control_stream(cs, params)

    if theta_prior is None and omega_prior is None:
        return None

    prior_rec = getattr(cs, "prior_record", None)
    nwpri = 1 if prior_rec is not None else 0

    return PriorSpec(
        theta_prior=None if theta_prior is None else theta_prior[0],
        theta_prior_cov=None if theta_prior is None else theta_prior[1],
        omega_prior=None if omega_prior is None else omega_prior[0],
        omega_prior_cov=None if omega_prior is None else omega_prior[1],
        nwpri=nwpri,
    )


def _theta_prior_from_control_stream(
    cs: ControlStream,
    params: ParameterSet,
) -> tuple[np.ndarray, np.ndarray] | None:
    thetap = getattr(cs, "thetap_record", None)
    thetapv = getattr(cs, "thetapv_record", None)
    if thetap is None and thetapv is None:
        return None
    if thetap is None or thetapv is None:
        raise ValueError("$THETAP and $THETAPV must both be provided for runtime THETA priors.")

    n_theta = params.n_theta()
    theta_prior = np.asarray(thetap.values, dtype=np.float64)
    if len(theta_prior) != n_theta:
        raise ValueError(f"$THETAP must provide exactly {n_theta} values; got {len(theta_prior)}.")
    theta_cov = _covariance_from_values(thetapv.values, n_theta, "$THETAPV")
    return theta_prior, theta_cov


def _omega_prior_from_control_stream(
    cs: ControlStream,
    params: ParameterSet,
) -> tuple[np.ndarray, np.ndarray] | None:
    omegap = getattr(cs, "omegap_record", None)
    omegapd = getattr(cs, "omegapd_record", None)
    if omegap is None and omegapd is None:
        return None
    if omegap is None or omegapd is None:
        raise ValueError("$OMEGAP and $OMEGAPD must both be provided for runtime OMEGA priors.")

    n_eta = params.n_eta()
    full_size = n_eta * (n_eta + 1) // 2
    diag_indices = _lower_triangle_diag_indices(n_eta)

    omega_values = np.asarray(omegap.values, dtype=np.float64)
    omega_prior = np.zeros(full_size, dtype=np.float64)
    inactive_variance = 1e12

    if len(omega_values) == full_size:
        omega_prior[:] = omega_values
        active_indices = list(range(full_size))
    elif len(omega_values) == n_eta:
        omega_prior[diag_indices] = omega_values
        active_indices = diag_indices
    else:
        raise ValueError(
            f"$OMEGAP must provide either {n_eta} diagonal values or {full_size} lower-triangle values; got {len(omega_values)}."
        )

    weights = _expand_positive_values(
        omegapd.values,
        expected=len(active_indices),
        label="$OMEGAPD",
    )
    omega_cov_diag = np.full(full_size, inactive_variance, dtype=np.float64)
    omega_cov_diag[active_indices] = 1.0 / weights
    omega_cov = np.diag(omega_cov_diag)
    return omega_prior, omega_cov


def _covariance_from_values(values: list[float], n: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    full_size = n * (n + 1) // 2
    if len(arr) == 1:
        if arr[0] <= 0:
            raise ValueError(f"{label} values must be positive.")
        return np.diag(np.full(n, arr[0], dtype=np.float64))
    if len(arr) == n:
        if np.any(arr <= 0):
            raise ValueError(f"{label} values must be positive.")
        return np.diag(arr)
    if len(arr) == full_size:
        cov = omega_from_lower_triangle(arr.tolist(), n)
        return cov
    raise ValueError(
        f"{label} must provide 1 scalar, {n} diagonal values, or {full_size} lower-triangle values; got {len(arr)}."
    )


def _expand_positive_values(
    values: list[float],
    *,
    expected: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        arr = np.full(expected, arr[0], dtype=np.float64)
    elif len(arr) != expected:
        raise ValueError(f"{label} must provide 1 scalar or {expected} values; got {len(arr)}.")
    if np.any(arr <= 0):
        raise ValueError(f"{label} values must be positive.")
    return arr


def _lower_triangle_diag_indices(n: int) -> list[int]:
    indices: list[int] = []
    idx = 0
    for col in range(n):
        for row in range(col, n):
            if row == col:
                indices.append(idx)
            idx += 1
    return indices


# ── Public exports ─────────────────────────────────────────────────────────────

__all__ = [
    "PriorSpec",
    "PriorAugmentedModel",
    "make_theta_prior",
    "prior_from_control_stream",
]
