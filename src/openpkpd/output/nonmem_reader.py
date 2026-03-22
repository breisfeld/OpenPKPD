"""
NONMEM result file reader.

Parses the four standard NONMEM output files and assembles them into a
:class:`NonmemResult` dataclass:

  ``.ext``  — iteration-by-iteration parameter estimates + final values
  ``.lst``  — human-readable listing (problem title, OFV, termination)
  ``.phi``  — individual empirical Bayes estimates (post-hoc ETAs)
  ``.cov``  — parameter covariance matrix

All four methods are independent; pass ``None`` for any file you don't have.

Usage::

    from openpkpd.output.nonmem_reader import read_nonmem_results

    result = read_nonmem_results(
        ext_path="run001.ext",
        lst_path="run001.lst",
        phi_path="run001.phi",
        cov_path="run001.cov",
    )
    print(result.theta_final)
    print(result.ofv)
    print(result.post_hoc_etas)

References
----------
NONMEM 7.5 User's Guide, Part II: Output Files.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class NonmemResult:
    """
    Parsed content from NONMEM output files.

    Attributes:
        theta_final:   Final THETA estimates (1-D array).
        omega_final:   Final OMEGA matrix (2-D array, lower-triangle filled).
        sigma_final:   Final SIGMA matrix (2-D array).
        ofv:           Final objective function value (OFV / -2LL).
        converged:     True when minimisation succeeded.
        method:        Estimation method string parsed from the .lst file.
        termination_message: Minimisation termination string.
        theta_names:   THETA column labels from the .ext file header.
        ofv_history:   List of (iteration, ofv) from the .ext file.
        post_hoc_etas: Dict mapping subject ID → ETA vector (from .phi).
        cov_matrix:    Parameter covariance matrix from .cov (optional).
        cov_names:     Parameter names for rows/cols of cov_matrix.
        se_theta:      Standard errors for THETA from .cov diagonal.
        source_files:  Dict of file paths that were successfully parsed.
    """

    theta_final: np.ndarray = field(default_factory=lambda: np.array([]))
    omega_final: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    sigma_final: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 0))
    ofv: float = float("nan")
    converged: bool = False
    method: str = ""
    termination_message: str = ""
    theta_names: list[str] = field(default_factory=list)
    ofv_history: list[tuple[int, float]] = field(default_factory=list)
    post_hoc_etas: dict[Any, np.ndarray] = field(default_factory=dict)
    cov_matrix: np.ndarray | None = None
    cov_names: list[str] = field(default_factory=list)
    se_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    source_files: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a compact text summary of the parsed results."""
        lines = [
            "NonmemResult",
            f"  Method:       {self.method or '(unknown)'}",
            f"  Converged:    {self.converged}",
            f"  OFV:          {self.ofv:.4f}" if not np.isnan(self.ofv) else "  OFV: (missing)",
            f"  THETA:        {np.round(self.theta_final, 4).tolist()}",
            f"  n_subjects:   {len(self.post_hoc_etas)}",
        ]
        if self.cov_matrix is not None:
            lines.append(f"  SE(THETA):    {np.round(self.se_theta, 4).tolist()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# .ext parser
# ---------------------------------------------------------------------------


def _parse_ext(path: str) -> dict:
    """
    Parse a NONMEM .ext file.

    Returns a dict with keys:
      theta_names, theta_final, omega_vector, sigma_vector, ofv, ofv_history
    """
    lines = Path(path).read_text(errors="replace").splitlines()

    header: list[str] = []
    rows: list[list[str]] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("TABLE NO"):
            in_table = True
            continue
        if not in_table:
            continue
        if stripped.startswith("ITERATION"):
            header = stripped.split()
            continue
        if stripped and header:
            rows.append(stripped.split())

    if not header or not rows:
        return {}

    col_map = {name: idx for idx, name in enumerate(header)}
    iter_idx = col_map.get("ITERATION", 0)
    obj_idx = col_map.get("OBJ", len(header) - 1)

    theta_cols = [i for i, n in enumerate(header) if re.match(r"^THETA\d+$", n)]
    omega_cols = [i for i, n in enumerate(header) if re.match(r"^OMEGA\(", n)]
    sigma_cols = [i for i, n in enumerate(header) if re.match(r"^SIGMA\(", n)]

    theta_names = [header[i] for i in theta_cols]
    ofv_history = []
    final_row: list[str] | None = None

    for row in rows:
        if len(row) <= iter_idx:
            continue
        try:
            it = int(float(row[iter_idx]))
        except ValueError:
            continue
        try:
            ofv_val = float(row[obj_idx])
        except (ValueError, IndexError):
            ofv_val = float("nan")
        ofv_history.append((it, ofv_val))
        if it == -1000000000:
            final_row = row

    if final_row is None and rows:
        final_row = rows[-1]

    def _extract(indices: list[int], row: list[str]) -> list[float]:
        result = []
        for i in indices:
            try:
                result.append(float(row[i]))
            except (ValueError, IndexError):
                result.append(float("nan"))
        return result

    theta_vals = _extract(theta_cols, final_row) if final_row else []
    omega_vals = _extract(omega_cols, final_row) if final_row else []
    sigma_vals = _extract(sigma_cols, final_row) if final_row else []
    final_ofv = float(final_row[obj_idx]) if final_row else float("nan")

    return {
        "theta_names": theta_names,
        "theta_final": np.array(theta_vals),
        "omega_vector": np.array(omega_vals),
        "sigma_vector": np.array(sigma_vals),
        "ofv": final_ofv,
        "ofv_history": ofv_history,
    }


def _vector_to_matrix(vec: np.ndarray) -> np.ndarray:
    """
    Reconstruct a symmetric matrix from a NONMEM lower-triangle vector.

    NONMEM stores OMEGA/SIGMA as OMEGA(1,1), OMEGA(2,1), OMEGA(2,2), …
    (row-major lower triangle including diagonal).
    """
    n_elem = len(vec)
    n = int(round((-1 + (1 + 8 * n_elem) ** 0.5) / 2))
    if n * (n + 1) // 2 != n_elem:
        return np.diag(vec)
    mat = np.zeros((n, n))
    idx = 0
    for row in range(n):
        for col in range(row + 1):
            mat[row, col] = vec[idx]
            mat[col, row] = vec[idx]
            idx += 1
    return mat


# ---------------------------------------------------------------------------
# .lst parser
# ---------------------------------------------------------------------------

_CONVERGED_PATTERNS = [
    r"MINIMIZATION\s+SUCCESSFUL",
    r"CONVERGENCE",
]
_FAILED_PATTERNS = [
    r"MINIMIZATION\s+TERMINATED",
    r"MINIMIZATION\s+ABORTED",
    r"ROUNDING\s+ERRORS",
]
_METHOD_PATTERN = re.compile(r"ESTIMATION\s+METHOD\s+USED:\s*(.+)", re.IGNORECASE)
_OFV_PATTERN = re.compile(r"#OBJV\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_TITLE_PATTERN = re.compile(r"\$PROB(?:LEM)?\s+(.+)", re.IGNORECASE)


def _parse_lst(path: str) -> dict:
    """Parse a NONMEM .lst listing file."""
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    converged = False
    termination_message = ""
    method = ""
    ofv = float("nan")
    title = ""

    for line in lines:
        stripped = line.strip()
        for pat in _CONVERGED_PATTERNS:
            if re.search(pat, stripped, re.IGNORECASE):
                converged = True
                termination_message = stripped
                break
        for pat in _FAILED_PATTERNS:
            if re.search(pat, stripped, re.IGNORECASE):
                converged = False
                if not termination_message:
                    termination_message = stripped
                break
        m = _METHOD_PATTERN.search(stripped)
        if m:
            method = m.group(1).strip()
        m = _OFV_PATTERN.search(stripped)
        if m:
            with contextlib.suppress(ValueError):
                ofv = float(m.group(1))
        m = _TITLE_PATTERN.match(stripped)
        if m and not title:
            title = m.group(1).strip()

    return {
        "converged": converged,
        "termination_message": termination_message,
        "method": method,
        "ofv": ofv,
        "title": title,
    }


# ---------------------------------------------------------------------------
# .phi parser
# ---------------------------------------------------------------------------


def _parse_phi(path: str) -> dict[Any, np.ndarray]:
    """
    Parse a NONMEM .phi file.

    Returns a dict mapping subject ID → ETA numpy array.
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    header: list[str] = []
    data: dict[Any, np.ndarray] = {}

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("TABLE"):
            continue
        if "ID" in stripped.upper() and "ETA" in stripped.upper():
            header = stripped.split()
            continue
        if not header or not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            id_val = int(float(parts[1])) if len(header) > 1 else int(float(parts[0]))
        except ValueError:
            continue
        eta_cols = [i for i, name in enumerate(header) if name.startswith("ETA")]
        eta_vals = []
        for col in eta_cols:
            try:
                eta_vals.append(float(parts[col]))
            except (ValueError, IndexError):
                eta_vals.append(float("nan"))
        data[id_val] = np.array(eta_vals)

    return data


# ---------------------------------------------------------------------------
# .cov parser
# ---------------------------------------------------------------------------


def _parse_cov(path: str) -> dict:
    """
    Parse a NONMEM .cov covariance matrix file.

    Returns {'names': [...], 'matrix': np.ndarray}.
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    names: list[str] = []
    rows: list[list[float]] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("TABLE"):
            in_table = True
            continue
        if not in_table or not stripped:
            continue
        parts = stripped.split()
        if parts and parts[0].upper() == "NAME":
            names = parts[1:]
            continue
        if names and parts:
            parts[0]
            try:
                row_vals = [float(v) for v in parts[1:]]
                rows.append(row_vals)
            except ValueError:
                continue

    if not names or not rows:
        return {}

    n = len(names)
    mat = np.full((n, n), float("nan"))
    for i, row in enumerate(rows):
        if i >= n:
            break
        for j, val in enumerate(row):
            if j < n:
                mat[i, j] = val

    # Symmetrise (NONMEM cov may be lower or full triangular)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(mat[i, j]) and not np.isnan(mat[j, i]):
                mat[i, j] = mat[j, i]
            elif np.isnan(mat[j, i]) and not np.isnan(mat[i, j]):
                mat[j, i] = mat[i, j]

    return {"names": names, "matrix": mat}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_nonmem_results(
    ext_path: str | None = None,
    lst_path: str | None = None,
    phi_path: str | None = None,
    cov_path: str | None = None,
) -> NonmemResult:
    """
    Parse NONMEM output files and return a unified :class:`NonmemResult`.

    Any of the four file paths may be ``None`` (or the file may not exist);
    missing files are silently skipped.

    Args:
        ext_path: Path to the ``.ext`` file (parameter history).
        lst_path: Path to the ``.lst`` listing file.
        phi_path: Path to the ``.phi`` file (individual ETAs).
        cov_path: Path to the ``.cov`` covariance file.

    Returns:
        :class:`NonmemResult` populated with all successfully parsed data.
    """
    result = NonmemResult()

    if ext_path and Path(ext_path).exists():
        try:
            ext = _parse_ext(ext_path)
            result.theta_final = ext.get("theta_final", np.array([]))
            result.theta_names = ext.get("theta_names", [])
            result.ofv_history = ext.get("ofv_history", [])
            if not np.isnan(ext.get("ofv", float("nan"))):
                result.ofv = ext["ofv"]
            omega_vec = ext.get("omega_vector", np.array([]))
            if len(omega_vec) > 0:
                result.omega_final = _vector_to_matrix(omega_vec)
            sigma_vec = ext.get("sigma_vector", np.array([]))
            if len(sigma_vec) > 0:
                result.sigma_final = _vector_to_matrix(sigma_vec)
            result.source_files["ext"] = ext_path
        except Exception:
            pass

    if lst_path and Path(lst_path).exists():
        try:
            lst = _parse_lst(lst_path)
            result.converged = lst.get("converged", False)
            result.termination_message = lst.get("termination_message", "")
            result.method = lst.get("method", "")
            if np.isnan(result.ofv) and not np.isnan(lst.get("ofv", float("nan"))):
                result.ofv = lst["ofv"]
            result.source_files["lst"] = lst_path
        except Exception:
            pass

    if phi_path and Path(phi_path).exists():
        try:
            result.post_hoc_etas = _parse_phi(phi_path)
            result.source_files["phi"] = phi_path
        except Exception:
            pass

    if cov_path and Path(cov_path).exists():
        try:
            cov = _parse_cov(cov_path)
            if cov:
                result.cov_matrix = cov.get("matrix")
                result.cov_names = cov.get("names", [])
                if result.cov_matrix is not None:
                    # Extract SE(THETA) = sqrt(diag) for THETA parameters
                    theta_indices = [
                        i for i, name in enumerate(result.cov_names) if name.startswith("THETA")
                    ]
                    diag = np.diag(result.cov_matrix)
                    se_vals = []
                    for idx in theta_indices:
                        v = diag[idx]
                        se_vals.append(float(np.sqrt(max(v, 0.0))))
                    result.se_theta = np.array(se_vals)
            result.source_files["cov"] = cov_path
        except Exception:
            pass

    return result
