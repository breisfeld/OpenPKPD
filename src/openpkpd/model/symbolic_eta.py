"""Narrow SymPy derivative-kernel support for selected analytical PK templates."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

import numpy as np

from openpkpd.model.derivative_kernels import (
    BaseSubjectDerivativeKernel,
    DerivativeKernelCapabilities,
)
from openpkpd.pk.analytical.advan3 import _biexp_central, _eigenvalues
from openpkpd.pk.analytical.advan4 import _triexp_oral
from openpkpd.utils.constants import LOG2PI, BLQMethod

try:
    import sympy as sp
    from sympy.printing.numpy import NumPyPrinter
except Exception:  # pragma: no cover - optional dependency
    sp = None
    NumPyPrinter = None

SYMPY_AVAILABLE = sp is not None
_KA_K_TOL = 1e-12
# Minimum variance floor applied to symbolic gradient variance estimates.
# Prevents division-by-zero in normalisation when predicted variance is near zero
# (e.g., at very early time points or for near-zero residual error models).
# Value chosen as ~1e-10 of typical sigma^2 ≈ 0.01–0.1 in population PK.
_VAR_FLOOR = 1e-10
_SYMBOLIC_SOURCE_CACHE_SCHEMA = "20260316j"
_PK_LINE_RE = re.compile(
    r"^([A-Z0-9]+)\s*=\s*theta\[(\d+)\]\s*\*\s*math\.exp\s*\(\s*eta\[(\d+)\]\s*\)\s*$",
    re.IGNORECASE,
)
_PK_THETA_LINE_RE = re.compile(r"^([A-Z0-9]+)\s*=\s*theta\[(\d+)\]\s*$", re.IGNORECASE)
_W_PROP_THETA_RE = re.compile(r"^w=f\*theta\[(\d+)\]$", re.IGNORECASE)
_W_THETA_RE = re.compile(r"^w=theta\[(\d+)\]$", re.IGNORECASE)
_W_SQRT_RE = re.compile(
    r"^w=math\.sqrt\(theta\[(\d+)\]\*\*2\+\(f\*theta\[(\d+)\]\)\*\*2\)$",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FLOAT_RE = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_PK_COV_POWER_RE = re.compile(
    rf"^([A-Z0-9]+)\s*=\s*\1\s*\*\s*\(([A-Z_][A-Z0-9_]*)\s*/\s*({_FLOAT_RE})\)\s*\*\*\s*theta\[(\d+)\]\s*$",
    re.IGNORECASE,
)
_PK_COV_LINEAR_RE = re.compile(
    rf"^([A-Z0-9]+)\s*=\s*\1\s*\*\s*\(1\s*\+\s*theta\[(\d+)\]\s*\*\s*\(([A-Z_][A-Z0-9_]*)\s*-\s*({_FLOAT_RE})\)\)\s*$",
    re.IGNORECASE,
)
_PK_COV_EXP_RE = re.compile(
    rf"^([A-Z0-9]+)\s*=\s*\1\s*\*\s*math\.exp\s*\(\s*theta\[(\d+)\]\s*\*\s*\(([A-Z_][A-Z0-9_]*)\s*-\s*({_FLOAT_RE})\)\s*\)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _CovariateAdjustment:
    kind: str
    theta_idx: int
    covariate_name: str
    reference: float


def _parse_pk_source(source: str, names: tuple[str, ...]) -> dict[str, tuple[int, int]] | None:
    wanted = {name.upper() for name in names}
    mapping: dict[str, tuple[int, int]] = {}
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if len(lines) != len(names):
        return None
    for line in lines:
        match = _PK_LINE_RE.fullmatch(line)
        if match is None:
            return None
        name = match.group(1).upper()
        if name not in wanted or name in mapping:
            return None
        mapping[name] = (int(match.group(2)), int(match.group(3)))
    return mapping if set(mapping) == wanted else None


def _parse_covariate_adjustment_line(line: str) -> tuple[str, _CovariateAdjustment] | None:
    power_match = _PK_COV_POWER_RE.fullmatch(line)
    if power_match is not None:
        return power_match.group(1).upper(), _CovariateAdjustment(
            kind="power",
            covariate_name=power_match.group(2).upper(),
            reference=float(power_match.group(3)),
            theta_idx=int(power_match.group(4)),
        )
    linear_match = _PK_COV_LINEAR_RE.fullmatch(line)
    if linear_match is not None:
        return linear_match.group(1).upper(), _CovariateAdjustment(
            kind="linear",
            theta_idx=int(linear_match.group(2)),
            covariate_name=linear_match.group(3).upper(),
            reference=float(linear_match.group(4)),
        )
    exp_match = _PK_COV_EXP_RE.fullmatch(line)
    if exp_match is not None:
        return exp_match.group(1).upper(), _CovariateAdjustment(
            kind="exponential",
            theta_idx=int(exp_match.group(2)),
            covariate_name=exp_match.group(3).upper(),
            reference=float(exp_match.group(4)),
        )
    return None


def _parse_pk_source_with_static_covariates(
    source: str,
    names: tuple[str, ...],
) -> dict[str, tuple[int, int, tuple[_CovariateAdjustment, ...]]] | None:
    wanted = {name.upper() for name in names}
    mapping: dict[str, tuple[int, int]] = {}
    adjustments: dict[str, list[_CovariateAdjustment]] = {name.upper(): [] for name in names}
    for line in [line.strip() for line in source.splitlines() if line.strip()]:
        base_match = _PK_LINE_RE.fullmatch(line)
        if base_match is not None:
            name = base_match.group(1).upper()
            if name in wanted and name not in mapping:
                mapping[name] = (int(base_match.group(2)), int(base_match.group(3)))
                continue
        cov_adjustment = _parse_covariate_adjustment_line(line)
        if cov_adjustment is None:
            return None
        name, adjustment = cov_adjustment
        if name not in wanted:
            return None
        adjustments[name].append(adjustment)
    if set(mapping) != wanted:
        return None
    return {
        name.upper(): (
            mapping[name.upper()][0],
            mapping[name.upper()][1],
            tuple(adjustments[name.upper()]),
        )
        for name in names
    }


def _parse_advan4_explicit_pk_source(source: str) -> dict[str, tuple[int, int | None]] | None:
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if len(lines) != 8:
        return None
    exp_specs = []
    for expected, line in zip(("KA", "CL", "V2"), lines[:3], strict=False):
        match = _PK_LINE_RE.fullmatch(line)
        if match is None or match.group(1).upper() != expected:
            return None
        exp_specs.append((expected, int(match.group(2)), int(match.group(3))))
    const_specs = []
    for expected, line in zip(("Q", "V3"), lines[3:5], strict=False):
        match = _PK_THETA_LINE_RE.fullmatch(line)
        if match is None or match.group(1).upper() != expected:
            return None
        const_specs.append((expected, int(match.group(2))))
    if tuple("".join(line.lower().split()) for line in lines[5:]) != (
        "k=cl/v2",
        "k12=q/v2",
        "k21=q/v3",
    ):
        return None
    result: dict[str, tuple[int, int | None]] = {
        name: (theta_idx, eta_idx) for name, theta_idx, eta_idx in exp_specs
    }
    result.update({name: (theta_idx, None) for name, theta_idx in const_specs})
    theta_positions = [theta_idx for theta_idx, _eta_idx in result.values()]
    eta_positions = [eta_idx for _theta_idx, eta_idx in result.values() if eta_idx is not None]
    if len(set(theta_positions)) != len(theta_positions) or len(set(eta_positions)) != len(
        eta_positions
    ):
        return None
    return result


def _parse_advan3_trans1_pk_source(source: str) -> dict[str, tuple[int, int | None]] | None:
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if len(lines) != 7:
        return None
    result = _parse_ordered_mixed_pk_source("\n".join(lines[:4]), ("CL", "V1", "Q", "V2"))
    if result is None:
        return None
    expected_tail = ("k=cl/v1", "k12=q/v1", "k21=q/v2")
    if tuple("".join(line.lower().split()) for line in lines[4:]) != expected_tail:
        return None
    return result


def _parse_ordered_mixed_pk_source(
    source: str, names: tuple[str, ...]
) -> dict[str, tuple[int, int | None]] | None:
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if len(lines) != len(names):
        return None
    result: dict[str, tuple[int, int | None]] = {}
    theta_positions: list[int] = []
    eta_positions: list[int] = []
    seen_fixed = False
    for expected, line in zip(names, lines, strict=False):
        exp_match = _PK_LINE_RE.fullmatch(line)
        if exp_match is not None and exp_match.group(1).upper() == expected:
            if seen_fixed:
                return None
            theta_idx = int(exp_match.group(2))
            eta_idx = int(exp_match.group(3))
            result[expected] = (theta_idx, eta_idx)
            theta_positions.append(theta_idx)
            eta_positions.append(eta_idx)
            continue
        theta_match = _PK_THETA_LINE_RE.fullmatch(line)
        if theta_match is not None and theta_match.group(1).upper() == expected:
            theta_idx = int(theta_match.group(2))
            result[expected] = (theta_idx, None)
            theta_positions.append(theta_idx)
            seen_fixed = True
            continue
        return None
    if len(set(theta_positions)) != len(theta_positions) or len(set(eta_positions)) != len(
        eta_positions
    ):
        return None
    if eta_positions != list(range(len(eta_positions))):
        return None
    return result


def _normalize_lines(source: str) -> tuple[str, ...]:
    lines = ["".join(line.lower().split()) for line in source.splitlines() if line.strip()]
    normalized: list[str] = []
    for line in lines:
        if line == "ipred=f":
            continue
        normalized.append(re.sub(r"\bipred\b", "f", line))
    return tuple(normalized)


def _parse_error_source(source: str, n_eps: int) -> tuple[str, tuple[int, ...]] | None:
    lines = _normalize_lines(source)
    if n_eps == 1:
        if lines in {("y=f*(1+eps[0])",), ("y=f+f*eps[0]",)}:
            return "proportional", ()
        if lines == ("y=f+eps[0]",):
            return "additive", ()
        if len(lines) == 2:
            prop_match = _W_PROP_THETA_RE.fullmatch(lines[0])
            if prop_match is not None and lines[1] == "y=f+w*eps[0]":
                return "proportional_theta", (int(prop_match.group(1)),)
            w_match = _W_THETA_RE.fullmatch(lines[0])
            if w_match is not None and lines[1] == "y=f+w*eps[0]":
                return "additive_theta", (int(w_match.group(1)),)
        if 2 <= len(lines) <= 4:
            sqrt_match = _W_SQRT_RE.fullmatch(lines[0])
            if sqrt_match is not None and lines[1:] in {
                ("y=f+w*eps[0]",),
                ("y=f+w*eps[0]", "ires=dv-f", "iwres=ires/w"),
            }:
                return "combined_theta", (int(sqrt_match.group(1)), int(sqrt_match.group(2)))
    elif n_eps == 2 and lines == ("y=f+eps[0]+f*eps[1]",):
        return "combined_eps", ()
    return None


def _variance_coefficients(
    error_model: str,
    error_theta_idx: tuple[int, ...],
    theta: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float, float]:
    sigma_arr = np.asarray(sigma, dtype=float)
    sigma00 = float(sigma_arr[0, 0]) if sigma_arr.size > 0 else 1.0
    if error_model == "additive":
        return sigma00, 0.0, 0.0
    if error_model == "proportional":
        return 0.0, 0.0, sigma00
    if error_model == "proportional_theta":
        w = float(theta[error_theta_idx[0]])
        return 0.0, 0.0, sigma00 * w * w
    if error_model == "additive_theta":
        w = float(theta[error_theta_idx[0]])
        return sigma00 * w * w, 0.0, 0.0
    if error_model == "combined_theta":
        add_w = float(theta[error_theta_idx[0]])
        prop_w = float(theta[error_theta_idx[1]])
        return sigma00 * add_w * add_w, 0.0, sigma00 * prop_w * prop_w
    if error_model == "combined_eps":
        sigma01 = (
            float(sigma_arr[0, 1]) if sigma_arr.shape[0] > 1 and sigma_arr.shape[1] > 1 else 0.0
        )
        sigma11 = (
            float(sigma_arr[1, 1]) if sigma_arr.shape[0] > 1 and sigma_arr.shape[1] > 1 else sigma00
        )
        return sigma00, 2.0 * sigma01, sigma11
    raise RuntimeError(f"unsupported symbolic error model: {error_model}")


def _common_symbolic_build_guards(
    indiv: Any, *, allow_pk_covariate_references: bool = False
) -> bool:
    if indiv.pk_callable is None or indiv.error_callable is None:
        logger.debug(
            "Symbolic gradient unavailable: pk_callable or error_callable is None "
            "(pk_callable=%s, error_callable=%s)",
            indiv.pk_callable,
            indiv.error_callable,
        )
        return False
    if (
        indiv.occasion_indices is not None
        or indiv.blq_method != BLQMethod.M1
        or indiv.lloq is not None
    ):
        logger.debug(
            "Symbolic gradient unavailable: IOV or non-M1 BLQ active "
            "(occasion_indices=%s, blq_method=%s, lloq=%s)",
            indiv.occasion_indices is not None,
            indiv.blq_method,
            indiv.lloq,
        )
        return False
    if indiv.des_callable is not None or indiv._error_requires_amounts:
        logger.debug(
            "Symbolic gradient unavailable: des_callable or error_requires_amounts active "
            "(des_callable=%s, error_requires_amounts=%s)",
            indiv.des_callable is not None,
            indiv._error_requires_amounts,
        )
        return False

    pk_source = getattr(indiv.pk_callable, "_source", None)
    error_source = getattr(indiv.error_callable, "_source", None)
    if not isinstance(pk_source, str) or not isinstance(error_source, str):
        logger.debug(
            "Symbolic gradient unavailable: pk_callable or error_callable has no _source "
            "(pk_source type=%s, error_source type=%s)",
            type(pk_source).__name__,
            type(error_source).__name__,
        )
        return False

    covariate_names = _collect_covariate_names(indiv)
    if not covariate_names:
        return True
    if _source_uses_covariate_names(error_source, covariate_names):
        logger.debug(
            "Symbolic gradient unavailable: error_callable references covariate(s) %s",
            covariate_names,
        )
        return False
    if not allow_pk_covariate_references and _source_uses_covariate_names(
        pk_source, covariate_names
    ):
        logger.debug(
            "Symbolic gradient unavailable: pk_callable references covariate(s) %s "
            "(allow_pk_covariate_references=%s)",
            covariate_names,
            allow_pk_covariate_references,
        )
        return False
    return True


def _collect_covariate_names(indiv: Any) -> tuple[str, ...]:
    names: set[str] = set()
    base_covariates = getattr(indiv, "_base_covariates", None) or {}
    names.update(str(name) for name in base_covariates)
    for covariates in getattr(indiv, "_observation_covariates", ()):
        names.update(str(name) for name in covariates)
    covariate_df = getattr(getattr(indiv, "subject_events", None), "covariate_df", None)
    if covariate_df is not None:
        names.update(str(name) for name in covariate_df.columns if str(name).upper() != "TIME")
    return tuple(sorted(names))


def _source_uses_covariate_names(source: str, covariate_names: tuple[str, ...]) -> bool:
    if not covariate_names:
        return False
    identifiers = {name.upper() for name in _IDENTIFIER_RE.findall(source)}
    return any(str(name).upper() in identifiers for name in covariate_names)


def _static_covariate_values(indiv: Any) -> dict[str, float] | None:
    values: dict[str, float] = {}

    def _merge_covariates(raw_covariates: dict[str, object] | None) -> bool:
        if not raw_covariates:
            return True
        for name, raw_value in raw_covariates.items():
            try:
                value = float(raw_value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return False
            key = str(name).upper()
            existing = values.get(key)
            if existing is None:
                values[key] = value
            elif not np.isclose(existing, value, rtol=1e-8, atol=0.0):
                return False
        return True

    if not _merge_covariates(getattr(indiv, "_base_covariates", None) or {}):
        return None
    for covariates in getattr(indiv, "_observation_covariates", ()):
        if not _merge_covariates(covariates):
            return None

    covariate_df = getattr(getattr(indiv, "subject_events", None), "covariate_df", None)
    if covariate_df is not None:
        for column in covariate_df.columns:
            if str(column).upper() == "TIME":
                continue
            try:
                arr = np.asarray(covariate_df[column], dtype=float)
            except (TypeError, ValueError):
                return None
            if arr.size == 0:
                continue
            if not np.allclose(arr, arr[0], rtol=1e-8, atol=0.0):
                return None
            key = str(column).upper()
            existing = values.get(key)
            if existing is None:
                values[key] = float(arr[0])
            elif not np.isclose(existing, float(arr[0]), rtol=1e-8, atol=0.0):
                return None

    return values


def _observation_arrays(
    indiv: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    doses = tuple(indiv.subject_events.dose_events)
    if any((not dose.is_bolus) or dose.reset or dose.compartment != 1 for dose in doses):
        raise ValueError("unsupported dose configuration")
    active_mask = indiv.subject_events.observation_mask()
    if not np.all(np.isfinite(indiv.subject_events.obs_dv[active_mask])):
        raise ValueError("non-finite DV")
    obs_times = np.asarray(indiv.subject_events.obs_times[active_mask], dtype=float)
    dv = np.asarray(indiv.subject_events.obs_dv[active_mask], dtype=float)
    dose_times = np.asarray([float(dose.time) for dose in doses], dtype=float)
    dose_amts = np.asarray([float(dose.amount) for dose in doses], dtype=float)
    dt_matrix = (
        obs_times[:, None] - dose_times[None, :]
        if len(dose_times)
        else np.empty((len(obs_times), 0), dtype=float)
    )
    positive_mask = dt_matrix > 0.0
    dt_eval_matrix = np.where(positive_mask, dt_matrix, 0.0)
    positive_weight = positive_mask.astype(float, copy=False)
    active_counts = positive_weight.sum(axis=1)
    return dt_matrix, dt_eval_matrix, positive_mask, positive_weight, active_counts, dose_amts, dv


def _param_from_spec(
    theta: np.ndarray, eta: np.ndarray, theta_idx: int, eta_idx: int | None
) -> float:
    base = float(theta[theta_idx])
    return base if eta_idx is None else float(base * np.exp(eta[eta_idx]))


def _covariate_multiplier(
    theta: np.ndarray,
    adjustments: tuple[_CovariateAdjustment, ...],
    static_covariates: dict[str, float],
) -> float:
    multiplier = 1.0
    for adjustment in adjustments:
        cov_value = static_covariates.get(adjustment.covariate_name)
        if cov_value is None:
            raise ValueError(f"missing covariate {adjustment.covariate_name}")
        theta_cov = float(theta[adjustment.theta_idx])
        if adjustment.kind == "power":
            ratio = max(cov_value / adjustment.reference, 1e-10)
            multiplier *= ratio**theta_cov
        elif adjustment.kind == "linear":
            multiplier *= 1.0 + theta_cov * (cov_value - adjustment.reference)
        elif adjustment.kind == "exponential":
            multiplier *= float(np.exp(theta_cov * (cov_value - adjustment.reference)))
        else:  # pragma: no cover - parser guarantees supported kinds
            raise RuntimeError(f"unsupported covariate adjustment kind: {adjustment.kind}")
    return multiplier


def _effective_theta_values(
    theta: np.ndarray,
    theta_idx: tuple[int, ...],
    covariate_adjustments: tuple[tuple[_CovariateAdjustment, ...], ...],
    static_covariates: dict[str, float],
) -> tuple[float, ...]:
    return tuple(
        float(theta[idx]) * _covariate_multiplier(theta, adjustments, static_covariates)
        for idx, adjustments in zip(theta_idx, covariate_adjustments, strict=False)
    )


def _eta_data_objective_values_from_predictions(
    predictions: np.ndarray,
    dv: np.ndarray,
    var_a: float,
    var_b: float,
    var_c: float,
) -> np.ndarray:
    pred_arr = np.asarray(predictions, dtype=float)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr[None, :]
    if pred_arr.shape[0] == 0:
        return np.array([], dtype=float)
    if len(dv) == 0:
        return np.zeros(pred_arr.shape[0], dtype=float)
    raw_var = var_a + var_b * pred_arr + var_c * (pred_arr**2)
    var = np.where(raw_var > _VAR_FLOOR, raw_var, _VAR_FLOOR)
    obs_term = LOG2PI + np.log(var) + (np.asarray(dv, dtype=float)[None, :] - pred_arr) ** 2 / var
    return np.sum(obs_term, axis=1, dtype=float)


def _symbolic_cache_dir() -> Path:
    override = os.getenv("OPENPKPD_SYMBOLIC_CACHE_DIR")
    if override:
        return Path(override)
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    base = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return base / "openpkpd" / "symbolic_eta"


def _symbolic_cache_file(cache_name: str) -> Path:
    digest = hashlib.sha256(
        f"{_SYMBOLIC_SOURCE_CACHE_SCHEMA}|{cache_name}|{getattr(sp, '__version__', 'nosympy')}|{np.__version__}".encode()
    ).hexdigest()[:16]
    return _symbolic_cache_dir() / f"{cache_name}_{digest}.py"


def _existing_symbolic_cache_file(cache_name: str) -> Path | None:
    exact = _symbolic_cache_file(cache_name)
    if exact.exists():
        return exact
    candidates = sorted(_symbolic_cache_dir().glob(f"{cache_name}_*.py"))
    return candidates[-1] if candidates else None


def _symbolic_runtime_available(*cache_names: str) -> bool:
    if SYMPY_AVAILABLE:
        return True
    return all(_existing_symbolic_cache_file(cache_name) is not None for cache_name in cache_names)


def _load_symbolic_functions_from_source(
    source: str, function_names: tuple[str, ...], filename: str
) -> dict[str, Any]:
    namespace = {"numpy": np}
    exec(compile(source, filename, "exec"), namespace, namespace)
    return {name: namespace[name] for name in function_names}


def _generate_symbolic_function_source(
    function_specs: dict[str, tuple[tuple[Any, ...], Any]],
) -> str:
    if NumPyPrinter is None:  # pragma: no cover - guarded by SYMPY_AVAILABLE on build path
        raise RuntimeError("sympy numpy printer is not available")
    printer = NumPyPrinter()
    lines = [
        "# Auto-generated by openpkpd symbolic_eta.py",
        "import numpy",
        "",
    ]
    for name, (args, expr) in function_specs.items():
        arg_list = ", ".join(str(arg) for arg in args)
        lines.append(f"def {name}({arg_list}):")
        exprs = tuple(expr) if isinstance(expr, (tuple, list)) else (expr,)
        replacements, reduced_exprs = sp.cse(exprs, symbols=sp.numbered_symbols("_cse"))
        for symbol, replacement in replacements:
            lines.append(f"    {symbol} = {printer.doprint(replacement)}")
        if len(reduced_exprs) == 1:
            lines.append(f"    return {printer.doprint(reduced_exprs[0])}")
        else:
            rendered = ", ".join(printer.doprint(reduced_expr) for reduced_expr in reduced_exprs)
            lines.append(f"    return ({rendered})")
        lines.append("")
    return "\n".join(lines)


def _write_symbolic_cache_source(path: Path, source: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(source, encoding="utf-8")
        temp_path.replace(path)
    except OSError:
        return


def _compile_or_load_symbolic_functions(
    cache_name: str,
    function_names: tuple[str, ...],
    builder: Any,
) -> dict[str, Any]:
    cache_path = _existing_symbolic_cache_file(cache_name) or _symbolic_cache_file(cache_name)
    if cache_path.exists():
        try:
            source = cache_path.read_text(encoding="utf-8")
            return _load_symbolic_functions_from_source(source, function_names, str(cache_path))
        except Exception:
            pass
    if sp is None:  # pragma: no cover - optional dependency path
        raise RuntimeError("sympy is not available and no symbolic cache source was found")
    function_specs = builder()
    source = _generate_symbolic_function_source(function_specs)
    _write_symbolic_cache_source(cache_path, source)
    return _load_symbolic_functions_from_source(source, function_names, str(cache_path))


def _prediction_cache_key(theta: np.ndarray, eta: np.ndarray) -> tuple[bytes, bytes]:
    theta_arr = np.ascontiguousarray(np.asarray(theta, dtype=float))
    eta_arr = np.ascontiguousarray(np.asarray(eta, dtype=float))
    return theta_arr.tobytes(), eta_arr.tobytes()


def _prediction_cache_lookup(
    kernel: Any,
    theta: np.ndarray,
    eta: np.ndarray,
    *,
    include_second: bool,
) -> tuple[tuple[bytes, bytes], tuple[np.ndarray, np.ndarray, np.ndarray | None] | None]:
    key = _prediction_cache_key(theta, eta)
    if getattr(kernel, "_prediction_cache_key", None) != key:
        return key, None
    f = getattr(kernel, "_prediction_cache_f", None)
    df = getattr(kernel, "_prediction_cache_df", None)
    d2f = getattr(kernel, "_prediction_cache_d2f", None)
    if f is None or df is None:
        return key, None
    if include_second and d2f is None:
        return key, None
    return key, (f, df, d2f)


def _prediction_cache_store(
    kernel: Any,
    key: tuple[bytes, bytes],
    f: np.ndarray,
    df: np.ndarray,
    d2f: np.ndarray | None,
) -> None:
    kernel._prediction_cache_key = key
    kernel._prediction_cache_f = f
    kernel._prediction_cache_df = df
    kernel._prediction_cache_d2f = d2f


def _supports_narrow_theta_gradients(
    error_model: str,
    covariate_adjustments: tuple[tuple[_CovariateAdjustment, ...], ...],
) -> bool:
    return error_model in {"proportional", "additive", "combined_eps"} and not any(
        covariate_adjustments
    )


def _masked_row_sum(
    values: Any, positive_weight: np.ndarray, active_counts: np.ndarray
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return float(arr) * active_counts
    if arr.shape == positive_weight.shape:
        return np.add.reduce(arr * positive_weight, axis=1)
    if arr.ndim == 1 and arr.shape[0] == positive_weight.shape[1]:
        return positive_weight @ arr
    if arr.ndim == 1 and arr.shape[0] == positive_weight.shape[0]:
        return arr * active_counts
    if arr.ndim == 2 and arr.shape == (positive_weight.shape[0], 1):
        return arr[:, 0] * active_counts
    return np.add.reduce(np.broadcast_to(arr, positive_weight.shape) * positive_weight, axis=1)


@lru_cache(maxsize=1)
def _compiled_terms() -> dict[str, Any]:
    function_names = (
        "contrib",
        "contrib_grad_0",
        "contrib_grad_1",
        "contrib_grad_2",
        "obs_term_df",
    )

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        t0, t1, t2, e0, e1, e2, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "t0 t1 t2 e0 e1 e2 dt amt var_a var_b var_c dv f",
            real=True,
        )
        ka = t0 * sp.exp(e0)
        cl = t1 * sp.exp(e1)
        v = t2 * sp.exp(e2)
        k = cl / v
        contrib = amt * ka / (ka - k) * (sp.exp(-k * dt) - sp.exp(-ka * dt)) / v
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (t0, t1, t2, e0, e1, e2, dt, amt)
        return {
            "contrib": (args, contrib),
            "contrib_grad_0": (args, sp.diff(contrib, e0)),
            "contrib_grad_1": (args, sp.diff(contrib, e1)),
            "contrib_grad_2": (args, sp.diff(contrib, e2)),
            "obs_term_df": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f))),
        }

    compiled = _compile_or_load_symbolic_functions("advan2_terms", function_names, _builder)
    return {
        "contrib": compiled["contrib"],
        "contrib_grad": tuple(compiled[f"contrib_grad_{idx}"] for idx in range(3)),
        "obs_term_df": compiled["obs_term_df"],
    }


@lru_cache(maxsize=1)
def _compiled_advan1_terms() -> dict[str, Any]:
    function_names = ("contrib", "contrib_grad_0", "contrib_grad_1", "obs_term_df")

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        t0, t1, e0, e1, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "t0 t1 e0 e1 dt amt var_a var_b var_c dv f",
            real=True,
        )
        cl = t0 * sp.exp(e0)
        v = t1 * sp.exp(e1)
        k = cl / v
        contrib = amt * sp.exp(-k * dt) / v
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (t0, t1, e0, e1, dt, amt)
        return {
            "contrib": (args, contrib),
            "contrib_grad_0": (args, sp.diff(contrib, e0)),
            "contrib_grad_1": (args, sp.diff(contrib, e1)),
            "obs_term_df": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f))),
        }

    compiled = _compile_or_load_symbolic_functions("advan1_terms", function_names, _builder)
    return {
        "contrib": compiled["contrib"],
        "contrib_grad": tuple(compiled[f"contrib_grad_{idx}"] for idx in range(2)),
        "obs_term_df": compiled["obs_term_df"],
    }


@lru_cache(maxsize=1)
def _compiled_advan1_hessian_terms() -> dict[str, Any]:
    function_names = (
        "contrib_hess_00",
        "contrib_hess_01",
        "contrib_hess_10",
        "contrib_hess_11",
        "obs_term_d2f",
    )

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        t0, t1, e0, e1, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "t0 t1 e0 e1 dt amt var_a var_b var_c dv f",
            real=True,
        )
        cl = t0 * sp.exp(e0)
        v = t1 * sp.exp(e1)
        k = cl / v
        contrib = amt * sp.exp(-k * dt) / v
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (t0, t1, e0, e1, dt, amt)
        return {
            "contrib_hess_00": (args, sp.diff(contrib, e0, e0)),
            "contrib_hess_01": (args, sp.diff(contrib, e0, e1)),
            "contrib_hess_10": (args, sp.diff(contrib, e1, e0)),
            "contrib_hess_11": (args, sp.diff(contrib, e1, e1)),
            "obs_term_d2f": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f, 2))),
        }

    compiled = _compile_or_load_symbolic_functions("advan1_hessian_terms", function_names, _builder)
    return {
        "contrib_hess": (
            (compiled["contrib_hess_00"], compiled["contrib_hess_01"]),
            (compiled["contrib_hess_10"], compiled["contrib_hess_11"]),
        ),
        "obs_term_d2f": compiled["obs_term_d2f"],
    }


@lru_cache(maxsize=1)
def _compiled_advan3_terms() -> dict[str, Any]:
    function_names = (
        "contrib",
        "contrib_grad_0",
        "contrib_grad_1",
        "contrib_grad_2",
        "contrib_grad_3",
        "obs_term_df",
    )

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        cl, v1, q, v2, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "cl v1 q v2 dt amt var_a var_b var_c dv f",
            real=True,
        )
        k = cl / v1
        k12 = q / v1
        k21 = q / v2
        s = k + k12 + k21
        disc = sp.sqrt(s**2 - 4 * k * k21)
        lam1 = (s - disc) / 2
        lam2 = (s + disc) / 2
        dl = lam2 - lam1
        contrib = (
            amt
            * ((k21 - lam1) * sp.exp(-lam1 * dt) + (lam2 - k21) * sp.exp(-lam2 * dt))
            / (dl * v1)
        )
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (cl, v1, q, v2, dt, amt)
        operators = (
            lambda ex: cl * sp.diff(ex, cl),
            lambda ex: v1 * sp.diff(ex, v1),
            lambda ex: q * sp.diff(ex, q),
            lambda ex: v2 * sp.diff(ex, v2),
        )
        return {
            "contrib": (args, contrib),
            **{f"contrib_grad_{idx}": (args, op(contrib)) for idx, op in enumerate(operators)},
            "obs_term_df": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f))),
        }

    compiled = _compile_or_load_symbolic_functions("advan3_terms", function_names, _builder)
    return {
        "contrib": compiled["contrib"],
        "contrib_grad": tuple(compiled[f"contrib_grad_{idx}"] for idx in range(4)),
        "obs_term_df": compiled["obs_term_df"],
    }


@lru_cache(maxsize=1)
def _compiled_advan3_hessian_terms() -> dict[str, Any]:
    upper_pairs = tuple((row, col) for row in range(4) for col in range(row, 4))
    function_names = ("contrib_hess_bundle", "obs_term_d2f")

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        cl, v1, q, v2, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "cl v1 q v2 dt amt var_a var_b var_c dv f",
            real=True,
        )
        k = cl / v1
        k12 = q / v1
        k21 = q / v2
        s = k + k12 + k21
        disc = sp.sqrt(s**2 - 4 * k * k21)
        lam1 = (s - disc) / 2
        lam2 = (s + disc) / 2
        dl = lam2 - lam1
        contrib = (
            amt
            * ((k21 - lam1) * sp.exp(-lam1 * dt) + (lam2 - k21) * sp.exp(-lam2 * dt))
            / (dl * v1)
        )
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (cl, v1, q, v2, dt, amt)
        operators = (
            lambda ex: cl * sp.diff(ex, cl),
            lambda ex: v1 * sp.diff(ex, v1),
            lambda ex: q * sp.diff(ex, q),
            lambda ex: v2 * sp.diff(ex, v2),
        )
        return {
            "contrib_hess_bundle": (
                args,
                tuple(operators[row](operators[col](contrib)) for row, col in upper_pairs),
            ),
            "obs_term_d2f": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f, 2))),
        }

    compiled = _compile_or_load_symbolic_functions("advan3_hessian_terms", function_names, _builder)
    return {
        "contrib_hess_bundle": compiled["contrib_hess_bundle"],
        "contrib_hess_pairs": upper_pairs,
        "obs_term_d2f": compiled["obs_term_d2f"],
    }


@lru_cache(maxsize=1)
def _compiled_advan4_terms() -> dict[str, Any]:
    function_names = (
        "contrib",
        "contrib_grad_0",
        "contrib_grad_1",
        "contrib_grad_2",
        "obs_term_df",
    )

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        ka, cl, v2, q, v3, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "ka cl v2 q v3 dt amt var_a var_b var_c dv f",
            real=True,
        )
        k = cl / v2
        k12 = q / v2
        k21 = q / v3
        s = k + k12 + k21
        disc = sp.sqrt(s**2 - 4 * k * k21)
        lam1 = (s - disc) / 2
        lam2 = (s + disc) / 2
        dl = lam2 - lam1
        h1 = (sp.exp(-lam1 * dt) - sp.exp(-ka * dt)) / (ka - lam1)
        h2 = (sp.exp(-lam2 * dt) - sp.exp(-ka * dt)) / (ka - lam2)
        c1 = (k21 - lam1) / dl
        c2 = (lam2 - k21) / dl
        contrib = amt * ka * (c1 * h1 + c2 * h2) / v2
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (ka, cl, v2, q, v3, dt, amt)
        operators = (
            lambda ex: ka * sp.diff(ex, ka),
            lambda ex: cl * sp.diff(ex, cl),
            lambda ex: v2 * sp.diff(ex, v2),
        )
        return {
            "contrib": (args, contrib),
            **{f"contrib_grad_{idx}": (args, op(contrib)) for idx, op in enumerate(operators)},
            "obs_term_df": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f))),
        }

    compiled = _compile_or_load_symbolic_functions("advan4_terms", function_names, _builder)
    return {
        "contrib": compiled["contrib"],
        "contrib_grad": tuple(compiled[f"contrib_grad_{idx}"] for idx in range(3)),
        "obs_term_df": compiled["obs_term_df"],
    }


@lru_cache(maxsize=1)
def _compiled_advan4_hessian_terms() -> dict[str, Any]:
    upper_pairs = tuple((row, col) for row in range(3) for col in range(row, 3))
    function_names = ("contrib_hess_bundle", "obs_term_d2f")

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        ka, cl, v2, q, v3, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "ka cl v2 q v3 dt amt var_a var_b var_c dv f",
            real=True,
        )
        k = cl / v2
        k12 = q / v2
        k21 = q / v3
        s = k + k12 + k21
        disc = sp.sqrt(s**2 - 4 * k * k21)
        lam1 = (s - disc) / 2
        lam2 = (s + disc) / 2
        dl = lam2 - lam1
        h1 = (sp.exp(-lam1 * dt) - sp.exp(-ka * dt)) / (ka - lam1)
        h2 = (sp.exp(-lam2 * dt) - sp.exp(-ka * dt)) / (ka - lam2)
        c1 = (k21 - lam1) / dl
        c2 = (lam2 - k21) / dl
        contrib = amt * ka * (c1 * h1 + c2 * h2) / v2
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (ka, cl, v2, q, v3, dt, amt)
        operators = (
            lambda ex: ka * sp.diff(ex, ka),
            lambda ex: cl * sp.diff(ex, cl),
            lambda ex: v2 * sp.diff(ex, v2),
        )
        return {
            "contrib_hess_bundle": (
                args,
                tuple(operators[row](operators[col](contrib)) for row, col in upper_pairs),
            ),
            "obs_term_d2f": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f, 2))),
        }

    compiled = _compile_or_load_symbolic_functions("advan4_hessian_terms", function_names, _builder)
    return {
        "contrib_hess_bundle": compiled["contrib_hess_bundle"],
        "contrib_hess_pairs": upper_pairs,
        "obs_term_d2f": compiled["obs_term_d2f"],
    }


@lru_cache(maxsize=1)
def _compiled_limit_terms() -> dict[str, Any]:
    function_names = (
        "contrib_limit",
        "contrib_limit_grad_0",
        "contrib_limit_grad_1",
        "contrib_limit_grad_2",
    )

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        ka, k, v, dt, amt, z = sp.symbols("ka k v dt amt z", real=True)
        expr = amt * ka / (ka - k) * (sp.exp(-k * dt) - sp.exp(-ka * dt)) / v
        args = (ka, k, v, dt, amt)

        def _limit(ex: Any) -> Any:
            return sp.simplify(sp.limit(ex.subs(ka, k + z), z, 0))

        operators = (
            lambda ex: sp.simplify(ka * sp.diff(ex, ka)),
            lambda ex: sp.simplify(k * sp.diff(ex, k)),
            lambda ex: sp.simplify(-k * sp.diff(ex, k) + v * sp.diff(ex, v)),
        )
        return {
            "contrib_limit": (args, _limit(expr)),
            "contrib_limit_grad_0": (args, _limit(operators[0](expr))),
            "contrib_limit_grad_1": (args, _limit(operators[1](expr))),
            "contrib_limit_grad_2": (args, _limit(operators[2](expr))),
        }

    compiled = _compile_or_load_symbolic_functions("advan2_limit_terms", function_names, _builder)
    return {
        "contrib_limit": compiled["contrib_limit"],
        "contrib_limit_grad": tuple(compiled[f"contrib_limit_grad_{idx}"] for idx in range(3)),
    }


@lru_cache(maxsize=1)
def _compiled_hessian_terms() -> dict[str, Any]:
    function_names = (
        "contrib_hess_00",
        "contrib_hess_01",
        "contrib_hess_02",
        "contrib_hess_10",
        "contrib_hess_11",
        "contrib_hess_12",
        "contrib_hess_20",
        "contrib_hess_21",
        "contrib_hess_22",
        "obs_term_d2f",
    )

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        t0, t1, t2, e0, e1, e2, dt, amt, var_a, var_b, var_c, dv, f = sp.symbols(
            "t0 t1 t2 e0 e1 e2 dt amt var_a var_b var_c dv f",
            real=True,
        )
        ka = t0 * sp.exp(e0)
        cl = t1 * sp.exp(e1)
        v = t2 * sp.exp(e2)
        k = cl / v
        contrib = amt * ka / (ka - k) * (sp.exp(-k * dt) - sp.exp(-ka * dt)) / v
        obs_var = var_a + var_b * f + var_c * f**2
        obs_term = sp.log(obs_var) + (dv - f) ** 2 / obs_var
        args = (t0, t1, t2, e0, e1, e2, dt, amt)
        return {
            **{
                f"contrib_hess_{row}{col}": (args, sp.diff(contrib, var_i, var_j))
                for row, var_i in enumerate((e0, e1, e2))
                for col, var_j in enumerate((e0, e1, e2))
            },
            "obs_term_d2f": ((f, dv, var_a, var_b, var_c), sp.simplify(sp.diff(obs_term, f, 2))),
        }

    compiled = _compile_or_load_symbolic_functions("advan2_hessian_terms", function_names, _builder)
    return {
        "contrib_hess": tuple(
            tuple(compiled[f"contrib_hess_{row}{col}"] for col in range(3)) for row in range(3)
        ),
        "obs_term_d2f": compiled["obs_term_d2f"],
    }


@lru_cache(maxsize=1)
def _compiled_limit_hessian_terms() -> dict[str, Any]:
    function_names = tuple(f"contrib_limit_hess_{row}{col}" for row in range(3) for col in range(3))

    def _builder() -> dict[str, tuple[tuple[Any, ...], Any]]:
        ka, k, v, dt, amt, z = sp.symbols("ka k v dt amt z", real=True)
        expr = amt * ka / (ka - k) * (sp.exp(-k * dt) - sp.exp(-ka * dt)) / v
        args = (ka, k, v, dt, amt)

        def _limit(ex: Any) -> Any:
            return sp.simplify(sp.limit(ex.subs(ka, k + z), z, 0))

        operators = (
            lambda ex: sp.simplify(ka * sp.diff(ex, ka)),
            lambda ex: sp.simplify(k * sp.diff(ex, k)),
            lambda ex: sp.simplify(-k * sp.diff(ex, k) + v * sp.diff(ex, v)),
        )
        return {
            f"contrib_limit_hess_{row}{col}": (args, _limit(op_i(op_j(expr))))
            for row, op_i in enumerate(operators)
            for col, op_j in enumerate(operators)
        }

    compiled = _compile_or_load_symbolic_functions(
        "advan2_limit_hessian_terms", function_names, _builder
    )
    return {
        "contrib_limit_hess": tuple(
            tuple(compiled[f"contrib_limit_hess_{row}{col}"] for col in range(3))
            for row in range(3)
        )
    }


def _symbolic_compiled_loaders() -> tuple[tuple[str, Any], ...]:
    return (
        ("advan2_terms", _compiled_terms),
        ("advan1_terms", _compiled_advan1_terms),
        ("advan1_hessian_terms", _compiled_advan1_hessian_terms),
        ("advan3_terms", _compiled_advan3_terms),
        ("advan3_hessian_terms", _compiled_advan3_hessian_terms),
        ("advan4_terms", _compiled_advan4_terms),
        ("advan4_hessian_terms", _compiled_advan4_hessian_terms),
        ("advan2_limit_terms", _compiled_limit_terms),
        ("advan2_hessian_terms", _compiled_hessian_terms),
        ("advan2_limit_hessian_terms", _compiled_limit_hessian_terms),
    )


def clear_symbolic_runtime_caches() -> None:
    for _cache_name, loader in _symbolic_compiled_loaders():
        loader.cache_clear()


def prewarm_symbolic_caches() -> list[dict[str, str | bool]]:
    """Compile/load all symbolic helper families and return cache metadata."""
    warmed: list[dict[str, str | bool]] = []
    for cache_name, loader in _symbolic_compiled_loaders():
        loader()
        cache_path = _symbolic_cache_file(cache_name)
        warmed.append(
            {
                "cache_name": cache_name,
                "cache_path": str(cache_path),
                "exists": cache_path.exists(),
            }
        )
    return warmed


@dataclass(slots=True)
class SympyAdvan2Trans2Objective(BaseSubjectDerivativeKernel):
    capabilities = DerivativeKernelCapabilities(
        eta_objective_gradient=True,
        eta_objective_hessian=True,
        prediction_eta_jacobian=True,
        theta_data_objective_gradient=True,
        prediction_theta_jacobian=True,
    )

    theta_idx: tuple[int, int, int]
    eta_idx: tuple[int, int, int]
    covariate_adjustments: tuple[
        tuple[_CovariateAdjustment, ...],
        tuple[_CovariateAdjustment, ...],
        tuple[_CovariateAdjustment, ...],
    ]
    static_covariates: dict[str, float]
    error_model: str
    error_theta_idx: tuple[int, ...]
    dt_matrix: np.ndarray
    dt_eval_matrix: np.ndarray
    positive_mask: np.ndarray
    positive_weight: np.ndarray
    active_counts: np.ndarray
    dose_amounts: np.ndarray
    dv: np.ndarray
    _prediction_cache_key: tuple[bytes, bytes] | None = field(default=None, init=False, repr=False)
    _prediction_cache_f: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_df: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_d2f: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def build(cls, indiv: Any, trans: int) -> SympyAdvan2Trans2Objective | None:
        if (
            not _symbolic_runtime_available(
                "advan2_terms",
                "advan2_limit_terms",
                "advan2_hessian_terms",
                "advan2_limit_hessian_terms",
            )
            or trans != 2
            or getattr(indiv.pk_subroutine, "advan", None) != 2
        ):
            return None
        if not _common_symbolic_build_guards(indiv, allow_pk_covariate_references=True):
            return None
        pk_source = getattr(indiv.pk_callable, "_source", None)
        error_source = getattr(indiv.error_callable, "_source", None)
        if not isinstance(pk_source, str) or not isinstance(error_source, str):
            return None
        mapping = _parse_pk_source_with_static_covariates(pk_source, ("KA", "CL", "V"))
        error_model = _parse_error_source(error_source, indiv.n_eps)
        if mapping is None or error_model is None:
            return None
        error_kind, error_theta_idx = error_model
        theta_idx = cast(
            tuple[int, int, int], tuple(mapping[name][0] for name in ("KA", "CL", "V"))
        )
        eta_idx = cast(tuple[int, int, int], tuple(mapping[name][1] for name in ("KA", "CL", "V")))
        covariate_adjustments = cast(
            tuple[
                tuple[_CovariateAdjustment, ...],
                tuple[_CovariateAdjustment, ...],
                tuple[_CovariateAdjustment, ...],
            ],
            tuple(mapping[name][2] for name in ("KA", "CL", "V")),
        )
        if sorted(theta_idx) != [0, 1, 2] or sorted(eta_idx) != [0, 1, 2]:
            return None
        static_covariates = _static_covariate_values(indiv) if any(covariate_adjustments) else {}
        if static_covariates is None:
            return None
        try:
            (
                dt_matrix,
                dt_eval_matrix,
                positive_mask,
                positive_weight,
                active_counts,
                dose_amounts,
                dv,
            ) = _observation_arrays(indiv)
        except ValueError:
            return None
        return cls(
            theta_idx=theta_idx,
            eta_idx=eta_idx,
            covariate_adjustments=covariate_adjustments,
            static_covariates=static_covariates,
            error_model=error_kind,
            error_theta_idx=error_theta_idx,
            dt_matrix=dt_matrix,
            dt_eval_matrix=dt_eval_matrix,
            positive_mask=positive_mask,
            positive_weight=positive_weight,
            active_counts=active_counts,
            dose_amounts=dose_amounts,
            dv=dv,
        )

    def _variance_coefficients(
        self, theta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, float, float]:
        return _variance_coefficients(self.error_model, self.error_theta_idx, theta, sigma)

    def _prediction_mean_direct(self, theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
        if self.dt_matrix.shape[0] == 0:
            return np.array([], dtype=float)
        theta_vals = _effective_theta_values(
            theta, self.theta_idx, self.covariate_adjustments, self.static_covariates
        )
        ka = theta_vals[0] * np.exp(float(eta[self.eta_idx[0]]))
        v = theta_vals[2] * np.exp(float(eta[self.eta_idx[2]]))
        k = theta_vals[1] * np.exp(float(eta[self.eta_idx[1]])) / v
        dt = self.dt_eval_matrix
        if abs(ka - k) < _KA_K_TOL:
            contrib = self.dose_amounts * ka * dt * np.exp(-k * dt) / v
        else:
            contrib = self.dose_amounts * ka * (np.exp(-k * dt) - np.exp(-ka * dt)) / (v * (ka - k))
        return _masked_row_sum(contrib, self.positive_weight, self.active_counts)

    def _prediction_mean_and_partials(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        *,
        include_second: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        cache_key, cached = _prediction_cache_lookup(
            self, theta, eta, include_second=include_second
        )
        if cached is not None:
            return cached
        first_order_cached = None
        if include_second:
            _, first_order_cached = _prediction_cache_lookup(self, theta, eta, include_second=False)
        theta_vals = _effective_theta_values(
            theta, self.theta_idx, self.covariate_adjustments, self.static_covariates
        )
        eta_vals = tuple(float(eta[i]) for i in self.eta_idx)
        if self.dt_matrix.shape[0] == 0:
            second = np.zeros((0, 3, 3), dtype=float) if include_second else None
            empty = (np.array([], dtype=float), np.zeros((0, 3), dtype=float), second)
            _prediction_cache_store(self, cache_key, empty[0], empty[1], empty[2])
            return empty
        ka = theta_vals[0] * np.exp(eta_vals[0])
        k = theta_vals[1] * np.exp(eta_vals[1]) / (theta_vals[2] * np.exp(eta_vals[2]))
        dt = self.dt_eval_matrix
        use_limit = abs(ka - k) < _KA_K_TOL
        if first_order_cached is not None:
            f, df, _ = first_order_cached
        else:
            if use_limit:
                limit_terms = _compiled_limit_terms()
                v = theta_vals[2] * np.exp(eta_vals[2])
                contrib = np.asarray(
                    limit_terms["contrib_limit"](ka, k, v, dt, self.dose_amounts), dtype=float
                )
                grad_terms = limit_terms["contrib_limit_grad"]
            else:
                terms = _compiled_terms()
                contrib = np.asarray(
                    terms["contrib"](*theta_vals, *eta_vals, dt, self.dose_amounts), dtype=float
                )
                grad_terms = terms["contrib_grad"]
            f = _masked_row_sum(contrib, self.positive_weight, self.active_counts)
            df = np.empty((self.dt_matrix.shape[0], 3), dtype=float)
            for col, grad_fn in enumerate(grad_terms):
                grad_matrix = np.asarray(
                    grad_fn(ka, k, theta_vals[2] * np.exp(eta_vals[2]), dt, self.dose_amounts)
                    if use_limit
                    else grad_fn(*theta_vals, *eta_vals, dt, self.dose_amounts),
                    dtype=float,
                )
                df[:, col] = _masked_row_sum(grad_matrix, self.positive_weight, self.active_counts)
        d2f = None
        if include_second:
            if use_limit:
                hess_terms = _compiled_limit_hessian_terms()["contrib_limit_hess"]
            else:
                hess_terms = _compiled_hessian_terms()["contrib_hess"]
            d2f = np.empty((self.dt_matrix.shape[0], 3, 3), dtype=float)
            for row, hess_row in enumerate(hess_terms):
                for col in range(row, len(hess_row)):
                    hess_fn = hess_row[col]
                    hess_matrix = np.asarray(
                        hess_fn(ka, k, theta_vals[2] * np.exp(eta_vals[2]), dt, self.dose_amounts)
                        if use_limit
                        else hess_fn(*theta_vals, *eta_vals, dt, self.dose_amounts),
                        dtype=float,
                    )
                    d2f[:, row, col] = _masked_row_sum(
                        hess_matrix, self.positive_weight, self.active_counts
                    )
                    if col != row:
                        d2f[:, col, row] = d2f[:, row, col]
        _prediction_cache_store(self, cache_key, f, df, d2f)
        return f, df, d2f

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        _f, df_used, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df_used.shape[0], len(eta)), dtype=float)
        for col, pos in enumerate(self.eta_idx):
            jac[:, pos] = df_used[:, col]
        return jac

    def supports_theta_data_objective_gradient(self) -> bool:
        return _supports_narrow_theta_gradients(self.error_model, self.covariate_adjustments)

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError("theta Jacobian is only supported on the narrow analytical subset")
        _f, df_used, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df_used.shape[0], len(theta)), dtype=float)
        for col, pos in enumerate(self.theta_idx):
            theta_val = float(theta[pos])
            if abs(theta_val) < 1e-12:
                raise NotImplementedError("theta Jacobian is undefined at theta=0")
            jac[:, pos] = df_used[:, col] / theta_val
        return jac

    def eta_data_objective_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        hess = np.zeros((len(eta), len(eta)), dtype=float)
        if len(self.dv) == 0:
            return hess
        hess_terms = _compiled_hessian_terms()
        terms = _compiled_terms()
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        f, df, d2f = self._prediction_mean_and_partials(theta, eta, include_second=True)
        assert d2f is not None
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        d2term_df2 = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(hess_terms["obs_term_d2f"](f, self.dv, var_a, var_b, var_c), dtype=float),
            np.full_like(f, 2.0 / _VAR_FLOOR),
        )
        used_hess = np.einsum("o,oi,oj->ij", d2term_df2, df, df) + np.einsum(
            "o,oij->ij", dterm_df, d2f
        )
        for row, pos_i in enumerate(self.eta_idx):
            for col, pos_j in enumerate(self.eta_idx):
                hess[pos_i, pos_j] = used_hess[row, col]
        return hess

    def eta_data_objective_value_grad(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        eta_grad = np.zeros(len(eta), dtype=float)
        if len(self.dv) == 0:
            return 0.0, eta_grad
        terms = _compiled_terms()
        f, df, _d2f = self._prediction_mean_and_partials(theta, eta)
        raw_var = var_a + var_b * f + var_c * (f**2)
        var = np.where(raw_var > _VAR_FLOOR, raw_var, _VAR_FLOOR)
        obs_term = LOG2PI + np.log(var) + (self.dv - f) ** 2 / var
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        eta_grad_used = df.T @ dterm_df
        for pos, value in zip(self.eta_idx, eta_grad_used, strict=False):
            eta_grad[pos] = float(value)
        return float(np.sum(obs_term)), eta_grad

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError(
                "theta data-objective gradient is only supported on the narrow analytical subset"
            )
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        theta_grad = np.zeros(len(theta), dtype=float)
        if len(self.dv) == 0:
            return theta_grad
        terms = _compiled_terms()
        f, _df, _d2f = self._prediction_mean_and_partials(theta, eta)
        theta_jac = self.prediction_theta_jacobian(theta, eta, sigma)
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        return theta_jac.T @ dterm_df

    def eta_data_objective_values(
        self,
        theta: np.ndarray,
        eta_batch: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        predictions = np.empty((len(eta_arr), len(self.dv)), dtype=float)
        for i, eta in enumerate(eta_arr):
            predictions[i] = self._prediction_mean_direct(theta, eta)
        return _eta_data_objective_values_from_predictions(
            predictions, self.dv, var_a, var_b, var_c
        )

    def evaluate(
        self, theta: np.ndarray, eta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return self.eta_data_objective_value_grad(theta, eta, sigma)


@dataclass(slots=True)
class SympyAdvan1Trans2Objective(BaseSubjectDerivativeKernel):
    capabilities = DerivativeKernelCapabilities(
        eta_objective_gradient=True,
        eta_objective_hessian=True,
        prediction_eta_jacobian=True,
        theta_data_objective_gradient=True,
        prediction_theta_jacobian=True,
    )

    theta_idx: tuple[int, int]
    eta_idx: tuple[int, int]
    covariate_adjustments: tuple[tuple[_CovariateAdjustment, ...], tuple[_CovariateAdjustment, ...]]
    static_covariates: dict[str, float]
    error_model: str
    error_theta_idx: tuple[int, ...]
    dt_matrix: np.ndarray
    dt_eval_matrix: np.ndarray
    positive_mask: np.ndarray
    positive_weight: np.ndarray
    active_counts: np.ndarray
    dose_amounts: np.ndarray
    dv: np.ndarray
    _prediction_cache_key: tuple[bytes, bytes] | None = field(default=None, init=False, repr=False)
    _prediction_cache_f: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_df: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_d2f: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def build(cls, indiv: Any, trans: int) -> SympyAdvan1Trans2Objective | None:
        if (
            not _symbolic_runtime_available("advan1_terms", "advan1_hessian_terms")
            or trans != 2
            or getattr(indiv.pk_subroutine, "advan", None) != 1
        ):
            return None
        if not _common_symbolic_build_guards(indiv, allow_pk_covariate_references=True):
            return None
        pk_source = getattr(indiv.pk_callable, "_source", None)
        error_source = getattr(indiv.error_callable, "_source", None)
        if not isinstance(pk_source, str) or not isinstance(error_source, str):
            return None
        mapping = _parse_pk_source_with_static_covariates(pk_source, ("CL", "V"))
        error_model = _parse_error_source(error_source, indiv.n_eps)
        if mapping is None or error_model is None:
            return None
        error_kind, error_theta_idx = error_model
        theta_idx = cast(tuple[int, int], tuple(mapping[name][0] for name in ("CL", "V")))
        eta_idx = cast(tuple[int, int], tuple(mapping[name][1] for name in ("CL", "V")))
        covariate_adjustments = cast(
            tuple[tuple[_CovariateAdjustment, ...], tuple[_CovariateAdjustment, ...]],
            tuple(mapping[name][2] for name in ("CL", "V")),
        )
        if sorted(theta_idx) != [0, 1] or sorted(eta_idx) != [0, 1]:
            return None
        static_covariates = _static_covariate_values(indiv) if any(covariate_adjustments) else {}
        if static_covariates is None:
            return None
        try:
            (
                dt_matrix,
                dt_eval_matrix,
                positive_mask,
                positive_weight,
                active_counts,
                dose_amounts,
                dv,
            ) = _observation_arrays(indiv)
        except ValueError:
            return None
        return cls(
            theta_idx=theta_idx,
            eta_idx=eta_idx,
            covariate_adjustments=covariate_adjustments,
            static_covariates=static_covariates,
            error_model=error_kind,
            error_theta_idx=error_theta_idx,
            dt_matrix=dt_matrix,
            dt_eval_matrix=dt_eval_matrix,
            positive_mask=positive_mask,
            positive_weight=positive_weight,
            active_counts=active_counts,
            dose_amounts=dose_amounts,
            dv=dv,
        )

    def _variance_coefficients(
        self, theta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, float, float]:
        return _variance_coefficients(self.error_model, self.error_theta_idx, theta, sigma)

    def _prediction_mean_direct(self, theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
        if self.dt_matrix.shape[0] == 0:
            return np.array([], dtype=float)
        theta_vals = _effective_theta_values(
            theta, self.theta_idx, self.covariate_adjustments, self.static_covariates
        )
        cl = theta_vals[0] * np.exp(float(eta[self.eta_idx[0]]))
        v = theta_vals[1] * np.exp(float(eta[self.eta_idx[1]]))
        dt = self.dt_eval_matrix
        contrib = self.dose_amounts * np.exp(-(cl / v) * dt) / v
        return _masked_row_sum(contrib, self.positive_weight, self.active_counts)

    def _prediction_mean_and_partials(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        *,
        include_second: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        cache_key, cached = _prediction_cache_lookup(
            self, theta, eta, include_second=include_second
        )
        if cached is not None:
            return cached
        first_order_cached = None
        if include_second:
            _, first_order_cached = _prediction_cache_lookup(self, theta, eta, include_second=False)
        theta_vals = _effective_theta_values(
            theta, self.theta_idx, self.covariate_adjustments, self.static_covariates
        )
        eta_vals = tuple(float(eta[i]) for i in self.eta_idx)
        if self.dt_matrix.shape[0] == 0:
            second = np.zeros((0, 2, 2), dtype=float) if include_second else None
            empty = (np.array([], dtype=float), np.zeros((0, 2), dtype=float), second)
            _prediction_cache_store(self, cache_key, empty[0], empty[1], empty[2])
            return empty
        dt = self.dt_eval_matrix
        if first_order_cached is not None:
            f, df, _ = first_order_cached
        else:
            terms = _compiled_advan1_terms()
            contrib = np.asarray(
                terms["contrib"](*theta_vals, *eta_vals, dt, self.dose_amounts), dtype=float
            )
            f = _masked_row_sum(contrib, self.positive_weight, self.active_counts)
            df = np.empty((self.dt_matrix.shape[0], 2), dtype=float)
            for col, grad_fn in enumerate(terms["contrib_grad"]):
                grad_matrix = np.asarray(
                    grad_fn(*theta_vals, *eta_vals, dt, self.dose_amounts), dtype=float
                )
                df[:, col] = _masked_row_sum(grad_matrix, self.positive_weight, self.active_counts)
        d2f = None
        if include_second:
            hess_terms = _compiled_advan1_hessian_terms()["contrib_hess"]
            d2f = np.empty((self.dt_matrix.shape[0], 2, 2), dtype=float)
            for row, hess_row in enumerate(hess_terms):
                for col in range(row, len(hess_row)):
                    hess_fn = hess_row[col]
                    hess_matrix = np.asarray(
                        hess_fn(*theta_vals, *eta_vals, dt, self.dose_amounts), dtype=float
                    )
                    d2f[:, row, col] = _masked_row_sum(
                        hess_matrix, self.positive_weight, self.active_counts
                    )
                    if col != row:
                        d2f[:, col, row] = d2f[:, row, col]
        _prediction_cache_store(self, cache_key, f, df, d2f)
        return f, df, d2f

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        _f, df_used, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df_used.shape[0], len(eta)), dtype=float)
        for col, pos in enumerate(self.eta_idx):
            jac[:, pos] = df_used[:, col]
        return jac

    def supports_theta_data_objective_gradient(self) -> bool:
        return _supports_narrow_theta_gradients(self.error_model, self.covariate_adjustments)

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError("theta Jacobian is only supported on the narrow analytical subset")
        _f, df_used, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df_used.shape[0], len(theta)), dtype=float)
        for col, pos in enumerate(self.theta_idx):
            theta_val = float(theta[pos])
            if abs(theta_val) < 1e-12:
                raise NotImplementedError("theta Jacobian is undefined at theta=0")
            jac[:, pos] = df_used[:, col] / theta_val
        return jac

    def eta_data_objective_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        hess = np.zeros((len(eta), len(eta)), dtype=float)
        if len(self.dv) == 0:
            return hess
        hess_terms = _compiled_advan1_hessian_terms()
        terms = _compiled_advan1_terms()
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        f, df, d2f = self._prediction_mean_and_partials(theta, eta, include_second=True)
        assert d2f is not None
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        d2term_df2 = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(hess_terms["obs_term_d2f"](f, self.dv, var_a, var_b, var_c), dtype=float),
            np.full_like(f, 2.0 / _VAR_FLOOR),
        )
        used_hess = np.einsum("o,oi,oj->ij", d2term_df2, df, df) + np.einsum(
            "o,oij->ij", dterm_df, d2f
        )
        for row, pos_i in enumerate(self.eta_idx):
            for col, pos_j in enumerate(self.eta_idx):
                hess[pos_i, pos_j] = used_hess[row, col]
        return hess

    def eta_data_objective_value_grad(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        eta_grad = np.zeros(len(eta), dtype=float)
        if len(self.dv) == 0:
            return 0.0, eta_grad
        terms = _compiled_advan1_terms()
        f, df, _d2f = self._prediction_mean_and_partials(theta, eta)
        raw_var = var_a + var_b * f + var_c * (f**2)
        var = np.where(raw_var > _VAR_FLOOR, raw_var, _VAR_FLOOR)
        obs_term = LOG2PI + np.log(var) + (self.dv - f) ** 2 / var
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        eta_grad_used = df.T @ dterm_df
        for pos, value in zip(self.eta_idx, eta_grad_used, strict=False):
            eta_grad[pos] = float(value)
        return float(np.sum(obs_term)), eta_grad

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError(
                "theta data-objective gradient is only supported on the narrow analytical subset"
            )
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        theta_grad = np.zeros(len(theta), dtype=float)
        if len(self.dv) == 0:
            return theta_grad
        terms = _compiled_advan1_terms()
        f, _df, _d2f = self._prediction_mean_and_partials(theta, eta)
        theta_jac = self.prediction_theta_jacobian(theta, eta, sigma)
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        return theta_jac.T @ dterm_df

    def eta_data_objective_values(
        self,
        theta: np.ndarray,
        eta_batch: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        predictions = np.empty((len(eta_arr), len(self.dv)), dtype=float)
        for i, eta in enumerate(eta_arr):
            predictions[i] = self._prediction_mean_direct(theta, eta)
        return _eta_data_objective_values_from_predictions(
            predictions, self.dv, var_a, var_b, var_c
        )

    def evaluate(
        self, theta: np.ndarray, eta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return self.eta_data_objective_value_grad(theta, eta, sigma)


@dataclass(slots=True)
class SympyAdvan3Trans4Objective(BaseSubjectDerivativeKernel):
    capabilities = DerivativeKernelCapabilities(
        eta_objective_gradient=True,
        eta_objective_hessian=True,
        prediction_eta_jacobian=True,
        theta_data_objective_gradient=True,
        prediction_theta_jacobian=True,
    )

    theta_idx: tuple[int, int, int, int]
    eta_idx: tuple[int | None, int | None, int | None, int | None]
    error_model: str
    error_theta_idx: tuple[int, ...]
    dt_matrix: np.ndarray
    dt_eval_matrix: np.ndarray
    positive_mask: np.ndarray
    positive_weight: np.ndarray
    active_counts: np.ndarray
    dose_amounts: np.ndarray
    dv: np.ndarray
    _prediction_cache_key: tuple[bytes, bytes] | None = field(default=None, init=False, repr=False)
    _prediction_cache_f: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_df: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_d2f: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def build(cls, indiv: Any, trans: int) -> SympyAdvan3Trans4Objective | None:
        if (
            not SYMPY_AVAILABLE
            or trans not in {1, 4}
            or getattr(indiv.pk_subroutine, "advan", None) != 3
        ):
            return None
        if not _common_symbolic_build_guards(indiv):
            return None
        pk_source = getattr(indiv.pk_callable, "_source", None)
        error_source = getattr(indiv.error_callable, "_source", None)
        if not isinstance(pk_source, str) or not isinstance(error_source, str):
            return None
        if trans == 4:
            mapping = _parse_ordered_mixed_pk_source(pk_source, ("CL", "V1", "Q", "V2"))
        else:
            mapping = _parse_advan3_trans1_pk_source(pk_source)
        error_model = _parse_error_source(error_source, indiv.n_eps)
        if mapping is None or error_model is None:
            return None
        error_kind, error_theta_idx = error_model
        theta_idx = cast(
            tuple[int, int, int, int],
            tuple(int(mapping[name][0]) for name in ("CL", "V1", "Q", "V2")),
        )
        eta_idx = cast(
            tuple[int | None, int | None, int | None, int | None],
            tuple(mapping[name][1] for name in ("CL", "V1", "Q", "V2")),
        )
        if sorted(theta_idx) != [0, 1, 2, 3]:
            return None
        try:
            (
                dt_matrix,
                dt_eval_matrix,
                positive_mask,
                positive_weight,
                active_counts,
                dose_amounts,
                dv,
            ) = _observation_arrays(indiv)
        except ValueError:
            return None
        return cls(
            theta_idx=theta_idx,
            eta_idx=eta_idx,
            error_model=error_kind,
            error_theta_idx=error_theta_idx,
            dt_matrix=dt_matrix,
            dt_eval_matrix=dt_eval_matrix,
            positive_mask=positive_mask,
            positive_weight=positive_weight,
            active_counts=active_counts,
            dose_amounts=dose_amounts,
            dv=dv,
        )

    def _variance_coefficients(
        self, theta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, float, float]:
        return _variance_coefficients(self.error_model, self.error_theta_idx, theta, sigma)

    def _used_eta_positions(self) -> tuple[int, ...]:
        return tuple(pos for pos in self.eta_idx if pos is not None)

    def _prediction_mean_direct(self, theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
        if self.dt_matrix.shape[0] == 0:
            return np.array([], dtype=float)
        cl = _param_from_spec(theta, eta, self.theta_idx[0], self.eta_idx[0])
        v1 = _param_from_spec(theta, eta, self.theta_idx[1], self.eta_idx[1])
        q = _param_from_spec(theta, eta, self.theta_idx[2], self.eta_idx[2])
        v2 = _param_from_spec(theta, eta, self.theta_idx[3], self.eta_idx[3])
        k = cl / v1
        k12 = q / v1
        k21 = q / v2
        lam1, lam2 = _eigenvalues(k, k12, k21)
        f = np.zeros(self.dt_matrix.shape[0], dtype=float)
        for dose_idx in range(self.dt_matrix.shape[1]):
            mask = self.positive_mask[:, dose_idx]
            if not np.any(mask):
                continue
            a1, _a2 = _biexp_central(
                float(self.dose_amounts[dose_idx]),
                k,
                k12,
                k21,
                lam1,
                lam2,
                self.dt_matrix[mask, dose_idx],
            )
            f[mask] += a1 / v1
        return f

    def _prediction_used_jacobian_fd(
        self, theta: np.ndarray, eta: np.ndarray, eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        used_eta_idx = self._used_eta_positions()
        used_eta = np.asarray([eta[pos] for pos in used_eta_idx], dtype=float)
        jac = np.empty((self.dt_matrix.shape[0], len(used_eta_idx)), dtype=float)

        def pred_of_used(used_value: np.ndarray) -> np.ndarray:
            eta_full = np.asarray(eta, dtype=float).copy()
            for idx, pos in enumerate(used_eta_idx):
                eta_full[pos] = used_value[idx]
            return self._prediction_mean_direct(theta, eta_full)

        base = pred_of_used(used_eta)
        for idx in range(len(used_eta_idx)):
            delta = np.zeros_like(used_eta)
            delta[idx] = eps
            jac[:, idx] = (pred_of_used(used_eta + delta) - pred_of_used(used_eta - delta)) / (
                2.0 * eps
            )
        return base, jac

    def _prediction_used_hessian_fd(
        self, theta: np.ndarray, eta: np.ndarray, eps: float = 1e-4
    ) -> np.ndarray:
        used_eta_idx = self._used_eta_positions()
        used_eta = np.asarray([eta[pos] for pos in used_eta_idx], dtype=float)
        hess = np.empty(
            (self.dt_matrix.shape[0], len(used_eta_idx), len(used_eta_idx)), dtype=float
        )

        def pred_of_used(used_value: np.ndarray) -> np.ndarray:
            eta_full = np.asarray(eta, dtype=float).copy()
            for idx, pos in enumerate(used_eta_idx):
                eta_full[pos] = used_value[idx]
            return self._prediction_mean_direct(theta, eta_full)

        for row in range(len(used_eta_idx)):
            for col in range(len(used_eta_idx)):
                delta_row = np.zeros_like(used_eta)
                delta_col = np.zeros_like(used_eta)
                delta_row[row] = eps
                delta_col[col] = eps
                hess[:, row, col] = (
                    pred_of_used(used_eta + delta_row + delta_col)
                    - pred_of_used(used_eta + delta_row - delta_col)
                    - pred_of_used(used_eta - delta_row + delta_col)
                    + pred_of_used(used_eta - delta_row - delta_col)
                ) / (4.0 * eps * eps)
        return hess

    def _prediction_mean_and_partials(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        *,
        include_second: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        cache_key, cached = _prediction_cache_lookup(
            self, theta, eta, include_second=include_second
        )
        if cached is not None:
            return cached
        first_order_cached = None
        if include_second:
            _, first_order_cached = _prediction_cache_lookup(self, theta, eta, include_second=False)
        theta_vals = tuple(float(theta[i]) for i in self.theta_idx)
        eta_vals = tuple(float(eta[pos]) if pos is not None else 0.0 for pos in self.eta_idx)
        if self.dt_matrix.shape[0] == 0:
            second = np.zeros((0, 4, 4), dtype=float) if include_second else None
            empty = (np.array([], dtype=float), np.zeros((0, 4), dtype=float), second)
            _prediction_cache_store(self, cache_key, empty[0], empty[1], empty[2])
            return empty
        cl = theta_vals[0] * np.exp(eta_vals[0])
        v1 = theta_vals[1] * np.exp(eta_vals[1])
        q = theta_vals[2] * np.exp(eta_vals[2])
        v2 = theta_vals[3] * np.exp(eta_vals[3])
        k = cl / v1
        k12 = q / v1
        k21 = q / v2
        lam1, lam2 = _eigenvalues(k, k12, k21)
        used_positions = self._used_eta_positions()
        if abs(lam2 - lam1) <= 1e-8:
            base, used_df = self._prediction_used_jacobian_fd(theta, eta)
            full_df = np.zeros((self.dt_matrix.shape[0], 4), dtype=float)
            for col, pos in enumerate(self.eta_idx):
                if pos is not None:
                    full_df[:, col] = used_df[:, used_positions.index(pos)]
            full_d2f = None
            if include_second:
                used_hess = self._prediction_used_hessian_fd(theta, eta)
                full_d2f = np.zeros((self.dt_matrix.shape[0], 4, 4), dtype=float)
                for row, pos_i in enumerate(self.eta_idx):
                    if pos_i is None:
                        continue
                    row_used = used_positions.index(pos_i)
                    for col, pos_j in enumerate(self.eta_idx):
                        if pos_j is None:
                            continue
                        col_used = used_positions.index(pos_j)
                        full_d2f[:, row, col] = used_hess[:, row_used, col_used]
            _prediction_cache_store(self, cache_key, base, full_df, full_d2f)
            return base, full_df, full_d2f
        dt = self.dt_eval_matrix
        term_args = (cl, v1, q, v2, dt, self.dose_amounts)
        if first_order_cached is not None:
            f, df, _ = first_order_cached
        else:
            terms = _compiled_advan3_terms()
            contrib = np.asarray(terms["contrib"](*term_args), dtype=float)
            f = _masked_row_sum(contrib, self.positive_weight, self.active_counts)
            df = np.empty((self.dt_matrix.shape[0], 4), dtype=float)
            for col, grad_fn in enumerate(terms["contrib_grad"]):
                grad_matrix = np.asarray(grad_fn(*term_args), dtype=float)
                df[:, col] = _masked_row_sum(grad_matrix, self.positive_weight, self.active_counts)
        d2f = None
        if include_second:
            hess_terms = _compiled_advan3_hessian_terms()
            hess_bundle = hess_terms["contrib_hess_bundle"](*term_args)
            d2f = np.empty((self.dt_matrix.shape[0], 4, 4), dtype=float)
            for idx, (row, col) in enumerate(hess_terms["contrib_hess_pairs"]):
                hess_matrix = np.asarray(hess_bundle[idx], dtype=float)
                d2f[:, row, col] = _masked_row_sum(
                    hess_matrix, self.positive_weight, self.active_counts
                )
                if col != row:
                    d2f[:, col, row] = d2f[:, row, col]
        _prediction_cache_store(self, cache_key, f, df, d2f)
        return f, df, d2f

    def prediction_eta_jacobian(
        self, theta: np.ndarray, eta: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        _f, df_full, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df_full.shape[0], len(eta)), dtype=float)
        for col, pos in enumerate(self.eta_idx):
            if pos is not None:
                jac[:, pos] = df_full[:, col]
        return jac

    def eta_data_objective_hessian(
        self, theta: np.ndarray, eta: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        hess = np.zeros((len(eta), len(eta)), dtype=float)
        if len(self.dv) == 0:
            return hess
        hess_terms = _compiled_advan3_hessian_terms()
        terms = _compiled_advan3_terms()
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        f, df_full, d2f_full = self._prediction_mean_and_partials(theta, eta, include_second=True)
        assert d2f_full is not None
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        d2term_df2 = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(hess_terms["obs_term_d2f"](f, self.dv, var_a, var_b, var_c), dtype=float),
            np.full_like(f, 2.0 / _VAR_FLOOR),
        )
        full_hess = np.einsum("o,oi,oj->ij", d2term_df2, df_full, df_full) + np.einsum(
            "o,oij->ij", dterm_df, d2f_full
        )
        for row, pos_i in enumerate(self.eta_idx):
            if pos_i is None:
                continue
            for col, pos_j in enumerate(self.eta_idx):
                if pos_j is None:
                    continue
                hess[pos_i, pos_j] = full_hess[row, col]
        return hess

    def eta_data_objective_value_grad(
        self, theta: np.ndarray, eta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, np.ndarray]:
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        eta_grad = np.zeros(len(eta), dtype=float)
        if len(self.dv) == 0:
            return 0.0, eta_grad
        terms = _compiled_advan3_terms()
        f, df_full, _d2f = self._prediction_mean_and_partials(theta, eta)
        raw_var = var_a + var_b * f + var_c * (f**2)
        var = np.where(raw_var > _VAR_FLOOR, raw_var, _VAR_FLOOR)
        obs_term = LOG2PI + np.log(var) + (self.dv - f) ** 2 / var
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        full_grad = df_full.T @ dterm_df
        for col, pos in enumerate(self.eta_idx):
            if pos is not None:
                eta_grad[pos] = float(full_grad[col])
        return float(np.sum(obs_term)), eta_grad

    def supports_theta_data_objective_gradient(self) -> bool:
        return self.error_model in {"proportional", "additive", "combined_eps"}

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError("theta Jacobian only supported on narrow analytical subset")
        _f, df_full, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df_full.shape[0], len(theta)), dtype=float)
        for col, pos in enumerate(self.theta_idx):
            theta_val = float(theta[pos])
            if abs(theta_val) < 1e-12:
                raise NotImplementedError("theta Jacobian undefined at theta=0")
            jac[:, pos] = df_full[:, col] / theta_val
        return jac

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError(
                "theta data-objective gradient is only supported on the narrow analytical subset"
            )
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        theta_grad = np.zeros(len(theta), dtype=float)
        if len(self.dv) == 0:
            return theta_grad
        terms = _compiled_advan3_terms()
        f, _df, _d2f = self._prediction_mean_and_partials(theta, eta)
        theta_jac = self.prediction_theta_jacobian(theta, eta, sigma)
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        return theta_jac.T @ dterm_df

    def eta_data_objective_values(
        self,
        theta: np.ndarray,
        eta_batch: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        predictions = np.empty((len(eta_arr), len(self.dv)), dtype=float)
        for i, eta in enumerate(eta_arr):
            predictions[i] = self._prediction_mean_direct(theta, eta)
        return _eta_data_objective_values_from_predictions(
            predictions, self.dv, var_a, var_b, var_c
        )

    def evaluate(
        self, theta: np.ndarray, eta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return self.eta_data_objective_value_grad(theta, eta, sigma)


@dataclass(slots=True)
class SympyAdvan4Trans1Objective(BaseSubjectDerivativeKernel):
    capabilities = DerivativeKernelCapabilities(
        eta_objective_gradient=True,
        eta_objective_hessian=True,
        prediction_eta_jacobian=True,
        theta_data_objective_gradient=True,
        prediction_theta_jacobian=True,
    )

    theta_idx: tuple[int, int, int, int, int]
    eta_idx: tuple[int, int, int]
    error_model: str
    error_theta_idx: tuple[int, ...]
    dt_matrix: np.ndarray
    dt_eval_matrix: np.ndarray
    positive_mask: np.ndarray
    positive_weight: np.ndarray
    active_counts: np.ndarray
    dose_amounts: np.ndarray
    dv: np.ndarray
    _prediction_cache_key: tuple[bytes, bytes] | None = field(default=None, init=False, repr=False)
    _prediction_cache_f: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_df: np.ndarray | None = field(default=None, init=False, repr=False)
    _prediction_cache_d2f: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def build(cls, indiv: Any, trans: int) -> SympyAdvan4Trans1Objective | None:
        if (
            not SYMPY_AVAILABLE
            or trans not in {1, 4}
            or getattr(indiv.pk_subroutine, "advan", None) != 4
        ):
            return None
        if not _common_symbolic_build_guards(indiv):
            return None
        pk_source = getattr(indiv.pk_callable, "_source", None)
        error_source = getattr(indiv.error_callable, "_source", None)
        if not isinstance(pk_source, str) or not isinstance(error_source, str):
            return None
        if trans == 1:
            mapping = _parse_advan4_explicit_pk_source(pk_source)
        else:
            mapping = _parse_ordered_mixed_pk_source(pk_source, ("KA", "CL", "V2", "Q", "V3"))
        error_model = _parse_error_source(error_source, indiv.n_eps)
        if mapping is None or error_model is None:
            return None
        error_kind, error_theta_idx = error_model
        theta_idx = cast(
            tuple[int, int, int, int, int],
            tuple(int(mapping[name][0]) for name in ("KA", "CL", "V2", "Q", "V3")),
        )
        eta_idx_raw = tuple(mapping[name][1] for name in ("KA", "CL", "V2"))
        if any(v is None for v in eta_idx_raw):
            return None
        eta_idx = cast(tuple[int, int, int], eta_idx_raw)
        if sorted(theta_idx) != [0, 1, 2, 3, 4] or sorted(eta_idx) != [0, 1, 2]:
            return None
        try:
            (
                dt_matrix,
                dt_eval_matrix,
                positive_mask,
                positive_weight,
                active_counts,
                dose_amounts,
                dv,
            ) = _observation_arrays(indiv)
        except ValueError:
            return None
        return cls(
            theta_idx=theta_idx,
            eta_idx=eta_idx,
            error_model=error_kind,
            error_theta_idx=error_theta_idx,
            dt_matrix=dt_matrix,
            dt_eval_matrix=dt_eval_matrix,
            positive_mask=positive_mask,
            positive_weight=positive_weight,
            active_counts=active_counts,
            dose_amounts=dose_amounts,
            dv=dv,
        )

    def _variance_coefficients(
        self, theta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, float, float]:
        return _variance_coefficients(self.error_model, self.error_theta_idx, theta, sigma)

    def _prediction_mean_direct(self, theta: np.ndarray, eta: np.ndarray) -> np.ndarray:
        if self.dt_matrix.shape[0] == 0:
            return np.array([], dtype=float)
        ka = float(theta[self.theta_idx[0]] * np.exp(eta[self.eta_idx[0]]))
        cl = float(theta[self.theta_idx[1]] * np.exp(eta[self.eta_idx[1]]))
        v2 = float(theta[self.theta_idx[2]] * np.exp(eta[self.eta_idx[2]]))
        q = float(theta[self.theta_idx[3]])
        v3 = float(theta[self.theta_idx[4]])
        k = cl / v2
        k12 = q / v2
        k21 = q / v3
        lam1, lam2 = _eigenvalues(k, k12, k21)
        f = np.zeros(self.dt_matrix.shape[0], dtype=float)
        for dose_idx in range(self.dt_matrix.shape[1]):
            mask = self.positive_mask[:, dose_idx]
            if not np.any(mask):
                continue
            _, a2, _ = _triexp_oral(
                float(self.dose_amounts[dose_idx]),
                ka,
                k,
                k12,
                k21,
                lam1,
                lam2,
                self.dt_matrix[mask, dose_idx],
            )
            f[mask] += a2 / v2
        return f

    def _prediction_used_jacobian_fd(
        self, theta: np.ndarray, eta: np.ndarray, eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        used_eta = np.asarray([eta[pos] for pos in self.eta_idx], dtype=float)
        jac = np.empty((self.dt_matrix.shape[0], len(self.eta_idx)), dtype=float)

        def pred_of_used(used_value: np.ndarray) -> np.ndarray:
            eta_full = np.asarray(eta, dtype=float).copy()
            for idx, pos in enumerate(self.eta_idx):
                eta_full[pos] = used_value[idx]
            return self._prediction_mean_direct(theta, eta_full)

        base = pred_of_used(used_eta)
        for idx in range(len(used_eta)):
            delta = np.zeros_like(used_eta)
            delta[idx] = eps
            jac[:, idx] = (pred_of_used(used_eta + delta) - pred_of_used(used_eta - delta)) / (
                2.0 * eps
            )
        return base, jac

    def _prediction_used_hessian_fd(
        self, theta: np.ndarray, eta: np.ndarray, eps: float = 1e-4
    ) -> np.ndarray:
        used_eta = np.asarray([eta[pos] for pos in self.eta_idx], dtype=float)
        hess = np.empty(
            (self.dt_matrix.shape[0], len(self.eta_idx), len(self.eta_idx)), dtype=float
        )

        def pred_of_used(used_value: np.ndarray) -> np.ndarray:
            eta_full = np.asarray(eta, dtype=float).copy()
            for idx, pos in enumerate(self.eta_idx):
                eta_full[pos] = used_value[idx]
            return self._prediction_mean_direct(theta, eta_full)

        for row in range(len(used_eta)):
            for col in range(len(used_eta)):
                delta_row = np.zeros_like(used_eta)
                delta_col = np.zeros_like(used_eta)
                delta_row[row] = eps
                delta_col[col] = eps
                hess[:, row, col] = (
                    pred_of_used(used_eta + delta_row + delta_col)
                    - pred_of_used(used_eta + delta_row - delta_col)
                    - pred_of_used(used_eta - delta_row + delta_col)
                    + pred_of_used(used_eta - delta_row - delta_col)
                ) / (4.0 * eps * eps)
        return hess

    def _prediction_mean_and_partials(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        *,
        include_second: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        cache_key, cached = _prediction_cache_lookup(
            self, theta, eta, include_second=include_second
        )
        if cached is not None:
            return cached
        first_order_cached = None
        if include_second:
            _, first_order_cached = _prediction_cache_lookup(self, theta, eta, include_second=False)
        theta_vals = tuple(float(theta[i]) for i in self.theta_idx)
        eta_vals = tuple(float(eta[i]) for i in self.eta_idx)
        if self.dt_matrix.shape[0] == 0:
            second = np.zeros((0, 3, 3), dtype=float) if include_second else None
            empty = (np.array([], dtype=float), np.zeros((0, 3), dtype=float), second)
            _prediction_cache_store(self, cache_key, empty[0], empty[1], empty[2])
            return empty
        ka = theta_vals[0] * np.exp(eta_vals[0])
        cl = theta_vals[1] * np.exp(eta_vals[1])
        v2 = theta_vals[2] * np.exp(eta_vals[2])
        q = theta_vals[3]
        v3 = theta_vals[4]
        k = cl / v2
        k12 = q / v2
        k21 = q / v3
        lam1, lam2 = _eigenvalues(k, k12, k21)
        if min(abs(lam2 - lam1), abs(ka - lam1), abs(ka - lam2)) <= 1e-8:
            f, df = self._prediction_used_jacobian_fd(theta, eta)
            d2f = self._prediction_used_hessian_fd(theta, eta) if include_second else None
            _prediction_cache_store(self, cache_key, f, df, d2f)
            return f, df, d2f
        dt = self.dt_eval_matrix
        term_args = (ka, cl, v2, q, v3, dt, self.dose_amounts)
        if first_order_cached is not None:
            f, df, _ = first_order_cached
        else:
            terms = _compiled_advan4_terms()
            contrib = np.asarray(terms["contrib"](*term_args), dtype=float)
            f = _masked_row_sum(contrib, self.positive_weight, self.active_counts)
            df = np.empty((self.dt_matrix.shape[0], 3), dtype=float)
            for col, grad_fn in enumerate(terms["contrib_grad"]):
                grad_matrix = np.asarray(grad_fn(*term_args), dtype=float)
                df[:, col] = _masked_row_sum(grad_matrix, self.positive_weight, self.active_counts)
        d2f = None
        if include_second:
            hess_terms = _compiled_advan4_hessian_terms()
            hess_bundle = hess_terms["contrib_hess_bundle"](*term_args)
            d2f = np.empty((self.dt_matrix.shape[0], 3, 3), dtype=float)
            for idx, (row, col) in enumerate(hess_terms["contrib_hess_pairs"]):
                hess_matrix = np.asarray(hess_bundle[idx], dtype=float)
                d2f[:, row, col] = _masked_row_sum(
                    hess_matrix, self.positive_weight, self.active_counts
                )
                if col != row:
                    d2f[:, col, row] = d2f[:, row, col]
        _prediction_cache_store(self, cache_key, f, df, d2f)
        return f, df, d2f

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        _f, df_used, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df_used.shape[0], len(eta)), dtype=float)
        for col, pos in enumerate(self.eta_idx):
            jac[:, pos] = df_used[:, col]
        return jac

    def eta_data_objective_hessian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        hess = np.zeros((len(eta), len(eta)), dtype=float)
        if len(self.dv) == 0:
            return hess
        hess_terms = _compiled_advan4_hessian_terms()
        terms = _compiled_advan4_terms()
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        f, df, d2f = self._prediction_mean_and_partials(theta, eta, include_second=True)
        assert d2f is not None
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        d2term_df2 = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(hess_terms["obs_term_d2f"](f, self.dv, var_a, var_b, var_c), dtype=float),
            np.full_like(f, 2.0 / _VAR_FLOOR),
        )
        used_hess = np.einsum("o,oi,oj->ij", d2term_df2, df, df) + np.einsum(
            "o,oij->ij", dterm_df, d2f
        )
        for row, pos_i in enumerate(self.eta_idx):
            for col, pos_j in enumerate(self.eta_idx):
                hess[pos_i, pos_j] = used_hess[row, col]
        return hess

    def eta_data_objective_value_grad(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        eta_grad = np.zeros(len(eta), dtype=float)
        if len(self.dv) == 0:
            return 0.0, eta_grad
        terms = _compiled_advan4_terms()
        f, df, _d2f = self._prediction_mean_and_partials(theta, eta)
        raw_var = var_a + var_b * f + var_c * (f**2)
        var = np.where(raw_var > _VAR_FLOOR, raw_var, _VAR_FLOOR)
        obs_term = LOG2PI + np.log(var) + (self.dv - f) ** 2 / var
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        eta_grad_used = df.T @ dterm_df
        for pos, value in zip(self.eta_idx, eta_grad_used, strict=False):
            eta_grad[pos] = float(value)
        return float(np.sum(obs_term)), eta_grad

    def supports_theta_data_objective_gradient(self) -> bool:
        return self.error_model in {"proportional", "additive", "combined_eps"}

    def prediction_theta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        eps: float = 1e-6,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError("theta Jacobian only supported on narrow analytical subset")
        _f, df, _d2f = self._prediction_mean_and_partials(theta, eta)
        jac = np.zeros((df.shape[0], len(theta)), dtype=float)
        # KA, CL, V2 have ETAs: df/d_theta_i = df/d_eta_i / theta_i
        for col in range(3):
            pos = self.theta_idx[col]
            theta_val = float(theta[pos])
            if abs(theta_val) < 1e-12:
                raise NotImplementedError("theta Jacobian undefined at theta=0")
            jac[:, pos] = df[:, col] / theta_val
        # Q, V3 are theta-only: use central finite differences on the analytical prediction
        for col in (3, 4):
            pos = self.theta_idx[col]
            theta_val = float(theta[pos])
            h = max(eps * abs(theta_val), 1e-8)
            theta_hi = np.array(theta, dtype=float)
            theta_lo = np.array(theta, dtype=float)
            theta_hi[pos] += h
            theta_lo[pos] -= h
            jac[:, pos] = (
                self._prediction_mean_direct(theta_hi, eta)
                - self._prediction_mean_direct(theta_lo, eta)
            ) / (2.0 * h)
        return jac

    def theta_data_objective_gradient(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_theta_data_objective_gradient():
            raise NotImplementedError(
                "theta data-objective gradient is only supported on the narrow analytical subset"
            )
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        theta_grad = np.zeros(len(theta), dtype=float)
        if len(self.dv) == 0:
            return theta_grad
        terms = _compiled_advan4_terms()
        f, _df, _d2f = self._prediction_mean_and_partials(theta, eta)
        theta_jac = self.prediction_theta_jacobian(theta, eta, sigma)
        raw_var = var_a + var_b * f + var_c * (f**2)
        dterm_df = np.where(
            raw_var > _VAR_FLOOR,
            np.asarray(terms["obs_term_df"](f, self.dv, var_a, var_b, var_c), dtype=float),
            2.0 * (f - self.dv) / _VAR_FLOOR,
        )
        return theta_jac.T @ dterm_df

    def eta_data_objective_values(
        self,
        theta: np.ndarray,
        eta_batch: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        eta_arr = np.asarray(eta_batch, dtype=float)
        if eta_arr.ndim == 1:
            eta_arr = eta_arr[None, :]
        var_a, var_b, var_c = self._variance_coefficients(theta, sigma)
        predictions = np.empty((len(eta_arr), len(self.dv)), dtype=float)
        for i, eta in enumerate(eta_arr):
            predictions[i] = self._prediction_mean_direct(theta, eta)
        return _eta_data_objective_values_from_predictions(
            predictions, self.dv, var_a, var_b, var_c
        )

    def evaluate(
        self, theta: np.ndarray, eta: np.ndarray, sigma: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return self.eta_data_objective_value_grad(theta, eta, sigma)
