"""
PK/PD and standalone PD models for pharmacometric analysis.

Implements indirect response models (IDR I-IV), effect compartment,
turnover, placebo response, and tumor growth inhibition (Simeoni 2004) models.
All models are fitted by minimising -2*log-likelihood via scipy.optimize.minimize.

References:
    Dayneka NL et al. (1993). Comparison of four basic models of indirect
        pharmacodynamic responses. J Pharmacokinet Biopharm 21(4):457-478.
    Simeoni M et al. (2004). Predictive pharmacokinetic-pharmacodynamic modeling
        of tumor growth kinetics in xenograft models after administration of
        anticancer agents. Cancer Res 64(3):1094-1101.
    Sheiner LB (1969). Modeling pharmacodynamics: parametric and nonparametric
        approaches. Clin Pharmacol Ther 23(3):322-334.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import integrate, optimize

if TYPE_CHECKING:
    pass


@dataclass
class PDData:
    """
    Data structure for PD model fitting.

    Attributes:
        subject_id:      Unique subject identifier.
        times:           Observation times (shape n).
        response:        Observed PD response values (shape n).
        concentrations:  Plasma PK concentrations at each time (shape n).
                         None for pure time-course models.
        baseline:        Baseline response R0. If None, inferred from R0=Kin/Kout.
    """

    subject_id: int | str
    times: np.ndarray
    response: np.ndarray
    concentrations: np.ndarray | None = None
    baseline: float | None = None

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.response = np.asarray(self.response, dtype=float)
        if self.concentrations is not None:
            self.concentrations = np.asarray(self.concentrations, dtype=float)


@dataclass
class PDResult:
    """
    Result from PD model maximum-likelihood estimation.

    Attributes:
        params:     Fitted parameter dictionary.
        ofv:        Objective function value (-2 * log-likelihood).
        se:         Standard errors for each parameter.
        converged:  Whether the optimizer converged.
        aic:        Akaike Information Criterion = OFV + 2 * n_params.
        predicted:  Array of model-predicted response values.
    """

    params: dict[str, float]
    ofv: float
    se: dict[str, float]
    converged: bool
    aic: float
    predicted: np.ndarray = field(default_factory=lambda: np.array([]))


class PDModel(ABC):
    """Abstract base class for PD models."""

    #: Names of model parameters (for initial value / bounds handling)
    param_names: list[str] = []

    @abstractmethod
    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        """
        Predict PD response at all times in data.

        Args:
            params: Parameter dictionary.
            data:   PDData with times, concentrations, and optionally baseline.

        Returns:
            Array of predicted responses, shape (n,).
        """

    def fit(
        self,
        data: PDData,
        initial_params: dict[str, float] | None = None,
        sigma2: float = 1.0,
        method: str = "L-BFGS-B",
    ) -> PDResult:
        """
        Fit the PD model to data by minimising OFV = -2 * sum(log-likelihood).

        Uses additive Gaussian error model: Y ~ N(f(t), sigma2).

        Args:
            data:           PDData object.
            initial_params: Starting values. If None, uses class defaults.
            sigma2:         Fixed residual variance (default 1.0).
            method:         scipy.optimize.minimize method.

        Returns:
            PDResult with fitted parameters and diagnostics.
        """
        p0 = initial_params or self._default_params(data)
        names = list(p0.keys())
        x0 = np.array([p0[k] for k in names], dtype=float)
        bounds = self._get_bounds(names)

        def objective(x: np.ndarray) -> float:
            params = dict(zip(names, x, strict=False))
            try:
                pred = self.predict(params, data)
                if np.any(~np.isfinite(pred)):
                    return 1e10
                resid = data.response - pred
                ofv = float(np.sum(resid**2) / sigma2 + len(resid) * np.log(sigma2))
                return ofv if np.isfinite(ofv) else 1e10
            except Exception:
                return 1e10

        result = optimize.minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        fitted_params = dict(zip(names, result.x, strict=False))
        ofv = float(result.fun)
        n_params = len(names)
        aic = ofv + 2.0 * n_params

        # SE from Hessian diagonal (if available)
        se: dict[str, float] = {}
        if hasattr(result, "hess_inv") and result.hess_inv is not None:
            try:
                if hasattr(result.hess_inv, "todense"):
                    hess_inv_diag = np.diag(result.hess_inv.todense())
                else:
                    hess_inv_diag = np.diag(result.hess_inv)
                for k, v in zip(names, hess_inv_diag, strict=False):
                    se[k] = float(np.sqrt(max(v, 0.0)))
            except Exception:
                se = {k: float("nan") for k in names}
        else:
            se = {k: float("nan") for k in names}

        predicted = self.predict(fitted_params, data)

        return PDResult(
            params=fitted_params,
            ofv=ofv,
            se=se,
            converged=bool(result.success),
            aic=aic,
            predicted=predicted,
        )

    def _default_params(self, data: PDData) -> dict[str, float]:
        """Default initial parameter values."""
        return dict.fromkeys(self.param_names, 1.0)

    def _get_bounds(self, names: list[str]) -> list[tuple[float | None, float | None]]:
        """Default bounds: all parameters > 0."""
        return [(1e-8, None) for _ in names]


# ---------------------------------------------------------------------------
# Simple algebraic direct PD models
# ---------------------------------------------------------------------------


class LinearPDModel(PDModel):
    """
    Linear direct-effect PD model.

    E(t) = E0 + slope * C(t)

    Parameters: E0, slope
    """

    param_names = ["E0", "slope"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        e0 = float(params.get("E0", 0.0))
        slope = float(params.get("slope", 1.0))
        c = (
            np.asarray(data.concentrations, dtype=float)
            if data.concentrations is not None
            else np.zeros(len(data.times))
        )
        return e0 + slope * c

    def _default_params(self, data: PDData) -> dict[str, float]:
        r_mean = float(np.nanmean(data.response)) if len(data.response) > 0 else 1.0
        return {"E0": r_mean * 0.5, "slope": 1.0}

    def _get_bounds(self, names: list[str]) -> list[tuple[float | None, float | None]]:
        return [(None, None), (None, None)]  # E0 and slope unconstrained


class EmaxModel(PDModel):
    """
    Direct Emax PD model.

    E(t) = E0 + Emax * C(t) / (EC50 + C(t))

    Parameters: E0, Emax, EC50
    """

    param_names = ["E0", "Emax", "EC50"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        e0 = float(params.get("E0", 0.0))
        emax = float(params.get("Emax", 1.0))
        ec50 = float(params.get("EC50", 1.0))
        c = (
            np.asarray(data.concentrations, dtype=float)
            if data.concentrations is not None
            else np.zeros(len(data.times))
        )
        denom = ec50 + c
        denom = np.where(denom <= 0, 1e-30, denom)
        return e0 + emax * c / denom

    def _default_params(self, data: PDData) -> dict[str, float]:
        r_max = float(np.nanmax(data.response)) if len(data.response) > 0 else 1.0
        return {"E0": 0.0, "Emax": r_max, "EC50": 1.0}

    def _get_bounds(self, names: list[str]) -> list[tuple[float | None, float | None]]:
        # E0 unconstrained, Emax unconstrained (can be negative = inhibition), EC50 > 0
        return [(None, None), (None, None), (1e-8, None)]


class HillModel(PDModel):
    """
    Sigmoidal Emax (Hill) PD model.

    E(t) = E0 + Emax * C(t)^gamma / (EC50^gamma + C(t)^gamma)

    Parameters: E0, Emax, EC50, gamma
    """

    param_names = ["E0", "Emax", "EC50", "gamma"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        e0 = float(params.get("E0", 0.0))
        emax = float(params.get("Emax", 1.0))
        ec50 = float(params.get("EC50", 1.0))
        gamma = float(params.get("gamma", 1.0))
        c = (
            np.asarray(data.concentrations, dtype=float)
            if data.concentrations is not None
            else np.zeros(len(data.times))
        )
        c_safe = np.maximum(c, 0.0)
        ec50_safe = max(ec50, 1e-30)
        c_n = c_safe**gamma
        ec50_n = ec50_safe**gamma
        denom = ec50_n + c_n
        denom = np.where(denom <= 0, 1e-30, denom)
        return e0 + emax * c_n / denom

    def _default_params(self, data: PDData) -> dict[str, float]:
        r_max = float(np.nanmax(data.response)) if len(data.response) > 0 else 1.0
        return {"E0": 0.0, "Emax": r_max, "EC50": 1.0, "gamma": 1.0}

    def _get_bounds(self, names: list[str]) -> list[tuple[float | None, float | None]]:
        return [(None, None), (None, None), (1e-8, None), (0.1, 10.0)]


class InhibEmaxModel(PDModel):
    """
    Inhibitory Emax PD model.

    E(t) = E0 * (1 - Imax * C(t)^gamma / (IC50^gamma + C(t)^gamma))

    Parameters: E0, Imax, IC50, gamma
    Imax is constrained to [0, 1].
    """

    param_names = ["E0", "Imax", "IC50", "gamma"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        e0 = float(params.get("E0", 1.0))
        imax = float(np.clip(params.get("Imax", 1.0), 0.0, 1.0))
        ic50 = float(params.get("IC50", 1.0))
        gamma = float(params.get("gamma", 1.0))
        c = (
            np.asarray(data.concentrations, dtype=float)
            if data.concentrations is not None
            else np.zeros(len(data.times))
        )
        c_safe = np.maximum(c, 0.0)
        ic50_safe = max(ic50, 1e-30)
        c_n = c_safe**gamma
        ic50_n = ic50_safe**gamma
        denom = ic50_n + c_n
        denom = np.where(denom <= 0, 1e-30, denom)
        return e0 * (1.0 - imax * c_n / denom)

    def _default_params(self, data: PDData) -> dict[str, float]:
        r_mean = float(np.nanmean(data.response)) if len(data.response) > 0 else 1.0
        return {"E0": r_mean, "Imax": 0.9, "IC50": 1.0, "gamma": 1.0}

    def _get_bounds(self, names: list[str]) -> list[tuple[float | None, float | None]]:
        bounds: list[tuple[float | None, float | None]] = []
        for n in names:
            if n == "Imax":
                bounds.append((0.0, 1.0))
            elif n == "gamma":
                bounds.append((0.1, 10.0))
            else:
                bounds.append((1e-8, None))
        return bounds


# ---------------------------------------------------------------------------
# Indirect Response Models (IDR I-IV)
# ---------------------------------------------------------------------------


class IndirectResponseModel(PDModel):
    """
    Indirect response model (Types I-IV, Dayneka 1993).

    Type I:   dR/dt = Kin*(1 + Emax*C/(EC50+C)) - Kout*R   (stimulate input)
    Type II:  dR/dt = Kin - Kout*(1 + Emax*C/(EC50+C))*R   (stimulate output)
    Type III: dR/dt = Kin*(1 - Imax*C/(IC50+C)) - Kout*R   (inhibit input)
    Type IV:  dR/dt = Kin - Kout*(1 - Imax*C/(IC50+C))*R   (inhibit output)

    Baseline: R0 = Kin / Kout (steady-state without drug)

    Args:
        idr_type: 1, 2, 3, or 4.
    """

    param_names = ["Kin", "Kout", "EC50", "Emax"]

    def __init__(self, idr_type: int = 1) -> None:
        if idr_type not in (1, 2, 3, 4):
            raise ValueError("idr_type must be 1, 2, 3, or 4.")
        self.idr_type = idr_type
        if idr_type in (1, 2):
            self.param_names = ["Kin", "Kout", "EC50", "Emax"]
        else:
            self.param_names = ["Kin", "Kout", "IC50", "Imax"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        kin = float(params.get("Kin", 1.0))
        kout = float(params.get("Kout", 1.0))
        if kout <= 0:
            return np.full(len(data.times), np.nan)
        r0 = data.baseline if data.baseline is not None else kin / kout

        if self.idr_type in (1, 2):
            ec50 = float(params.get("EC50", 1.0))
            emax = float(params.get("Emax", 1.0))

            def _drug_effect(c: float) -> float:
                return emax * c / (ec50 + c) if ec50 + c > 0 else 0.0

        else:
            ic50 = float(params.get("IC50", 1.0))
            imax = min(float(params.get("Imax", 1.0)), 1.0)

            def _drug_effect(c: float) -> float:
                return imax * c / (ic50 + c) if ic50 + c > 0 else 0.0

        conc_func = self._make_conc_func(data)

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            r = max(y[0], 0.0)
            c = max(conc_func(t), 0.0)
            e = _drug_effect(c)
            if self.idr_type == 1:
                drdt = kin * (1.0 + e) - kout * r
            elif self.idr_type == 2:
                drdt = kin - kout * (1.0 + e) * r
            elif self.idr_type == 3:
                drdt = kin * (1.0 - e) - kout * r
            else:  # type 4
                drdt = kin - kout * (1.0 - e) * r
            return np.array([drdt])

        t_span = (0.0, float(np.max(data.times)))
        sol = integrate.solve_ivp(
            odes,
            [0.0, t_span[1]],
            [r0],
            t_eval=data.times,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            dense_output=False,
        )
        if sol.success:
            return sol.y[0]
        return np.full(len(data.times), np.nan)

    def _make_conc_func(self, data: PDData):
        """Build a piecewise-linear concentration interpolant."""
        if data.concentrations is None:
            return lambda t: 0.0
        times = data.times
        concs = data.concentrations

        def interp(t: float) -> float:
            return float(np.interp(t, times, concs, left=0.0, right=0.0))

        return interp

    def _default_params(self, data: PDData) -> dict[str, float]:
        r_mean = float(np.nanmean(data.response)) if len(data.response) > 0 else 1.0
        if self.idr_type in (1, 2):
            return {"Kin": r_mean, "Kout": 1.0, "EC50": 1.0, "Emax": 1.0}
        else:
            return {"Kin": r_mean, "Kout": 1.0, "IC50": 1.0, "Imax": 0.9}

    def _get_bounds(self, names: list[str]) -> list[tuple[float | None, float | None]]:
        bounds: list[tuple[float | None, float | None]] = []
        for n in names:
            if n == "Imax":
                bounds.append((0.0, 1.0))
            else:
                bounds.append((1e-8, None))
        return bounds


# ---------------------------------------------------------------------------
# Effect Compartment Model
# ---------------------------------------------------------------------------


class EffectCompartmentModel(PDModel):
    """
    Effect compartment (biophase) PD model.

    dCe/dt = Ke0 * (C_plasma - Ce)
    E = Emax * Ce^n / (EC50^n + Ce^n)

    Parameters: Ke0, Emax, EC50, n (Hill coefficient)
    """

    param_names = ["Ke0", "Emax", "EC50", "n"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        ke0 = float(params.get("Ke0", 1.0))
        emax = float(params.get("Emax", 1.0))
        ec50 = float(params.get("EC50", 1.0))
        hill = float(params.get("n", 1.0))

        if data.concentrations is None:
            return np.zeros(len(data.times))

        _concentrations = data.concentrations

        def conc_func(t: float) -> float:
            return float(np.interp(t, data.times, _concentrations, left=0.0, right=0.0))

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            ce = max(y[0], 0.0)
            c_plasma = max(conc_func(t), 0.0)
            dce = ke0 * (c_plasma - ce)
            return np.array([dce])

        sol = integrate.solve_ivp(
            odes,
            [float(data.times[0]), float(data.times[-1])],
            [0.0],
            t_eval=data.times,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            return np.full(len(data.times), np.nan)

        ce_arr = np.maximum(sol.y[0], 0.0)
        ce_n = ce_arr**hill
        ec50_n = ec50**hill
        effect = emax * ce_n / (ec50_n + ce_n + 1e-30)
        return effect

    def _default_params(self, data: PDData) -> dict[str, float]:
        return {
            "Ke0": 0.5,
            "Emax": float(np.nanmax(data.response)) if len(data.response) > 0 else 1.0,
            "EC50": 1.0,
            "n": 1.0,
        }


# ---------------------------------------------------------------------------
# Turnover Model
# ---------------------------------------------------------------------------


class TurnoverModel(PDModel):
    """
    Turnover (production/degradation) model.

    dR/dt = Kin*(1 + stim_in) - Kout*(1 + stim_out)*R

    where stim_in and stim_out are Emax-type functions of concentration.
    Both can be zero (pure turnover without drug effect).

    Parameters: Kin, Kout, EC50_in, Emax_in, EC50_out, Emax_out
    """

    param_names = ["Kin", "Kout", "EC50_in", "Emax_in", "EC50_out", "Emax_out"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        kin = float(params.get("Kin", 1.0))
        kout = float(params.get("Kout", 1.0))
        ec50_in = float(params.get("EC50_in", 1.0))
        emax_in = float(params.get("Emax_in", 0.0))
        ec50_out = float(params.get("EC50_out", 1.0))
        emax_out = float(params.get("Emax_out", 0.0))

        if kout <= 0:
            return np.full(len(data.times), np.nan)
        r0 = data.baseline if data.baseline is not None else kin / kout

        def conc_func(t: float) -> float:
            if data.concentrations is not None:
                return float(np.interp(t, data.times, data.concentrations, left=0.0, right=0.0))
            return 0.0

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            r = max(y[0], 0.0)
            c = max(conc_func(t), 0.0)
            s_in = emax_in * c / (ec50_in + c) if ec50_in + c > 0 else 0.0
            s_out = emax_out * c / (ec50_out + c) if ec50_out + c > 0 else 0.0
            drdt = kin * (1.0 + s_in) - kout * (1.0 + s_out) * r
            return np.array([drdt])

        sol = integrate.solve_ivp(
            odes,
            [0.0, float(np.max(data.times))],
            [r0],
            t_eval=data.times,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        if sol.success:
            return sol.y[0]
        return np.full(len(data.times), np.nan)

    def _default_params(self, data: PDData) -> dict[str, float]:
        r_mean = float(np.nanmean(data.response)) if len(data.response) > 0 else 1.0
        return {
            "Kin": r_mean,
            "Kout": 1.0,
            "EC50_in": 1.0,
            "Emax_in": 0.5,
            "EC50_out": 1.0,
            "Emax_out": 0.5,
        }


# ---------------------------------------------------------------------------
# Placebo Response Model
# ---------------------------------------------------------------------------


class PlaceboResponseModel(PDModel):
    """
    Placebo response model (pure time-course, no concentration dependence).

    E(t) = E0 * exp(-kdeg * t) + Eplacebo * (1 - exp(-kpl * t))

    Parameters: E0, kdeg, Eplacebo, kpl
    """

    param_names = ["E0", "kdeg", "Eplacebo", "kpl"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        e0 = float(params.get("E0", 1.0))
        kdeg = float(params.get("kdeg", 0.1))
        eplacebo = float(params.get("Eplacebo", 0.5))
        kpl = float(params.get("kpl", 0.1))
        t = data.times
        return e0 * np.exp(-kdeg * t) + eplacebo * (1.0 - np.exp(-kpl * t))

    def _default_params(self, data: PDData) -> dict[str, float]:
        r0 = float(data.response[0]) if len(data.response) > 0 else 1.0
        return {"E0": r0, "kdeg": 0.1, "Eplacebo": r0 * 0.5, "kpl": 0.05}


# ---------------------------------------------------------------------------
# Tumor Growth Inhibition Model (Simeoni 2004)
# ---------------------------------------------------------------------------


class TumorGrowthInhibitionModel(PDModel):
    """
    Simeoni (2004) TGI model.

    Compartment system (4 damage compartments):
      dX1/dt = lambda0*X1 / (1 + (lambda0/lambda1 * TotX)^psi)^(1/psi) - K2*C*X1
      dXi/dt = K2*C*X(i-1) - K1*X(i)   for i = 2,3,4
      TotX = X1 + X2 + X3 + X4

    Parameters: lambda0, lambda1, K1, K2, psi, X0
    IPRED = TotX
    """

    param_names = ["lambda0", "lambda1", "K1", "K2", "psi", "X0"]

    def predict(self, params: dict[str, float], data: PDData) -> np.ndarray:
        lam0 = float(params.get("lambda0", 0.1))
        lam1 = float(params.get("lambda1", 1.0))
        k1 = float(params.get("K1", 0.1))
        k2 = float(params.get("K2", 0.01))
        psi = float(params.get("psi", 20.0))
        x0 = float(params.get("X0", 1.0))

        def conc_func(t: float) -> float:
            if data.concentrations is not None:
                return float(np.interp(t, data.times, data.concentrations, left=0.0, right=0.0))
            return 0.0

        def odes(t: float, y: np.ndarray) -> np.ndarray:
            x1, x2, x3, x4 = max(y[0], 0.0), max(y[1], 0.0), max(y[2], 0.0), max(y[3], 0.0)
            tot = x1 + x2 + x3 + x4
            c = max(conc_func(t), 0.0)
            inner = lam0 / lam1 * tot if lam1 > 0 else 0.0
            growth_denom = (1.0 + inner**psi) ** (1.0 / psi) if psi > 0 and inner >= 0 else 1.0
            dx1 = lam0 * x1 / growth_denom - k2 * c * x1
            dx2 = k2 * c * x1 - k1 * x2
            dx3 = k1 * x2 - k1 * x3
            dx4 = k1 * x3 - k1 * x4
            return np.array([dx1, dx2, dx3, dx4])

        sol = integrate.solve_ivp(
            odes,
            [0.0, float(np.max(data.times))],
            [x0, 0.0, 0.0, 0.0],
            t_eval=data.times,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        if sol.success:
            return sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3]
        return np.full(len(data.times), np.nan)

    def _default_params(self, data: PDData) -> dict[str, float]:
        x0_est = float(data.response[0]) if len(data.response) > 0 else 1.0
        return {"lambda0": 0.1, "lambda1": 1.0, "K1": 0.1, "K2": 0.01, "psi": 20.0, "X0": x0_est}

    def _get_bounds(self, names: list[str]) -> list[tuple[float | None, float | None]]:
        return [(1e-8, None) for _ in names]


# ---------------------------------------------------------------------------
# Sequential PK/PD Workflow
# ---------------------------------------------------------------------------


class SequentialPKPDWorkflow:
    """
    Sequential PK/PD fitting workflow.

    Step 1: Uses post-hoc IPRED values from a fitted PK result as the
            concentration driver for PD fitting.
    Step 2: Fits the PD model using scipy.optimize (no PK re-estimation).

    Args:
        pk_result:  Fitted EstimationResult containing post_hoc_etas and model predictions.
        pk_model:   The fitted PopulationModel (for generating IPRED).
    """

    def __init__(self, pk_result: Any, pk_model: Any) -> None:
        self.pk_result = pk_result
        self.pk_model = pk_model

    def _extract_pk_concentrations(self, pd_data: PDData) -> np.ndarray:
        """Best-effort extraction of subject PK concentrations from fitted PK outputs."""
        try:
            individual = self.pk_model.individual_model(pd_data.subject_id)
            theta = np.asarray(self.pk_result.theta_final, dtype=float)
            sigma = np.asarray(getattr(self.pk_result, "sigma_final", np.eye(1)), dtype=float)
            post_hoc = getattr(self.pk_result, "post_hoc_etas", {})
            eta = np.asarray(post_hoc.get(pd_data.subject_id, np.array([])), dtype=float)

            obs_times = np.asarray(pd_data.times, dtype=float)
            if len(obs_times) == 0:
                return np.array([], dtype=float)

            subject_obs_times = np.asarray(individual.subject_events.obs_times, dtype=float)
            if len(subject_obs_times) > 0 and np.array_equal(obs_times, subject_obs_times):
                ipred, _obs_mask, _f = individual.evaluate(
                    theta, eta, sigma, trans=self.pk_model.trans
                )
                return np.asarray(ipred, dtype=float)

            from openpkpd.model.individual import _theta_to_pk_params

            if individual.pk_callable is not None:
                base_covariates = individual.subject_events.covariate_at(0.0)
                raw_params = individual.pk_callable(
                    list(theta),
                    list(eta),
                    t=0.0,
                    covariates=base_covariates,
                )
            else:
                raw_params = _theta_to_pk_params(theta, eta, self.pk_model.trans)

            try:
                micro_params = individual.pk_subroutine.apply_trans(raw_params, self.pk_model.trans)
            except Exception:
                micro_params = raw_params

            solve_kwargs: dict[str, object] = {}
            has_time_varying_covariates = (
                individual.subject_events.covariate_df is not None
                and individual.pk_callable is not None
                and individual.des_callable is not None
            )
            if has_time_varying_covariates:

                def _covariate_fn(t: float) -> dict:
                    covs = individual.subject_events.covariate_at(t)
                    raw = individual.pk_callable(list(theta), list(eta), t=t, covariates=covs)
                    try:
                        return individual.pk_subroutine.apply_trans(raw, self.pk_model.trans)
                    except Exception:
                        return raw

                solve_kwargs["covariate_fn"] = _covariate_fn
                solve_kwargs["covariate_change_times"] = (
                    individual.subject_events.covariate_change_times()
                )

            pk_sol = individual.pk_subroutine.solve(
                micro_params,
                individual.subject_events.dose_events,
                obs_times,
                pk_callable=None,
                des_callable=individual.des_callable,
                **solve_kwargs,
            )
            return np.asarray(pk_sol.ipred, dtype=float)
        except Exception:
            return np.zeros(len(pd_data.times), dtype=float)

    def fit_pd(
        self,
        pd_data: PDData,
        pd_model_cls: type[PDModel],
        initial_params: dict[str, float] | None = None,
        fixed_pk: bool = True,
    ) -> PDResult:
        """
        Fit a PD model using post-hoc PK predictions as concentrations.

        Args:
            pd_data:        PDData with times and response. If concentrations
                            is None, attempts to extract from pk_result IPRED.
            pd_model_cls:   PD model class (e.g. IndirectResponseModel).
            initial_params: Starting parameter values.
            fixed_pk:       If True, PK is not re-estimated (default True).

        Returns:
            PDResult from PD model fitting.
        """
        # If concentrations not provided, try to extract them from the fitted PK model.
        if pd_data.concentrations is None:
            concs = self._extract_pk_concentrations(pd_data)
            pd_data = PDData(
                subject_id=pd_data.subject_id,
                times=pd_data.times,
                response=pd_data.response,
                concentrations=concs,
                baseline=pd_data.baseline,
            )

        pd_model = pd_model_cls()
        return pd_model.fit(pd_data, initial_params=initial_params)
