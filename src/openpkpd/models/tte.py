"""
Time-to-event (survival) models for pharmacometric analysis.

Implements parametric survival models with exposure-dependent hazards.
Supports single events and repeated TTE (multiple events per subject).

The likelihood contributions are:
  - Observed event at t_i:  log h(t_i)  +  log S(t_i)  =  log h(t_i) - H(t_i)
  - Censored at t_i:         log S(t_i)                 =  -H(t_i)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import integrate, optimize


class HazardFunction(Enum):
    """Enumeration of supported parametric hazard function families."""

    CONSTANT = "constant"
    WEIBULL = "weibull"
    GOMPERTZ = "gompertz"
    LOG_LOGISTIC = "loglogistic"
    LOG_NORMAL = "lognormal"
    CUSTOM = "custom"


@dataclass
class TTEData:
    """Data structure for time-to-event analysis.

    Attributes:
        subject_id: Unique subject identifier.
        event_times: Array of event or censoring times (length n_events).
        event_indicator: 1 = event observed, 0 = censored (length n_events).
        concentration_times: Optional times at which PK concentrations were measured.
        concentrations: PK concentrations matched to concentration_times.
    """

    subject_id: int
    event_times: np.ndarray
    event_indicator: np.ndarray
    concentration_times: np.ndarray | None = None
    concentrations: np.ndarray | None = None


@dataclass
class TTEResult:
    """Result from TTE model maximum-likelihood estimation.

    Attributes:
        hazard_params: Fitted hazard-function parameter vector.
        baseline_hazard: Baseline hazard rate (first element of hazard_params).
        exposure_effect: Effect size of concentration on hazard.
        ofv: Objective function value (−2 × log-likelihood).
        converged: Whether the optimizer reported convergence.
        aic: Akaike Information Criterion (OFV + 2 × n_params).
        survival_function: Callable S(t) -> float using fitted parameters.
    """

    hazard_params: np.ndarray
    baseline_hazard: float
    exposure_effect: float
    ofv: float
    converged: bool
    aic: float
    survival_function: Callable[[float], float]


class TTEModel(ABC):
    """Abstract base class for parametric time-to-event models.

    Subclasses must implement :meth:`hazard`.  Default implementations of
    :meth:`cumulative_hazard` and :meth:`survival` use numerical quadrature
    so that subclasses with analytical expressions may override them for
    efficiency.
    """

    @abstractmethod
    def hazard(
        self,
        t: float | np.ndarray,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float | np.ndarray:
        """Instantaneous hazard rate h(t).

        Args:
            t: Time(s) at which to evaluate the hazard.
            params: Parameter vector for the hazard function.
            concentration: Drug concentration at time t (default 0).

        Returns:
            Hazard value(s) >= 0.
        """

    def cumulative_hazard(
        self,
        t: float,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float:
        """Cumulative hazard H(t) = integral_0^t h(s) ds.

        Uses adaptive quadrature.  Subclasses with analytical expressions
        should override this method for better performance.

        Args:
            t: Upper integration limit.
            params: Hazard-function parameter vector.
            concentration: Drug concentration (held constant over [0, t]).

        Returns:
            Non-negative cumulative hazard.
        """
        if t <= 0.0:
            return 0.0
        result, _ = integrate.quad(
            lambda s: float(self.hazard(s, params, concentration)),
            0.0,
            t,
            limit=100,
            epsabs=1e-8,
            epsrel=1e-6,
        )
        return max(result, 0.0)

    def survival(
        self,
        t: float,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float:
        """Survival function S(t) = exp(-H(t)).

        Args:
            t: Time at which to evaluate survival.
            params: Hazard-function parameter vector.
            concentration: Drug concentration at time t.

        Returns:
            Survival probability in (0, 1].
        """
        return float(np.exp(-self.cumulative_hazard(t, params, concentration)))

    def _subject_log_likelihood(self, data: TTEData, params: np.ndarray) -> float:
        """Log-likelihood contribution for a single subject.

        For each observation interval:
          - Event: log h(t_event) - H(t_event)  [starts from 0 for single TTE,
            or from previous event time for repeated TTE]
          - Censored: -H(t_censor)

        Args:
            data: TTEData for one subject.
            params: Current parameter vector.

        Returns:
            Log-likelihood contribution (scalar).
        """
        ll = 0.0
        for _, (t, ev) in enumerate(zip(data.event_times, data.event_indicator, strict=False)):
            # Concentration at this time (constant approximation)
            conc = self._concentration_at(t, data)
            h = float(self.hazard(max(t, 1e-12), params, conc))
            H = self.cumulative_hazard(max(t, 0.0), params, conc)
            if ev == 1:
                ll += np.log(max(h, 1e-300)) - H
            else:
                ll -= H
        return ll

    @staticmethod
    def _concentration_at(t: float, data: TTEData) -> float:
        """Interpolate PK concentration at time t.

        Args:
            t: Query time.
            data: TTEData potentially containing PK profile.

        Returns:
            Interpolated concentration, or 0.0 if no PK data available.
        """
        if data.concentration_times is None or data.concentrations is None:
            return 0.0
        return float(np.interp(t, data.concentration_times, data.concentrations))

    def _fit_bounds(self, init_params: np.ndarray) -> list[tuple[float | None, float | None]]:
        """Return optimizer bounds for ``fit()``.

        By default, all parameters are constrained positive. Hazard-model
        subclasses with unconstrained trend or exposure-effect parameters
        should override this helper so repeated-TTE fitting can inherit the
        same feasible region as the underlying base model.
        """
        return [(1e-9, None)] * len(init_params)

    def log_likelihood(self, data: list[TTEData], params: np.ndarray) -> float:
        """Compute total log-likelihood across all subjects.

        Args:
            data: List of per-subject TTEData records.
            params: Parameter vector for the hazard function.

        Returns:
            Total log-likelihood (sum over subjects).
        """
        return sum(self._subject_log_likelihood(d, params) for d in data)

    def fit(self, data: list[TTEData], init_params: np.ndarray) -> TTEResult:
        """Fit model via maximum-likelihood estimation.

        Minimises −2 × log-likelihood using L-BFGS-B with parameter bounds
        enforcing positivity where required.

        Args:
            data: List of per-subject TTEData records.
            init_params: Initial parameter vector (must be feasible).

        Returns:
            TTEResult with fitted parameters and diagnostics.
        """

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        bounds = self._fit_bounds(init_params)

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        n_params = len(params_hat)
        ofv = float(result.fun)
        aic = ofv + 2.0 * n_params

        exposure_effect = float(params_hat[2]) if n_params > 2 else 0.0

        # Build closure capturing fitted params
        fitted_params = params_hat.copy()
        model_ref = self

        def survival_fn(t: float, c: float = 0.0) -> float:
            return model_ref.survival(t, fitted_params, c)

        return TTEResult(
            hazard_params=params_hat,
            baseline_hazard=float(params_hat[0]),
            exposure_effect=exposure_effect,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
            survival_function=survival_fn,
        )


class ConstantHazardModel(TTEModel):
    """Exponential survival model with optional linear concentration effect.

    Hazard function:
        h(t) = lambda * (1 + beta * C)

    where lambda = params[0] is the baseline hazard and beta = params[1]
    (optional) is the slope of the concentration effect.

    The cumulative hazard is analytical: H(t) = h * t.
    """

    def hazard(
        self,
        t: float | np.ndarray,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float | np.ndarray:
        """Constant hazard with optional proportional concentration modifier.

        Args:
            t: Time (unused; hazard is constant).
            params: [lambda] or [lambda, beta].
            concentration: Drug concentration.

        Returns:
            Hazard value >= 1e-10.
        """
        lam = float(params[0])
        if len(params) > 1:
            lam = lam * (1.0 + float(params[1]) * concentration)
        return max(lam, 1e-10)

    def cumulative_hazard(
        self,
        t: float,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float:
        """Analytical cumulative hazard: H(t) = lambda * t.

        Overrides the numerical quadrature from the base class.

        Args:
            t: Upper limit (time).
            params: [lambda] or [lambda, beta].
            concentration: Drug concentration.

        Returns:
            H(t) >= 0.
        """
        return float(self.hazard(t, params, concentration)) * max(t, 0.0)


class WeibullHazardModel(TTEModel):
    """Weibull survival model with optional log-linear concentration effect.

    Hazard function:
        h(t) = (p / scale) * (t / scale)^(p-1) * exp(beta * C)

    Parameters:
        params[0] = scale (lambda)  > 0
        params[1] = shape (p)       > 0
        params[2] = beta (optional) in R  — log-linear concentration effect

    Special cases:
        p == 1 : reduces to exponential (constant hazard)
        p >  1 : increasing hazard (e.g. ageing)
        p <  1 : decreasing hazard

    The cumulative hazard has an analytical form:
        H(t) = (t / scale)^p * exp(beta * C)
    """

    def hazard(
        self,
        t: float | np.ndarray,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float | np.ndarray:
        """Weibull instantaneous hazard.

        Args:
            t: Time (scalar or array).
            params: [scale, shape] or [scale, shape, beta].
            concentration: Drug concentration.

        Returns:
            Hazard value(s) >= 0.
        """
        scale = max(float(params[0]), 1e-10)
        p = max(float(params[1]), 1e-10)
        t_safe = np.maximum(np.asarray(t, dtype=float), 1e-12)
        baseline = (p / scale) * (t_safe / scale) ** (p - 1.0)
        if len(params) > 2:
            baseline = baseline * np.exp(float(params[2]) * concentration)
        return baseline

    def cumulative_hazard(
        self,
        t: float,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float:
        """Analytical Weibull cumulative hazard: H(t) = (t/scale)^p * exp(beta*C).

        Overrides the numerical quadrature from the base class.

        Args:
            t: Upper limit (time).
            params: [scale, shape] or [scale, shape, beta].
            concentration: Drug concentration.

        Returns:
            H(t) >= 0.
        """
        scale = max(float(params[0]), 1e-10)
        p = max(float(params[1]), 1e-10)
        t_safe = max(t, 0.0)
        H = (t_safe / scale) ** p
        if len(params) > 2:
            H *= np.exp(float(params[2]) * concentration)
        return float(H)

    def _fit_bounds(self, init_params: np.ndarray) -> list[tuple[float | None, float | None]]:
        """scale and shape remain positive; exposure effects are unconstrained."""
        n_p = len(init_params)
        if n_p >= 3:
            return [(1e-9, None), (1e-9, None)] + [(None, None)] * (n_p - 2)
        return [(1e-9, None)] * n_p

    def fit(self, data: list[TTEData], init_params: np.ndarray) -> TTEResult:
        """Fit Weibull model via MLE.

        Allows negative beta (concentration can be protective or hazardous).

        Args:
            data: Per-subject TTEData records.
            init_params: Initial [scale, shape] or [scale, shape, beta].

        Returns:
            TTEResult with fitted parameters.
        """

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        bounds = self._fit_bounds(init_params)

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        n_params = len(params_hat)
        ofv = float(result.fun)
        aic = ofv + 2.0 * n_params
        exposure_effect = float(params_hat[2]) if n_params > 2 else 0.0

        fitted_params = params_hat.copy()
        model_ref = self

        def survival_fn(t: float, c: float = 0.0) -> float:
            return model_ref.survival(t, fitted_params, c)

        return TTEResult(
            hazard_params=params_hat,
            baseline_hazard=float(params_hat[0]),
            exposure_effect=exposure_effect,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
            survival_function=survival_fn,
        )


class GompertzHazardModel(TTEModel):
    """Gompertz survival model with optional log-linear concentration effect.

    Hazard function:
        h(t) = alpha * exp(beta * t) * exp(gamma * C)

    Parameters:
        params[0] = alpha  > 0   baseline rate
        params[1] = beta   (R)   time-trend (> 0: increasing, < 0: decreasing)
        params[2] = gamma  (R, optional)  log-linear concentration effect

    Cumulative hazard:
        H(t) = (alpha / beta) * (exp(beta * t) - 1)  * exp(gamma * C)
        when beta == 0: H(t) = alpha * t * exp(gamma * C)
    """

    def hazard(
        self,
        t: float | np.ndarray,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float | np.ndarray:
        alpha = max(float(params[0]), 1e-10)
        beta = float(params[1]) if len(params) > 1 else 0.0
        gamma = float(params[2]) if len(params) > 2 else 0.0
        t_arr = np.asarray(t, dtype=float)
        return alpha * np.exp(beta * t_arr + gamma * concentration)

    def cumulative_hazard(
        self,
        t: float,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float:
        alpha = max(float(params[0]), 1e-10)
        beta = float(params[1]) if len(params) > 1 else 0.0
        gamma = float(params[2]) if len(params) > 2 else 0.0
        t_safe = max(t, 0.0)
        conc_factor = np.exp(gamma * concentration)
        if abs(beta) < 1e-12:
            return float(alpha * t_safe * conc_factor)
        return float(alpha / beta * (np.exp(beta * t_safe) - 1.0) * conc_factor)

    def _fit_bounds(self, init_params: np.ndarray) -> list[tuple[float | None, float | None]]:
        """alpha stays positive; trend and exposure effects remain unconstrained."""
        return [(1e-9, None)] + [(None, None)] * (len(init_params) - 1)

    def fit(self, data: list[TTEData], init_params: np.ndarray) -> TTEResult:
        """Fit Gompertz model — beta is unconstrained (sign indicates trend direction)."""

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        bounds = self._fit_bounds(init_params)

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)
        exposure_effect = float(params_hat[2]) if len(params_hat) > 2 else 0.0

        fitted_params = params_hat.copy()
        model_ref = self

        def survival_fn(t: float, c: float = 0.0) -> float:
            return model_ref.survival(t, fitted_params, c)

        return TTEResult(
            hazard_params=params_hat,
            baseline_hazard=float(params_hat[0]),
            exposure_effect=exposure_effect,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
            survival_function=survival_fn,
        )


class LogLogisticHazardModel(TTEModel):
    """Log-logistic survival model with optional log-linear concentration effect.

    Hazard function:
        h(t) = (p / scale) * (t / scale)^(p-1) / (1 + (t / scale)^p)
               * exp(gamma * C)

    The log-logistic distribution has a non-monotone hazard (increases then
    decreases) when p > 1, making it useful for cancer survival data.

    Parameters:
        params[0] = scale (lambda) > 0
        params[1] = shape (p)      > 0
        params[2] = gamma (optional, unconstrained) — log-linear conc effect

    Cumulative hazard (analytical):
        H(t) = log(1 + (t / scale)^p) * exp(gamma * C)
    """

    def hazard(
        self,
        t: float | np.ndarray,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float | np.ndarray:
        scale = max(float(params[0]), 1e-10)
        p = max(float(params[1]), 1e-10)
        gamma = float(params[2]) if len(params) > 2 else 0.0
        t_arr = np.maximum(np.asarray(t, dtype=float), 1e-12)
        ratio = t_arr / scale
        baseline = (p / scale) * ratio ** (p - 1.0) / (1.0 + ratio**p)
        return baseline * np.exp(gamma * concentration)

    def cumulative_hazard(
        self,
        t: float,
        params: np.ndarray,
        concentration: float = 0.0,
    ) -> float:
        scale = max(float(params[0]), 1e-10)
        p = max(float(params[1]), 1e-10)
        gamma = float(params[2]) if len(params) > 2 else 0.0
        t_safe = max(t, 0.0)
        H = float(np.log1p((t_safe / scale) ** p))
        return H * np.exp(gamma * concentration)

    def _fit_bounds(self, init_params: np.ndarray) -> list[tuple[float | None, float | None]]:
        """scale and shape stay positive; concentration effects are unconstrained."""
        return [(1e-9, None), (1e-9, None)] + [(None, None)] * max(len(init_params) - 2, 0)

    def fit(self, data: list[TTEData], init_params: np.ndarray) -> TTEResult:
        """Fit log-logistic model; gamma (concentration effect) is unconstrained."""

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        bounds = self._fit_bounds(init_params)

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)
        exposure_effect = float(params_hat[2]) if len(params_hat) > 2 else 0.0

        fitted_params = params_hat.copy()
        model_ref = self

        def survival_fn(t: float, c: float = 0.0) -> float:
            return model_ref.survival(t, fitted_params, c)

        return TTEResult(
            hazard_params=params_hat,
            baseline_hazard=float(params_hat[0]),
            exposure_effect=exposure_effect,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
            survival_function=survival_fn,
        )


class RepeatedTTEModel:
    """Repeated time-to-event model supporting multiple events per subject.

    Handles recurrent events by modelling either:
    - Gap time: time since the last event (inter-event intervals).
    - Calendar time: absolute time from start of follow-up.

    For gap-time analysis, each inter-event interval is treated as an
    independent survival time with the same hazard model.  For calendar-time
    analysis, the cumulative hazard is integrated over the entire follow-up.

    Attributes:
        base_model: The underlying parametric hazard model.
        time_scale: ``'gap'`` or ``'calendar'``.
    """

    def __init__(self, base_model: TTEModel, time_scale: str = "gap") -> None:
        """Initialise repeated TTE model.

        Args:
            base_model: Parametric hazard model for individual inter-event
                intervals.
            time_scale: ``'gap'`` uses time-since-last-event; ``'calendar'``
                uses absolute time.
        """
        if time_scale not in ("gap", "calendar"):
            raise ValueError("time_scale must be 'gap' or 'calendar'.")
        self.base_model = base_model
        self.time_scale = time_scale

    def _gap_times(self, event_times: np.ndarray) -> np.ndarray:
        """Convert absolute event times to inter-event gap times.

        Args:
            event_times: Sorted array of absolute event/censoring times.

        Returns:
            Gap times starting from 0.
        """
        times = np.concatenate(([0.0], event_times))
        return np.diff(times)

    def log_likelihood(self, data: list[TTEData], params: np.ndarray) -> float:
        """Compute total log-likelihood for repeated TTE data.

        Under the gap-time model each inter-event interval (t_{i-1}, t_i]
        contributes independently.  Under the calendar-time model, the hazard
        process is never reset.

        Args:
            data: Per-subject TTEData.  event_times may contain multiple
                entries per subject for recurrent events.
            params: Parameter vector passed to base_model.

        Returns:
            Total log-likelihood across all subjects.
        """
        ll = 0.0
        for subj in data:
            if self.time_scale == "gap":
                gaps = self._gap_times(subj.event_times)
                for i, (gap, ev) in enumerate(zip(gaps, subj.event_indicator, strict=False)):
                    # Absolute event time for concentration lookup
                    abs_t = float(subj.event_times[i])
                    conc = TTEModel._concentration_at(abs_t, subj)
                    gap_safe = max(gap, 1e-12)
                    h = float(self.base_model.hazard(gap_safe, params, conc))
                    H = self.base_model.cumulative_hazard(gap_safe, params, conc)
                    if ev == 1:
                        ll += np.log(max(h, 1e-300)) - H
                    else:
                        ll -= H
            else:
                # Calendar time: integrate over [0, t_i] for each event
                prev_t = 0.0
                for _, (t, ev) in enumerate(
                    zip(subj.event_times, subj.event_indicator, strict=False)
                ):
                    conc = TTEModel._concentration_at(float(t), subj)
                    t_safe = max(float(t), 1e-12)
                    h = float(self.base_model.hazard(t_safe, params, conc))
                    H_full = self.base_model.cumulative_hazard(t_safe, params, conc)
                    H_prev = self.base_model.cumulative_hazard(max(prev_t, 0.0), params, conc)
                    delta_H = max(H_full - H_prev, 0.0)
                    if ev == 1:
                        ll += np.log(max(h, 1e-300)) - delta_H
                    else:
                        ll -= delta_H
                    prev_t = float(t)
        return ll

    def fit(self, data: list[TTEData], init_params: np.ndarray) -> TTEResult:
        """Fit repeated TTE model via maximum-likelihood estimation.

        Args:
            data: Per-subject TTEData records (potentially multi-event).
            init_params: Initial parameter vector.

        Returns:
            TTEResult with fitted parameters and diagnostics.
        """

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        bounds = self.base_model._fit_bounds(init_params)
        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        n_params = len(params_hat)
        ofv = float(result.fun)
        aic = ofv + 2.0 * n_params
        exposure_effect = float(params_hat[2]) if n_params > 2 else 0.0

        fitted_params = params_hat.copy()
        base = self.base_model

        def survival_fn(t: float, c: float = 0.0) -> float:
            return base.survival(t, fitted_params, c)

        return TTEResult(
            hazard_params=params_hat,
            baseline_hazard=float(params_hat[0]),
            exposure_effect=exposure_effect,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
            survival_function=survival_fn,
        )
