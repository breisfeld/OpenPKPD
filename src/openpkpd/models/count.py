"""
Count data models for pharmacometric analysis.

Models for integer count outcomes such as seizure counts, lesion counts,
adverse event frequencies, and other non-negative integer responses.

Supported models:
  - Poisson (equi-dispersion)
  - Negative Binomial (over-dispersion)
  - Zero-Inflated Poisson (excess zeros)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class CountData:
    """Data container for count-model fitting.

    Attributes:
        subject_id: Unique subject identifier.
        counts: Integer count observations.
        times: Observation times aligned with counts.
        offsets: Log-exposure offsets (e.g. log(observation_window)); added
            to the linear predictor before taking exp.  If ``None``, no
            offset is applied.
        covariates: Dictionary mapping covariate names to arrays of values
            aligned with counts and times.
    """

    subject_id: int
    counts: np.ndarray
    times: np.ndarray
    offsets: np.ndarray | None = None
    covariates: dict[str, np.ndarray] | None = None


@dataclass
class CountResult:
    """Result from count model maximum-likelihood estimation.

    Attributes:
        rate_params: Fitted log-rate parameter vector (intercept + covariate
            slopes).
        dispersion: Fitted dispersion parameter for Negative Binomial; ``None``
            for Poisson and ZIP.
        ofv: Objective function value (−2 × log-likelihood).
        converged: Whether the optimiser reported convergence.
        aic: Akaike Information Criterion.
    """

    rate_params: np.ndarray
    dispersion: float | None
    ofv: float
    converged: bool
    aic: float


class CountModel(ABC):
    """Abstract base class for count data models.

    Subclasses must implement :meth:`log_pmf`.  Default implementations of
    :meth:`log_likelihood` and :meth:`fit` are provided and rely only on
    ``log_pmf`` and :meth:`mean_rate`.
    """

    @abstractmethod
    def log_pmf(self, k: int, mu: float, **kwargs: float) -> float:
        """Log probability mass function: log P(Y = k | mu, ...).

        Args:
            k: Observed integer count.
            mu: Expected (mean) count.
            **kwargs: Additional distribution parameters (e.g. dispersion).

        Returns:
            Log probability (−∞, 0].
        """

    def mean_rate(
        self,
        params: np.ndarray,
        covariates: dict[str, np.ndarray] | None = None,
        offset: float = 0.0,
    ) -> np.ndarray:
        """Compute expected rate from a log-linear model.

        The linear predictor is:
            eta = params[0] + sum_j params[j+1] * cov_j + offset
            mu  = exp(eta)

        Covariates are iterated in sorted key order to ensure a deterministic
        parameter–covariate mapping.

        Args:
            params: Parameter vector: [intercept, slope_1, slope_2, ...].
            covariates: Dict of covariate arrays, one value per observation.
            offset: Log-exposure offset (added to log-mean).

        Returns:
            Array of expected counts, shape (n_obs,).
        """
        eta = np.full(1, float(params[0]))

        if covariates is not None:
            for j, key in enumerate(sorted(covariates.keys())):
                if j + 1 < len(params):
                    eta = eta + float(params[j + 1]) * np.asarray(covariates[key], dtype=float)
        eta = eta + offset
        return np.exp(np.clip(eta, -500.0, 500.0))

    def _extra_kwargs(self, params: np.ndarray) -> dict[str, float]:
        """Extract extra model-specific kwargs from params.

        The base implementation returns an empty dict.  Subclasses that
        embed additional parameters (e.g. dispersion) at the tail of
        ``params`` should override this method.

        Args:
            params: Full parameter vector.

        Returns:
            Dict passed as **kwargs to log_pmf.
        """
        return {}

    def _n_rate_params(self, params: np.ndarray) -> int:
        """Number of log-rate parameters (excludes distribution parameters).

        Base implementation treats the entire vector as rate parameters.
        Subclasses that append extra parameters should override.

        Args:
            params: Full parameter vector.

        Returns:
            Number of rate parameters.
        """
        return len(params)

    def log_likelihood(self, data: list[CountData], params: np.ndarray) -> float:
        """Compute total log-likelihood across all subjects and observations.

        Args:
            data: List of per-subject CountData records.
            params: Combined parameter vector (rate params + extras).

        Returns:
            Total log-likelihood.
        """
        n_rate = self._n_rate_params(params)
        rate_params = params[:n_rate]
        extra = self._extra_kwargs(params)

        ll = 0.0
        for subj in data:
            for i, (k, _) in enumerate(zip(subj.counts, subj.times, strict=False)):
                cov_i: dict[str, np.ndarray] | None = None
                if subj.covariates is not None:
                    cov_i = {key: np.array([subj.covariates[key][i]]) for key in subj.covariates}
                offset = float(subj.offsets[i]) if subj.offsets is not None else 0.0
                mu_arr = self.mean_rate(rate_params, cov_i, offset)
                mu = float(mu_arr[0])
                ll += self.log_pmf(int(k), mu, **extra)

        return ll if np.isfinite(ll) else -1e300

    def fit(self, data: list[CountData], init_params: np.ndarray) -> CountResult:
        """Fit count model via maximum-likelihood estimation.

        Minimises −2 × log-likelihood using L-BFGS-B.

        Args:
            data: Per-subject CountData records.
            init_params: Initial parameter vector.

        Returns:
            CountResult with fitted parameters and diagnostics.
        """

        def neg_ll(params: np.ndarray) -> float:
            try:
                ll = self.log_likelihood(data, params)
                return -2.0 * ll if np.isfinite(ll) else 1e12
            except Exception:
                return 1e12

        bounds = self._default_bounds(init_params)
        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        n_rate = self._n_rate_params(params_hat)
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)
        dispersion = self._extract_dispersion(params_hat)

        return CountResult(
            rate_params=params_hat[:n_rate],
            dispersion=dispersion,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
        )

    def _default_bounds(self, params: np.ndarray) -> list[tuple[float | None, float | None]]:
        """Default parameter bounds (unconstrained).

        Subclasses may override to impose constraints (e.g. positivity of
        the dispersion parameter).

        Args:
            params: Initial parameter vector (used only for length).

        Returns:
            List of (lower, upper) bound tuples.
        """
        return [(None, None)] * len(params)

    def _extract_dispersion(self, params: np.ndarray) -> float | None:
        """Extract dispersion from fitted parameters, if present.

        Args:
            params: Fitted parameter vector.

        Returns:
            Dispersion value, or ``None`` if not applicable.
        """
        return None


class PoissonModel(CountModel):
    """Poisson count model with log-linear mean function.

    Y_i ~ Poisson(lambda_i),  log(lambda_i) = X_i @ beta + offset_i

    The Poisson distribution assumes equi-dispersion (Var = mean).
    Use :class:`NegativeBinomialModel` when counts show over-dispersion.
    """

    def log_pmf(self, k: int, mu: float, **kwargs: float) -> float:
        """Log Poisson PMF: log P(Y = k | Poisson(mu)).

        Args:
            k: Observed count.
            mu: Expected count (lambda > 0).
            **kwargs: Ignored.

        Returns:
            Log probability.
        """
        mu_safe = max(mu, 1e-300)
        return float(stats.poisson.logpmf(k, mu_safe))

    def mean_rate(
        self,
        params: np.ndarray,
        covariates: dict[str, np.ndarray] | None = None,
        offset: float = 0.0,
    ) -> np.ndarray:
        """Compute log-linear Poisson rate.

        Args:
            params: [intercept, slope_1, ...].
            covariates: Dict of covariate arrays.
            offset: Log-exposure offset.

        Returns:
            Expected count array.
        """
        return super().mean_rate(params, covariates, offset)


class NegativeBinomialModel(CountModel):
    """Negative Binomial model for over-dispersed counts.

    Y_i ~ NegBin(mu_i, r) where:
        E[Y] = mu
        Var[Y] = mu + mu^2 / r

    As r → ∞, NB approaches Poisson.

    The NB parameterisation used here follows SciPy's ``nbinom``:
        P(Y=k) = C(k+r-1, k) * p^r * (1-p)^k
    with p = r / (r + mu).

    Attributes:
        r: Fixed over-dispersion parameter.  If ``None``, ``r`` is estimated
           as the last element of the parameter vector.
    """

    def __init__(self, r: float | None = None) -> None:
        """Initialise Negative Binomial model.

        Args:
            r: Fixed dispersion parameter.  Pass ``None`` to estimate from
               data (appended to the end of ``init_params``).
        """
        self.r = r

    def log_pmf(self, k: int, mu: float, r: float = 1.0, **kwargs: float) -> float:
        """Log NegBin PMF.

        Args:
            k: Observed count.
            mu: Expected count (> 0).
            r: Over-dispersion parameter (> 0).  Ignored if ``self.r`` is set.

        Returns:
            Log probability.
        """
        r_use = self.r if self.r is not None else r
        r_safe = max(r_use, 1e-6)
        mu_safe = max(mu, 1e-300)
        p = r_safe / (r_safe + mu_safe)
        return float(stats.nbinom.logpmf(k, n=r_safe, p=p))

    def _n_rate_params(self, params: np.ndarray) -> int:
        """Rate params are all but last element when r is estimated.

        Args:
            params: Full parameter vector.

        Returns:
            Number of rate parameters.
        """
        if self.r is None:
            return max(len(params) - 1, 1)
        return len(params)

    def _extra_kwargs(self, params: np.ndarray) -> dict[str, float]:
        """Extract dispersion from params tail if not fixed.

        Args:
            params: Full parameter vector.

        Returns:
            Dict with key ``'r'`` when estimating dispersion.
        """
        if self.r is None and len(params) > 1:
            return {"r": float(np.exp(params[-1]))}
        return {}

    def _default_bounds(self, params: np.ndarray) -> list[tuple[float | None, float | None]]:
        """Rate params unconstrained; log(r) unconstrained (r > 0 by exp).

        Args:
            params: Initial parameter vector.

        Returns:
            Bounds list.
        """
        return [(None, None)] * len(params)

    def _extract_dispersion(self, params: np.ndarray) -> float | None:
        """Return fitted dispersion r.

        Args:
            params: Fitted parameter vector.

        Returns:
            Dispersion value or ``None``.
        """
        if self.r is not None:
            return self.r
        if len(params) > 1:
            return float(np.exp(params[-1]))
        return None

    def fit(self, data: list[CountData], init_params: np.ndarray) -> CountResult:
        """Fit Negative Binomial model via MLE.

        When ``r`` is ``None``, the last element of ``init_params`` should be
        log(r_init) (e.g. 0.0 for r=1).

        Args:
            data: Per-subject CountData records.
            init_params: Initial parameter vector.

        Returns:
            CountResult with fitted dispersion.
        """

        def neg_ll(params: np.ndarray) -> float:
            n_rate = self._n_rate_params(params)
            rate_params = params[:n_rate]
            extra: dict[str, float] = {}
            if self.r is None and len(params) > n_rate:
                extra = {"r": float(np.exp(params[-1]))}
            ll = 0.0
            for subj in data:
                for i, k in enumerate(subj.counts):
                    cov_i: dict[str, np.ndarray] | None = None
                    if subj.covariates is not None:
                        cov_i = {
                            key: np.array([subj.covariates[key][i]]) for key in subj.covariates
                        }
                    offset = float(subj.offsets[i]) if subj.offsets is not None else 0.0
                    mu_arr = self.mean_rate(rate_params, cov_i, offset)
                    mu = float(mu_arr[0])
                    ll += self.log_pmf(int(k), mu, **extra)
            return -2.0 * ll if np.isfinite(ll) else 1e12

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=[(None, None)] * len(init_params),
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        n_rate = self._n_rate_params(params_hat)
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)
        dispersion = self._extract_dispersion(params_hat)

        return CountResult(
            rate_params=params_hat[:n_rate],
            dispersion=dispersion,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
        )


class ZeroInflatedPoissonModel(CountModel):
    """Zero-inflated Poisson (ZIP) model for data with excess zeros.

    Mixture model:
        P(Y = 0) = pi + (1 - pi) * exp(-lambda)
        P(Y = k) = (1 - pi) * Poisson(k | lambda)   for k >= 1

    where pi is the zero-inflation probability (structural zeros) and
    lambda is the Poisson mean for the count component.

    The zero-inflation probability ``pi`` can be either fixed or estimated.
    When estimated, the last parameter in ``init_params`` is logit(pi).
    """

    def __init__(self, zero_prob: float | None = None) -> None:
        """Initialise ZIP model.

        Args:
            zero_prob: Fixed zero-inflation probability in [0, 1).  Pass
                ``None`` to estimate from data (appended to params as
                logit(pi)).
        """
        self.zero_prob = zero_prob

    def log_pmf(self, k: int, mu: float, zero_prob: float = 0.0, **kwargs: float) -> float:
        """Log ZIP PMF.

        Args:
            k: Observed count.
            mu: Poisson mean of count component (> 0).
            zero_prob: Zero-inflation probability in [0, 1).  Ignored if
                ``self.zero_prob`` is set.

        Returns:
            Log probability.
        """
        pi = self.zero_prob if self.zero_prob is not None else zero_prob
        pi = float(np.clip(pi, 0.0, 1.0 - 1e-10))
        mu_safe = max(mu, 1e-300)

        if k == 0:
            log_p = np.log(pi + (1.0 - pi) * np.exp(-mu_safe) + 1e-300)
        else:
            log_p = np.log(1.0 - pi + 1e-300) + stats.poisson.logpmf(k, mu_safe)
        return float(log_p)

    def _n_rate_params(self, params: np.ndarray) -> int:
        """Rate params are all but last element when pi is estimated.

        Args:
            params: Full parameter vector.

        Returns:
            Number of rate parameters.
        """
        if self.zero_prob is None:
            return max(len(params) - 1, 1)
        return len(params)

    def _extra_kwargs(self, params: np.ndarray) -> dict[str, float]:
        """Extract zero_prob from params tail when not fixed.

        Args:
            params: Full parameter vector.

        Returns:
            Dict with key ``'zero_prob'`` when estimating pi.
        """
        if self.zero_prob is None and len(params) > 1:
            logit_pi = float(params[-1])
            pi = 1.0 / (1.0 + np.exp(-logit_pi))
            return {"zero_prob": float(pi)}
        return {}

    def _default_bounds(self, params: np.ndarray) -> list[tuple[float | None, float | None]]:
        """All parameters unconstrained (logit transform handles [0,1]).

        Args:
            params: Initial parameter vector.

        Returns:
            Bounds list.
        """
        return [(None, None)] * len(params)

    def _extract_dispersion(self, params: np.ndarray) -> float | None:
        """Return None — ZIP has no dispersion parameter per se.

        Args:
            params: Fitted parameter vector.

        Returns:
            Always ``None``.
        """
        return None

    def fit(self, data: list[CountData], init_params: np.ndarray) -> CountResult:
        """Fit ZIP model via MLE.

        When ``zero_prob`` is ``None``, the last element of ``init_params``
        should be logit(pi_init).

        Args:
            data: Per-subject CountData records.
            init_params: Initial parameter vector.

        Returns:
            CountResult (dispersion is None for ZIP).
        """

        def neg_ll(params: np.ndarray) -> float:
            n_rate = self._n_rate_params(params)
            rate_params = params[:n_rate]
            extra: dict[str, float] = {}
            if self.zero_prob is None and len(params) > n_rate:
                logit_pi = float(params[-1])
                pi = 1.0 / (1.0 + np.exp(-logit_pi))
                extra = {"zero_prob": float(pi)}
            ll = 0.0
            for subj in data:
                for i, k in enumerate(subj.counts):
                    cov_i: dict[str, np.ndarray] | None = None
                    if subj.covariates is not None:
                        cov_i = {
                            key: np.array([subj.covariates[key][i]]) for key in subj.covariates
                        }
                    offset = float(subj.offsets[i]) if subj.offsets is not None else 0.0
                    mu_arr = self.mean_rate(rate_params, cov_i, offset)
                    mu = float(mu_arr[0])
                    ll += self.log_pmf(int(k), mu, **extra)
            return -2.0 * ll if np.isfinite(ll) else 1e12

        result = optimize.minimize(
            neg_ll,
            init_params,
            method="L-BFGS-B",
            bounds=[(None, None)] * len(init_params),
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        params_hat = result.x
        n_rate = self._n_rate_params(params_hat)
        ofv = float(result.fun)
        aic = ofv + 2.0 * len(params_hat)

        return CountResult(
            rate_params=params_hat[:n_rate],
            dispersion=None,
            ofv=ofv,
            converged=bool(result.success),
            aic=aic,
        )
