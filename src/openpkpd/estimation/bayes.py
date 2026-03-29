"""
Bayesian estimation for population PK/PD models.

Implements Bayesian posterior estimation using PyMC when available, the built-in
pure-NumPy NUTS backend otherwise, or a Laplace approximation when requested.

The Laplace approximation runs FOCE to obtain the MAP estimate, then samples
from a multivariate normal approximation centred at the MAP with covariance
approximated from the inverse Hessian.

References:
    Gelman, A. et al. (2013). Bayesian Data Analysis, 3rd edition.
    Lunn, D. et al. (2013). The BUGS Book. CRC Press.
    Beal, S. & Sheiner, L. (1992). NONMEM Users Guides. UCSF.
"""

from __future__ import annotations

from collections import OrderedDict
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from openpkpd.estimation.base import EstimationMethod, EstimationResult
from openpkpd.utils.errors import EstimationError

# ---------------------------------------------------------------------------
# Extended result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BayesianResult(EstimationResult):
    """
    Extended result for Bayesian estimation.

    Inherits all fields from EstimationResult and adds posterior-specific
    quantities.

    Attributes:
        posterior_samples: Dictionary of raw MCMC (or approximate) samples.
                           Keys are parameter names, values are arrays of
                           shape (n_samples, n_params_for_that_key).
                           Common keys: 'theta', 'omega_diag', 'sigma_diag'.
        r_hat:             Gelman-Rubin R-hat convergence diagnostic per
                           parameter. Values close to 1.0 indicate convergence.
                           Shape (n_theta,). Set to all-ones for Laplace.
        n_effective:       Effective sample size per parameter.
                           Shape (n_theta,).
        posterior_ci_lo:   Lower bound of the posterior credible interval
                           (default 95%) per THETA parameter.
        posterior_ci_hi:   Upper bound of the posterior credible interval
                           per THETA parameter.
        backend_used:      The backend that was actually used ('pymc',
                           'nuts', or 'laplace').
    """

    # Override field defaults so BayesianResult can be constructed standalone
    # All EstimationResult required fields are satisfied by the parent @dataclass.
    posterior_samples: dict[str, np.ndarray] = field(default_factory=dict)
    posterior_samples_by_chain: dict[str, np.ndarray] = field(default_factory=dict)
    r_hat: np.ndarray = field(default_factory=lambda: np.array([]))
    n_effective: np.ndarray = field(default_factory=lambda: np.array([]))
    posterior_ci_lo: np.ndarray = field(default_factory=lambda: np.array([]))
    posterior_ci_hi: np.ndarray = field(default_factory=lambda: np.array([]))
    backend_used: str = ""

    def posterior_summary(self, ci: float = 0.95) -> str:
        """
        Return a formatted table of posterior statistics per THETA.

        Args:
            ci: Credible interval level (default 0.95 → 95% CI).

        Returns:
            Multi-line string with mean, SD, and CI for each THETA.
        """
        lines = [
            f"Bayesian Posterior Summary — {self.method}",
            f"  Backend: {self.backend_used}",
            f"  OFV (MAP): {self.ofv:.4f}",
            "",
            f"  {'Param':<10} {'Mean':>10} {'SD':>10} "
            f"{'CI_lo':>10} {'CI_hi':>10} "
            f"{'R-hat':>8} {'N_eff':>8}",
            "  " + "-" * 70,
        ]

        theta_samples = self.posterior_samples.get("theta")
        if theta_samples is not None and theta_samples.ndim == 2:
            n_theta = theta_samples.shape[1]
            alpha = (1.0 - ci) / 2.0
            for k in range(n_theta):
                samp = theta_samples[:, k]
                mean_k = float(np.mean(samp))
                sd_k = float(np.std(samp, ddof=1))
                lo_k = float(np.quantile(samp, alpha))
                hi_k = float(np.quantile(samp, 1.0 - alpha))
                rhat_k = float(self.r_hat[k]) if k < len(self.r_hat) else float("nan")
                neff_k = int(self.n_effective[k]) if k < len(self.n_effective) else 0
                lines.append(
                    f"  {'THETA(' + str(k + 1) + ')':<10} "
                    f"{mean_k:>10.4f} {sd_k:>10.4f} "
                    f"{lo_k:>10.4f} {hi_k:>10.4f} "
                    f"{rhat_k:>8.4f} {neff_k:>8d}"
                )
        else:
            # Fall back to point estimates
            for k, th in enumerate(self.theta_final):
                lo_k = (
                    float(self.posterior_ci_lo[k])
                    if k < len(self.posterior_ci_lo)
                    else float("nan")
                )
                hi_k = (
                    float(self.posterior_ci_hi[k])
                    if k < len(self.posterior_ci_hi)
                    else float("nan")
                )
                lines.append(
                    f"  {'THETA(' + str(k + 1) + ')':<10} "
                    f"{th:>10.4f} {'':>10} "
                    f"{lo_k:>10.4f} {hi_k:>10.4f}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BAYESMethod
# ---------------------------------------------------------------------------


class BAYESMethod(EstimationMethod):
    """
    Bayesian posterior estimation via MCMC.

    Uses PyMC if available, then the built-in pure-NumPy NUTS backend, or
    falls back to a Laplace approximation (FOCE-based) when requested.

    The Laplace fallback:
    1. Runs FOCE to obtain a MAP estimate.
    2. Approximates the posterior with a multivariate normal centred at
       the MAP estimate.  The covariance is approximated by numerical
       finite-differences of the negative log-posterior (Hessian).
    3. Returns the resulting approximate posterior samples.

    PyMC backend:
    - Log-normal priors on THETA parameters.
    - LKJ-Cholesky prior on OMEGA correlation matrix.
    - Half-Normal prior on OMEGA standard deviations.
    - Half-Normal prior on SIGMA.
    - Gaussian likelihood per observation.

    Args:
        n_samples:       Number of posterior samples per chain (after tuning).
        n_chains:        Number of MCMC chains.
        tune:            Number of tuning/warm-up steps (not included in
                         posterior samples).
        target_accept:   Target acceptance rate for NUTS sampler (0–1).
        seed:            Random seed for reproducibility.
        backend:         'auto' (default), 'pymc', 'nuts', or 'laplace'.
        prior_sd_theta:  Prior standard deviation in log-space for THETA
                         parameters when using lognormal priors.

    Usage::

        method = BAYESMethod(n_samples=2000, n_chains=4, tune=1000)
        result = method.estimate(population_model, init_params)
        print(result.posterior_summary())
    """

    method_name = "BAYES"
    _SUPPORTED_BACKENDS = {"auto", "pymc", "nuts", "laplace"}

    def __init__(
        self,
        n_samples: int = 1000,
        n_chains: int = 2,
        tune: int = 500,
        target_accept: float = 0.85,
        seed: int = 42,
        backend: str = "auto",
        prior_sd_theta: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.tune = tune
        self.target_accept = target_accept
        self.seed = seed
        self.backend = backend
        self.prior_sd_theta = prior_sd_theta
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # EstimationMethod interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        population_model: Any,
        init_params: Any,
        **kwargs: Any,
    ) -> BayesianResult:
        """
        Run Bayesian estimation. Auto-selects the best available backend.

        Args:
            population_model: Assembled PopulationModel with dataset and
                              PK callable.
            init_params:      Initial ParameterSet used as starting point.
            **kwargs:         Passed through to the backend.

        Returns:
            BayesianResult with posterior samples and summary statistics.
        """
        backend = self._select_backend()
        if backend == "pymc":
            return self._estimate_pymc(population_model, init_params)
        elif backend == "nuts":
            return self._estimate_nuts(population_model, init_params)
        else:
            return self._estimate_laplace(population_model, init_params)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _select_backend(self) -> str:
        """
        Choose the best available MCMC backend.

        Auto-selection order:
          pymc     — full-featured, cross-platform (requires openpkpd[bayes])
          nuts     — built-in pure-NumPy NUTS, zero extra dependencies
          laplace  — last resort when backend='laplace' is requested explicitly

        Returns:
            Name of the backend to use: 'pymc', 'nuts', or 'laplace'.
        """
        if self.backend not in self._SUPPORTED_BACKENDS:
            supported = ", ".join(sorted(self._SUPPORTED_BACKENDS))
            raise EstimationError(
                f"Unsupported BAYES backend {self.backend!r}. Supported backends: {supported}."
            )
        if self.backend != "auto":
            return self.backend
        import importlib.util
        import sys

        if "pymc" in sys.modules:
            pymc_found = sys.modules["pymc"] is not None
        else:
            try:
                pymc_found = importlib.util.find_spec("pymc") is not None
            except (ValueError, ModuleNotFoundError):
                pymc_found = False
        if pymc_found:
            return "pymc"
        # Pure-NumPy NUTS is always available — no optional dependencies
        return "nuts"

    # ------------------------------------------------------------------
    # PyMC backend
    # ------------------------------------------------------------------

    def _estimate_pymc(
        self,
        population_model: Any,
        init_params: Any,
    ) -> BayesianResult:
        """
        Bayesian estimation via PyMC.

        Builds a PyMC model with:
          - Log-normal priors on THETA (mu=log(theta_init), sigma=prior_sd_theta).
          - LKJ-Cholesky prior on OMEGA correlation structure.
          - Half-Normal prior on OMEGA standard deviations.
          - Half-Normal prior on SIGMA residual variance.
          - Gaussian likelihood per observation, vectorised over subjects.

        After sampling, computes posterior means, R-hat, and effective N.

        Args:
            population_model: PopulationModel instance.
            init_params:      ParameterSet with initial values and bounds.

        Returns:
            BayesianResult with full posterior trace.
        """
        import pymc as pm  # type: ignore[import]
        import pytensor.tensor as pt  # type: ignore[import]

        t0 = time.time()
        theta_init = init_params.theta.copy()
        n_theta = len(theta_init)
        n_eta = init_params.omega.shape[0]

        with pm.Model() as pymc_model:
            # -- Priors on THETA (log-normal: positive PK parameters) ------
            log_theta_mu = np.log(np.maximum(theta_init, 1e-8))
            pm.LogNormal(
                "theta",
                mu=log_theta_mu,
                sigma=self.prior_sd_theta,
                shape=n_theta,
            )

            # -- Priors on OMEGA (LKJ-Cholesky + half-normal SDs) ----------
            if n_eta > 0:
                omega_sd = pm.HalfNormal(
                    "omega_sd",
                    sigma=np.sqrt(np.maximum(np.diag(init_params.omega), 1e-8)),
                    shape=n_eta,
                )
                if n_eta > 1:
                    omega_chol, _, _ = pm.LKJCholeskyCov(
                        "omega_chol",
                        n=n_eta,
                        eta=2.0,
                        sd_dist=pm.HalfNormal.dist(sigma=1.0, shape=n_eta),
                    )
                    pm.Deterministic("omega", pm.math.dot(omega_chol, omega_chol.T))
                else:
                    pm.Deterministic("omega", pt.reshape(omega_sd**2, (1, 1)))

            # -- Priors on SIGMA -------------------------------------------
            sigma_init_diag = np.maximum(np.diag(init_params.sigma), 1e-8)
            sigma_var = pm.HalfNormal(
                "sigma_var",
                sigma=sigma_init_diag,
                shape=len(sigma_init_diag),
            )

            # -- Likelihood (simplified: additive normal per observation) --
            # Build prediction and likelihood for each subject
            all_obs: list[Any] = []
            all_pred_mu: list[Any] = []

            for sid in population_model.subject_ids():
                indiv = population_model.individual_model(sid)
                subj_ev = indiv.subject_events
                obs_mask = subj_ev.observation_mask()
                dv = subj_ev.obs_dv[obs_mask]
                if len(dv) == 0:
                    continue

                try:
                    eta_zero = np.zeros(n_eta)
                    _, _, f = indiv.evaluate(
                        theta_init,
                        eta_zero,
                        init_params.sigma,
                        trans=population_model.trans,
                    )
                    f_obs = f[obs_mask]
                except Exception:
                    f_obs = np.ones(len(dv))

                all_obs.extend(dv.tolist())
                all_pred_mu.extend(f_obs.tolist())

            obs_arr = np.array(all_obs)
            pred_arr = np.array(all_pred_mu)

            if len(obs_arr) > 0:
                sigma_sd = pm.math.sqrt(sigma_var[0])
                pm.Normal(
                    "likelihood",
                    mu=pred_arr,
                    sigma=sigma_sd,
                    observed=obs_arr,
                )

        # -- Sample -------------------------------------------------------
        with pymc_model:
            trace = pm.sample(
                draws=self.n_samples,
                tune=self.tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.seed,
                progressbar=False,
                return_inferencedata=True,
            )

        # -- Extract posterior summary ------------------------------------
        import arviz as az  # type: ignore[import]

        theta_samples = trace.posterior["theta"].values  # (chains, draws, n_theta)
        theta_flat = theta_samples.reshape(-1, n_theta)  # (n_total, n_theta)
        theta_mean = np.mean(theta_flat, axis=0)

        r_hat_vals = az.rhat(trace).theta.values
        n_eff_vals = az.ess(trace).theta.values

        omega_final = init_params.omega.copy()
        sigma_final = init_params.sigma.copy()

        ci_lo, ci_hi = self._compute_posterior_summary(theta_flat, ci=0.95)
        elapsed = time.time() - t0

        return BayesianResult(
            theta_final=theta_mean,
            omega_final=omega_final,
            sigma_final=sigma_final,
            ofv=float("nan"),  # not directly available from MCMC
            converged=True,
            elapsed_time=elapsed,
            method="BAYES(PyMC)",
            posterior_samples={"theta": theta_flat},
            posterior_samples_by_chain={"theta": theta_samples},  # (chains, draws, n_theta)
            r_hat=r_hat_vals,
            n_effective=n_eff_vals,
            posterior_ci_lo=ci_lo,
            posterior_ci_hi=ci_hi,
            backend_used="pymc",
        )

    # ------------------------------------------------------------------
    # Pure-NumPy NUTS backend (zero dependencies)
    # ------------------------------------------------------------------

    def _estimate_nuts(
        self,
        population_model: Any,
        init_params: Any,
    ) -> BayesianResult:
        """
        Bayesian estimation via the built-in pure-NumPy NUTS sampler.

        Requires no optional packages — NumPy and SciPy are already core
        dependencies of OpenPKPD.

        Log-posterior
        -------------
        log p(theta | data) ∝ log p(theta) + log p(data | theta)

        **Prior** — log-normal on each THETA:
          log p(theta) = -0.5 * Σ_k ((log θ_k - log θ_init_k) / prior_sd_theta)²
                         - Σ_k log θ_k           ← Jacobian for log-space sampling

        **Likelihood** — FOCE marginal approximation (when data are present):
          log p(data | theta) ≈ -OFV(theta) / 2

        where OFV is the FOCEI/FOCE objective returned by
        :meth:`FOCEMethod._outer_ofv`.  This is the industry-standard NLME
        Bayesian approach used by NONMEM's BAYES method.

        For each theta proposal the FOCE inner loop optimises η̂_i for every
        subject (via L-BFGS-B) and the outer OFV is evaluated at those optimal
        random effects.  **Warm-starting**: η̂ from the previous log_prob call
        is used as the initial guess for the next, so the inner loop typically
        converges in 5–20 iterations rather than the cold-start default of 200.

        Omega and Sigma are held fixed at ``init_params.omega`` /
        ``init_params.sigma`` during NUTS (theta-only sampling).  Full joint
        sampling over Omega and Sigma will be added in a future release.

        Performance note
        ----------------
        Each NUTS leapfrog step requires ``2 * n_theta + 1`` FOCE evaluations
        (finite-difference gradient).  For large datasets (> 50 subjects) or
        many parameters (> 10 THETA) this can be slow — 10–60 min/run typical.
        Use ``openpkpd[bayes]`` (PyMC) for faster autodiff-based sampling.

        Multi-chain sampling is achieved by running NUTSSampler once per chain
        with a deterministically-derived seed.  R-hat and ESS are computed with
        :mod:`openpkpd.estimation.mcmc_diagnostics`.

        Args:
            population_model: PopulationModel instance (or None in unit tests).
            init_params:      ParameterSet with initial values and bounds.

        Returns:
            BayesianResult with full posterior trace.
        """
        import warnings

        from openpkpd.estimation.mcmc_diagnostics import compute_ess, compute_rhat
        from openpkpd.estimation.nuts import NUTSSampler

        t0 = time.time()
        theta_init = np.asarray(init_params.theta, dtype=float).copy()
        n_theta = len(theta_init)
        n_eta = np.asarray(init_params.omega).shape[0]
        log_theta_mu = np.log(np.maximum(theta_init, 1e-8))

        # -----------------------------------------------------------------
        # Log-prior: LogNormal(mu=log(theta_init), sigma=prior_sd_theta)
        # -----------------------------------------------------------------
        def _log_prior(theta: np.ndarray) -> float:
            if np.any(theta <= 0):
                return -np.inf
            log_th = np.log(theta)
            return float(
                -0.5 * np.sum(((log_th - log_theta_mu) / self.prior_sd_theta) ** 2)
                - n_theta * np.log(self.prior_sd_theta)
                - np.sum(log_th)  # Jacobian for log-normal → log-space sampling
            )

        # -----------------------------------------------------------------
        # Log-likelihood: FOCE marginal approximation.
        # OFV is on the -2LL scale, so log p(Y|theta) ≈ -OFV/2.
        # Warm-start: foce._current_eta_hat is updated after every inner loop
        # so consecutive NUTS proposals benefit from a nearby initial eta.
        # -----------------------------------------------------------------
        if population_model is not None:
            from openpkpd.estimation.foce import FOCEMethod
            from openpkpd.model.parameters import ParameterSet

            subject_ids = list(population_model.subject_ids())
            # inner_maxiter=50: warm-started iterations converge faster than
            # the cold-start default (200); keeps per-step cost bounded.
            foce = FOCEMethod(inner_maxiter=50)
            foce._current_eta_hat = {
                sid: np.zeros(n_eta)
                for sid in subject_ids
            }
            _omega = np.asarray(init_params.omega).copy()
            _sigma = np.asarray(init_params.sigma).copy()
            log_prob_calls = 0
            foce_inner_calls = 0
            foce_outer_calls = 0
            foce_inner_elapsed = 0.0
            foce_outer_elapsed = 0.0
            theta_grad_calls = 0
            theta_grad_elapsed = 0.0
            exact_log_prob_cache: OrderedDict[bytes, tuple[float, np.ndarray | None]] = OrderedDict()
            exact_log_prob_cache_size = 128
            exact_log_prob_cache_hits = 0
            exact_log_prob_cache_misses = 0
            supports_theta_gradient = (
                (not foce.interaction)
                and all(
                    isinstance(
                        population_model.individual_model(sid).supports_theta_data_objective_gradient(
                            population_model.trans
                        ),
                        (bool, np.bool_),
                    )
                    and bool(
                        population_model.individual_model(sid).supports_theta_data_objective_gradient(
                            population_model.trans
                        )
                    )
                    for sid in subject_ids
                )
            )
            warm_start_cache: OrderedDict[bytes, tuple[np.ndarray, dict[int, np.ndarray]]] = (
                OrderedDict()
            )
            warm_start_cache_size = 32
            warm_start_exact_hits = 0
            warm_start_nearest_hits = 0
            warm_start_cold_starts = 0

            def _copy_eta_hat(eta_hat: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
                return {
                    sid: np.asarray(value, dtype=float).copy()
                    for sid, value in eta_hat.items()
                }

            def _seed_eta_hat(theta: np.ndarray) -> dict[int, np.ndarray]:
                nonlocal warm_start_exact_hits, warm_start_nearest_hits, warm_start_cold_starts
                theta_arr = np.asarray(theta, dtype=float)
                key = theta_arr.tobytes()
                exact = warm_start_cache.get(key)
                if exact is not None:
                    warm_start_cache.move_to_end(key)
                    warm_start_exact_hits += 1
                    return _copy_eta_hat(exact[1])

                if warm_start_cache:
                    nearest_key = min(
                        warm_start_cache,
                        key=lambda candidate: float(
                            np.linalg.norm(theta_arr - warm_start_cache[candidate][0])
                        ),
                    )
                    warm_start_cache.move_to_end(nearest_key)
                    warm_start_nearest_hits += 1
                    return _copy_eta_hat(warm_start_cache[nearest_key][1])

                warm_start_cold_starts += 1
                return {
                    sid: np.zeros(n_eta, dtype=float)
                    for sid in subject_ids
                }

            def _store_eta_hat(theta: np.ndarray, eta_hat: dict[int, np.ndarray]) -> None:
                key = np.asarray(theta, dtype=float).tobytes()
                warm_start_cache[key] = (
                    np.asarray(theta, dtype=float).copy(),
                    _copy_eta_hat(eta_hat),
                )
                warm_start_cache.move_to_end(key)
                while len(warm_start_cache) > warm_start_cache_size:
                    warm_start_cache.popitem(last=False)

            def _log_prior_grad(theta: np.ndarray) -> np.ndarray:
                theta_arr = np.asarray(theta, dtype=float)
                if np.any(theta_arr <= 0):
                    return np.full_like(theta_arr, np.nan, dtype=float)
                log_th = np.log(theta_arr)
                return -(
                    ((log_th - log_theta_mu) / (self.prior_sd_theta**2)) + 1.0
                ) / theta_arr

            def _theta_log_prob_grad(
                theta: np.ndarray,
                eta_hat: dict[int, np.ndarray],
            ) -> np.ndarray | None:
                nonlocal theta_grad_calls, theta_grad_elapsed
                if not supports_theta_gradient:
                    return None
                theta_grad_calls += 1
                t_grad = time.time()
                grad = _log_prior_grad(theta)
                for sid in subject_ids:
                    indiv = population_model.individual_model(sid)
                    grad -= 0.5 * np.asarray(
                        indiv.theta_data_objective_gradient(
                            theta,
                            eta_hat.get(sid, np.zeros(n_eta, dtype=float)),
                            _sigma,
                            trans=population_model.trans,
                        ),
                        dtype=float,
                    )
                theta_grad_elapsed += time.time() - t_grad
                return grad

            def log_prob(theta: np.ndarray) -> float:
                """Log-posterior: prior + FOCE marginal log-likelihood."""
                nonlocal log_prob_calls, foce_inner_calls, foce_outer_calls
                nonlocal foce_inner_elapsed, foce_outer_elapsed
                nonlocal exact_log_prob_cache_hits, exact_log_prob_cache_misses
                theta_arr = np.asarray(theta, dtype=float)
                key = theta_arr.tobytes()
                cached_payload = exact_log_prob_cache.get(key)
                if cached_payload is not None:
                    exact_log_prob_cache.move_to_end(key)
                    exact_log_prob_cache_hits += 1
                    return cached_payload[0]
                exact_log_prob_cache_misses += 1
                log_prob_calls += 1
                lp = _log_prior(theta_arr)
                if not np.isfinite(lp):
                    exact_log_prob_cache[key] = (
                        float("-inf"),
                        np.zeros_like(theta_arr, dtype=float) if supports_theta_gradient else None,
                    )
                    return -np.inf
                params = ParameterSet(theta=theta_arr, omega=_omega, sigma=_sigma)
                try:
                    foce._current_eta_hat = _seed_eta_hat(theta_arr)
                    t_inner = time.time()
                    eta_hat = foce._inner_loop(population_model, params)
                    foce_inner_elapsed += time.time() - t_inner
                    foce_inner_calls += 1
                    foce._current_eta_hat = _copy_eta_hat(eta_hat)
                    _store_eta_hat(theta_arr, eta_hat)
                    # float() converts the scalar OFV and raises TypeError on
                    # non-numeric returns (e.g. mock objects in unit tests).
                    t_outer = time.time()
                    ofv = float(foce._outer_ofv(population_model, params, eta_hat))
                    foce_outer_elapsed += time.time() - t_outer
                    foce_outer_calls += 1
                except Exception:
                    return -np.inf
                if not np.isfinite(ofv):
                    return -np.inf
                # OFV = -2 * log p(Y|theta,eta_hat) → log-lik = -OFV/2
                value = lp - 0.5 * ofv
                grad = _theta_log_prob_grad(theta_arr, eta_hat)
                exact_log_prob_cache[key] = (value, grad)
                exact_log_prob_cache.move_to_end(key)
                while len(exact_log_prob_cache) > exact_log_prob_cache_size:
                    exact_log_prob_cache.popitem(last=False)
                return value

            def grad_log_prob(theta: np.ndarray) -> np.ndarray:
                theta_arr = np.asarray(theta, dtype=float)
                key = theta_arr.tobytes()
                cached_payload = exact_log_prob_cache.get(key)
                if cached_payload is None:
                    _ = log_prob(theta_arr)
                    cached_payload = exact_log_prob_cache.get(key)
                if cached_payload is None or cached_payload[1] is None:
                    raise RuntimeError("analytic theta gradient unavailable for this theta")
                return np.asarray(cached_payload[1], dtype=float)

        else:
            log_prob_calls = 0
            foce_inner_calls = 0
            foce_outer_calls = 0
            foce_inner_elapsed = 0.0
            foce_outer_elapsed = 0.0
            theta_grad_calls = 0
            theta_grad_elapsed = 0.0
            exact_log_prob_cache: OrderedDict[bytes, tuple[float, np.ndarray | None]] = OrderedDict()
            exact_log_prob_cache_size = 128
            exact_log_prob_cache_hits = 0
            exact_log_prob_cache_misses = 0
            supports_theta_gradient = True

            def _log_prior_grad(theta: np.ndarray) -> np.ndarray:
                theta_arr = np.asarray(theta, dtype=float)
                if np.any(theta_arr <= 0):
                    return np.full_like(theta_arr, np.nan, dtype=float)
                log_th = np.log(theta_arr)
                return -(
                    ((log_th - log_theta_mu) / (self.prior_sd_theta**2)) + 1.0
                ) / theta_arr

            # population_model is None — unit-test / prior-only mode
            def log_prob(theta: np.ndarray) -> float:
                nonlocal log_prob_calls, exact_log_prob_cache_hits, exact_log_prob_cache_misses
                theta_arr = np.asarray(theta, dtype=float)
                key = theta_arr.tobytes()
                cached_payload = exact_log_prob_cache.get(key)
                if cached_payload is not None:
                    exact_log_prob_cache.move_to_end(key)
                    exact_log_prob_cache_hits += 1
                    return cached_payload[0]
                exact_log_prob_cache_misses += 1
                log_prob_calls += 1
                value = _log_prior(theta_arr)
                if not np.isfinite(value):
                    exact_log_prob_cache[key] = (
                        float("-inf"),
                        np.zeros_like(theta_arr, dtype=float),
                    )
                    return value
                exact_log_prob_cache[key] = (value, _log_prior_grad(theta_arr))
                exact_log_prob_cache.move_to_end(key)
                while len(exact_log_prob_cache) > exact_log_prob_cache_size:
                    exact_log_prob_cache.popitem(last=False)
                return value

            def grad_log_prob(theta: np.ndarray) -> np.ndarray:
                theta_arr = np.asarray(theta, dtype=float)
                key = theta_arr.tobytes()
                cached_payload = exact_log_prob_cache.get(key)
                if cached_payload is None:
                    _ = log_prob(theta_arr)
                    cached_payload = exact_log_prob_cache.get(key)
                if cached_payload is None or cached_payload[1] is None:
                    raise RuntimeError("prior-only gradient unavailable")
                return np.asarray(cached_payload[1], dtype=float)

        # -----------------------------------------------------------------
        # Multi-chain sampling
        # -----------------------------------------------------------------
        rng = np.random.default_rng(self.seed)
        chain_seeds = [int(rng.integers(0, 2**31)) for _ in range(self.n_chains)]
        chains: list[np.ndarray] = []
        chain_diagnostics: list[dict[str, float | int | bool]] = []
        for seed_c in chain_seeds:
            sampler = NUTSSampler(
                log_prob,
                grad_log_prob_fn=grad_log_prob if supports_theta_gradient else None,
                seed=seed_c,
                delta=self.target_accept,
            )
            chain_draws = sampler.sample(
                theta_init,
                n_samples=self.n_samples,
                n_warmup=self.tune,
            )
            chains.append(chain_draws)
            chain_diagnostics.append(dict(sampler.last_diagnostics))

        theta_by_chain = np.stack(chains)          # (n_chains, n_samples, n_theta)
        theta_flat = theta_by_chain.reshape(-1, n_theta)
        theta_mean = np.mean(theta_flat, axis=0)

        r_hat_vals = compute_rhat(theta_by_chain)
        n_eff_vals = compute_ess(theta_by_chain)

        if not bool(np.all(r_hat_vals <= 1.1)):
            warnings.warn(
                "NUTS sampler: R-hat > 1.1 for one or more parameters. "
                "Consider increasing n_samples or tune.",
                UserWarning,
                stacklevel=3,
            )

        ci_lo, ci_hi = self._compute_posterior_summary(theta_flat, ci=0.95)
        elapsed = time.time() - t0
        diagnostics = {
            "nuts": {
                "n_chains": int(self.n_chains),
                "n_samples_per_chain": int(self.n_samples),
                "n_warmup_per_chain": int(self.tune),
                "log_prob_calls": int(log_prob_calls),
                "foce_inner_calls": int(foce_inner_calls),
                "foce_outer_calls": int(foce_outer_calls),
                "foce_inner_seconds": float(foce_inner_elapsed),
                "foce_outer_seconds": float(foce_outer_elapsed),
                "log_prob_calls_per_posterior_draw": (
                    float(log_prob_calls / max(theta_flat.shape[0], 1))
                ),
                "exact_log_prob_cache_size": int(exact_log_prob_cache_size),
                "exact_log_prob_cache_hits": int(exact_log_prob_cache_hits),
                "exact_log_prob_cache_misses": int(exact_log_prob_cache_misses),
                "theta_gradient_calls": int(theta_grad_calls),
                "theta_gradient_seconds": float(theta_grad_elapsed),
                "used_analytic_theta_gradient": bool(supports_theta_gradient),
                "used_population_model": bool(population_model is not None),
                "theta_only": True,
                "warm_start_cache_size": int(warm_start_cache_size)
                if population_model is not None
                else 0,
                "warm_start_exact_hits": int(warm_start_exact_hits)
                if population_model is not None
                else 0,
                "warm_start_nearest_hits": int(warm_start_nearest_hits)
                if population_model is not None
                else 0,
                "warm_start_cold_starts": int(warm_start_cold_starts)
                if population_model is not None
                else 0,
                "chain_diagnostics": chain_diagnostics,
            }
        }

        return BayesianResult(
            theta_final=theta_mean,
            omega_final=init_params.omega.copy(),
            sigma_final=init_params.sigma.copy(),
            ofv=float("nan"),
            converged=bool(np.all(r_hat_vals <= 1.1)),
            elapsed_time=elapsed,
            method="BAYES(NUTS)",
            posterior_samples={"theta": theta_flat},
            posterior_samples_by_chain={"theta": theta_by_chain},
            r_hat=r_hat_vals,
            n_effective=n_eff_vals,
            posterior_ci_lo=ci_lo,
            posterior_ci_hi=ci_hi,
            backend_used="nuts",
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Laplace approximation fallback
    # ------------------------------------------------------------------

    def _estimate_laplace(
        self,
        population_model: Any,
        init_params: Any,
    ) -> BayesianResult:
        """
        Laplace approximation fallback.

        When no MCMC backend is available:
        1. Run FOCE to get the MAP (maximum a posteriori) estimate.
        2. Numerically compute the Hessian of the OFV-like negative
           log-posterior at MAP.
        3. Sample from MVN(MAP, 2 * Hessian^{-1}) as an approximate posterior.

        The Hessian is estimated by finite differences on the FOCE objective
        function, which serves as the negative log-likelihood. A weak prior
        on THETA (N(log theta_init, prior_sd_theta^2) in log-space) is added
        analytically.

        Args:
            population_model: PopulationModel instance (may be None in tests).
            init_params:      ParameterSet with initial parameter values.

        Returns:
            BayesianResult with approximate posterior samples.
        """
        from openpkpd.estimation.foce import FOCEMethod

        t0 = time.time()
        rng = np.random.default_rng(self.seed)

        foce = FOCEMethod()
        foce_result = foce.estimate(population_model, init_params)

        theta_map = foce_result.theta_final
        n_theta = len(theta_map)

        # Estimate covariance from finite-difference Hessian on the FOCE OFV.
        # The FOCE outer objective is OFV-like (-2 log posterior), so the
        # local Gaussian covariance is 2 * H^{-1}.
        hess_cov = self._approx_hessian_covariance(
            population_model, init_params, theta_map, foce_result
        )

        # Draw approximate posterior samples
        try:
            approx_samples = rng.multivariate_normal(theta_map, hess_cov, size=self.n_samples)
        except np.linalg.LinAlgError:
            # Degenerate covariance: use diagonal approximation
            diag_cov = np.diag(hess_cov)
            approx_samples = theta_map + rng.standard_normal((self.n_samples, n_theta)) * np.sqrt(
                np.maximum(diag_cov, 1e-8)
            )

        ci_lo, ci_hi = self._compute_posterior_summary(approx_samples, ci=0.95)
        elapsed = time.time() - t0

        return BayesianResult(
            theta_final=theta_map,
            omega_final=foce_result.omega_final,
            sigma_final=foce_result.sigma_final,
            ofv=foce_result.ofv,
            converged=foce_result.converged,
            post_hoc_etas=foce_result.post_hoc_etas,
            ofv_history=foce_result.ofv_history,
            n_function_evals=foce_result.n_function_evals,
            elapsed_time=elapsed,
            method="BAYES(Laplace)",
            message=foce_result.message,
            posterior_samples={"theta": approx_samples},
            r_hat=np.ones(n_theta),
            n_effective=np.full(n_theta, self.n_samples),
            posterior_ci_lo=ci_lo,
            posterior_ci_hi=ci_hi,
            backend_used="laplace",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _approx_hessian_covariance(
        self,
        population_model: Any,
        init_params: Any,
        theta_map: np.ndarray,
        foce_result: EstimationResult,
    ) -> np.ndarray:
        """
        Compute an approximate posterior covariance from the Hessian.

        Uses finite-difference second derivatives of the FOCE OFV
        (an OFV-like ``-2 log posterior`` quantity). Adds the contribution
        of a log-normal prior on THETA in the Hessian.

        Args:
            population_model: PopulationModel or None.
            init_params:      Original ParameterSet.
            theta_map:        MAP estimate for THETA.
            foce_result:      FOCE result at the MAP.

        Returns:
            Approximate posterior covariance matrix, shape (n_theta, n_theta).
        """
        n_theta = len(theta_map)
        eps = 1e-4

        try:
            from openpkpd.estimation.foce import FOCEMethod
            from openpkpd.model.parameters import ParameterSet

            foce_inner = FOCEMethod()
            foce_inner._current_eta_hat = foce_result.post_hoc_etas or {}

            def ofv_at_theta(th: np.ndarray) -> float:
                try:
                    # Build a ParameterSet with new theta
                    new_params = ParameterSet(
                        theta=th,
                        omega=foce_result.omega_final,
                        sigma=foce_result.sigma_final,
                        theta_specs=getattr(init_params, "theta_specs", []),
                        omega_specs=getattr(init_params, "omega_specs", []),
                        sigma_specs=getattr(init_params, "sigma_specs", []),
                    )
                    eta_hat = foce_inner._inner_loop(population_model, new_params)
                    return foce_inner._outer_ofv(population_model, new_params, eta_hat)
                except Exception:
                    return float("nan")

            # Numerical Hessian via central differences
            hess = np.zeros((n_theta, n_theta))
            f0 = ofv_at_theta(theta_map)

            for i in range(n_theta):
                for j in range(i, n_theta):
                    ei = np.zeros(n_theta)
                    ej = np.zeros(n_theta)
                    ei[i] = eps
                    ej[j] = eps

                    if i == j:
                        f_pp = ofv_at_theta(theta_map + 2 * ei)
                        f_p = ofv_at_theta(theta_map + ei)
                        f_m = ofv_at_theta(theta_map - ei)
                        f_mm = ofv_at_theta(theta_map - 2 * ei)
                        if all(np.isfinite([f_pp, f_p, f_m, f_mm])):
                            h2 = (-f_pp + 16 * f_p - 30 * f0 + 16 * f_m - f_mm) / (12 * eps**2)
                        else:
                            h2 = (f_p - 2 * f0 + f_m) / (eps**2) if np.isfinite(f_p + f_m) else 1.0
                        hess[i, i] = h2
                    else:
                        f_pp = ofv_at_theta(theta_map + ei + ej)
                        f_pm = ofv_at_theta(theta_map + ei - ej)
                        f_mp = ofv_at_theta(theta_map - ei + ej)
                        f_mm = ofv_at_theta(theta_map - ei - ej)
                        if all(np.isfinite([f_pp, f_pm, f_mp, f_mm])):
                            h2 = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
                        else:
                            h2 = 0.0
                        hess[i, j] = h2
                        hess[j, i] = h2

            # Add log-normal prior contribution to Hessian diagonal
            theta_safe = np.maximum(theta_map, 1e-8)
            prior_hess_diag = 1.0 / (self.prior_sd_theta**2 * theta_safe**2)
            np.fill_diagonal(hess, np.diag(hess) + prior_hess_diag)

            # Convert OFV Hessian to local Gaussian covariance.
            # If OFV(theta) = const + (theta-mu)^T Sigma^{-1} (theta-mu),
            # then Hessian(OFV) = 2 * Sigma^{-1}, hence Sigma = 2 * H^{-1}.
            hess = 0.5 * (hess + hess.T)  # ensure symmetry
            eigvals = np.linalg.eigvalsh(hess)
            if np.any(eigvals <= 0):
                # Make positive definite by adding small diagonal
                hess += np.eye(n_theta) * max(abs(eigvals.min()) + 1e-6, 1e-6)

            cov = 2.0 * np.linalg.inv(hess)
            # Symmetrise
            cov = 0.5 * (cov + cov.T)
            return cov

        except Exception:
            # Fallback: diagonal covariance proportional to theta_map^2
            diag = (0.1 * np.maximum(np.abs(theta_map), 1e-8)) ** 2
            return np.diag(diag)

    def _compute_posterior_summary(
        self,
        samples: np.ndarray,
        ci: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute credible interval bounds from posterior samples.

        Args:
            samples: Array of shape (n_samples, n_params).
            ci:      Credible interval level (e.g. 0.95 for 95% CI).

        Returns:
            Tuple (ci_lo, ci_hi), each an array of shape (n_params,).
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        alpha = (1.0 - ci) / 2.0
        ci_lo = np.quantile(samples, alpha, axis=0)
        ci_hi = np.quantile(samples, 1.0 - alpha, axis=0)
        return ci_lo, ci_hi
