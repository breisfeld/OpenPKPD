"""
Example 15: Bayesian Estimation via MAP and Laplace Posterior Approximation.

Demonstrates:
  - Fitting a 1-compartment oral PK model with Bayesian estimation
  - Using the Laplace approximation fallback (works without PyMC)
  - Displaying posterior credible intervals for each THETA parameter
  - Comparing FOCE (MAP) and Bayesian (Laplace) estimates side by side
  - Note on enabling true MCMC sampling via the built-in NUTS backend or PyMC

Dataset: Deterministic synthetic oral-PK data (12 subjects) generated from
         known population means KA=1.5, CL=2.8, V=32.9.

The Bayesian method implemented here:
  1. Runs FOCE to obtain the MAP (maximum a posteriori) estimate.
  2. Approximates the posterior as a multivariate normal centred at the MAP
     with covariance computed from the numerical Hessian of the objective.
  3. Samples from this approximate posterior to produce credible intervals.

For full MCMC sampling with NUTS (No-U-Turn Sampler), OpenPKPD ships a built-in
pure-NumPy backend:
    BAYESMethod(backend='nuts', n_samples=1000, n_chains=4)

Optional external backend:
    pip install pymc
    BAYESMethod(backend='pymc', n_samples=2000, n_chains=4)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.bayes import BAYESMethod, BayesianResult
from openpkpd.estimation.foce import FOCEMethod


def _print_laplace_boundary_diagnostics(
    theta_samples: np.ndarray | None,
    theta_specs: list[object],
    param_names: list[str],
) -> None:
    """Highlight when Laplace samples collapse onto parameter bounds."""
    if theta_samples is None or theta_samples.ndim != 2:
        return

    lines: list[str] = []
    for k, (name, spec) in enumerate(zip(param_names, theta_specs)):
        if k >= theta_samples.shape[1]:
            continue
        values = theta_samples[:, k]
        at_lower = float(np.mean(np.isclose(values, spec.lower, atol=1e-6, rtol=0.0)))
        at_upper = float(
            np.mean(
                np.isclose(values, spec.upper, atol=1e-6, rtol=0.0),
            )
        ) if math.isfinite(spec.upper) else 0.0
        if at_lower >= 0.05 or at_upper >= 0.05:
            lines.append(
                f"  - {name}: {at_lower:.1%} at lower bound, {at_upper:.1%} at upper bound"
            )

    if not lines:
        return

    print("\n" + "=" * 70)
    print("Laplace Approximation Diagnostics")
    print("=" * 70)
    print("  Posterior samples are saturating one or more parameter bounds.")
    print("  On this tiny dataset, treat those intervals as support-limited diagnostics,")
    print("  not as trustworthy posterior uncertainty.")
    print("  Prefer the NUTS/PyMC backends or a richer dataset for Bayesian inference.")
    for line in lines:
        print(line)


# ---------------------------------------------------------------------------
# Deterministic oral-PK dataset (12 subjects)
# ---------------------------------------------------------------------------
def _build_dataset() -> NONMEMDataset:
    rng = np.random.default_rng(42)
    ka_pop, cl_pop, v_pop = 1.5, 2.8, 32.9
    dose = 320.0
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0])

    rows = []
    for sid in range(1, 13):
        ka = ka_pop * math.exp(rng.normal(0, 0.3))
        cl = cl_pop * math.exp(rng.normal(0, 0.25))
        v = v_pop * math.exp(rng.normal(0, 0.15))
        k = cl / v

        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            }
        )
        for t in obs_times:
            if abs(ka - k) < 1e-6:
                conc = dose * ka / v * t * math.exp(-k * t)
            else:
                conc = dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
            dv = max(conc * (1 + rng.normal(0, 0.1)), 0.01)
            rows.append(
                {
                    "ID": sid,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": dv,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def main() -> None:
    print("=" * 70)
    print("Example 15: Bayesian Estimation via MAP and Laplace Posterior Approximation")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load dataset and build model
    # -----------------------------------------------------------------------
    ds = _build_dataset()

    built = (
        ModelBuilder()
        .problem("Synthetic 1-cmt oral PK — Bayesian")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.3, 1.5, 8.0), (0.5, 3.0, 15.0), (10.0, 35.0, 80.0)])
        .omega([0.09, 0.06, 0.04])
        .sigma(0.02)
        .estimation(method="FOCE", maxeval=500)
        .build()
    )

    population_model = built.population_model
    init_params = built.params

    param_names = ["KA (hr⁻¹)", "CL (L/hr)", "V (L)"]
    print(f"\nDataset: {len(ds.subject_ids())} subjects, {len(ds.df)} rows")

    # -----------------------------------------------------------------------
    # 2. FOCE estimation (MAP estimate)
    # -----------------------------------------------------------------------
    print("\n[Step 1] Running FOCE for MAP estimate ...")
    foce = FOCEMethod(maxeval=500, print_interval=100)
    foce_result = foce.estimate(population_model, init_params)

    print(f"  FOCE OFV:    {foce_result.ofv:.4f}")
    print(f"  Converged:   {foce_result.converged}")
    print("  FOCE THETA estimates:")
    for k, (name, th) in enumerate(zip(param_names, foce_result.theta_final)):
        print(f"    THETA({k+1}) [{name}] = {th:.4f}")

    # -----------------------------------------------------------------------
    # 3. Bayesian estimation (Laplace approximation)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Running Bayesian estimation (Laplace approximation) ...")
    print("  (This uses FOCE MAP + numerical Hessian for posterior approximation)")

    bayes = BAYESMethod(
        n_samples=2000,
        n_chains=1,           # Laplace: chains don't matter
        seed=42,
        backend="laplace",    # Force Laplace fallback (no MCMC required)
        prior_sd_theta=2.0,   # Weakly informative log-normal prior on THETA
    )

    bayes_result = bayes.estimate(population_model, init_params)

    print(f"\n  Backend used: {bayes_result.backend_used}")
    print(f"  OFV (MAP):   {bayes_result.ofv:.4f}")
    print(f"  n_samples:   {len(bayes_result.posterior_samples.get('theta', []))}")

    # -----------------------------------------------------------------------
    # 4. Posterior summary: credible intervals for each THETA
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Posterior Summary (95% Credible Intervals)")
    print("=" * 70)
    print(bayes_result.posterior_summary(ci=0.95))

    # -----------------------------------------------------------------------
    # 5. Side-by-side comparison: FOCE vs Bayesian
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparison: FOCE Point Estimates vs Bayesian Posterior")
    print("=" * 70)

    theta_samples = bayes_result.posterior_samples.get("theta")

    header = (
        f"  {'Param':<20}  {'FOCE_MAP':>10}  "
        f"{'Bayes_Mean':>12}  {'95% CI lower':>14}  {'95% CI upper':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    if theta_samples is not None and theta_samples.ndim == 2:
        for k, name in enumerate(param_names):
            foce_val = foce_result.theta_final[k]
            if k < theta_samples.shape[1]:
                samp_k = theta_samples[:, k]
                post_mean = float(np.mean(samp_k))
                ci_lo = float(np.quantile(samp_k, 0.025))
                ci_hi = float(np.quantile(samp_k, 0.975))
                print(
                    f"  {name:<20}  {foce_val:>10.4f}  "
                    f"{post_mean:>12.4f}  {ci_lo:>14.4f}  {ci_hi:>14.4f}"
                )
            else:
                print(f"  {name:<20}  {foce_val:>10.4f}  (no samples)")
    else:
        print("  No posterior samples available.")

    _print_laplace_boundary_diagnostics(
        theta_samples,
        init_params.theta_specs,
        param_names,
    )

    # -----------------------------------------------------------------------
    # 6. OMEGA and SIGMA
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Population Parameters (OMEGA, SIGMA from MAP/FOCE)")
    print("=" * 70)
    print(f"  OMEGA diagonal: {np.diag(bayes_result.omega_final)}")
    print(f"  SIGMA diagonal: {np.diag(bayes_result.sigma_final)}")

    # -----------------------------------------------------------------------
    # 7. Note on MCMC backends
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Notes on Full MCMC Sampling")
    print("=" * 70)
    print("""
  The Laplace approximation is a fast, practical fallback that works
  without any additional dependencies. For true posterior sampling, use one
  of the MCMC backends:

  Built-in pure-NumPy NUTS:
    BAYESMethod(backend='nuts', n_samples=1000, n_chains=4)
    - no optional dependency required
    - best for small-to-moderate models where runtime is acceptable
    - multi-chain diagnostics come from BAYESMethod, not the standalone
      nuts_estimate() helper

  PyMC (recommended for general use):
    pip install pymc
    BAYESMethod(backend='pymc', n_samples=2000, n_chains=4)

  Key advantages of full MCMC over Laplace:
    - Properly samples from the true posterior (not a Gaussian approx)
    - Provides R-hat and effective sample size diagnostics
    - Handles multimodal or skewed posterior distributions
    - Required for hierarchical population PK/PD models with >5 subjects

  Current OpenPKPD Bayesian limitations:
    - the built-in NUTS backend currently samples THETA only; OMEGA and SIGMA
      remain fixed at their starting values
    - finite-difference gradients can be slow on larger ODE-heavy models
    - use benchmarked workflows and diagnostics before implying parity with
      mature Bayesian engines such as Monolix or Pumas
""")

    print("Done.")


if __name__ == "__main__":
    main()
