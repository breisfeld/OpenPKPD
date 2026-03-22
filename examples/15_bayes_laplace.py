"""
Example 15: Bayesian Estimation via MAP and Laplace Posterior Approximation.

Demonstrates:
  - Fitting a 1-compartment oral PK model with Bayesian estimation
  - Using the Laplace approximation fallback (works without PyMC/NumPyro)
  - Displaying posterior credible intervals for each THETA parameter
  - Comparing FOCE (MAP) and Bayesian (Laplace) estimates side by side
  - Note on enabling true MCMC sampling via PyMC or NumPyro backends

Dataset: Embedded theophylline-like data (3 subjects, simulated).

The Bayesian method implemented here:
  1. Runs FOCE to obtain the MAP (maximum a posteriori) estimate.
  2. Approximates the posterior as a multivariate normal centred at the MAP
     with covariance computed from the numerical Hessian of the objective.
  3. Samples from this approximate posterior to produce credible intervals.

For full MCMC sampling with NUTS (No-U-Turn Sampler), install PyMC:
    pip install pymc
or NumPyro (JAX-based):
    pip install numpyro jax jaxlib
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.bayes import BAYESMethod, BayesianResult
from openpkpd.estimation.foce import FOCEMethod


# ---------------------------------------------------------------------------
# Embedded theophylline-like dataset (3 subjects)
# ---------------------------------------------------------------------------
THEO_DATA = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,4.02,0,1,1,79.6
1,0.27,0,0.74,0,0,79.6
1,0.57,0,1.72,0,0,79.6
1,1.02,0,7.91,0,0,79.6
1,1.92,0,8.31,0,0,79.6
1,3.5,0,8.33,0,0,79.6
1,5.02,0,6.85,0,0,79.6
1,7.03,0,6.08,0,0,79.6
1,9.0,0,5.4,0,0,79.6
1,12.05,0,4.55,0,0,79.6
1,24.37,0,1.25,0,0,79.6
2,0,4.4,0,1,1,72.4
2,0.35,0,0.96,0,0,72.4
2,0.6,0,2.33,0,0,72.4
2,1.07,0,4.71,0,0,72.4
2,2.13,0,8.33,0,0,72.4
2,3.5,0,9.02,0,0,72.4
2,5.02,0,7.14,0,0,72.4
2,7.02,0,5.68,0,0,72.4
2,9.1,0,4.55,0,0,72.4
2,12.1,0,3.01,0,0,72.4
2,25.0,0,0.9,0,0,72.4
3,0,4.95,0,1,1,70.5
3,0.27,0,0.64,0,0,70.5
3,0.58,0,1.92,0,0,70.5
3,1.02,0,4.44,0,0,70.5
3,1.92,0,7.03,0,0,70.5
3,3.5,0,9.07,0,0,70.5
3,5.02,0,7.56,0,0,70.5
3,7.02,0,6.59,0,0,70.5
3,9.0,0,5.88,0,0,70.5
3,12.15,0,4.73,0,0,70.5
3,24.17,0,1.25,0,0,70.5
"""


def main() -> None:
    print("=" * 70)
    print("Example 15: Bayesian Estimation via MAP and Laplace Posterior Approximation")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load dataset and build model
    # -----------------------------------------------------------------------
    df = pd.read_csv(io.StringIO(THEO_DATA))
    ds = NONMEMDataset.from_dataframe(df)

    built = (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — Bayesian")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .estimation(method="FOCE", maxeval=500)
        .build()
    )

    population_model = built.population_model
    init_params = built.params

    param_names = ["KA (hr⁻¹)", "CL (L/hr)", "V (L)"]

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
  without any additional dependencies. For production-quality Bayesian
  inference with proper uncertainty quantification, use the MCMC backends:

  PyMC (recommended for general use):
    pip install pymc
    BAYESMethod(backend='pymc', n_samples=2000, n_chains=4)

  NumPyro (JAX-based, faster on GPU/TPU):
    pip install numpyro jax jaxlib
    BAYESMethod(backend='numpyro', n_samples=2000, n_chains=4)

  Key advantages of full MCMC over Laplace:
    - Properly samples from the true posterior (not a Gaussian approx)
    - Provides R-hat and effective sample size diagnostics
    - Handles multimodal or skewed posterior distributions
    - Required for hierarchical population PK/PD models with >5 subjects
""")

    print("Done.")


if __name__ == "__main__":
    main()
