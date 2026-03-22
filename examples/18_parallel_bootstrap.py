"""
Example 18: Parallel execution backend and bootstrap resampling.

Demonstrates:
  - Using get_backend() to distribute computation across CPUs
  - Selecting between multiprocessing, Dask, and Ray backends
  - Running a parameter bootstrap manually with the parallel backend
  - Building a bootstrap confidence interval from repeated fits

Background
----------
OpenPKPD's parallel module provides a unified map() interface:

    from openpkpd.parallel import get_backend

    backend = get_backend(n_jobs=4)                 # auto-select best backend
    results = backend.map(fit_function, arg_list)   # returns list[Any] in order

Supported backends (in auto-selection priority):
  1. Dask distributed  (pip install dask[distributed])
  2. Ray               (pip install ray)
  3. multiprocessing   (always available, no extras)

This example demonstrates a simple non-parametric bootstrap of a
one-compartment pharmacokinetic model fit.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Embedded mini-dataset: 3 subjects, theophylline-like data
# ------------------------------------------------------------------
THEO_MINI = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,4.02,0,1,1,79.6
1,0.57,0,1.72,0,0,79.6
1,1.92,0,8.31,0,0,79.6
1,3.5,0,8.33,0,0,79.6
1,7.03,0,6.08,0,0,79.6
1,12.05,0,4.55,0,0,79.6
2,0,4.4,0,1,1,72.4
2,0.6,0,2.33,0,0,72.4
2,2.13,0,8.33,0,0,72.4
2,3.5,0,9.02,0,0,72.4
2,7.02,0,5.68,0,0,72.4
2,12.1,0,3.01,0,0,72.4
3,0,4.95,0,1,1,70.5
3,0.58,0,1.92,0,0,70.5
3,1.92,0,7.03,0,0,70.5
3,3.5,0,9.07,0,0,70.5
3,7.02,0,6.59,0,0,70.5
3,12.15,0,4.73,0,0,70.5
"""


# ---------------------------------------------------------------------------
# 1. A simple fit function that can be pickled and sent to workers
# ---------------------------------------------------------------------------

def fit_bootstrap_replicate(replicate_data: pd.DataFrame) -> dict | None:
    """
    Fit a one-compartment oral FO model on a bootstrap replicate.

    Parameters
    ----------
    replicate_data : pd.DataFrame
        NONMEM-format data for the replicate (subjects sampled with replacement).

    Returns
    -------
    dict with keys theta_final, ofv, converged; or None on failure.
    """
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    try:
        ds = NONMEMDataset.from_dataframe(replicate_data)
        built = (
            ModelBuilder()
            .problem("Bootstrap replicate")
            .dataset(ds)
            .subroutines(advan=2, trans=2)
            .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
            .error("Y = F*(1 + EPS(1))")
            .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
            .omega([0.3, 0.3, 0.3])
            .sigma(0.1)
            .estimation(method="FO", maxeval=300)
            .build()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = built.fit()
        return {
            "theta_final": result.theta_final.tolist(),
            "ofv": result.ofv,
            "converged": result.converged,
        }
    except Exception as e:
        return {"error": str(e), "converged": False}


# ---------------------------------------------------------------------------
# 2. Generate bootstrap replicates
# ---------------------------------------------------------------------------

def generate_replicates(df: pd.DataFrame, n_replicates: int, seed: int = 42):
    """
    Sample subjects with replacement to generate bootstrap replicates.

    Each replicate has the same number of subjects as the original dataset
    but with subjects re-labeled 1..N.
    """
    rng = np.random.default_rng(seed)
    subjects = df["ID"].unique()
    n_subjects = len(subjects)

    replicates = []
    for _ in range(n_replicates):
        sampled = rng.choice(subjects, size=n_subjects, replace=True)
        frames = []
        for new_id, orig_id in enumerate(sampled, start=1):
            subj_df = df[df["ID"] == orig_id].copy()
            subj_df["ID"] = new_id
            frames.append(subj_df)
        rep_df = pd.concat(frames, ignore_index=True)
        replicates.append(rep_df)
    return replicates


# ---------------------------------------------------------------------------
# 3. Summarise bootstrap results
# ---------------------------------------------------------------------------

def bootstrap_ci(results: list[dict], param_index: int, label: str, alpha: float = 0.05):
    """Compute percentile bootstrap CI for a given THETA index."""
    values = [r["theta_final"][param_index] for r in results if r.get("converged")]
    if not values:
        print(f"  {label}: no converged replicates")
        return
    lo = np.percentile(values, 100 * alpha / 2)
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    median = np.median(values)
    print(f"  {label:<6}: median={median:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  "
          f"(n={len(values)} converged)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from openpkpd.parallel import get_backend

    print("=" * 60)
    print("Example 18: Parallel bootstrap with get_backend()")
    print("=" * 60)

    # Load data
    df = pd.read_csv(io.StringIO(THEO_MINI))
    print(f"\nDataset: {df['ID'].nunique()} subjects, {(df['EVID']==0).sum()} observations")

    # Number of bootstrap replicates (small for the example; use 200+ in practice)
    n_boot = 10

    print(f"\nGenerating {n_boot} bootstrap replicates...")
    replicates = generate_replicates(df, n_replicates=n_boot, seed=99)

    # ------------------------------------------------------------------
    # Select backend — auto-selects Dask > Ray > multiprocessing
    # ------------------------------------------------------------------
    n_jobs = 2  # Use 2 workers for the example; set to -1 for all CPUs
    backend = get_backend(n_jobs=n_jobs, backend="multiprocessing")
    print(f"Backend: {type(backend).__name__}  (n_jobs={backend.n_jobs})")

    print(f"\nFitting {n_boot} bootstrap replicates in parallel...")
    with backend:
        boot_results = backend.map(fit_bootstrap_replicate, replicates)

    n_converged = sum(1 for r in boot_results if r and r.get("converged"))
    print(f"Converged: {n_converged}/{n_boot}")

    # ------------------------------------------------------------------
    # Bootstrap CIs (95% percentile method)
    # ------------------------------------------------------------------
    converged_results = [r for r in boot_results if r and r.get("converged")]
    if converged_results:
        print("\nBootstrap 95% confidence intervals:")
        bootstrap_ci(converged_results, 0, "KA")
        bootstrap_ci(converged_results, 1, "CL")
        bootstrap_ci(converged_results, 2, "V")
    else:
        print("\nNo converged replicates to summarise.")

    # ------------------------------------------------------------------
    # Context manager usage (alternative)
    # ------------------------------------------------------------------
    print("\nContext manager usage (auto-close backend):")
    from openpkpd.parallel import _MultiprocessingBackend

    simple_results = []
    with _MultiprocessingBackend(n_jobs=1) as b:
        simple_results = b.map(lambda x: x * 2, [1, 2, 3, 4, 5])
    print(f"  map([1..5], x*2) = {simple_results}")

    print("\nDone.")


if __name__ == "__main__":
    main()
