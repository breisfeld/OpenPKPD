"""
Example 14: Simulation-based VPC — Theophylline 1-compartment oral model.

Demonstrates:
  1. Building and fitting a 1-compartment oral model from embedded data (FOCE).
  2. Computing time-binned simulated percentiles via the simulation engine.
  3. Creating a VPC-style prediction interval plot.
  4. Displaying individual simulated profiles alongside observed data.

This example is self-contained — no external data files are required.
All plots are returned as Figure objects and optionally saved to disk
when the OPENPKPD_EXAMPLE_OUTPUT environment variable is set.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset

# ---------------------------------------------------------------------------
# Theophylline dataset (12 subjects, embedded inline)
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
4,0,4.53,0,1,1,72.7
4,0.3,0,1.03,0,0,72.7
4,0.52,0,2.02,0,0,72.7
4,1.0,0,5.63,0,0,72.7
4,1.92,0,8.6,0,0,72.7
4,3.5,0,8.38,0,0,72.7
4,5.02,0,7.54,0,0,72.7
4,7.07,0,6.88,0,0,72.7
4,9.0,0,5.78,0,0,72.7
4,12.12,0,3.99,0,0,72.7
4,24.08,0,1.17,0,0,72.7
5,0,5.68,0,1,1,54.6
5,0.27,0,2.24,0,0,54.6
5,0.58,0,4.57,0,0,54.6
5,1.02,0,9.29,0,0,54.6
5,2.02,0,9.66,0,0,54.6
5,3.62,0,8.58,0,0,54.6
5,5.08,0,8.36,0,0,54.6
5,7.07,0,7.47,0,0,54.6
5,9.0,0,6.89,0,0,54.6
5,12.15,0,5.94,0,0,54.6
5,24.17,0,3.28,0,0,54.6
6,0,4.0,0,1,1,80.0
6,0.35,0,0.4,0,0,80.0
6,0.6,0,1.15,0,0,80.0
6,1.07,0,4.0,0,0,80.0
6,2.13,0,6.6,0,0,80.0
6,3.5,0,7.99,0,0,80.0
6,5.02,0,7.31,0,0,80.0
6,7.02,0,6.72,0,0,80.0
6,9.1,0,5.76,0,0,80.0
6,12.1,0,4.06,0,0,80.0
6,25.0,0,1.25,0,0,80.0
"""


def build_and_fit_model(df: pd.DataFrame) -> tuple:
    """
    Build and fit a 1-compartment oral model to theophylline data.

    Uses ADVAN2 (TRANS2) with IIV on KA, CL, V and proportional error.

    Args:
        df: Theophylline pharmacokinetic dataset.

    Returns:
        Tuple of (BuiltModel, EstimationResult).
    """
    ds = NONMEMDataset.from_dataframe(df)

    built = (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — FOCE (Example 14)")
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
        .estimation(method="FOCE", interaction=True, maxeval=600)
        .build()
    )

    print("Fitting theophylline FOCE model...")
    result = built.fit()
    print(result.summary())
    return built, result


def compute_vpc_percentiles_manual(
    built,
    result,
    n_rep: int = 200,
    seed: int = 42,
) -> dict:
    """
    Compute time-binned VPC percentiles without using VPCEngine directly.

    Runs the SimulationEngine to generate n_rep replicate datasets,
    then computes the 5th, 50th, and 95th percentiles at each unique
    observed time point.

    Args:
        built:  BuiltModel (has .population_model attribute).
        result: EstimationResult from model.fit().
        n_rep:  Number of simulation replicates.
        seed:   Random seed.

    Returns:
        Dictionary with keys:
          - times: sorted unique observed times
          - obs_p5, obs_p50, obs_p95: observed percentiles at each time
          - sim_p5, sim_p50, sim_p95: simulated percentiles at each time
    """
    from openpkpd.simulation.engine import SimulationEngine

    pop_model = built.population_model
    print(f"Simulating {n_rep} replicates (seed={seed})...")
    engine = SimulationEngine(pop_model, result, seed=seed)
    sim_result = engine.simulate(n_replicates=n_rep)
    full_df = sim_result.simulated_df

    # Observed data: REP=0, MDV=0
    obs_df = full_df[(full_df["REP"] == 0) & (full_df["MDV"] == 0)]
    sim_df = full_df[(full_df["REP"] >= 1) & (full_df["MDV"] == 0)]

    unique_times = np.sort(obs_df["TIME"].unique())

    obs_p5: list[float] = []
    obs_p50: list[float] = []
    obs_p95: list[float] = []
    sim_p5: list[float] = []
    sim_p50: list[float] = []
    sim_p95: list[float] = []

    for t in unique_times:
        obs_at_t = obs_df[obs_df["TIME"] == t]["DV"].values
        if len(obs_at_t) == 0:
            continue
        obs_p5.append(float(np.percentile(obs_at_t, 5)))
        obs_p50.append(float(np.percentile(obs_at_t, 50)))
        obs_p95.append(float(np.percentile(obs_at_t, 95)))

        sim_at_t = sim_df[np.isclose(sim_df["TIME"], t)]["DV"].values
        if len(sim_at_t) > 0:
            sim_p5.append(float(np.percentile(sim_at_t, 5)))
            sim_p50.append(float(np.percentile(sim_at_t, 50)))
            sim_p95.append(float(np.percentile(sim_at_t, 95)))
        else:
            sim_p5.append(np.nan)
            sim_p50.append(np.nan)
            sim_p95.append(np.nan)

    print(f"VPC percentiles computed for {len(unique_times)} time points.")
    return {
        "times":   np.array(unique_times),
        "obs_p5":  np.array(obs_p5),
        "obs_p50": np.array(obs_p50),
        "obs_p95": np.array(obs_p95),
        "sim_p5":  np.array(sim_p5),
        "sim_p50": np.array(sim_p50),
        "sim_p95": np.array(sim_p95),
    }


def compute_vpc_via_engine(built, result, n_rep: int = 200, seed: int = 42):
    """
    Compute VPC using VPCEngine (preferred approach).

    Falls back gracefully to None if the engine is unavailable.

    Args:
        built:  BuiltModel.
        result: EstimationResult.
        n_rep:  Number of replicates.
        seed:   Random seed.

    Returns:
        VPCResult or None if VPCEngine is unavailable.
    """
    try:
        from openpkpd.simulation import SimulationEngine, VPCEngine

        pop_model = built.population_model
        print(f"Running VPCEngine with {n_rep} replicates...")
        sim_engine = SimulationEngine(pop_model, result, seed=seed)
        vpc_engine = VPCEngine(sim_engine)
        vpc_result = vpc_engine.compute(n_replicates=n_rep, n_bins=8)
        print("VPCEngine computation complete.")
        return vpc_result
    except ImportError as exc:
        print(f"VPCEngine not available ({exc}); using manual percentile computation.")
        return None
    except Exception as exc:
        print(f"VPCEngine failed ({exc}); falling back to manual computation.")
        return None


def make_plots(
    df: pd.DataFrame,
    built,
    result,
    vpc_pct: dict,
    vpc_result=None,
    sim_result=None,
) -> dict:
    """
    Create all VPC-related figures.

    Args:
        df:         Original theophylline DataFrame.
        built:      BuiltModel.
        result:     EstimationResult.
        vpc_pct:    Dictionary of manually computed VPC percentiles.
        vpc_result: VPCResult from VPCEngine (optional).
        sim_result: SimulationResult (optional, used for simulation_panel).

    Returns:
        Dictionary mapping filename -> Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from openpkpd.plots.simulation import prediction_interval_plot
    from openpkpd.plots.diagnostics import compute_diagnostics

    figs: dict = {}

    # ── Figure 1: Prediction Interval Plot (manual percentiles) ──────────────
    times = vpc_pct["times"]
    obs_conc_for_vpc = vpc_pct["obs_p50"]  # observed median as "representative"

    print("Creating prediction interval plot...")
    fig1 = prediction_interval_plot(
        times=times,
        obs_conc=obs_conc_for_vpc,
        sim_pi_lo=vpc_pct["sim_p5"],
        sim_median=vpc_pct["sim_p50"],
        sim_pi_hi=vpc_pct["sim_p95"],
        title="Theophylline VPC — Prediction Intervals (Manual)",
        figsize=(10, 6),
    )
    figs["14_vpc_prediction_interval.png"] = fig1

    # ── Figure 2: VPC via simulation.vpc_plot (if VPCResult available) ───────
    if vpc_result is not None:
        from openpkpd.plots.simulation import vpc_plot
        print("Creating vpc_plot from VPCResult...")
        fig2 = vpc_plot(
            vpc_result,
            title="Theophylline VPC — VPCEngine Result",
            figsize=(10, 6),
        )
        figs["14_vpc_engine.png"] = fig2

    # ── Figure 3: Classic VPC from model_perf.vpc (raw Monte Carlo) ──────────
    try:
        from openpkpd.plots.model_perf import vpc as vpc_modelperf
        from openpkpd.plots.diagnostics import compute_diagnostics

        print("Computing diagnostic DataFrame...")
        diag_df = compute_diagnostics(built.population_model, result)

        print("Creating VPC via model_perf.vpc() (n_sim=100)...")
        fig3 = vpc_modelperf(
            diag_df,
            built.population_model,
            result,
            n_sim=100,
            title="Theophylline VPC — model_perf.vpc()",
        )
        figs["14_vpc_model_perf.png"] = fig3
    except Exception as exc:
        print(f"model_perf.vpc() skipped: {exc}")

    # ── Figure 4: Simulation panel (individual simulated profiles) ────────────
    if sim_result is not None:
        from openpkpd.plots.simulation import simulation_panel

        sim_df = sim_result.simulated_df
        obs_df = sim_df[sim_df["REP"] == 0][["ID", "TIME", "DV"]].copy()
        obs_df = obs_df[obs_df["DV"] > 0]  # exclude dose rows

        print("Creating simulation panel (individual profiles)...")
        fig4 = simulation_panel(
            simulated_df=sim_df[sim_df["REP"] >= 1],
            observed_df=obs_df,
            n_subjects=6,
            figsize=(15, 10),
            title="Theophylline — Simulated vs Observed Individual Profiles",
        )
        figs["14_simulation_panel.png"] = fig4

    # ── Figure 5: DV and PRED profiles on same axes (observed + predicted) ────
    try:
        from openpkpd.plots.diagnostics import compute_diagnostics
        from openpkpd.plots.pk import spaghetti_plot

        diag_df = compute_diagnostics(built.population_model, result)
        print("Creating spaghetti plot...")
        fig5 = spaghetti_plot(
            diag_df,
            title="Theophylline — Individual Concentration-Time Profiles",
        )
        figs["14_spaghetti.png"] = fig5
    except Exception as exc:
        print(f"Spaghetti plot skipped: {exc}")

    return figs


def main() -> None:
    """
    Main entry point for Example 14.

    Runs the full pipeline:
      1. Load embedded theophylline data.
      2. Fit a 1-compartment oral model (FOCE).
      3. Simulate 200 replicates using SimulationEngine.
      4. Compute VPC percentile bands.
      5. Create VPC-style plots.
      6. Save or display figures.
    """
    # Check matplotlib availability upfront
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed. Install with: uv pip install matplotlib")
        print("Plots will not be generated.")
        return

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Example 14: Simulation-based VPC")
    print("=" * 60)
    df = pd.read_csv(io.StringIO(THEO_DATA))
    print(f"Loaded {len(df)} rows, {df['ID'].nunique()} subjects")

    # ── Step 2: Build and fit model ───────────────────────────────────────────
    built, result = build_and_fit_model(df)
    print(f"\nFitted parameters:")
    print(f"  KA = {result.theta_final[0]:.4f} hr^-1")
    print(f"  CL = {result.theta_final[1]:.4f} L/hr")
    print(f"  V  = {result.theta_final[2]:.4f} L")

    # ── Step 3: Simulate replicates ───────────────────────────────────────────
    n_rep = 200
    sim_result = None
    try:
        from openpkpd.simulation.engine import SimulationEngine
        sim_engine = SimulationEngine(built.population_model, result, seed=42)
        sim_result = sim_engine.simulate(n_replicates=n_rep)
        print(f"\nGenerated {n_rep} replicate datasets "
              f"({len(sim_result.simulated_df)} total rows)")
    except ImportError as exc:
        print(f"SimulationEngine not available: {exc}")

    # ── Step 4: Compute VPC percentiles ───────────────────────────────────────
    # Try VPCEngine first (preferred); fall back to manual computation
    vpc_result = compute_vpc_via_engine(built, result, n_rep=n_rep)

    if sim_result is not None:
        vpc_pct = compute_vpc_percentiles_manual(built, result, n_rep=n_rep)
    else:
        # Fallback: synthetic percentiles for demonstration
        print("Using synthetic percentiles for demonstration (simulation not available).")
        obs_df = df[df["EVID"] == 0][["TIME", "DV"]].copy()
        unique_t = np.sort(obs_df["TIME"].unique())
        vpc_pct = {
            "times":   unique_t,
            "obs_p5":  np.full(len(unique_t), np.nan),
            "obs_p50": np.full(len(unique_t), np.nan),
            "obs_p95": np.full(len(unique_t), np.nan),
            "sim_p5":  np.full(len(unique_t), np.nan),
            "sim_p50": np.full(len(unique_t), np.nan),
            "sim_p95": np.full(len(unique_t), np.nan),
        }

    # ── Step 5: Create plots ──────────────────────────────────────────────────
    print("\nGenerating plots...")
    figs = make_plots(df, built, result, vpc_pct,
                      vpc_result=vpc_result, sim_result=sim_result)

    # ── Step 6: Save or display ───────────────────────────────────────────────
    print(f"\nCreated {len(figs)} figures.")
    out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")

    if out_dir:
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        for fname, fig in figs.items():
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")
        print(f"\nAll figures saved to: {out_dir}")
    else:
        print("Set OPENPKPD_EXAMPLE_OUTPUT environment variable to save figures.")
        print("Example: OPENPKPD_EXAMPLE_OUTPUT=/tmp/figs python 14_simulation_vpc.py")
        import matplotlib.pyplot as plt
        for fig in figs.values():
            plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
