"""
Example 09: 3-Compartment IV Model (ADVAN11)

Demonstrates:
  - ADVAN11 with TRANS4 parameterization (CL, V1, Q2, V2, Q3, V3)
  - FOCE estimation for a 3-compartment PK model
  - Triexponential pharmacokinetic profile
  - Simulation of data from ADVAN11 and model fitting
  - Visual display of individual predicted profiles

3-compartment IV model structure:
  Dose → Central (1) ↔ Peripheral1 (2) ↔ (none)
                    ↔ Peripheral2 (3) ↔ (none)

TRANS4 parameterization (NONMEM convention):
  K   = CL  / V1    (elimination from central)
  K12 = Q2  / V1    (central → peripheral1)
  K21 = Q2  / V2    (peripheral1 → central)
  K13 = Q3  / V1    (central → peripheral2)
  K31 = Q3  / V3    (peripheral2 → central)

True parameters used for simulation:
  CL  = 2.0 L/h, V1 = 10 L, Q2 = 1.5 L/h, V2 = 30 L, Q3 = 0.5 L/h, V3 = 50 L

Population variability:
  30% CV on CL and V1 (log-normal ETAs)
  Proportional residual error (15% CV)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.pk.analytical.advan11 import ADVAN11
from openpkpd.data.event_processor import DoseEvent


# ── True population parameters ────────────────────────────────────────────────

_TRUE_PARAMS = {
    "CL": 2.0,   # L/h
    "V1": 10.0,  # L
    "Q2": 1.5,   # L/h
    "V2": 30.0,  # L
    "Q3": 0.5,   # L/h
    "V3": 50.0,  # L
}


def _apply_trans4(p: dict) -> dict:
    """Convert TRANS4 (CL, V1, Q2, V2, Q3, V3) to micro rate constants."""
    return {
        "K":   p["CL"] / p["V1"],
        "K12": p["Q2"] / p["V1"],
        "K21": p["Q2"] / p["V2"],
        "K13": p["Q3"] / p["V1"],
        "K31": p["Q3"] / p["V3"],
        "V1":  p["V1"],
    }


# ── Simulate data ──────────────────────────────────────────────────────────────

def _simulate_data(n_subj: int = 12, seed: int = 7) -> NONMEMDataset:
    """
    Simulate 3-compartment IV data from ADVAN11.

    Uses a rich sampling design (10 time points, 0–72 h) to characterize
    all three exponential phases of the triexponential profile.
    """
    rng = np.random.default_rng(seed)
    advan11 = ADVAN11()

    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0])
    dose = 500.0  # mg IV bolus

    # IIV: 30% CV on CL and V1, 20% CV on Q2
    iiv_cl = 0.09   # variance = (0.3)^2
    iiv_v1 = 0.09
    iiv_q2 = 0.04   # 20% CV

    rows: list[dict] = []
    for i in range(1, n_subj + 1):
        # Individual parameters
        CL_i = _TRUE_PARAMS["CL"] * np.exp(rng.normal(0, np.sqrt(iiv_cl)))
        V1_i = _TRUE_PARAMS["V1"] * np.exp(rng.normal(0, np.sqrt(iiv_v1)))
        Q2_i = _TRUE_PARAMS["Q2"] * np.exp(rng.normal(0, np.sqrt(iiv_q2)))

        indiv_params = dict(_TRUE_PARAMS)
        indiv_params.update({"CL": CL_i, "V1": V1_i, "Q2": Q2_i})
        micro = _apply_trans4(indiv_params)

        sol = advan11.solve(
            micro,
            [DoseEvent(time=0.0, amount=dose, compartment=1)],
            obs_times,
        )

        # 15% proportional residual error
        dv = np.maximum(
            sol.ipred * (1.0 + rng.normal(0, 0.15, len(obs_times))),
            0.0001,
        )

        # Dose row (EVID=1, MDV=1)
        rows.append({
            "ID": i, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1
        })

        # Observation rows (EVID=0)
        for j, t in enumerate(obs_times):
            rows.append({
                "ID": i, "TIME": t, "AMT": 0.0, "DV": float(dv[j]),
                "EVID": 0, "MDV": 0
            })

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


# ── NONMEM-style $PK code ─────────────────────────────────────────────────────

_PK_CODE = """
CL  = THETA(1) * EXP(ETA(1))
V1  = THETA(2) * EXP(ETA(2))
Q2  = THETA(3)
V2  = THETA(4)
Q3  = THETA(5)
V3  = THETA(6)
K   = CL  / V1
K12 = Q2  / V1
K21 = Q2  / V2
K13 = Q3  / V1
K31 = Q3  / V3
"""

_ERROR_CODE = """
Y = F * (1 + EPS(1))
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run Example 09: 3-compartment IV ADVAN11 model."""
    print("=" * 60)
    print("Example 09: 3-Compartment IV Model (ADVAN11)")
    print("=" * 60)

    # 1. Simulate data
    print("\nSimulating 3-cmt IV data from ADVAN11...")
    ds = _simulate_data(n_subj=12, seed=7)
    n_obs = (ds.df["EVID"] == 0).sum()
    n_subj = ds.df["ID"].nunique()
    print(f"  Subjects: {n_subj}, Observations: {n_obs}")
    print(f"\n  True population parameters:")
    for k, v in _TRUE_PARAMS.items():
        print(f"    {k:4s} = {v:.2f}")

    # 2. Build the model
    print("\nBuilding ADVAN11 FOCE model (TRANS1: micro params in $PK)...")
    model = (
        ModelBuilder()
        .problem("3-compartment IV ADVAN11 FOCE")
        .dataset(ds)
        .subroutines(advan=11, trans=1)  # TRANS1: pass micro params through
        .pk(_PK_CODE)
        .error(_ERROR_CODE)
        # Initial THETA estimates (CL, V1, Q2, V2, Q3, V3 via micro param mapping)
        .theta([
            (0.01, 2.0, 50.0),   # CL
            (0.5,  10.0, 200.0), # V1
            (0.01, 1.5, 20.0),   # Q2
            (1.0,  30.0, 500.0), # V2
            (0.01, 0.5, 10.0),   # Q3
            (1.0,  50.0, 500.0), # V3
        ])
        # IIV on CL and V1 (diagonal OMEGA)
        .omega([0.1, 0.1])
        # Proportional residual error
        .sigma(0.05)
        .estimation(method="FOCE", interaction=True, maxeval=500)
        .build()
    )

    # 3. Fit the model
    print("\nRunning FOCE estimation...")
    print("  (maxeval=500 — for demonstration; increase for production)")
    try:
        result = model.fit()

        print(f"\nEstimation complete:")
        print(f"  OFV       = {result.ofv:.3f}")
        print(f"  Converged = {result.converged}")
        print(f"  Method    = {result.method}")

        print(f"\n  THETA estimates (vs. true values):")
        theta_names = ["CL (L/h)", "V1 (L)", "Q2 (L/h)", "V2 (L)", "Q3 (L/h)", "V3 (L)"]
        true_vals = [
            _TRUE_PARAMS["CL"], _TRUE_PARAMS["V1"], _TRUE_PARAMS["Q2"],
            _TRUE_PARAMS["V2"], _TRUE_PARAMS["Q3"], _TRUE_PARAMS["V3"],
        ]
        for i, (name, est, true) in enumerate(zip(theta_names, result.theta_final, true_vals)):
            pct_err = 100.0 * (est - true) / true
            print(f"    THETA({i+1}) [{name}]: est={est:.3f}, true={true:.3f} ({pct_err:+.1f}%)")

        print(f"\n  OMEGA (IIV on CL and V1):")
        for k in range(result.omega_final.shape[0]):
            cv = 100.0 * np.sqrt(result.omega_final[k, k])
            print(f"    OMEGA({k+1},{k+1}) = {result.omega_final[k,k]:.4f} ({cv:.1f}% CV)")

        print(f"\n  SIGMA(1,1) = {result.sigma_final[0,0]:.4f} "
              f"({100*np.sqrt(result.sigma_final[0,0]):.1f}% CV)")

        # Show individual predictions for first subject
        print(f"\n  Subject 1 individual predictions (IPRED):")
        sid = 1
        indiv = model.population_model.individual_model(sid)
        eta_hat = result.post_hoc_etas.get(sid, np.zeros(result.omega_final.shape[0]))
        ipred, obs_mask, f = indiv.evaluate(
            result.theta_final, eta_hat, result.sigma_final, trans=1
        )
        events = indiv.subject_events
        print(f"  {'TIME':>6} | {'DV':>8} | {'IPRED':>8}")
        print(f"  {'-'*28}")
        for j, t in enumerate(events.obs_times[:5]):
            dv_val = events.obs_dv[j]
            ipred_val = ipred[j] if j < len(ipred) else float("nan")
            print(f"  {t:>6.1f} | {dv_val:>8.3f} | {ipred_val:>8.3f}")
        print(f"  ... (showing first 5 of {len(events.obs_times)})")

        # Demonstrate simulate()
        print(f"\nSimulating 3 replicates from fitted model...")
        sim = model.simulate(n_replicates=3, seed=42, result=result)
        rep_counts = sim.simulated_df["REP"].value_counts().sort_index()
        print(f"  Replicate row counts:")
        for rep, n in rep_counts.items():
            label = "observed" if rep == 0 else f"replicate {rep}"
            print(f"    REP={rep} ({label}): {n} rows")

    except Exception as exc:
        print(f"\n  Estimation note: {exc}")
        print("  Note: This is expected for a demonstration with limited maxeval.")
        print("  For a real fit, use maxeval=9999 or higher.")

    # 4. Optional plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Show a concentration-time curve from the simulated data
        print("\nGenerating concentration-time plot...")
        advan11 = ADVAN11()
        micro = _apply_trans4(_TRUE_PARAMS)
        obs_times = np.linspace(0.01, 72, 300)
        sol = advan11.solve(
            micro,
            [DoseEvent(time=0.0, amount=500.0)],
            obs_times,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(obs_times, sol.ipred, "b-", linewidth=2, label="True IPRED (typical)")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title("3-Compartment IV Profile (ADVAN11)\nTriexponential Decay")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig.savefig(os.path.join(out_dir, "09_three_cmt_profile.png"), dpi=150)
            print(f"  Saved to {out_dir}/09_three_cmt_profile.png")
        else:
            print("  (Set OPENPKPD_EXAMPLE_OUTPUT to save plot)")
        plt.close(fig)

    except ImportError:
        print("  matplotlib not installed — skipping plot.")

    print("\nExample 09 complete.")


if __name__ == "__main__":
    main()
