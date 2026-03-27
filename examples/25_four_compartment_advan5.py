"""
Example 25: 4-Compartment General Linear Model (ADVAN5)

Demonstrates:
  - ADVAN5 with TRANS1 (Kij micro rate constants)
  - N-compartment structure inferred automatically from parameter keys
  - FOCE estimation for a 4-compartment PK model
  - Convergence check: ADVAN5 (N=4, Q4=0) == ADVAN11 (N=3)
  - Comparison with ADVAN11 as a special case

4-compartment IV model structure:
  Dose → Central (1) ↔ Peripheral 1 (2)
                    ↔ Peripheral 2 (3)
                    ↔ Peripheral 3 (4)

True parameters (macro → micro via _apply_trans_4cmt):
  CL = 2.0 L/h, V1 = 10 L
  Q2 = 1.5 L/h, V2 = 30 L
  Q3 = 0.5 L/h, V3 = 50 L
  Q4 = 0.1 L/h, V4 = 100 L

Population variability:
  30% CV on CL and V1 (log-normal ETAs)
  Proportional residual error (15% CV)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan5 import ADVAN5
from openpkpd.pk.analytical.advan11 import ADVAN11


# ── True population parameters ────────────────────────────────────────────────

_TRUE_PARAMS = {
    "CL": 2.0,   # L/h — elimination clearance
    "V1": 10.0,  # L   — central volume
    "Q2": 1.5,   # L/h — inter-compartmental clearance (cmt 1 ↔ 2)
    "V2": 30.0,  # L   — peripheral 1 volume
    "Q3": 0.5,   # L/h — inter-compartmental clearance (cmt 1 ↔ 3)
    "V3": 50.0,  # L   — peripheral 2 volume
    "Q4": 0.1,   # L/h — inter-compartmental clearance (cmt 1 ↔ 4)
    "V4": 100.0, # L   — peripheral 3 volume
}


def _apply_trans_4cmt(p: dict) -> dict:
    """Convert macro parameters to micro rate constants for ADVAN5/TRANS1."""
    return {
        "K":   p["CL"] / p["V1"],
        "K12": p["Q2"] / p["V1"],
        "K21": p["Q2"] / p["V2"],
        "K13": p["Q3"] / p["V1"],
        "K31": p["Q3"] / p["V3"],
        "K14": p["Q4"] / p["V1"],
        "K41": p["Q4"] / p["V4"],
        "V1":  p["V1"],
    }


# ── Simulate data ──────────────────────────────────────────────────────────────

def _simulate_data(n_subj: int = 12, seed: int = 42) -> NONMEMDataset:
    """
    Simulate 4-compartment IV data from ADVAN5.

    Uses 11 sampling time points (0.25–96 h) to characterise all phases of
    the quadra-exponential profile.
    """
    rng = np.random.default_rng(seed)
    advan5 = ADVAN5()

    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0])
    dose = 500.0  # mg IV bolus

    iiv_cl = 0.09   # 30% CV  (variance = (0.3)^2)
    iiv_v1 = 0.09

    rows: list[dict] = []
    for i in range(1, n_subj + 1):
        CL_i = _TRUE_PARAMS["CL"] * np.exp(rng.normal(0, np.sqrt(iiv_cl)))
        V1_i = _TRUE_PARAMS["V1"] * np.exp(rng.normal(0, np.sqrt(iiv_v1)))

        indiv_params = dict(_TRUE_PARAMS)
        indiv_params.update({"CL": CL_i, "V1": V1_i})
        micro = _apply_trans_4cmt(indiv_params)

        sol = advan5.solve(
            micro,
            [DoseEvent(time=0.0, amount=dose, compartment=1)],
            obs_times,
        )

        dv = np.maximum(
            sol.ipred * (1.0 + rng.normal(0, 0.15, len(obs_times))),
            0.0001,
        )

        rows.append({"ID": i, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1})
        for j, t in enumerate(obs_times):
            rows.append({"ID": i, "TIME": t, "AMT": 0.0, "DV": float(dv[j]), "EVID": 0, "MDV": 0})

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


# ── NONMEM-style $PK code ─────────────────────────────────────────────────────

_PK_CODE = """
CL  = THETA(1) * EXP(ETA(1))
V1  = THETA(2) * EXP(ETA(2))
Q2  = THETA(3)
V2  = THETA(4)
Q3  = THETA(5)
V3  = THETA(6)
Q4  = THETA(7)
V4  = THETA(8)
K   = CL  / V1
K12 = Q2  / V1
K21 = Q2  / V2
K13 = Q3  / V1
K31 = Q3  / V3
K14 = Q4  / V1
K41 = Q4  / V4
"""

_ERROR_CODE = """
Y = F * (1 + EPS(1))
"""



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run Example 25: 4-compartment IV ADVAN5 model."""
    print("=" * 60)
    print("Example 25: 4-Compartment General Linear Model (ADVAN5)")
    print("=" * 60)

    # --- Simulate data ---
    print("\n[1] Simulating 4-compartment data from ADVAN5...")
    dataset = _simulate_data(n_subj=12, seed=42)
    n_obs = (dataset.df["EVID"] == 0).sum()
    print(f"    {len(dataset.df['ID'].unique())} subjects, {n_obs} observations")

    # --- Build and fit model ---
    print("\n[2] Fitting ADVAN5 (N=4) via FOCE...")
    model = (
        ModelBuilder()
        .data(dataset)
        .subroutines(advan=5, trans=1)
        .pk(_PK_CODE)
        .error(_ERROR_CODE)
        .theta(
            [2.0, 10.0, 1.5, 30.0, 0.5, 50.0, 0.1, 100.0],
            lower=[0.1, 1.0, 0.01, 5.0, 0.01, 10.0, 0.001, 20.0],
        )
        .omega([[0.09, 0.0], [0.0, 0.09]])
        .sigma([[0.04]])
        .estimation(method="FOCE", maxeval=500)
        .build()
    )

    result = model.fit()

    print("\n    Parameter estimates vs. true values:")
    true_vals = [_TRUE_PARAMS[k] for k in ("CL", "V1", "Q2", "V2", "Q3", "V3", "Q4", "V4")]
    names = ["CL", "V1", "Q2", "V2", "Q3", "V3", "Q4", "V4"]
    thetas = result.estimates.get("THETA", [])
    for name, true_val, est_val in zip(names, true_vals, thetas):
        pct_err = 100.0 * (est_val - true_val) / true_val if true_val else float("nan")
        print(f"    {name:4s}: true={true_val:6.2f}  est={est_val:6.2f}  ({pct_err:+.1f}%)")

    # --- ADVAN5 vs ADVAN11 convergence check ---
    print("\n[3] Convergence check: ADVAN5 (K14=K41=0) vs ADVAN11...")
    micro_3cmt = _apply_trans_4cmt({**_TRUE_PARAMS, "Q4": 0.0})   # zero out 4th cmt
    micro_3cmt_11 = {
        "K":   _TRUE_PARAMS["CL"] / _TRUE_PARAMS["V1"],
        "K12": _TRUE_PARAMS["Q2"] / _TRUE_PARAMS["V1"],
        "K21": _TRUE_PARAMS["Q2"] / _TRUE_PARAMS["V2"],
        "K13": _TRUE_PARAMS["Q3"] / _TRUE_PARAMS["V1"],
        "K31": _TRUE_PARAMS["Q3"] / _TRUE_PARAMS["V3"],
        "V1":  _TRUE_PARAMS["V1"],
    }

    dose = [DoseEvent(time=0.0, amount=500.0)]
    check_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0])

    sol5 = ADVAN5().solve(micro_3cmt, dose, check_times)
    sol11 = ADVAN11().solve(micro_3cmt_11, dose, check_times)

    max_diff = np.max(np.abs(sol5.ipred - sol11.ipred))
    print(f"    Max |ΔIPRED| = {max_diff:.2e}  (should be < 1e-10)")

    if max_diff < 1e-8:
        print("    ✓ ADVAN5 and ADVAN11 agree to machine precision when Q4=0")
    else:
        print("    ✗ Unexpected discrepancy — investigate!")

    print("\nDone.")


if __name__ == "__main__":
    main()
