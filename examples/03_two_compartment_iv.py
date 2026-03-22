"""
Example 03: 2-compartment IV model (ADVAN3 TRANS4) — FO estimation.

Demonstrates:
  - ADVAN3 with TRANS4 parameterization (CL, V1, Q, V2)
  - Log-scale concentration-time plot
  - spaghetti plot
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.data.event_processor import DoseEvent


def _simulate_data(n_subj: int = 8, seed: int = 0) -> NONMEMDataset:
    """Simulate 2-cmt IV dataset from ADVAN3."""
    rng = np.random.default_rng(seed)
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    advan3 = ADVAN3()
    rows = []
    for i in range(1, n_subj + 1):
        dose = 100.0
        k = 0.2 * np.exp(rng.normal(0, 0.2))
        k12 = 0.08 * np.exp(rng.normal(0, 0.15))
        k21 = 0.04 * np.exp(rng.normal(0, 0.15))
        v1 = 8.0 * np.exp(rng.normal(0, 0.2))
        pk_params = {"K": k, "K12": k12, "K21": k21, "V1": v1}
        sol = advan3.solve(pk_params, [DoseEvent(0.0, dose, 1)], obs_times)
        dv = np.maximum(sol.ipred * (1 + rng.normal(0, 0.1, len(obs_times))), 1e-4)
        rows.append({"ID": i, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1})
        for j, t in enumerate(obs_times):
            rows.append({"ID": i, "TIME": t, "AMT": 0.0, "DV": float(dv[j]), "EVID": 0, "MDV": 0})
    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def main():
    ds = _simulate_data()

    built = (
        ModelBuilder()
        .problem("2-cmt IV ADVAN3 FO")
        .dataset(ds)
        .subroutines(advan=3, trans=1)
        .pk("""
CL = THETA(1)*EXP(ETA(1))
V1 = THETA(2)*EXP(ETA(2))
Q  = THETA(3)
V2 = THETA(4)
K  = CL/V1
K12 = Q/V1
K21 = Q/V2
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 1.6, 30), (1.0, 8.0, 100), (0.1, 0.64, 10), (1.0, 8.0, 100)])
        .omega([0.4, 0.4])
        .sigma(0.05)
        .estimation(method="FO", maxeval=600)
        .build()
    )

    print("Running FO on 2-cmt IV model...")
    result = built.fit()
    print(result.summary())

    try:
        import matplotlib
        matplotlib.use("Agg")
        from openpkpd.plots.diagnostics import compute_diagnostics
        from openpkpd.plots.pk import concentration_time, spaghetti_plot

        diag_df = compute_diagnostics(built.population_model, result)
        fig1 = concentration_time(diag_df, log_y=True,
                                  title="2-cmt IV — Log Conc-Time (FO)")
        fig2 = spaghetti_plot(diag_df, log_y=True,
                              title="2-cmt IV — Spaghetti (Log)")

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig1.savefig(os.path.join(out_dir, "03_log_conc_time.png"))
            fig2.savefig(os.path.join(out_dir, "03_spaghetti_log.png"))
        else:
            print("Figures created (set OPENPKPD_EXAMPLE_OUTPUT to save).")

    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
