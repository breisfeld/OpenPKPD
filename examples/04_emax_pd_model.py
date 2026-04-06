"""
Example 04: 1-cmt IV PK + direct Emax PD in $ERROR — FOCE estimation.

Demonstrates:
  - Emax model defined entirely in $ERROR block
  - emax_curve, effect_time, diagnostic_panel
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.data.event_processor import DoseEvent


# Emax PD $ERROR block
_ERROR_CODE = """\
E0   = THETA(3)
EMAX = THETA(4)
EC50 = THETA(5)
GAMMA = THETA(6)
W    = THETA(7)
IPRED = E0 + EMAX*F**GAMMA / (EC50**GAMMA + F**GAMMA)
Y    = IPRED + W*EPS(1)
IRES = DV - IPRED
IWRES = IRES / W
"""


def _simulate_data(n_subj: int = 6, seed: int = 99) -> NONMEMDataset:
    """Simulate PK/PD data."""
    rng = np.random.default_rng(seed)
    obs_times = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    advan1 = ADVAN1()
    dose_levels = [50.0, 50.0, 100.0, 100.0, 200.0, 200.0]
    rows = []
    for i in range(1, n_subj + 1):
        dose = dose_levels[i - 1]
        k_i = 0.15 * np.exp(rng.normal(0, 0.2))
        v_i = 10.0 * np.exp(rng.normal(0, 0.2))
        pk_params = {"K": k_i, "V": v_i}
        sol = advan1.solve(pk_params, [DoseEvent(0.0, dose, 1)], obs_times)
        c = sol.ipred

        # Emax PD: Hill exponent gamma=1.2
        e0, emax, ec50, gamma = 2.0, 15.0, 8.0, 1.2
        w = 1.5
        effect = e0 + emax * c**gamma / (ec50**gamma + c**gamma)
        dv = effect + rng.normal(0, w, len(obs_times))

        rows.append({"ID": i, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1})
        for j, t in enumerate(obs_times):
            rows.append({"ID": i, "TIME": t, "AMT": 0.0, "DV": float(dv[j]), "EVID": 0, "MDV": 0})

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def main():
    ds = _simulate_data()

    built = (
        ModelBuilder()
        .problem("1-cmt IV + Emax PD (Hill)")
        .dataset(ds)
        .subroutines(advan=1, trans=1)
        .pk("""
K = THETA(1)*EXP(ETA(1))
V = THETA(2)*EXP(ETA(2))
""")
        .error(_ERROR_CODE)
        .theta([
            (0.01, 0.15, 5.0),   # K
            (1.0, 10.0, 100.0),  # V
            (0.0, 2.0, 20.0),    # E0
            (1.0, 15.0, 100.0),  # Emax
            (0.1, 8.0, 100.0),   # EC50
            (0.1, 1.2, 5.0),     # GAMMA
            (0.1, 1.5, 20.0),    # W
        ])
        .omega([0.3, 0.3])
        .sigma(1.0, fixed=True)
        .estimation(method="FO", maxeval=600)
        .build()
    )

    print("Running FO on Emax PD model...")
    result = built.fit()
    print(result.summary())
    th = result.theta_final
    print(f"\nK={th[0]:.3f}, V={th[1]:.2f}, E0={th[2]:.2f}, "
          f"Emax={th[3]:.2f}, EC50={th[4]:.2f}, γ={th[5]:.2f}, W={th[6]:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        from openpkpd.plots.diagnostics import compute_diagnostics
        from openpkpd.plots.gof import diagnostic_panel
        from openpkpd.plots.pd import emax_curve, effect_time

        diag_df = compute_diagnostics(built.population_model, result)

        # Add IPRED (PK concentration) as a column for PD plots
        diag_df["CONC"] = diag_df["IPRED"]

        fig1 = emax_curve(
            diag_df, "CONC", "DV",
            emax=float(th[3]), ec50=float(th[4]), gamma=float(th[5]), e0=float(th[2]),
            title="Emax Curve — Observed vs Model",
        )
        fig2 = effect_time(diag_df, "DV", title="Effect-Time Profile")
        fig3 = diagnostic_panel(diag_df, title="Emax PD — GOF Panel")

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig1.savefig(os.path.join(out_dir, "04_emax_curve.png"))
            fig2.savefig(os.path.join(out_dir, "04_effect_time.png"))
            fig3.savefig(os.path.join(out_dir, "04_gof_panel.png"))
        else:
            print("Figures created (set OPENPKPD_EXAMPLE_OUTPUT to save).")

    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
