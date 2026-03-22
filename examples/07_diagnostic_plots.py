"""
Example 07: Full diagnostic plot gallery — Theophylline FOCE.

Demonstrates all available plot functions:
  - GOF panel (6 plots)
  - PK: spaghetti, concentration_time, mean_profile
  - ETA: histograms, pairs, vs covariate
  - OFV history
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset

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


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed. Install with: uv pip install matplotlib")
        return

    df = pd.read_csv(io.StringIO(THEO_DATA))
    ds = NONMEMDataset.from_dataframe(df)

    built = (
        ModelBuilder()
        .problem("Theophylline FOCE — full diagnostics")
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

    print("Running FOCE...")
    result = built.fit()
    print(result.summary())

    from openpkpd.plots.diagnostics import compute_diagnostics
    from openpkpd.plots.gof import (
        dv_vs_ipred, dv_vs_pred, cwres_vs_time, cwres_vs_pred,
        cwres_qq, abs_iwres_vs_ipred, diagnostic_panel,
    )
    from openpkpd.plots.pk import concentration_time, spaghetti_plot, mean_profile
    from openpkpd.plots.eta import eta_histograms, eta_pairs, eta_vs_covariate
    from openpkpd.plots.model_perf import ofv_history

    diag_df = compute_diagnostics(built.population_model, result)

    # Merge WT covariate from original data for eta_vs_covariate
    wt_df = df[df["EVID"] == 1][["ID", "WT"]].drop_duplicates()
    diag_df = diag_df.merge(wt_df, on="ID", how="left")

    out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")

    plots = {
        "07_gof_panel.png": diagnostic_panel(diag_df, title="Theophylline FOCE — GOF Panel"),
        "07_dv_vs_ipred.png": dv_vs_ipred(diag_df),
        "07_dv_vs_pred.png": dv_vs_pred(diag_df),
        "07_cwres_time.png": cwres_vs_time(diag_df),
        "07_cwres_pred.png": cwres_vs_pred(diag_df),
        "07_cwres_qq.png": cwres_qq(diag_df),
        "07_abs_iwres.png": abs_iwres_vs_ipred(diag_df),
        "07_spaghetti.png": spaghetti_plot(diag_df),
        "07_conc_time.png": concentration_time(diag_df),
        "07_mean_profile.png": mean_profile(diag_df, sd_band=True),
        "07_ofv_history.png": ofv_history(result),
    }

    # ETA plots (only if ETAs present)
    eta_cols = [c for c in diag_df.columns if c.startswith("ETA")]
    if eta_cols:
        plots["07_eta_hist.png"] = eta_histograms(diag_df, result.omega_final,
                                                   title="Theophylline — ETA Histograms")
        plots["07_eta_pairs.png"] = eta_pairs(diag_df, title="Theophylline — ETA Pairs")
        if "WT" in diag_df.columns:
            plots["07_eta1_vs_wt.png"] = eta_vs_covariate(diag_df, "WT", "ETA1",
                                                            title="ETA1 vs Body Weight")

    print(f"\nCreated {len(plots)} figures.")
    if out_dir:
        import matplotlib.pyplot as plt
        for fname, fig in plots.items():
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=120)
            plt.close(fig)
        print(f"Figures saved to {out_dir}")
    else:
        print("Set OPENPKPD_EXAMPLE_OUTPUT env var to save figures to disk.")


if __name__ == "__main__":
    main()
