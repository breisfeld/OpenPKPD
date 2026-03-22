"""
Example 02: Warfarin 1-compartment oral model — FOCE estimation.

Demonstrates:
  - FOCE with interaction
  - diagnostic_panel, eta_histograms
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset

WARFARIN_DATA = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,70.0,0,1,1,70.0
1,0.5,0,1.42,0,0,70.0
1,2.0,0,4.01,0,0,70.0
1,8.0,0,5.40,0,0,70.0
1,24.0,0,3.56,0,0,70.0
1,72.0,0,1.19,0,0,70.0
1,120.0,0,0.44,0,0,70.0
2,0,65.0,0,1,1,65.0
2,0.5,0,1.30,0,0,65.0
2,2.0,0,3.80,0,0,65.0
2,8.0,0,5.20,0,0,65.0
2,24.0,0,3.40,0,0,65.0
2,72.0,0,1.10,0,0,65.0
2,120.0,0,0.40,0,0,65.0
3,0,80.0,0,1,1,80.0
3,0.5,0,1.55,0,0,80.0
3,2.0,0,4.30,0,0,80.0
3,8.0,0,5.65,0,0,80.0
3,24.0,0,3.75,0,0,80.0
3,72.0,0,1.25,0,0,80.0
3,120.0,0,0.48,0,0,80.0
4,0,75.0,0,1,1,75.0
4,0.5,0,1.48,0,0,75.0
4,2.0,0,4.10,0,0,75.0
4,8.0,0,5.45,0,0,75.0
4,24.0,0,3.60,0,0,75.0
4,72.0,0,1.20,0,0,75.0
4,120.0,0,0.45,0,0,75.0
"""


def main():
    df = pd.read_csv(io.StringIO(WARFARIN_DATA))
    ds = NONMEMDataset.from_dataframe(df)

    built = (
        ModelBuilder()
        .problem("Warfarin 1-cmt oral FOCE")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 0.9, 20), (0.001, 0.13, 5), (0.1, 8.7, 200)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(method="FOCE", interaction=True, maxeval=800)
        .build()
    )

    print("Running FOCE estimation...")
    result = built.fit()
    print(result.summary())

    try:
        import matplotlib
        matplotlib.use("Agg")
        from openpkpd.plots.diagnostics import compute_diagnostics
        from openpkpd.plots.gof import diagnostic_panel
        from openpkpd.plots.eta import eta_histograms

        diag_df = compute_diagnostics(built.population_model, result)
        fig1 = diagnostic_panel(diag_df, title="Warfarin — GOF Diagnostics (FOCE)")

        eta_cols = [c for c in diag_df.columns if c.startswith("ETA")]
        if eta_cols:
            fig2 = eta_histograms(diag_df, result.omega_final,
                                  title="Warfarin — ETA Distributions")

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig1.savefig(os.path.join(out_dir, "02_gof_panel.png"))
            if eta_cols:
                fig2.savefig(os.path.join(out_dir, "02_eta_hist.png"))
            print(f"Figures saved to {out_dir}")
        else:
            print("Figures created (set OPENPKPD_EXAMPLE_OUTPUT to save).")

    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
