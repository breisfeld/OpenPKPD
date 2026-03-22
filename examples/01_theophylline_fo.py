"""
Example 01: Theophylline 1-compartment oral model — FO estimation.

Demonstrates:
  - Embedding data directly in the script
  - ModelBuilder fluent API
  - spaghetti_plot, concentration_time from openpkpd.plots
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset

# ---------------------------------------------------------------------------
# Theophylline dataset (first 6 subjects)
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


def main():
    # 1. Load dataset
    df = pd.read_csv(io.StringIO(THEO_DATA))
    ds = NONMEMDataset.from_dataframe(df)

    # 2. Build model
    built = (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral FO")
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
        .estimation(method="FO", maxeval=500)
        .build()
    )

    # 3. Fit
    print("Running FO estimation...")
    result = built.fit()
    print(result.summary())
    print(f"\nKA = {result.theta_final[0]:.4f} hr⁻¹")
    print(f"CL = {result.theta_final[1]:.4f} L/hr")
    print(f"V  = {result.theta_final[2]:.4f} L")

    # 4. Diagnostics
    try:
        import matplotlib
        matplotlib.use("Agg")
        from openpkpd.plots.diagnostics import compute_diagnostics
        from openpkpd.plots.pk import spaghetti_plot, concentration_time

        diag_df = compute_diagnostics(built.population_model, result)

        fig1 = spaghetti_plot(diag_df, title="Theophylline — Spaghetti Plot (FO)")
        fig2 = concentration_time(diag_df, log_y=False,
                                  title="Theophylline — Conc-Time (FO)")

        # Save if output dir specified, else show
        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig1.savefig(os.path.join(out_dir, "01_spaghetti.png"))
            fig2.savefig(os.path.join(out_dir, "01_conc_time.png"))
            print(f"Figures saved to {out_dir}")
        elif os.environ.get("DISPLAY") or os.name == "nt":
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("No display available; figures created but not shown.")

    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
