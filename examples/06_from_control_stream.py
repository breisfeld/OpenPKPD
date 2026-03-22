"""
Example 06: Run a model from a NONMEM control stream string.

Demonstrates:
  - ControlStream.from_string()
  - cli.runner.run_from_control_stream() (or manual assembly)
  - result.summary()
"""

from __future__ import annotations

import io
import os
import tempfile

import pandas as pd

from openpkpd.parser.control_stream import ControlStream
from openpkpd.data.dataset import NONMEMDataset

# Minimal theophylline control stream for dose-normalized data
CTL_TEXT = """\
$PROBLEM Theophylline via control stream
$DATA theophylline.csv IGNORE=#
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  KA = THETA(1)*EXP(ETA(1))
  CL = THETA(2)*EXP(ETA(2))
  V  = THETA(3)*EXP(ETA(3))
$ERROR
  IPRED = F
  W = THETA(4) * IPRED
  Y = IPRED + W * EPS(1)
$THETA (0.01, 1.5, 20)  ; KA
$THETA (0, 0.04, 2)     ; CL/WT
$THETA (0, 0.50, 5)     ; V/WT
$THETA (0.01, 0.10, 0.50) ; proportional residual SD
$OMEGA 0.48 0.07 0.02
$SIGMA 1 FIXED
$ESTIMATION METHOD=ZERO MAXEVAL=500
"""

THEO_DATA = """\
ID,TIME,AMT,DV,EVID,MDV
1,0,4.02,0,1,1
1,0.27,0,0.74,0,0
1,0.57,0,1.72,0,0
1,1.02,0,7.91,0,0
1,1.92,0,8.31,0,0
1,3.5,0,8.33,0,0
1,5.02,0,6.85,0,0
1,7.03,0,6.08,0,0
1,9.0,0,5.4,0,0
1,12.05,0,4.55,0,0
1,24.37,0,1.25,0,0
2,0,4.4,0,1,1
2,0.35,0,0.96,0,0
2,0.6,0,2.33,0,0
2,1.07,0,4.71,0,0
2,2.13,0,8.33,0,0
2,3.5,0,9.02,0,0
2,5.02,0,7.14,0,0
2,7.02,0,5.68,0,0
2,9.1,0,4.55,0,0
2,12.1,0,3.01,0,0
2,25.0,0,0.9,0,0
3,0,4.95,0,1,1
3,0.27,0,0.64,0,0
3,0.58,0,1.92,0,0
3,1.02,0,4.44,0,0
3,1.92,0,7.03,0,0
3,3.5,0,9.07,0,0
3,5.02,0,7.56,0,0
3,7.02,0,6.59,0,0
3,9.0,0,5.88,0,0
3,12.15,0,4.73,0,0
3,24.17,0,1.25,0,0
"""


def main():
    # Parse control stream
    cs = ControlStream.from_string(CTL_TEXT)
    print("Parsed control stream:")
    print(f"  Problem: {cs.problem.title if cs.problem else 'N/A'}")
    print(f"  ADVAN: {cs.subroutines.advan if cs.subroutines else 'N/A'}")
    print(f"  n_theta: {sum(len(tr.specs) for tr in cs.theta_records)}")

    # Load dataset
    df = pd.read_csv(io.StringIO(THEO_DATA))
    ds = NONMEMDataset.from_dataframe(df)

    # Assemble model using ModelBuilder (same parameters as control stream)
    from openpkpd import ModelBuilder
    built = (
        ModelBuilder()
        .problem("Theophylline from .ctl")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
        .error("IPRED = F\nW = THETA(4) * IPRED\nY = IPRED + W * EPS(1)")
        .theta([(0.01, 1.5, 20), (0, 0.04, 2), (0, 0.50, 5), (0.01, 0.10, 0.50)])
        .omega([0.48, 0.07, 0.02])
        .sigma(1.0, fixed=True)
        .estimation(method="FO", maxeval=500)
        .build()
    )

    print("\nRunning estimation (equivalent to $ESTIMATION METHOD=ZERO)...")
    result = built.fit()
    print(result.summary())


if __name__ == "__main__":
    main()
