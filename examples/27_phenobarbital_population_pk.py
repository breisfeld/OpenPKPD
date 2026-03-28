"""Example 27 — Phenobarbital neonatal population PK."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset


DATA_FILE = Path("examples/shared_data/phenobarbital/phenobarbital_simulated.csv")


def main() -> None:
    ds = NONMEMDataset.from_csv(str(DATA_FILE))

    model = (
        ModelBuilder()
        .problem("Phenobarbital neonatal population PK")
        .dataset(ds)
        .covariates(["WT"])
        .subroutines(advan=1, trans=1)
        .pk(
            """
TVCL = THETA(1) * WT
TVV  = THETA(2) * WT
CL   = TVCL * EXP(ETA(1))
V    = TVV  * EXP(ETA(2))
K    = CL / V
S1   = V
"""
        )
        .error(
            """
IPRED = F
W     = IPRED * THETA(3)
Y     = IPRED + W * EPS(1)
"""
        )
        .theta(
            [
                (0.001, 0.0047, 0.05),
                (0.1, 0.96, 5.0),
                (0.001, 0.1, 1.0),
            ]
        )
        .omega([[0.0361, 0.0], [0.0, 0.0256]])
        .sigma([[1.0]])
        .estimation(method="FO", maxeval=500)
        .build()
    )

    print("Running phenobarbital FO estimation...")
    result = model.fit()
    print(result.summary())

    cl_per_kg = float(result.theta_final[0])
    v_per_kg = float(result.theta_final[1])
    half_life_h = v_per_kg * np.log(2.0) / cl_per_kg
    print(f"CL/kg = {cl_per_kg:.5f} L/h/kg")
    print(f"V/kg  = {v_per_kg:.4f} L/kg")
    print(f"t1/2  = {half_life_h:.1f} h")


if __name__ == "__main__":
    main()
