"""Example 32 — Nonparametric support-point estimation on phenobarbital PK."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset


DATA_FILE = Path("examples/shared_data/phenobarbital/phenobarbital_simulated.csv")


def main() -> None:
    dataset = NONMEMDataset.from_csv(str(DATA_FILE))

    model = (
        ModelBuilder()
        .problem("Phenobarbital neonatal population PK — nonparametric")
        .dataset(dataset)
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
        .estimation(method="NONPARAMETRIC", base_method="FOCEI", maxeval=300, max_iter=80)
        .build()
    )

    print("=" * 72)
    print("Example 32: Nonparametric support-point estimation on phenobarbital")
    print("=" * 72)
    result = model.fit()
    print(result.summary())

    support_weights = np.asarray(result.support_weights)
    top = np.argsort(support_weights)[-5:][::-1]

    print("\nTop support points")
    for idx in top:
        eta = result.support_points[idx]
        print(
            f"  rank={np.where(top == idx)[0][0] + 1} "
            f"weight={support_weights[idx]:.4f} eta={eta}"
        )

    print("\nEmpirical support distribution")
    print(f"  mean ETA:     {result.empirical_mean()}")
    print(f"  variance ETA: {result.empirical_variance()}")


if __name__ == "__main__":
    main()
