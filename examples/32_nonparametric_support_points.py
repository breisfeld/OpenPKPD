"""Example 32 — Nonparametric support-point estimation on synthetic oral PK."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset


N_SUBJECTS = 12
OBS_TIMES = np.array([0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0])
DOSE = 320.0


def _build_synthetic_dataset() -> NONMEMDataset:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int]] = []

    for sid in range(1, N_SUBJECTS + 1):
        eta_ka = rng.normal(0.0, 0.30)
        eta_cl = rng.normal(0.0, 0.20)
        eta_v = rng.normal(0.0, 0.15)
        ka = 1.5 * math.exp(eta_ka)
        cl = 2.8 * math.exp(eta_cl)
        v = 32.9 * math.exp(eta_v)
        k = cl / v
        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": DOSE,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            }
        )
        for time in OBS_TIMES:
            conc = DOSE * ka / (v * (ka - k)) * (math.exp(-k * time) - math.exp(-ka * time))
            dv = max(conc * (1.0 + rng.normal(0.0, 0.10)), 0.01)
            rows.append(
                {
                    "ID": sid,
                    "TIME": float(time),
                    "AMT": 0.0,
                    "DV": float(dv),
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def main() -> None:
    dataset = _build_synthetic_dataset()

    model = (
        ModelBuilder()
        .problem("Synthetic oral PK — nonparametric support points")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk(
            """
KA = THETA(1)
CL = THETA(2) * EXP(ETA(1))
V  = THETA(3)
"""
        )
        .error("Y = F * (1 + EPS(1))")
        .theta(
            [
                (0.5, 1.5, 8.0),
                (0.5, 3.0, 15.0),
                (10.0, 35.0, 80.0),
            ]
        )
        .omega([0.09])
        .sigma(0.01)
        .estimation(
            method="NONPARAMETRIC",
            base_method="FOCE",
            maxeval=200,
            max_iter=40,
        )
        .build()
    )

    print("=" * 72)
    print("Example 32: Nonparametric support-point estimation")
    print("=" * 72)
    print(f"Synthetic dataset: {N_SUBJECTS} subjects, 1 ETA on CL, seed=42.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
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
    print(f"  mean ETA:     {np.round(result.empirical_mean(), 4)}")
    print(f"  variance ETA: {np.round(result.empirical_variance(), 4)}")


if __name__ == "__main__":
    main()
