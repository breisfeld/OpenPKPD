"""Example 25 — FOCEI optimizer controls from the Python API."""

from __future__ import annotations

from pathlib import Path

from openpkpd.api.model_builder import ModelBuilder


def main() -> None:
    dataset_path = Path("examples/shared_data/theophylline/theophylline.csv")

    built = (
        ModelBuilder()
        .data(str(dataset_path))
        .subroutines(advan=2, trans=2)
        .pk(
            "\n".join(
                [
                    "KA = THETA(1) * EXP(ETA(1))",
                    "CL = THETA(2) * EXP(ETA(2))",
                    "V  = THETA(3) * EXP(ETA(3))",
                ]
            )
        )
        .error("Y = F * (1 + EPS(1))")
        .theta([(0.01, 1.5, 20.0), (0.001, 0.08, 5.0), (0.1, 30.0, 500.0)])
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .estimation(
            method="FOCEI",
            maxeval=40,
            n_starts=2,
            outer_optimizer="L-BFGS-B",
            outer_fallback_optimizer="Powell",
            outer_fallback_maxeval=15,
            retain_best_iterate=True,
            retry_on_abnormal=True,
            retry_omega_scales=(0.5, 0.25),
        )
        .build()
    )

    result = built.fit()
    print(result.summary())


if __name__ == "__main__":
    main()
