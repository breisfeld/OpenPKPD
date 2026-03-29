"""Example 31 — IMPMAP warm-started importance sampling on warfarin PK."""

from __future__ import annotations

from pathlib import Path

from openpkpd.api.model_builder import ModelBuilder


def _build(method: str):
    dataset_path = Path("examples/shared_data/warfarin/warfarin.csv")
    return (
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
        .theta([(0.01, 0.9, 20.0), (0.001, 0.13, 5.0), (0.1, 8.7, 200.0)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(method=method, isample=60, maxeval=6, seed=42)
        .build()
    )


def main() -> None:
    imp_result = _build("IMP").fit()
    impmap_result = _build("IMPMAP").fit()

    print("IMP theta:", imp_result.theta_final)
    print("IMP diagnostics:", imp_result.diagnostics)
    print()
    print("IMPMAP theta:", impmap_result.theta_final)
    print("IMPMAP diagnostics:", impmap_result.diagnostics)


if __name__ == "__main__":
    main()
