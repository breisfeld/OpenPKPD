"""Example 29 — Optimal design with PFIM."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openpkpd import ModelBuilder


DATA_FILE = Path("examples/shared_data/theophylline/theophylline.csv")


def main() -> None:
    built = (
        ModelBuilder()
        .problem("Theophylline optimal design demo")
        .data(str(DATA_FILE))
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
        .theta([1.5, 0.08, 30.0])
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .build()
    )

    engine = built.design()
    reference_times = np.array([0.5, 1.0, 2.0, 8.0], dtype=float)
    reference_fim = engine.compute_fim(reference_times, n_subjects=12)
    optimal = engine.optimize_design(
        n_samples=4,
        t_min=0.25,
        t_max=24.0,
        n_subjects=12,
        criterion="D",
        method="L-BFGS-B",
        n_starts=8,
    )
    d_eff = engine.efficiency(
        optimal.sampling_times,
        reference_times,
        criterion="D",
        n_subjects=12,
    )

    print("Example 29: Optimal design with PFIM")
    print(f"Reference times: {reference_times.tolist()}")
    print(f"Reference det(FIM): {np.linalg.det(reference_fim):.6g}")
    print(f"Optimized times: {np.round(optimal.sampling_times, 3).tolist()}")
    print(f"D-efficiency vs reference: {d_eff:.4f}")
    print(f"A-criterion: {optimal.a_efficiency:.6f}")
    print(f"Condition number: {optimal.condition_number:.4f}")
    print("Expected SE:", np.round(optimal.se_theta, 4).tolist())


if __name__ == "__main__":
    main()
