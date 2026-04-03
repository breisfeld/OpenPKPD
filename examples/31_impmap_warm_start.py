"""Example 31 — IMPMAP warm-started importance sampling on warfarin PK."""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

from openpkpd.api.model_builder import ModelBuilder


def _build():
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
        .estimation(method="IMPMAP", isample=20, maxeval=1, seed=42)
        .build()
    )


def main() -> None:
    logging.getLogger("openpkpd.estimation.imp").setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        result = _build().fit()
    warm_start = result.diagnostics.get("warm_start", {})

    print("Example 31: IMPMAP warm-start diagnostics on warfarin PK")
    print(f"Method: {result.method}")
    print(f"Short-run converged: {result.converged}")
    print(f"Short-run OFV: {result.ofv:.4f}")
    print("Short-run THETA:", result.theta_final)
    print("Warm start used:", warm_start.get("used"))
    print("Warm start method:", warm_start.get("method"))
    print("Warm start converged:", warm_start.get("converged"))
    print("Warm start OFV:", warm_start.get("ofv"))
    print("Warm start message:", warm_start.get("message"))
    print(f"Recorded OFV evaluations: {len(result.ofv_history or [])}")


if __name__ == "__main__":
    main()
