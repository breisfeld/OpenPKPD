"""Writers for mixture-model runtime artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from openpkpd.mixture import MixtureResult


def write_mixture_summary(path: str, result: MixtureResult, *, method: str) -> None:
    """Write a compact JSON summary for a mixture run."""
    payload = {
        "n_subpop": int(result.n_subpop),
        "mixture_probs": [float(x) for x in result.mixture_probs],
        "ofv": float(result.ofv),
        "converged": bool(result.converged),
        "estimation_method": method,
        "subpopulations": [
            {
                "index": k + 1,
                "theta": [float(x) for x in sub.theta_final],
                "omega_diag": [float(x) for x in sub.omega_final.diagonal()],
                "sigma_diag": [float(x) for x in sub.sigma_final.diagonal()],
                "ofv": float(sub.ofv),
            }
            for k, sub in enumerate(result.subpop_results)
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


def write_mixture_assignments(path: str, result: MixtureResult) -> None:
    """Write subject-level posterior probabilities and hard assignments."""
    assignments = result.subject_assignments()
    rows: list[dict[str, float | int]] = []
    for sid in sorted(result.subpop_probabilities):
        probs = result.subpop_probabilities[sid]
        row: dict[str, float | int] = {
            "ID": int(sid),
            "assigned_subpop": int(assignments[sid]),
        }
        for k, prob in enumerate(probs, start=1):
            row[f"P_SUBPOP{k}"] = float(prob)
        rows.append(row)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
