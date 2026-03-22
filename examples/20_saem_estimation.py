"""
Example 20: SAEM Estimation.

Demonstrates:
  - Stochastic Approximation EM (SAEM) estimation
  - Two-phase algorithm (stochastic exploration + convergence phases)
  - Comparing SAEM vs FOCE parameter estimates on the same dataset
  - OFV convergence history plot
  - SAEMMethod constructor options (n_iter_phase1, n_iter_phase2, seed)

Dataset: Embedded theophylline-like data (12 subjects, simulated from
         known population means KA=1.5, CL=2.8, V=32.9).

Background:
  SAEM is a stochastic EM algorithm that avoids the need to evaluate the
  full likelihood integral.  The E-step samples individual parameters via
  Metropolis-Hastings; the M-step updates population parameters from the
  stochastic sufficient statistics.  Phase 1 uses γ=1 (large step) for
  global exploration; Phase 2 decreases γ_k ∝ k^{-0.7} for convergence.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.estimation.saem import SAEMMethod
from openpkpd.model.parameters import ParameterSet, ThetaSpec, OmegaSpec, SigmaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2


# ---------------------------------------------------------------------------
# Embedded dataset (12 subjects)
# ---------------------------------------------------------------------------

def _build_dataset() -> NONMEMDataset:
    rng = np.random.default_rng(42)
    ka_pop, cl_pop, v_pop = 1.5, 2.8, 32.9
    dose = 320.0
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0])

    rows = []
    for sid in range(1, 13):
        ka = ka_pop * math.exp(rng.normal(0, 0.3))
        cl = cl_pop * math.exp(rng.normal(0, 0.25))
        v  = v_pop  * math.exp(rng.normal(0, 0.15))
        k  = cl / v

        rows.append({"ID": sid, "TIME": 0.0, "AMT": dose, "DV": 0.0,
                     "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0,
                     "ADDL": 0, "II": 0, "SS": 0})
        for t in obs_times:
            if abs(ka - k) < 1e-6:
                c = dose * ka / v * t * math.exp(-k * t)
            else:
                c = dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
            dv = max(c * (1 + rng.normal(0, 0.1)), 0.01)
            rows.append({"ID": sid, "TIME": t, "AMT": 0.0, "DV": dv,
                         "EVID": 0, "MDV": 0, "CMT": 1, "RATE": 0.0,
                         "ADDL": 0, "II": 0, "SS": 0})

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

def _build_model(dataset: NONMEMDataset) -> tuple[PopulationModel, ParameterSet]:
    theta_specs = [
        ThetaSpec(init=1.5, lower=0.3, upper=8.0),    # KA  hr⁻¹
        ThetaSpec(init=3.0, lower=0.5, upper=15.0),   # CL  L/hr
        ThetaSpec(init=35.0, lower=10.0, upper=80.0), # V   L
    ]
    omega_specs = [
        OmegaSpec(block_size=1, values=[0.09]),  # ω²_KA
        OmegaSpec(block_size=1, values=[0.06]),  # ω²_CL
        OmegaSpec(block_size=1, values=[0.04]),  # ω²_V
    ]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.02])]
    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
    pop_model = PopulationModel(
        dataset=dataset, pk_subroutine=ADVAN2(), params=params, trans=2, advan=2,
    )
    return pop_model, params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Example 20: SAEM Estimation")
    print("=" * 60)

    dataset = _build_dataset()
    n_subj = len(dataset.subject_ids())
    print(f"Dataset: {n_subj} subjects, {len(dataset.df)} rows\n")

    # ── SAEM ──────────────────────────────────────────────────────────────────
    print("Running SAEM (K1=150, K2=100)...")
    pop_model_saem, params_saem = _build_model(dataset)
    saem = SAEMMethod(
        n_iter_phase1=150,  # stochastic exploration phase
        n_iter_phase2=100,  # convergence phase
        seed=42,
        print_interval=50,
    )
    saem_result = saem.estimate(pop_model_saem, params_saem)

    print("\n--- SAEM results ---")
    print(saem_result.summary())

    # ── FOCE (reference) ──────────────────────────────────────────────────────
    print("\nRunning FOCE (reference comparison)...")
    pop_model_foce, params_foce = _build_model(dataset)
    foce = FOCEMethod(interaction=True, maxeval=500, print_interval=100)
    foce_result = foce.estimate(pop_model_foce, params_foce)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print("\n--- Parameter comparison ---")
    labels = ["KA (hr⁻¹)", "CL (L/hr)", "V (L)"]
    true_vals = [1.5, 2.8, 32.9]
    print(f"{'Parameter':<15} {'True':>8} {'FOCE':>10} {'SAEM':>10}")
    print("-" * 45)
    for lbl, truth, t_foce, t_saem in zip(
        labels, true_vals, foce_result.theta_final, saem_result.theta_final
    ):
        print(f"{lbl:<15} {truth:>8.3f} {t_foce:>10.3f} {t_saem:>10.3f}")

    print(f"\nOFV  FOCE = {foce_result.ofv:.2f}")
    print(f"OFV  SAEM = {saem_result.ofv:.2f}  (stochastic OFV — not directly comparable)")

    # ── OFV convergence history ───────────────────────────────────────────────
    if hasattr(saem_result, "ofv_history") and saem_result.ofv_history:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(saem_result.ofv_history, lw=1.5, color="steelblue")
            ax.axvline(150, color="red", ls="--", lw=1, label="Phase 2 start")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("OFV (stochastic)")
            ax.set_title("SAEM OFV Convergence History")
            ax.legend()
            plt.tight_layout()

            out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
            if out_dir:
                fig.savefig(os.path.join(out_dir, "20_saem_convergence.png"), dpi=120)
                print(f"\nFigure saved to {out_dir}")
            elif os.environ.get("DISPLAY") or os.name == "nt":
                plt.show()
            else:
                print("\nFigure created (no display available).")
        except ImportError:
            print("matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
