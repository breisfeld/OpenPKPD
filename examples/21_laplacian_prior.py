"""
Example 21: Laplacian Estimation with Prior Augmentation.

Demonstrates:
  - LaplacianMethod (FOCE + log|Hessian| correction)
  - PriorSpec / PriorAugmentedModel for MAP estimation with informative priors
  - Comparing FOCE, Laplacian, and prior-augmented Laplacian on the same data
  - Prior penalty shrinks estimates toward prior mean when data are sparse

Background:
  The Laplacian objective adds a second-order Hessian correction:

      OFV_Laplace_i = OFV_FOCE_i + log|H_i|

  where H_i is the Hessian of the individual objective at η̂_i.
  This gives a better approximation to the marginal likelihood than FOCE
  for non-normal data or large between-subject variability.

  Prior augmentation (equivalent to NONMEM $PRIOR NWPRI) adds a Gaussian
  penalty:

      OFV_total = OFV_data + (θ − θ_prior)ᵀ Σ_θ⁻¹ (θ − θ_prior)

  pushing estimates toward prior means when the data are uninformative.

Dataset: Warfarin 1-cmt oral, 4 subjects (sparse — only 7 obs each).
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.estimation.laplacian import LaplacianMethod
from openpkpd.model.parameters import ParameterSet, ThetaSpec, OmegaSpec, SigmaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.prior import PriorSpec, PriorAugmentedModel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

WARFARIN_DATA = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,70.0,0,1,1,70.0
1,0.5,0,1.42,0,0,70.0
1,2.0,0,4.01,0,0,70.0
1,8.0,0,5.40,0,0,70.0
1,24.0,0,3.56,0,0,70.0
1,72.0,0,1.19,0,0,70.0
1,120.0,0,0.44,0,0,70.0
2,0,65.0,0,1,1,65.0
2,0.5,0,1.30,0,0,65.0
2,2.0,0,3.80,0,0,65.0
2,8.0,0,5.20,0,0,65.0
2,24.0,0,3.40,0,0,65.0
2,72.0,0,1.10,0,0,65.0
2,120.0,0,0.40,0,0,65.0
3,0,80.0,0,1,1,80.0
3,0.5,0,1.55,0,0,80.0
3,2.0,0,4.30,0,0,80.0
3,8.0,0,5.65,0,0,80.0
3,24.0,0,3.75,0,0,80.0
3,72.0,0,1.25,0,0,80.0
3,120.0,0,0.48,0,0,80.0
4,0,75.0,0,1,1,75.0
4,0.5,0,1.48,0,0,75.0
4,2.0,0,4.10,0,0,75.0
4,8.0,0,5.45,0,0,75.0
4,24.0,0,3.60,0,0,75.0
4,72.0,0,1.20,0,0,75.0
4,120.0,0,0.45,0,0,75.0
"""


def _build_dataset() -> NONMEMDataset:
    df = pd.read_csv(io.StringIO(WARFARIN_DATA))
    return NONMEMDataset.from_dataframe(df)


def _build_model(dataset: NONMEMDataset) -> tuple[PopulationModel, ParameterSet]:
    theta_specs = [
        ThetaSpec(init=1.0, lower=0.01, upper=20.0),    # KA  hr⁻¹
        ThetaSpec(init=0.13, lower=0.01, upper=5.0),    # CL  L/hr
        ThetaSpec(init=8.0, lower=1.0, upper=200.0),    # V   L
    ]
    omega_specs = [
        OmegaSpec(block_size=1, values=[0.16]),
        OmegaSpec(block_size=1, values=[0.09]),
        OmegaSpec(block_size=1, values=[0.09]),
    ]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.05])]
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
    print("Example 21: Laplacian Estimation with Prior Augmentation")
    print("=" * 60)

    dataset = _build_dataset()

    # ── 1. FOCE (baseline) ────────────────────────────────────────────────────
    print("\nRunning FOCE (baseline)...")
    pop_model_foce, params_foce = _build_model(dataset)
    foce_result = FOCEMethod(interaction=True, maxeval=600, print_interval=200).estimate(
        pop_model_foce, params_foce
    )
    print(f"  FOCE OFV = {foce_result.ofv:.4f}")

    # ── 2. Laplacian (no prior) ───────────────────────────────────────────────
    print("\nRunning Laplacian (no prior)...")
    pop_model_lap, params_lap = _build_model(dataset)
    lap_result = LaplacianMethod(interaction=True, maxeval=600, print_interval=200).estimate(
        pop_model_lap, params_lap
    )
    print(f"  Laplacian OFV = {lap_result.ofv:.4f}")

    # ── 3. Laplacian + informative prior ─────────────────────────────────────
    # Prior means from warfarin population literature (Holford 1986):
    #   KA ~ 0.9 hr⁻¹, CL ~ 0.13 L/hr, V ~ 8.7 L
    # Moderately informative (30% CV on each parameter)
    prior = PriorSpec(
        theta_prior=np.array([0.9, 0.13, 8.7]),
        theta_prior_cov=np.diag([(0.9 * 0.30) ** 2,
                                  (0.13 * 0.30) ** 2,
                                  (8.7 * 0.30) ** 2]),
    )

    print("\nRunning Laplacian + prior (KA=0.9, CL=0.13, V=8.7; 30% CV)...")
    pop_model_prior, params_prior = _build_model(dataset)
    aug_model = PriorAugmentedModel(population_model=pop_model_prior, prior=prior)
    lap_prior_result = LaplacianMethod(
        interaction=True, maxeval=600, print_interval=200,
    ).estimate(aug_model, params_prior)
    print(f"  Laplacian+Prior OFV = {lap_prior_result.ofv:.4f}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n--- THETA comparison ---")
    labels = ["KA (hr⁻¹)", "CL (L/hr)", "V (L)"]
    prior_vals = [0.9, 0.13, 8.7]
    print(f"{'Parameter':<15} {'Prior':>8} {'FOCE':>10} {'Laplacian':>12} {'Lap+Prior':>12}")
    print("-" * 62)
    for lbl, pv, t_f, t_l, t_lp in zip(
        labels, prior_vals,
        foce_result.theta_final,
        lap_result.theta_final,
        lap_prior_result.theta_final,
    ):
        print(f"{lbl:<15} {pv:>8.4f} {t_f:>10.4f} {t_l:>12.4f} {t_lp:>12.4f}")

    print(f"\nOFV:   FOCE={foce_result.ofv:.2f}  Laplacian={lap_result.ofv:.2f}  "
          f"Laplacian+Prior={lap_prior_result.ofv:.2f}")
    print("\nNote: Laplacian OFV includes log|H_i| correction; "
          "Prior OFV also includes prior penalty term.\n"
          "These OFVs are not directly comparable across methods.")

    # ── Plot: prior shrinkage ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, lbl, pv, t_f, t_lp in zip(
            axes, labels, prior_vals,
            foce_result.theta_final, lap_prior_result.theta_final,
        ):
            ax.bar(["FOCE", "Lap+Prior"], [t_f, t_lp], color=["steelblue", "tomato"])
            ax.axhline(pv, color="black", ls="--", lw=1.5, label="Prior mean")
            ax.set_title(lbl)
            ax.set_ylabel("Estimate")
            ax.legend(fontsize=8)
        plt.suptitle("Prior Shrinkage: FOCE vs Laplacian+Prior", y=1.02)
        plt.tight_layout()

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig.savefig(os.path.join(out_dir, "21_prior_shrinkage.png"), dpi=120)
            print(f"Figure saved to {out_dir}")
        elif os.environ.get("DISPLAY") or os.name == "nt":
            plt.show()
        else:
            print("Figure created (no display available).")
    except ImportError:
        print("matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
