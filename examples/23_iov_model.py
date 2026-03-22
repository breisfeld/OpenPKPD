"""
Example 23: Inter-Occasion Variability (IOV) Modelling.

Demonstrates:
  - Incorporating IOV via the OCC column in the dataset
  - Custom pk_callable that routes occasion-specific ETAs
  - Block-diagonal OMEGA structure: BSV + IOV contributions
  - Comparing a BSV-only model vs a BSV+IOV model (ΔOFV, LRT)
  - IOV variance estimation

Background:
  IOV models separate random effects into:
    η_BSV  ~ N(0, Ω_BSV)   — between-subject variability (time-invariant)
    κ_occ  ~ N(0, Ω_IOV)   — between-occasion variability (re-drawn each occasion)

  The OCC column marks which dosing occasion each observation belongs to.
  IndividualModel evaluates $PK once per unique occasion, passing
  covariates={"OCC": occ_val} to the pk_callable.  The pk_callable selects
  the appropriate occasion-specific ETA.

  ETA structure (2-occasion design):
    ETA(1) = BSV on CL (all occasions, time-invariant)
    ETA(2) = IOV on CL, occasion 1
    ETA(3) = IOV on CL, occasion 2

  $PK:  CL = THETA(2) * EXP(ETA(1) + ETA(OCC))
           ↑ ETA(2) for occ=1, ETA(3) for occ=2

Dataset: Synthetic 8-subject, 2-occasion oral design.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.fo import FOMethod
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.model.parameters import ParameterSet, ThetaSpec, OmegaSpec, SigmaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2


# ---------------------------------------------------------------------------
# Dataset: 2-occasion oral PK design
# ---------------------------------------------------------------------------

def _build_iov_dataset(n_subjects: int = 8, seed: int = 17) -> NONMEMDataset:
    """
    Simulate an IOV dataset with 2 occasions.

    OCC=1: first dosing occasion
    OCC=2: second dosing occasion (treated as independent in the model)
    """
    rng = np.random.default_rng(seed)
    ka_pop, cl_pop, v_pop = 1.2, 3.0, 30.0
    omega_bsv_ka  = 0.25    # BSV CV on KA
    omega_bsv_cl  = 0.20    # BSV CV on CL
    omega_iov_cl  = 0.15    # IOV CV on CL
    dose = 250.0
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])

    rows = []
    for sid in range(1, n_subjects + 1):
        eta_ka_bsv = rng.normal(0, omega_bsv_ka)
        eta_cl_bsv = rng.normal(0, omega_bsv_cl)
        eta_v      = rng.normal(0, 0.10)

        for occ in [1, 2]:
            kappa_cl = rng.normal(0, omega_iov_cl)   # occasion-specific
            ka = ka_pop * math.exp(eta_ka_bsv)
            cl = cl_pop * math.exp(eta_cl_bsv + kappa_cl)
            v  = v_pop  * math.exp(eta_v)
            k  = cl / v

            rows.append({"ID": sid, "TIME": 0.0, "AMT": dose, "DV": 0.0,
                         "EVID": 1, "MDV": 1, "CMT": 1, "RATE": 0.0,
                         "ADDL": 0, "II": 0, "SS": 0, "OCC": occ})
            for t in obs_times:
                if abs(ka - k) < 1e-6:
                    c = dose * ka / v * t * math.exp(-k * t)
                else:
                    c = dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
                dv = max(c * (1 + rng.normal(0, 0.08)), 0.001)
                rows.append({"ID": sid, "TIME": t, "AMT": 0.0, "DV": dv,
                             "EVID": 0, "MDV": 0, "CMT": 1, "RATE": 0.0,
                             "ADDL": 0, "II": 0, "SS": 0, "OCC": occ})

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _bsv_pk_callable(theta, eta, t=0.0, covariates=None):
    """BSV-only $PK (TRANS2): ETA(1)=BSV on CL only."""
    eta_cl = eta[0] if len(eta) > 0 else 0.0
    ka = theta[0]
    cl = theta[1] * math.exp(eta_cl)
    v  = theta[2]
    return {"KA": ka, "CL": cl, "V": v}


def _iov_pk_callable(theta, eta, t=0.0, covariates=None):
    """
    BSV+IOV $PK (TRANS2):
      ETA(1) = BSV on CL (all occasions)
      ETA(2) = IOV on CL, occasion 1
      ETA(3) = IOV on CL, occasion 2
    """
    occ = int(covariates.get("OCC", 1)) if covariates else 1
    eta_cl_bsv = eta[0] if len(eta) > 0 else 0.0
    # Select IOV ETA: index 1 for occ=1, index 2 for occ=2
    iov_idx = occ   # occ=1 → idx=1, occ=2 → idx=2
    eta_cl_iov = eta[iov_idx] if len(eta) > iov_idx else 0.0
    ka = theta[0]
    cl = theta[1] * math.exp(eta_cl_bsv + eta_cl_iov)
    v  = theta[2]
    return {"KA": ka, "CL": cl, "V": v}


def _build_bsv_model(dataset: NONMEMDataset) -> tuple[PopulationModel, ParameterSet]:
    """BSV-only model: 1 ETA on CL (KA and V fixed)."""
    theta_specs = [
        ThetaSpec(init=1.2, lower=0.2, upper=6.0),
        ThetaSpec(init=3.0, lower=0.5, upper=15.0),
        ThetaSpec(init=30.0, lower=8.0, upper=80.0),
    ]
    omega_specs = [
        OmegaSpec(block_size=1, values=[0.04]),   # ω²_BSV_CL
    ]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
    pop_model = PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
        pk_callable=_bsv_pk_callable,
    )
    return pop_model, params


def _build_iov_model(dataset: NONMEMDataset) -> tuple[PopulationModel, ParameterSet]:
    """BSV+IOV model: 3 ETAs (BSV_CL, IOV_CL_occ1, IOV_CL_occ2)."""
    theta_specs = [
        ThetaSpec(init=1.2, lower=0.2, upper=6.0),
        ThetaSpec(init=3.0, lower=0.5, upper=15.0),
        ThetaSpec(init=30.0, lower=8.0, upper=80.0),
    ]
    omega_specs = [
        OmegaSpec(block_size=1, values=[0.04]),   # ω²_BSV_CL
        OmegaSpec(block_size=1, values=[0.02]),   # ω²_IOV_CL occ=1
        OmegaSpec(block_size=1, values=[0.02]),   # ω²_IOV_CL occ=2
    ]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    params = ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)
    pop_model = PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
        pk_callable=_iov_pk_callable,
    )
    return pop_model, params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("Example 23: Inter-Occasion Variability (IOV) Modelling")
    print("=" * 62)

    dataset = _build_iov_dataset()
    n_subj = len(dataset.subject_ids())
    print(f"\nDataset: {n_subj} subjects, 2 occasions, {len(dataset.df)} rows")
    print("OCC column present:", "OCC" in dataset.df.columns)

    # ── BSV-only model ────────────────────────────────────────────────────────
    print("\nFitting BSV-only model (1 ETA on CL, FO method)...")
    pop_bsv, params_bsv = _build_bsv_model(dataset)
    bsv_result = FOMethod(maxeval=400, print_interval=150).estimate(pop_bsv, params_bsv)
    print(f"  OFV = {bsv_result.ofv:.4f}  converged={bsv_result.converged}")

    # ── BSV + IOV model ───────────────────────────────────────────────────────
    print("\nFitting BSV+IOV model (BSV on CL + per-occasion IOV, FO method)...")
    pop_iov, params_iov = _build_iov_model(dataset)
    iov_result = FOMethod(maxeval=400, print_interval=150).estimate(pop_iov, params_iov)
    print(f"  OFV = {iov_result.ofv:.4f}  converged={iov_result.converged}")

    # ── Model comparison (LRT) ────────────────────────────────────────────────
    import scipy.stats as stats
    delta_ofv = bsv_result.ofv - iov_result.ofv
    delta_df  = 2  # 2 extra IOV variance parameters (occ1, occ2)
    p_value   = stats.chi2.sf(delta_ofv, df=delta_df) if delta_ofv > 0 else 1.0

    print("\n--- Likelihood ratio test ---")
    print(f"ΔOFV = {delta_ofv:.4f}  (df={delta_df})  p={p_value:.4f}")
    if p_value < 0.05:
        print("-> IOV model significantly better (p < 0.05)")
    else:
        print("-> BSV-only model not significantly improved by adding IOV")

    # ── Parameter summary ─────────────────────────────────────────────────────
    print("\n--- THETA estimates ---")
    labels = ["KA fixed (hr⁻¹)", "CL (L/hr)", "V fixed (L)"]
    true_theta = [1.2, 3.0, 30.0]
    print(f"{'Parameter':<15} {'True':>8} {'BSV-only':>12} {'BSV+IOV':>12}")
    print("-" * 50)
    for lbl, tv, t_b, t_i in zip(labels, true_theta,
                                   bsv_result.theta_final, iov_result.theta_final):
        print(f"{lbl:<15} {tv:>8.3f} {t_b:>12.4f} {t_i:>12.4f}")

    print("\n--- OMEGA diagonal ---")
    print(f"  BSV-only:  ω²_BSV_CL = {np.diag(bsv_result.omega_final)[0]:.4f}")
    iov_diag = np.diag(iov_result.omega_final)
    print(f"  BSV+IOV:   ω²_BSV_CL = {iov_diag[0]:.4f}  "
          f"ω²_IOV_occ1 = {iov_diag[1]:.4f}  "
          f"ω²_IOV_occ2 = {iov_diag[2]:.4f}")
    print(f"\n  True ω²_BSV_CL ≈ {0.20**2:.4f}  True ω²_IOV_CL ≈ {0.15**2:.4f}")

    # ── Plot: per-occasion ETA scatter ────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.suptitle("IOV Model — ETA Distributions", y=1.02)

        # BSV ETAs from BSV-only model (single ETA on CL)
        bsv_etas = np.array([v[0] for v in bsv_result.post_hoc_etas.values()])
        axes[0].hist(bsv_etas, bins=6, color="steelblue", edgecolor="white")
        axes[0].axvline(0, color="black", lw=1.5, ls="--")
        axes[0].set_xlabel("η_CL (BSV)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("BSV-only: η_CL distribution")

        # IOV ETAs from BSV+IOV model
        iov_etas = np.array([v for v in iov_result.post_hoc_etas.values()])
        if iov_etas.shape[1] >= 3:
            axes[1].scatter(iov_etas[:, 1], iov_etas[:, 2],
                            color="tomato", s=40, zorder=5)
            lim = max(abs(iov_etas[:, 1:3]).max() * 1.3, 0.05)
            axes[1].set_xlim(-lim, lim)
            axes[1].set_ylim(-lim, lim)
            axes[1].axhline(0, color="gray", lw=0.8)
            axes[1].axvline(0, color="gray", lw=0.8)
            axes[1].set_xlabel("κ_CL (occasion 1)")
            axes[1].set_ylabel("κ_CL (occasion 2)")
            axes[1].set_title("IOV: κ_CL occasion 1 vs 2")

        plt.tight_layout()

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig.savefig(os.path.join(out_dir, "23_iov_etas.png"), dpi=120)
            print(f"\nFigure saved to {out_dir}")
        elif os.environ.get("DISPLAY") or os.name == "nt":
            plt.show()
        else:
            print("\nFigure created (no display available).")
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
