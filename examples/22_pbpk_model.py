"""
Example 22: Physiologically-Based Pharmacokinetic (PBPK) Modelling.

Demonstrates:
  - FiveOrganPBPK — 5-organ (lung, liver, kidney, gut, central) template
  - Defining organ blood flows, volumes, and metabolic clearances
  - Forward simulation: solving the PBPK ODE system for a single dose
  - Comparing PBPK-predicted plasma vs peripheral-tissue concentrations
  - PBPKModel.compartment_index() for named-organ access

Organs and equations:
  dA_lung/dt   = Q_lung  * (C_central − C_lung/Kp_lung)
  dA_liver/dt  = Q_liver * (C_central − C_liver/Kp_liver) − CL_liver*C_liver
  dA_kidney/dt = Q_kidney*(C_central − C_kidney/Kp_kidney) − CL_kidney*C_kidney
  dA_gut/dt    = Q_gut   * (C_central − C_gut/Kp_gut)
  dA_central/dt = -(Q_total)*C_central
                  + Q_lung*C_lung/Kp_lung
                  + Q_liver*C_liver/Kp_liver
                  + Q_kidney*C_kidney/Kp_kidney
                  + Q_gut*C_gut/Kp_gut

where C_organ = A_organ / V_organ  and  C_central = A_central / V_central.

Reference: Rowland M et al. (2011) Physiologically-based pharmacokinetics in
  drug development and regulatory science. Annu Rev Pharmacol Toxicol 51:45-73.
"""

from __future__ import annotations

import os

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.pbpk import FiveOrganPBPK


# ---------------------------------------------------------------------------
# PBPK parameters (generic small-molecule, human physiology)
# ---------------------------------------------------------------------------

PK_PARAMS = {
    # Organ blood flows (L/h)
    "Q_lung":   350.0,
    "Q_liver":  90.0,
    "Q_kidney": 72.0,
    "Q_gut":    60.0,
    # Organ volumes (L)
    "V_lung":   0.5,
    "V_liver":  1.8,
    "V_kidney": 0.3,
    "V_gut":    1.0,
    "V_central": 5.0,   # blood/plasma volume
    # Tissue:plasma partition coefficients
    "Kp_lung":   2.5,
    "Kp_liver":  8.0,
    "Kp_kidney": 4.0,
    "Kp_gut":    3.0,
    # Metabolic / excretory clearances (L/h)
    "CL_liver":  15.0,
    "CL_kidney":  8.0,
}


def des_callable(t, a, pk_params, theta=None, eta=None):
    """
    5-organ PBPK right-hand side.

    a[0] = lung, a[1] = liver, a[2] = kidney, a[3] = gut, a[4] = central
    """
    q_lung   = pk_params.get("Q_lung", 350.0)
    q_liver  = pk_params.get("Q_liver", 90.0)
    q_kidney = pk_params.get("Q_kidney", 72.0)
    q_gut    = pk_params.get("Q_gut", 60.0)

    v_lung   = max(pk_params.get("V_lung", 0.5), 1e-12)
    v_liver  = max(pk_params.get("V_liver", 1.8), 1e-12)
    v_kidney = max(pk_params.get("V_kidney", 0.3), 1e-12)
    v_gut    = max(pk_params.get("V_gut", 1.0), 1e-12)
    v_central= max(pk_params.get("V_central", 5.0), 1e-12)

    kp_lung   = pk_params.get("Kp_lung", 2.5)
    kp_liver  = pk_params.get("Kp_liver", 8.0)
    kp_kidney = pk_params.get("Kp_kidney", 4.0)
    kp_gut    = pk_params.get("Kp_gut", 3.0)

    cl_liver  = pk_params.get("CL_liver", 15.0)
    cl_kidney = pk_params.get("CL_kidney", 8.0)

    # Organ concentrations
    c_lung   = a[0] / v_lung
    c_liver  = a[1] / v_liver
    c_kidney = a[2] / v_kidney
    c_gut    = a[3] / v_gut
    c_central= a[4] / v_central

    # Flows back to central (venous return)
    ret_lung   = q_lung   * c_lung   / kp_lung
    ret_liver  = q_liver  * c_liver  / kp_liver
    ret_kidney = q_kidney * c_kidney / kp_kidney
    ret_gut    = q_gut    * c_gut    / kp_gut

    q_total = q_lung + q_liver + q_kidney + q_gut

    dadt = [
        q_lung   * (c_central - c_lung   / kp_lung),                    # lung
        q_liver  * (c_central - c_liver  / kp_liver)  - cl_liver*c_liver,  # liver
        q_kidney * (c_central - c_kidney / kp_kidney) - cl_kidney*c_kidney, # kidney
        q_gut    * (c_central - c_gut    / kp_gut),                      # gut
        -q_total * c_central + ret_lung + ret_liver + ret_kidney + ret_gut,  # central
    ]
    return dadt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Example 22: PBPK Model — 5-Organ Human Template")
    print("=" * 60)

    model = FiveOrganPBPK()
    print(f"\nCompartments: {model.compartment_names}")
    print(f"Output compartment: '{model.output_compartment_name}' "
          f"(index {model.compartment_index('central')})")

    # IV bolus: 100 mg administered into central (blood) compartment
    dose_event = DoseEvent(
        time=0.0,
        amount=100.0,   # mg
        compartment=5,  # central (1-indexed)
        rate=0.0,
    )

    # Observation times (0.25 h to 24 h)
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0])

    print(f"\nDose: {dose_event.amount:.0f} mg IV bolus at t=0 into blood/central")
    print(f"Observation times: {obs_times} h")
    print("\nSolving PBPK ODE system...")

    sol = model.solve(
        pk_params=PK_PARAMS,
        dose_events=[dose_event],
        obs_times=obs_times,
        des_callable=des_callable,
    )

    # Extract plasma and tissue concentrations
    c_central = sol.ipred  # central (plasma/blood) = output compartment
    print("\n--- Plasma (central) concentration profile ---")
    print(f"{'Time (h)':>10} {'C_plasma (mg/L)':>18}")
    print("-" * 30)
    for t, c in zip(obs_times, c_central):
        print(f"{t:>10.2f} {c:>18.4f}")

    # Named-compartment extraction from amounts matrix
    if sol.amounts is not None:
        idx = {name: i for i, name in enumerate(model.compartment_names)}

        v_liver = PK_PARAMS["V_liver"]
        v_kidney = PK_PARAMS["V_kidney"]
        c_liver  = sol.amounts[:, idx["liver"]]  / v_liver
        c_kidney = sol.amounts[:, idx["kidney"]] / v_kidney

        print("\n--- Tissue concentration comparison at key times ---")
        print(f"{'Time (h)':>10} {'Plasma':>12} {'Liver':>12} {'Kidney':>12}")
        print("-" * 50)
        for i, t in enumerate(obs_times):
            print(f"{t:>10.2f} {c_central[i]:>12.4f} "
                  f"{c_liver[i]:>12.4f} {c_kidney[i]:>12.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(obs_times, c_central, "o-", label="Plasma (central)", lw=2)
        if sol.amounts is not None:
            ax.semilogy(obs_times, c_liver, "s--", label="Liver", lw=1.5)
            ax.semilogy(obs_times, c_kidney, "^--", label="Kidney", lw=1.5)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title("5-Organ PBPK — Plasma and Tissue Concentrations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig.savefig(os.path.join(out_dir, "22_pbpk.png"), dpi=120)
            print(f"\nFigure saved to {out_dir}")
        elif os.environ.get("DISPLAY") or os.name == "nt":
            plt.show()
        else:
            print("\nFigure created (no display available).")
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
