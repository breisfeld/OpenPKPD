"""
Example 33: Target-Mediated Drug Disposition (TMDD).

Demonstrates:
  - Full TMDD model (Mager & Jusko 2001): drug–target binding, internalisation
  - QSSA approximation (Gibiansky et al. 2008): valid when binding kinetics fast
  - Michaelis-Menten approximation: further simplification for target saturation
  - Comparing the three models on the same dosing scenario
  - Plotting drug (C), target (R), and complex (RC) concentration-time profiles

Background:
  TMDD arises when target binding significantly influences drug disposition.
  The full model tracks free drug (C), free target (R), drug-target complex (RC):
      dC/dt  = -kel*C - kon*C*R + koff*RC + kin_dose/Vc
      dR/dt  = ksyn - kdeg*R - kon*C*R + koff*RC + kint*RC
      dRC/dt =  kon*C*R - (koff + kint)*RC

  QSSA collapses the complex: RC ≈ C*R / (Km + C), Km = (koff + kint)/kon.
  Michaelis-Menten further approximates non-linear elimination.

Reference:
  Mager DE, Jusko WJ (2001). J Pharmacokinet Pharmacodyn 28(6):507-32.
  Gibiansky L et al. (2008). J Pharmacokinet Pharmacodyn 35(5):573-91.
"""

from __future__ import annotations

import os

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.models.tmdd import FullTMDD, MichaelisMentenTMDD, QSSATMDDModel

# ---------------------------------------------------------------------------
# TMDD parameters (hypothetical monoclonal antibody)
# ---------------------------------------------------------------------------

PK_PARAMS_FULL = {
    "KON":   0.091,   # L/(nmol·h)   association rate
    "KOFF":  0.001,   # 1/h          dissociation rate
    "KINT":  0.1,     # 1/h          internalisation rate
    "KSYN":  0.11,    # nmol/(L·h)   target synthesis rate (ksyn = kdeg * R0)
    "KDEG":  0.01,    # 1/h          target degradation rate
    "KEL":   0.02,    # 1/h          linear drug elimination
    "VC":    3.0,     # L            central volume
}

# QSSA uses the same params — KM derived internally as (KOFF + KINT) / KON
PK_PARAMS_QSSA = {**PK_PARAMS_FULL}

# MM params: Vmax = KINT * R0 * VC, Km = (KOFF + KINT) / KON
R0 = PK_PARAMS_FULL["KSYN"] / PK_PARAMS_FULL["KDEG"]
KM = (PK_PARAMS_FULL["KOFF"] + PK_PARAMS_FULL["KINT"]) / PK_PARAMS_FULL["KON"]
VMAX = PK_PARAMS_FULL["KINT"] * R0
PK_PARAMS_MM = {
    "KEL":  PK_PARAMS_FULL["KEL"],
    "VC":   PK_PARAMS_FULL["VC"],
    "VMAX": VMAX,
    "KM":   KM,
}

DOSE_AMOUNT = 10.0  # nmol
TIMES = np.concatenate([np.linspace(0.01, 2, 20), np.linspace(2, 168, 80)])

# ---------------------------------------------------------------------------
# Solve all three models
# ---------------------------------------------------------------------------

dose = DoseEvent(time=0.0, amount=DOSE_AMOUNT, compartment=1)

full_model = FullTMDD()
sol_full = full_model.solve(PK_PARAMS_FULL, [dose], TIMES)

qssa_model = QSSATMDDModel()
sol_qssa = qssa_model.solve(PK_PARAMS_QSSA, [dose], TIMES)

mm_model = MichaelisMentenTMDD()
sol_mm = mm_model.solve(PK_PARAMS_MM, [dose], TIMES)

print("Full TMDD  peak C:", f"{sol_full.ipred.max():.3f} nmol/L")
print("QSSA       peak C:", f"{sol_qssa.ipred.max():.3f} nmol/L")
print("MM         peak C:", f"{sol_mm.ipred.max():.3f} nmol/L")
print()
print("Full TMDD  AUC(0-168h) ≈", f"{np.trapz(sol_full.ipred, TIMES):.1f} nmol·h/L")
print("QSSA       AUC(0-168h) ≈", f"{np.trapz(sol_qssa.ipred, TIMES):.1f} nmol·h/L")
print("MM         AUC(0-168h) ≈", f"{np.trapz(sol_mm.ipred, TIMES):.1f} nmol·h/L")

# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

_out = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT")
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(TIMES, sol_full.ipred, label="Full TMDD", lw=2)
    ax.semilogy(TIMES, sol_qssa.ipred, label="QSSA", lw=2, ls="--")
    ax.semilogy(TIMES, sol_mm.ipred, label="Michaelis-Menten", lw=2, ls=":")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Free drug C (nmol/L)")
    ax.set_title("TMDD: drug concentration")
    ax.legend()

    if sol_full.amounts.shape[1] >= 2:
        ax2 = axes[1]
        r_full = sol_full.amounts[:, 1]  # free target (cmt 2)
        ax2.plot(TIMES, r_full, label="Free target R", lw=2)
        if sol_full.amounts.shape[1] >= 3:
            rc_full = sol_full.amounts[:, 2]  # complex (cmt 3)
            ax2.plot(TIMES, rc_full, label="Complex RC", lw=2, ls="--")
        ax2.set_xlabel("Time (h)")
        ax2.set_ylabel("Concentration (nmol/L)")
        ax2.set_title("TMDD: target and complex")
        ax2.legend()

    plt.tight_layout()
    if _out:
        fig.savefig(os.path.join(_out, "33_tmdd_model.png"), dpi=120)
    else:
        plt.show()
    plt.close(fig)
except ImportError:
    pass
