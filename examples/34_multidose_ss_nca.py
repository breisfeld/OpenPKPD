"""
Example 34: Multi-dose steady-state NCA.

Demonstrates:
  - Generating a steady-state concentration-time profile via SS=1 dosing
  - Running NCA on the last dosing interval at SS
  - Key SS NCA parameters: AUCtau, Ctrough, accumulation ratio (R_ac),
    fluctuation (%Fluct), and degree of fluctuation (%DF)
  - Comparing single-dose vs steady-state PK parameters

Background:
  At steady state (SS), drug accumulates until input equals output over
  each dosing interval τ.  NCA at SS uses observations from a single
  dosing interval (t_last to t_last + τ) and computes:
    AUCtau    = AUC over one dosing interval at SS
    Ctrough   = concentration just before the dose (t = t_last)
    Cpeak_ss  = maximum concentration at SS
    R_ac      = AUCtau_ss / AUC(0-∞)_single   (accumulation ratio)
    %Fluct    = 100 * (Cpeak_ss - Ctrough) / Cavg_ss
"""

from __future__ import annotations

import os

import numpy as np

from openpkpd.data.event_processor import DoseEvent
from openpkpd.nca import NCAEngine
from openpkpd.pk.analytical.advan2 import ADVAN2

# ---------------------------------------------------------------------------
# Model parameters — 1-cmt oral
# ---------------------------------------------------------------------------

PARAMS = {"KA": 1.2, "K": 0.15, "V": 12.0, "F1": 0.9}
DOSE   = 100.0    # mg
TAU    = 12.0     # h  dosing interval
N_DOSES = 8       # enough for near-SS convergence (K*tau ≈ 1.8 → 95% SS after ~3 doses)

# ---------------------------------------------------------------------------
# Single-dose profile
# ---------------------------------------------------------------------------

times_sd = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0])
dose_sd = DoseEvent(time=0.0, amount=DOSE, compartment=1)
sol_sd = ADVAN2().solve(PARAMS, [dose_sd], times_sd)

auc_sd_inf = float(np.trapezoid(sol_sd.ipred, times_sd)) + sol_sd.ipred[-1] / PARAMS["K"]

# ---------------------------------------------------------------------------
# Repeated-dose profile and last-interval steady-state approximation
# ---------------------------------------------------------------------------

times_interval = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
times_all = np.concatenate(
    [dose_index * TAU + times_interval for dose_index in range(N_DOSES)]
)
doses = [
    DoseEvent(time=dose_index * TAU, amount=DOSE, compartment=1)
    for dose_index in range(N_DOSES)
]
sol_md = ADVAN2().solve(PARAMS, doses, times_all)

t_start_last = (N_DOSES - 1) * TAU
last_interval_mask = (times_all >= t_start_last) & (times_all <= t_start_last + TAU)
times_ss = times_all[last_interval_mask] - t_start_last
conc_ss = sol_md.ipred[last_interval_mask]

ss_nca = NCAEngine().compute_multidose_subject(
    times_ss,
    conc_ss,
    dose=DOSE,
    tau=TAU,
    ss=True,
    sd_auc_inf=auc_sd_inf,
)

ctrough = float(conc_ss[0])
cpeak_ss = float(np.max(conc_ss))
cavg_ss = float(ss_nca.c_avg)
auc_tau = float(ss_nca.auc_tau)
r_ac = float(ss_nca.r_ac)
fluct = float(ss_nca.fluctuation)

print("Single-dose NCA")
print(f"  Cmax     = {sol_sd.ipred.max():.3f} mg/L")
print(f"  AUC(0-∞) = {auc_sd_inf:.2f} mg·h/L")
print()
print("Steady-state NCA (SS=1, tau=12 h)")
print(f"  Ctrough  = {ctrough:.3f} mg/L")
print(f"  Cpeak_ss = {cpeak_ss:.3f} mg/L")
print(f"  Cavg_ss  = {cavg_ss:.3f} mg/L")
print(f"  AUCtau   = {auc_tau:.2f} mg·h/L")
print(f"  R_ac     = {r_ac:.3f}  (accumulation ratio)")
print(f"  %%Fluct  = {fluct:.1f}%%")

# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

_out = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT")
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_sd, sol_sd.ipred, "o--", label="Single dose", lw=1.5)
    ax.plot(times_ss, conc_ss, "s-", label=f"Last interval (~SS, tau={TAU}h)", lw=2)
    ax.axhline(ctrough, color="gray", ls=":", lw=1, label=f"Ctrough = {ctrough:.3f}")
    ax.axhline(cavg_ss, color="orange", ls=":", lw=1, label=f"Cavg_ss = {cavg_ss:.3f}")
    ax.set_xlabel("Time within interval (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Multi-dose SS NCA")
    ax.legend()
    plt.tight_layout()
    if _out:
        fig.savefig(os.path.join(_out, "34_multidose_ss_nca.png"), dpi=120)
    else:
        plt.show()
    plt.close(fig)
except ImportError:
    pass
