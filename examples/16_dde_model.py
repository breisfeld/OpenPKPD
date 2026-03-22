"""
Example 16: Delay Differential Equation (DDE) PK model.

Demonstrates:
  - DDESubroutine for systems where the elimination rate depends on
    drug concentration at a past time (transit / feedback delay)
  - Simulating a one-compartment DDE model analytically vs numerically
  - Using TAU (delay parameter) in a $DES-compatible callable
  - Comparing DDE output with the standard ODE (TAU = 0) reference

Background
----------
In certain PK/PD models (e.g., target-mediated drug disposition, maturation
transit models) the ODE right-hand side depends on the state at a prior time
τ.  OpenPKPD exposes this through DDESubroutine (ADVAN16).

The history function is injected into pk_params under the key ``"_AHISTORY"``:

    def des(t, A, pk_params, theta, eta):
        hist   = pk_params["_AHISTORY"]       # callable: t_lag -> A_array
        tau    = pk_params["TAU"]
        A_lag  = hist(max(t - tau, 0.0))      # A at time t - tau
        CL, V  = pk_params["CL"], pk_params["V"]
        ke     = CL / V
        dAdt1  = -ke * A_lag[0]              # rate driven by past amount
        return [dAdt1]
"""

from __future__ import annotations

import os

import numpy as np


# ---------------------------------------------------------------------------
# 1. Output path helper
# ---------------------------------------------------------------------------

def _resolve_output_path(filename: str) -> str:
    out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT")
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


# ---------------------------------------------------------------------------
# 2. Define model callables
# ---------------------------------------------------------------------------

def ode_des(t, A, pk_params, theta, eta):
    """Standard one-compartment ODE (no delay): dA/dt = -ke * A(t)."""
    ke = pk_params["CL"] / pk_params["V"]
    return [-ke * A[0]]


def dde_des(t, A, pk_params, theta, eta):
    """
    One-compartment DDE: elimination driven by A(t - tau).

    When tau > 0 this introduces a lag in elimination — drug "cleared"
    at a rate proportional to how much was present tau hours ago.
    """
    hist = pk_params.get("_AHISTORY")
    tau = pk_params.get("TAU", 0.0)
    ke = pk_params["CL"] / pk_params["V"]

    if hist is not None and tau > 0:
        A_lag = hist(max(t - tau, 0.0))
        return [-ke * A_lag[0]]
    # Degenerate to ODE when no history or tau = 0
    return [-ke * A[0]]


# ---------------------------------------------------------------------------
# 3. Solve for a single-dose scenario
# ---------------------------------------------------------------------------

def run_comparison(dose=100.0, cl=2.0, v=10.0, tau=0.5, t_obs=None):
    """Solve ODE and DDE models and return result arrays."""
    from openpkpd.data.event_processor import DoseEvent
    from openpkpd.pk.ode.advan6 import ADVAN6
    from openpkpd.pk.ode.dde import DDESubroutine

    if t_obs is None:
        t_obs = np.linspace(0.1, 12.0, 60)

    dose_events = [DoseEvent(time=0.0, amount=dose, rate=0.0, duration=0.0, compartment=1)]
    pk_ode = {"CL": cl, "V": v}
    pk_dde = {"CL": cl, "V": v, "TAU": tau}

    # ODE solver (ADVAN6)
    ode_solver = ADVAN6(n_compartments=1)
    ode_sol = ode_solver.solve(pk_ode, dose_events, t_obs, des_callable=ode_des)

    # DDE solver (ADVAN16 / DDESubroutine)
    dde_solver = DDESubroutine(n_compartments=1)
    dde_sol = dde_solver.solve(pk_dde, dose_events, t_obs, des_callable=dde_des)

    # Analytical ODE reference: C(t) = Dose / V * exp(-CL/V * t)
    ke = cl / v
    analytical = (dose / v) * np.exp(-ke * t_obs)

    return t_obs, ode_sol.ipred, dde_sol.ipred, analytical


def main():
    t_obs, c_ode, c_dde, c_analytical = run_comparison(
        dose=100.0, cl=2.0, v=10.0, tau=0.5
    )

    print("=" * 60)
    print("Example 16: Delay Differential Equation (DDE) PK model")
    print("=" * 60)
    print(f"{'Time':>8}  {'ODE (no delay)':>16}  {'DDE (tau=0.5h)':>16}  {'Analytical':>12}")
    print("-" * 60)
    for i in range(0, len(t_obs), 6):
        print(
            f"{t_obs[i]:8.2f}  {c_ode[i]:16.4f}  {c_dde[i]:16.4f}  {c_analytical[i]:12.4f}"
        )
    print()

    # Verify ODE matches analytical (should be very close)
    max_err = np.max(np.abs(c_ode - c_analytical))
    print(f"Max ODE vs analytical error: {max_err:.2e}  (should be < 1e-4)")
    assert max_err < 1e-3, f"ODE mismatch: {max_err}"

    # DDE should differ from ODE when tau > 0
    max_diff = np.max(np.abs(c_dde - c_ode))
    print(f"Max DDE vs ODE difference:  {max_diff:.2e}  (should be > 0 with tau=0.5)")
    assert max_diff > 0.01, "DDE should differ from ODE when tau > 0"

    print("\nDDE model with tau=0.5 h produces delayed elimination — as expected.")

    # ------------------------------------------------------------------
    # Optional: plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_obs, c_analytical, "k--", label="Analytical ODE", linewidth=1)
        ax.plot(t_obs, c_ode, label="ODE (TAU=0)", linewidth=1.5)
        ax.plot(t_obs, c_dde, label="DDE (TAU=0.5 h)", linewidth=1.5)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Concentration (mg/L)")
        ax.set_title("DDE vs ODE — one-compartment elimination delay")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = _resolve_output_path("16_dde_model.png")
        fig.savefig(out_path, dpi=120)
        print(f"\nFigure saved to {out_path}")
        plt.close(fig)
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
