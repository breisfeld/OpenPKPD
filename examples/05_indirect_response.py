"""
Example 05: Indirect response model — scipy ODE simulation + FOCE fit.

Demonstrates:
  - Simulating an indirect response model with scipy.integrate.solve_ivp
  - Fitting the simulated data with a direct Emax PD approximation in $ERROR
  - effect_time, hysteresis_loop plots
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.data.event_processor import DoseEvent


# Indirect response parameters
_KIN = 5.0    # Zero-order production rate
_KOUT = 1.0   # First-order elimination rate
_IMAX = 0.8   # Maximum inhibition fraction
_IC50 = 5.0   # IC50 (same units as PK concentration)
_E0_TURNOVER = _KIN / _KOUT  # Baseline = Kin/Kout


def _pkpd_simulate(conc_fn, t_end: float, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate indirect response model:
      dR/dt = Kin * (1 - IMAX * C(t) / (IC50 + C(t))) - Kout * R
    """
    def ode(t, y):
        c = conc_fn(t)
        inh = _IMAX * c / (_IC50 + c)
        dR = _KIN * (1 - inh) - _KOUT * y[0]
        return [dR]

    t_eval = np.arange(0, t_end + dt, dt)
    sol = solve_ivp(ode, [0, t_end], [_E0_TURNOVER], t_eval=t_eval, method="RK45")
    return sol.t, sol.y[0]


def _simulate_data(n_subj: int = 4, seed: int = 42) -> NONMEMDataset:
    """Simulate PK-IDR data for several subjects."""
    rng = np.random.default_rng(seed)
    dose = 100.0
    obs_times = np.array([1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    advan1 = ADVAN1()
    rows = []

    for i in range(1, n_subj + 1):
        k_i = 0.2 * np.exp(rng.normal(0, 0.2))
        v_i = 10.0 * np.exp(rng.normal(0, 0.2))
        pk_params = {"K": k_i, "V": v_i}
        sol_pk = advan1.solve(pk_params, [DoseEvent(0.0, dose, 1)], np.linspace(0, 24, 200))

        # Interpolation function for C(t)
        from scipy.interpolate import interp1d
        c_fn = interp1d(sol_pk.times, sol_pk.ipred, kind="linear",
                        bounds_error=False, fill_value=(0.0, sol_pk.ipred[-1]))

        # IDR simulation
        t_ode, r_ode = _pkpd_simulate(c_fn, 24.0)
        from scipy.interpolate import interp1d as interp1d2
        r_fn = interp1d2(t_ode, r_ode, kind="linear", bounds_error=False,
                         fill_value=(r_ode[0], r_ode[-1]))
        r_obs = r_fn(obs_times) + rng.normal(0, 0.5, len(obs_times))

        # PK observations (concentration, for the hysteresis loop)
        sol_pk_obs = advan1.solve(pk_params, [DoseEvent(0.0, dose, 1)], obs_times)

        rows.append({"ID": i, "TIME": 0.0, "AMT": dose, "DV": 0.0, "EVID": 1, "MDV": 1})
        for j, t in enumerate(obs_times):
            rows.append({
                "ID": i, "TIME": t, "AMT": 0.0,
                "DV": max(float(r_obs[j]), 0.01),
                "EVID": 0, "MDV": 0,
                "CONC": float(sol_pk_obs.ipred[j]),
            })

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def main():
    print("Simulating indirect response data...")
    ds = _simulate_data()
    obs_df = ds.df[ds.df["EVID"] == 0]

    # Fit with a direct Emax PD approximation (simplified IDR as turnover)
    built = (
        ModelBuilder()
        .problem("Indirect response — approximate Emax fit")
        .dataset(ds)
        .subroutines(advan=1, trans=2)
        .pk("""
K = THETA(1)*EXP(ETA(1))
V = THETA(2)*EXP(ETA(2))
""")
        .error("""
E0   = THETA(3)
EMAX = THETA(4)
EC50 = THETA(5)
W    = THETA(6)
IPRED = E0 + EMAX*F / (EC50 + F)
Y    = IPRED + W*EPS(1)
IRES = DV - IPRED
IWRES = IRES / W
""")
        .theta([
            (0.01, 0.2, 5.0),
            (1.0, 10.0, 100.0),
            (0.0, 4.0, 20.0),
            (0.0, 3.0, 30.0),
            (0.1, 5.0, 100.0),
            (0.1, 0.5, 10.0),
        ])
        .omega([0.3, 0.3])
        .sigma(1.0, fixed=True)
        .estimation(method="FO", maxeval=400)
        .build()
    )

    print("Fitting...")
    result = built.fit()
    print(result.summary())

    try:
        import matplotlib
        matplotlib.use("Agg")
        from openpkpd.plots.diagnostics import compute_diagnostics
        from openpkpd.plots.pd import effect_time, hysteresis_loop

        diag_df = compute_diagnostics(built.population_model, result)

        # Merge CONC column from simulated data
        conc_obs = obs_df[["ID", "TIME", "CONC"]].copy()
        diag_df = diag_df.merge(conc_obs, on=["ID", "TIME"], how="left")

        fig1 = effect_time(diag_df, "DV", title="Indirect Response — Effect-Time")

        if "CONC" in diag_df.columns:
            fig2 = hysteresis_loop(diag_df, "CONC", "DV",
                                   title="Indirect Response — Hysteresis Loop")

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig1.savefig(os.path.join(out_dir, "05_effect_time.png"))
            if "CONC" in diag_df.columns:
                fig2.savefig(os.path.join(out_dir, "05_hysteresis.png"))
        else:
            print("Figures created (set OPENPKPD_EXAMPLE_OUTPUT to save).")

    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
