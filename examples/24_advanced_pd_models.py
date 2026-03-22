"""
Example 24: Advanced PD Models.

Demonstrates:
  - EffectCompartmentModel (biophase / Ce compartment with Hill equation)
  - TurnoverModel (production stimulation + degradation inhibition)
  - TumorGrowthInhibitionModel (Simeoni 2004)
  - PlaceboResponseModel (time-course without drug)
  - Fitting each model to simulated data, printing AIC comparison
  - Overlay plot: observed vs model-predicted PD response

Models covered:
  ┌────────────────────────────┬───────────────────────────────────────────────┐
  │ EffectCompartmentModel     │ dCe/dt = Ke0*(C−Ce); E = Emax*Ce^n/(EC50^n+Ce^n) │
  │ TurnoverModel              │ dR/dt = Kin*(1+stim) − Kout*(1+inh)*R          │
  │ TumorGrowthInhibitionModel │ Simeoni 4-compartment damage cascade           │
  │ PlaceboResponseModel       │ E(t) = E0*exp(−kdeg*t) + Epl*(1−exp(−kpl*t))  │
  └────────────────────────────┴───────────────────────────────────────────────┘

Reference: Gabrielsson J & Weiner D (2016). Pharmacokinetic and
  Pharmacodynamic Data Analysis, 5th ed. Swedish Pharmaceutical Press.
"""

from __future__ import annotations

import math
import os

import numpy as np

from openpkpd.models.pkpd import (
    PDData,
    EffectCompartmentModel,
    TurnoverModel,
    TumorGrowthInhibitionModel,
    PlaceboResponseModel,
)


# ---------------------------------------------------------------------------
# Synthetic PD datasets
# ---------------------------------------------------------------------------

def _simulate_effect_compartment(seed: int = 1) -> PDData:
    """Simulate biophase effect data: hysteresis between plasma and effect."""
    rng = np.random.default_rng(seed)
    times  = np.linspace(0, 12, 25)
    # Plasma: 1-cmt IV bolus decline
    c_plasma = 10.0 * np.exp(-0.3 * times)
    # True effect-compartment response
    ke0, emax, ec50, n = 0.8, 90.0, 2.0, 2.0
    # Simulate Ce numerically
    from scipy.integrate import solve_ivp
    def ce_ode(t, y):
        cp = float(np.interp(t, times, c_plasma))
        return [ke0 * (cp - y[0])]
    sol = solve_ivp(ce_ode, [0, 12], [0.0], t_eval=times, dense_output=False)
    ce = np.maximum(sol.y[0], 0.0)
    ce_n = ce**n; ec50_n = ec50**n
    true_effect = emax * ce_n / (ec50_n + ce_n)
    obs = true_effect + rng.normal(0, 3, size=len(times))
    return PDData(subject_id=1, times=times, response=obs, concentrations=c_plasma)


def _simulate_turnover(seed: int = 2) -> PDData:
    """Simulate turnover model response (IDR type 1: stimulation of production)."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 24, 49)
    c = 5.0 * np.exp(-0.2 * times)   # simple 1-cmt PK

    kin, kout, ec50, emax = 2.0, 0.5, 1.5, 1.0
    r0 = kin / kout
    from scipy.integrate import solve_ivp
    def rhs(t, y):
        cp = float(np.interp(t, times, c))
        s_in = emax * cp / (ec50 + cp)
        return [kin * (1 + s_in) - kout * y[0]]
    sol = solve_ivp(rhs, [0, 24], [r0], t_eval=times, dense_output=False)
    true_r = sol.y[0]
    obs = true_r + rng.normal(0, 0.2, size=len(times))
    return PDData(subject_id=1, times=times, response=obs, concentrations=c,
                  baseline=r0)


def _simulate_tgi(seed: int = 3) -> PDData:
    """Simulate Simeoni TGI tumor growth/inhibition data."""
    rng = np.random.default_rng(seed)
    times = np.array([0, 3, 7, 10, 14, 17, 21, 25, 28])
    # Pulsed drug administration: declining concentration from day 0
    c = 2.0 * np.exp(-0.15 * times)

    from openpkpd.models.pkpd import TumorGrowthInhibitionModel
    model = TumorGrowthInhibitionModel()
    true_params = {"lambda0": 0.25, "lambda1": 2.0, "K1": 0.2, "K2": 0.025,
                   "psi": 20.0, "X0": 150.0}
    data = PDData(subject_id=1, times=times, response=np.ones(len(times)),
                  concentrations=c)
    true_vals = model.predict(true_params, data)
    obs = np.maximum(true_vals + rng.normal(0, 5, size=len(times)), 1.0)
    return PDData(subject_id=1, times=times, response=obs, concentrations=c)


def _simulate_placebo(seed: int = 4) -> PDData:
    """Simulate placebo time-course (disease progression + placebo response)."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0, 52, 14)  # weekly for a year
    e0, kdeg, epl, kpl = 60.0, 0.02, 20.0, 0.05
    true_vals = e0 * np.exp(-kdeg * times) + epl * (1 - np.exp(-kpl * times))
    obs = true_vals + rng.normal(0, 2.0, size=len(times))
    return PDData(subject_id=1, times=times, response=obs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Example 24: Advanced PD Models")
    print("=" * 65)

    # ── EffectCompartmentModel ─────────────────────────────────────────────
    print("\n1. Effect Compartment Model (biophase, Hill equation)")
    data_ce = _simulate_effect_compartment()
    model_ce = EffectCompartmentModel()
    res_ce = model_ce.fit(
        data_ce,
        initial_params={"Ke0": 0.5, "Emax": 80.0, "EC50": 2.0, "n": 1.5},
        sigma2=9.0,
    )
    print(f"   Ke0  = {res_ce.params['Ke0']:.3f}  (true: 0.8)")
    print(f"   Emax = {res_ce.params['Emax']:.2f}  (true: 90.0)")
    print(f"   EC50 = {res_ce.params['EC50']:.3f}  (true: 2.0)")
    print(f"   n    = {res_ce.params['n']:.3f}  (true: 2.0)")
    print(f"   OFV  = {res_ce.ofv:.2f}  AIC = {res_ce.aic:.2f}  "
          f"converged = {res_ce.converged}")

    # ── TurnoverModel ─────────────────────────────────────────────────────
    print("\n2. Turnover Model (production stimulation, IDR type 1)")
    data_to = _simulate_turnover()
    model_to = TurnoverModel()
    res_to = model_to.fit(
        data_to,
        initial_params={"Kin": 2.0, "Kout": 0.5, "EC50_in": 1.5, "Emax_in": 0.8},
        sigma2=0.04,
        method="Powell",
    )
    print(f"   Kin     = {res_to.params['Kin']:.3f}  (true: 2.0)")
    print(f"   Kout    = {res_to.params['Kout']:.3f}  (true: 0.5)")
    print(f"   EC50_in = {res_to.params['EC50_in']:.3f}  (true: 1.5)")
    print(f"   Emax_in = {res_to.params['Emax_in']:.3f}  (true: 1.0)")
    print(f"   OFV = {res_to.ofv:.2f}  AIC = {res_to.aic:.2f}  "
          f"converged = {res_to.converged}")

    # ── TumorGrowthInhibitionModel ─────────────────────────────────────────
    print("\n3. Tumor Growth Inhibition (Simeoni 2004)")
    data_tgi = _simulate_tgi()
    model_tgi = TumorGrowthInhibitionModel()
    res_tgi = model_tgi.fit(
        data_tgi,
        initial_params={"lambda0": 0.3, "lambda1": 2.0, "K1": 0.2,
                         "K2": 0.02, "psi": 15.0, "X0": 150.0},
        sigma2=25.0,
    )
    print(f"   lambda0 = {res_tgi.params['lambda0']:.4f}  (true: 0.25)")
    print(f"   lambda1 = {res_tgi.params['lambda1']:.3f}   (true: 2.0)")
    print(f"   K1      = {res_tgi.params['K1']:.3f}   (true: 0.2)")
    print(f"   K2      = {res_tgi.params['K2']:.4f}  (true: 0.025)")
    print(f"   X0      = {res_tgi.params['X0']:.1f}  (true: 150.0)")
    print(f"   OFV = {res_tgi.ofv:.2f}  AIC = {res_tgi.aic:.2f}  "
          f"converged = {res_tgi.converged}")

    # ── PlaceboResponseModel ───────────────────────────────────────────────
    print("\n4. Placebo Response Model (disease progression)")
    data_pl = _simulate_placebo()
    model_pl = PlaceboResponseModel()
    res_pl = model_pl.fit(
        data_pl,
        initial_params={"E0": 55.0, "kdeg": 0.03, "Eplacebo": 15.0, "kpl": 0.04},
        sigma2=4.0,
    )
    print(f"   E0       = {res_pl.params['E0']:.2f}  (true: 60.0)")
    print(f"   kdeg     = {res_pl.params['kdeg']:.4f}  (true: 0.02)")
    print(f"   Eplacebo = {res_pl.params['Eplacebo']:.2f}  (true: 20.0)")
    print(f"   kpl      = {res_pl.params['kpl']:.4f}  (true: 0.05)")
    print(f"   OFV = {res_pl.ofv:.2f}  AIC = {res_pl.aic:.2f}  "
          f"converged = {res_pl.converged}")

    # ── AIC comparison summary ─────────────────────────────────────────────
    print("\n--- Model AIC summary ---")
    for name, res in [("EffectCompartment", res_ce), ("Turnover", res_to),
                       ("TGI", res_tgi), ("Placebo", res_pl)]:
        print(f"  {name:<20} AIC = {res.aic:.2f}  (converged={res.converged})")

    # ── 4-panel plot ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        datasets = [data_ce, data_to, data_tgi, data_pl]
        results  = [res_ce, res_to, res_tgi, res_pl]
        titles   = ["Effect Compartment", "Turnover (IDR I)",
                    "Tumor Growth Inhibition", "Placebo Response"]

        for ax, data, res, title in zip(axes.flat, datasets, results, titles):
            ax.scatter(data.times, data.response, s=25, color="steelblue",
                       label="Observed", zorder=5)
            ax.plot(data.times, res.predicted, "r-", lw=2, label="Predicted")
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Response")
            ax.legend(fontsize=8)

        plt.suptitle("Advanced PD Models — Observed vs Predicted", y=1.01)
        plt.tight_layout()

        out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
        if out_dir:
            fig.savefig(os.path.join(out_dir, "24_advanced_pd.png"), dpi=120)
            print(f"\nFigure saved to {out_dir}")
        elif os.environ.get("DISPLAY") or os.name == "nt":
            plt.show()
        else:
            print("\nFigure created (no display available).")
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
