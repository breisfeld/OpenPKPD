"""
OpenPKPD — Notebook 07: PK/PD Models

Covers:
  - Model library: one_cmt_oral, two_cmt_iv, emax_direct, etc.
  - Direct effect models: Emax, sigmoid Emax, inhibitory Emax
  - Indirect response models (Types I–IV)
  - Effect compartment (Ce) models
  - Population PD model (mixed-effects)
  - TMDD (target-mediated drug disposition)
  - DDI (drug-drug interaction) models
  - Markov models
  - TTE (time-to-event)
  - Count and categorical models
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — PK/PD Models")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # PK/PD Models

        OpenPKPD includes a library of pre-built models covering the most
        common pharmacokinetic and pharmacodynamic scenarios.

        ## Model Library

        The `openpkpd.library` module provides factory functions that return
        configured `ModelBuilder` instances:
        """
    )
    return


@app.cell
def _():
    from openpkpd.library import list_models, get_model, show_model

    return get_model, list_models, show_model


@app.cell
def _(list_models, mo):
    models = list_models()
    mo.vstack(
        [
            mo.md("### Available Models in the Library"),
            mo.ui.table(
                models
                if hasattr(models, "columns")
                else __import__("pandas").DataFrame({"model": models})
            ),
        ]
    )
    return (models,)


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, np, plt


@app.cell
def _():
    from openpkpd.models.pkpd import (
        EmaxModel,
        EffectCompartmentModel,
        HillModel,
        IndirectResponseModel,
        InhibEmaxModel,
        PDData,
    )

    return EmaxModel, EffectCompartmentModel, HillModel, IndirectResponseModel, InhibEmaxModel, PDData


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Direct Emax Model

        The Emax model describes a maximum drug effect with concentration-dependent
        saturation:

        $$E(C) = E_0 + \frac{E_{\max} \cdot C}{EC_{50} + C}$$

        For **inhibitory** effects:

        $$E(C) = E_0 \cdot \left(1 - \frac{I_{\max} \cdot C}{IC_{50} + C}\right)$$
        """
    )
    return


@app.cell
def _(EmaxModel, HillModel, InhibEmaxModel, PDData, np, plt):
    C = np.linspace(0, 100, 400)
    dummy = PDData(subject_id=1, times=C, response=np.zeros_like(C), concentrations=C)

    emax_model = EmaxModel()
    sig_model = HillModel()
    inh_model = InhibEmaxModel()
    emax_effect = emax_model.predict({"E0": 1.0, "Emax": 10.0, "EC50": 20.0}, dummy)
    sig_effect = sig_model.predict({"E0": 1.0, "Emax": 10.0, "EC50": 20.0, "gamma": 3.0}, dummy)
    inh_effect = inh_model.predict({"E0": 10.0, "Imax": 1.0, "IC50": 15.0, "gamma": 1.5}, dummy)

    fig_emax, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(C, emax_effect, label="Emax (Hill=1)", lw=2)
    axes[0].plot(C, sig_effect, label="Hill model (gamma=3)", lw=2, ls="--")
    axes[0].axhline(11.0, color="grey", ls=":", lw=1)
    axes[0].axvline(20.0, color="red", ls=":", lw=1, label="EC₅₀")
    axes[0].set_xlabel("Concentration")
    axes[0].set_ylabel("Effect")
    axes[0].set_title("Emax and Sigmoid Emax Models")
    axes[0].legend()

    axes[1].plot(C, inh_effect, color="darkorange", lw=2)
    axes[1].axhline(0.0, color="grey", ls=":", lw=1)
    axes[1].axvline(15.0, color="red", ls=":", lw=1, label="IC₅₀")
    axes[1].set_xlabel("Concentration")
    axes[1].set_ylabel("Effect")
    axes[1].set_title("Inhibitory Emax Model (Hill=1.5)")
    axes[1].legend()

    fig_emax.tight_layout()
    fig_emax
    return (
        C,
        EmaxModel,
        HillModel,
        InhibEmaxModel,
        axes,
        emax_model,
        emax_effect,
        fig_emax,
        inh_model,
        inh_effect,
        sig_model,
        sig_effect,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Indirect Response Models

        Indirect response (IDR) models describe drugs that stimulate or inhibit
        the production or degradation of a response variable $R$:

        $$\frac{dR}{dt} = k_{\text{in}} \cdot (1 + S_{\text{in}}(C)) - k_{\text{out}} \cdot R \cdot (1 + I_{\text{out}}(C))$$

        | Type | Mechanism | Control |
        |------|-----------|---------|
        | I    | Inhibit $k_{\text{in}}$ | Production inhibited |
        | II   | Inhibit $k_{\text{out}}$ | Degradation inhibited |
        | III  | Stimulate $k_{\text{in}}$ | Production stimulated |
        | IV   | Stimulate $k_{\text{out}}$ | Degradation stimulated |
        """
    )
    return


@app.cell
def _(IndirectResponseModel, PDData, np, plt):
    t_idr = np.linspace(0, 48, 300)

    idr_i = IndirectResponseModel(idr_type=3)  # inhibit input
    idr_iii = IndirectResponseModel(idr_type=1)  # stimulate input

    # Simulate a declining concentration profile
    C_t = 5.0 * np.exp(-0.3 * t_idr)
    idr_data = PDData(
        subject_id=1,
        times=t_idr,
        response=np.zeros_like(t_idr),
        concentrations=C_t,
        baseline=20.0,
    )

    R_i = idr_i.predict({"Kin": 2.0, "Kout": 0.1, "IC50": 1.0, "Imax": 0.8}, idr_data)
    R_iii = idr_iii.predict({"Kin": 2.0, "Kout": 0.1, "EC50": 1.0, "Emax": 5.0}, idr_data)

    fig_idr, ax_idr = plt.subplots(1, 2, figsize=(12, 4))
    ax_idr[0].plot(t_idr, C_t, color="grey", lw=1.5, ls="--", label="C(t)")
    ax_idr[0].set_xlabel("Time (h)")
    ax_idr[0].set_ylabel("Concentration")
    ax_idr[0].set_title("Input: declining concentration")
    ax_idr[0].legend()

    ax_idr[1].plot(t_idr, R_i, lw=2, label="IDR Type I  (inhib. prod.)")
    ax_idr[1].plot(t_idr, R_iii, lw=2, ls="--", label="IDR Type III (stim. prod.)")
    ax_idr[1].axhline(20.0, color="grey", ls=":", lw=1, label="Baseline")
    ax_idr[1].set_xlabel("Time (h)")
    ax_idr[1].set_ylabel("Response R")
    ax_idr[1].set_title("Indirect Response Model")
    ax_idr[1].legend()

    fig_idr.tight_layout()
    fig_idr
    return (
        C_t,
        IndirectResponseModel,
        R_i,
        R_iii,
        ax_idr,
        fig_idr,
        idr_i,
        idr_iii,
        t_idr,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Effect Compartment (Ce) Model

        The effect compartment model accounts for the temporal delay between
        plasma PK and PD effect by introducing a hypothetical effect site:

        $$\frac{dC_e}{dt} = k_{e0} (C_p - C_e)$$

        The PD effect is then a function of $C_e$ rather than $C_p$.
        """
    )
    return


@app.cell
def _(EffectCompartmentModel, PDData, np, plt):
    t_ce = np.linspace(0, 24, 300)
    Cp = 10 * (np.exp(-0.2 * t_ce) - np.exp(-1.5 * t_ce))

    ce_model = EffectCompartmentModel()
    ce_data = PDData(subject_id=1, times=t_ce, response=np.zeros_like(t_ce), concentrations=Cp)
    E = ce_model.predict({"Ke0": 0.5, "Emax": 8.0, "EC50": 3.0, "n": 1.0}, ce_data)

    # Reconstruct Ce for illustration with the same ODE used by the model.
    from scipy.integrate import solve_ivp

    def ce_ode(t, y):
        cp = float(np.interp(t, t_ce, Cp, left=0.0, right=0.0))
        return [0.5 * (cp - y[0])]

    ce_sol = solve_ivp(ce_ode, [float(t_ce[0]), float(t_ce[-1])], [0.0], t_eval=t_ce)
    Ce = ce_sol.y[0] if ce_sol.success else np.zeros_like(t_ce)

    fig_ce, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(t_ce, Cp, lw=2, label="Cp (plasma)")
    a1.plot(t_ce, Ce, lw=2, ls="--", label="Ce (effect site)")
    a1.set_ylabel("Concentration")
    a1.legend()
    a1.set_title("Plasma vs Effect-Site Concentration")

    a2.plot(t_ce, E, lw=2, color="darkorange")
    a2.set_xlabel("Time (h)")
    a2.set_ylabel("Effect")
    a2.set_title("PD Effect (via Ce)")

    fig_ce.tight_layout()
    fig_ce
    return Ce, Cp, E, EffectCompartmentModel, a1, a2, ce_model, ce_data, fig_ce, t_ce


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Population PD Model (Mixed Effects)

        `PopulationPDModel` wraps a PD model with between-subject variability
        and residual error, analogous to the PK `PopulationModel`.

        ```python
        from openpkpd.models.population_pd import PopulationPDModel
        from openpkpd.models.pkpd import EmaxModel

        pd_model = PopulationPDModel(
            pd_struct=EmaxModel,
            params={
                "EMAX": ThetaSpec(init=10.0, lower=0, upper=100),
                "EC50": ThetaSpec(init=20.0, lower=0, upper=200),
                "E0":   ThetaSpec(init=1.0,  lower=0, upper=20),
            },
            omega=[0.3, 0.2],   # IIV on log(EMAX) and log(EC50)
            sigma=0.1,
        )
        result = pd_model.fit(dataset)
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. TMDD — Target-Mediated Drug Disposition

        TMDD describes drugs that bind tightly to pharmacological targets,
        causing the target to influence the drug's PK:

        ```python
        from openpkpd.models.tmdd import TMDDModel

        model = TMDDModel(approximation="QSS")   # "full" | "QSS" | "wagner"
        # Builds the appropriate ODE system automatically
        ```

        Key parameters: `kin` (target synthesis), `kdeg` (target degradation),
        `kon` (association), `koff` (dissociation), `kint` (complex internalisation).

        ## 6. DDI — Drug-Drug Interaction

        Model the effect of a perpetrator drug on the PK of a victim drug:

        ```python
        from openpkpd.models.ddi import DDIModel

        ddi = DDIModel(
            mechanism="competitive_inhibition",
            ki=2.5,        # inhibition constant (µM)
            victim_cl=5.0, # victim drug clearance
        )
        # Computes AUC ratio, Cmax ratio, and fold-change
        ratio = ddi.auc_ratio(inhibitor_conc=Ci)
        ```

        ## 7. Markov Models

        Markov models describe ordered categorical states (e.g., disease
        progression, adverse event grades) using transition probability matrices:

        ```python
        from openpkpd.models.markov import MarkovModel

        model = MarkovModel(n_states=3)
        model.fit(dataset, state_col="GRADE", time_col="TIME")
        probs = model.predict_proba(times=[0, 6, 12, 24])
        ```

        ## 8. TTE — Time-to-Event Models

        Survival/TTE models are available for endpoints like first occurrence
        of an adverse event or treatment discontinuation:

        ```python
        from openpkpd.models.tte import TTEModel

        tte = TTEModel(hazard="weibull")  # "weibull" | "exponential" | "log-logistic"
        result = tte.fit(dataset, event_col="EVENT", time_col="TIME")
        ```

        Drug effect can be incorporated as a time-varying covariate on the hazard.

        ## 9. Count and Categorical Models

        ```python
        from openpkpd.models.count import PoissonModel, NegativeBinomialModel
        from openpkpd.models.categorical import OrdinalLogisticModel

        count = PoissonModel()
        count.fit(dataset, count_col="N_EVENTS")

        cat = OrdinalLogisticModel(n_categories=5)
        cat.fit(dataset, response_col="GRADE")
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary

        | Model class | Module | Use case |
        |-------------|--------|----------|
        | `EmaxModel` | `models.pkpd` | Direct saturable effect |
        | `HillModel` | `models.pkpd` | Sigmoid concentration-response |
        | `InhibEmaxModel` | `models.pkpd` | Inhibitory effects |
        | `IndirectResponseModel` | `models.pkpd` | Indirect (Kin/Kout) PD |
        | `EffectCompartmentModel` | `models.pkpd` | Hysteresis via Ce |
        | `PopulationPDModel` | `models.population_pd` | Mixed-effects PD |
        | `TMDDModel` | `models.tmdd` | Target-mediated PK |
        | `DDIModel` | `models.ddi` | Drug-drug interaction |
        | `MarkovModel` | `models.markov` | State-transition |
        | `TTEModel` | `models.tte` | Survival / time-to-event |
        | `PoissonModel` | `models.count` | Count outcomes |
        | `OrdinalLogisticModel` | `models.categorical` | Ordered categories |

        **Next:** `08_diagnostics_plots.py` — GOF diagnostics and ETA analysis.
        """
    )
    return


if __name__ == "__main__":
    app.run()
