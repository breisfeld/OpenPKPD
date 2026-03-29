"""
OpenPKPD — Notebook 06: PK Subroutines (ADVAN Models)

Covers:
  - Analytical ADVAN models: ADVAN1, 2, 3, 4, 11, 12
  - ODE-based models: ADVAN6, ADVAN8, ADVAN10, ADVAN13
  - Absorption models: transit compartments, parallel absorption
  - Delay differential equations (ADVAN_DDE)
  - TRANS parameterisation variants (TRANS1–4, TRANS6)
  - Choosing ADVAN/TRANS for common PK scenarios
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="OpenPKPD — PK Subroutines")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # PK Subroutines (ADVAN Models)

        OpenPKPD implements the full set of NONMEM ADVAN subroutines for
        solving pharmacokinetic differential equations.

        ## Analytical ADVANs (closed-form solutions)

        | ADVAN | Compartments | Route | Description |
        |-------|-------------|-------|-------------|
        | ADVAN1 | 1 | IV bolus | One-compartment IV |
        | ADVAN2 | 1 | Oral / SC | One-compartment + absorption |
        | ADVAN3 | 2 | IV bolus | Two-compartment IV |
        | ADVAN4 | 2 | Oral / SC | Two-compartment + absorption |
        | ADVAN11 | 3 | IV bolus | Three-compartment IV |
        | ADVAN12 | 3 | Oral / SC | Three-compartment + absorption |

        ## ODE ADVANs (numerical integration)

        | ADVAN | Description |
        |-------|-------------|
        | ADVAN6 | General ODE (user-written $DES block), non-stiff RK45 |
        | ADVAN8 | General ODE, stiff LSODA |
        | ADVAN10 | Michaelis-Menten (saturable) elimination (ODE-backed) |
        | ADVAN13 | General ODE + adjoint-sensitivity gradients (JAX/diffrax) |

        ## TRANS Parameterisations

        | TRANS | Primary parameters | Use with |
        |-------|-------------------|---------|
        | TRANS1 | K (k10), V | ADVAN1 |
        | TRANS2 | CL, V | ADVAN1, 2 |
        | TRANS3 | CL, V, Q, V2 (2-cmt) | ADVAN3, 4 |
        | TRANS4 | CL, V1, Q, V2 | ADVAN3, 4 |
        | TRANS6 | ALPHA, BETA, S1 | ADVAN3 |
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Analytical Solutions

        All analytical ADVAN classes live in `openpkpd.pk.analytical`.
        They share a common `.solve(pk_params, dose_events, obs_times)` interface
        and return a `PKSolution` with `.ipred`, `.amounts`, and `.times`.
        """
    )
    return


@app.cell
def _(np, plt):
    from openpkpd.pk.analytical.advan1 import ADVAN1
    from openpkpd.pk.analytical.advan2 import ADVAN2
    from openpkpd.pk.analytical.advan3 import ADVAN3
    from openpkpd.pk.analytical.advan4 import ADVAN4
    from openpkpd.data.event_processor import DoseEvent

    t_obs = np.linspace(0.01, 24, 200)

    # --- ADVAN1: 1-cmt IV bolus ---
    advan1 = ADVAN1()
    dose_iv = [DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)]
    sol1 = advan1.solve(
        pk_params={"K": 3.5 / 40.0, "V": 40.0},
        dose_events=dose_iv,
        obs_times=t_obs,
    )

    # --- ADVAN2: 1-cmt oral ---
    advan2 = ADVAN2()
    dose_oral = [DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)]
    sol2 = advan2.solve(
        pk_params={"KA": 1.2, "K": 3.5 / 40.0, "V": 40.0},
        dose_events=dose_oral,
        obs_times=t_obs,
    )

    # --- ADVAN3: 2-cmt IV bolus ---
    advan3 = ADVAN3()
    sol3 = advan3.solve(
        pk_params={"CL": 3.5, "V1": 20.0, "Q": 2.0, "V2": 40.0},
        dose_events=dose_iv,
        obs_times=t_obs,
    )

    # --- ADVAN4: 2-cmt oral ---
    advan4 = ADVAN4()
    sol4 = advan4.solve(
        pk_params={
            "KA": 1.2,
            "K": 3.5 / 20.0,
            "K12": 2.0 / 20.0,
            "K21": 2.0 / 40.0,
            "V2": 20.0,
        },
        dose_events=dose_oral,
        obs_times=t_obs,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(t_obs, sol1.ipred, label="ADVAN1 — 1-cmt IV bolus", lw=2)
    ax.semilogy(t_obs, sol2.ipred, label="ADVAN2 — 1-cmt oral", lw=2, ls="--")
    ax.semilogy(t_obs, sol3.ipred, label="ADVAN3 — 2-cmt IV bolus", lw=2, ls="-.")
    ax.semilogy(t_obs, sol4.ipred, label="ADVAN4 — 2-cmt oral", lw=2, ls=":")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Analytical ADVAN solutions (100 mg dose, CL=3.5 L/h)")
    ax.legend()
    ax.set_ylim(1e-2, None)
    fig.tight_layout()
    fig
    return (
        ADVAN1,
        ADVAN2,
        ADVAN3,
        ADVAN4,
        DoseEvent,
        advan1,
        advan2,
        advan3,
        advan4,
        ax,
        dose_iv,
        dose_oral,
        fig,
        sol1,
        sol2,
        sol3,
        sol4,
        t_obs,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Three-Compartment Models (ADVAN11 / ADVAN12)

        ADVAN11 and ADVAN12 implement the three-compartment model analytically
        using the Bateman function with three exponential terms:

        $$C(t) = A \cdot e^{-\alpha t} + B \cdot e^{-\beta t} + C \cdot e^{-\gamma t}$$
        """
    )
    return


@app.cell
def _(DoseEvent, np, plt):
    from openpkpd.pk.analytical.advan11 import ADVAN11

    advan11 = ADVAN11()
    t3 = np.linspace(0.01, 48, 300)
    sol11 = advan11.solve(
        pk_params={
            "K": 3.5 / 10.0,
            "V1": 10.0,
            "K12": 2.0 / 10.0,
            "K21": 2.0 / 30.0,
            "K13": 0.5 / 10.0,
            "K31": 0.5 / 60.0,
        },
        dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)],
        obs_times=t3,
    )

    fig11, ax11 = plt.subplots(figsize=(8, 4))
    ax11.semilogy(t3, sol11.ipred, lw=2, color="steelblue")
    ax11.set_xlabel("Time (h)")
    ax11.set_ylabel("Concentration (mg/L)")
    ax11.set_title("ADVAN11 — 3-compartment IV (semi-log)")
    ax11.set_ylim(1e-3, None)
    fig11.tight_layout()
    fig11
    return ADVAN11, advan11, ax11, fig11, sol11, t3


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. ODE Models (ADVAN6 / ADVAN8)

        For models with user-defined differential equations — written in a
        `$DES` block in the control stream or as a Python callable — use
        ADVAN6 (non-stiff) or ADVAN8 (stiff, LSODA).

        In the `ModelBuilder` API, the ODE is specified by selecting
        `advan=6` or `advan=8` in `.subroutines()` and providing the
        right-hand-side in a `$DES`-like `des()` block (or embedded code).

        ```python
        ModelBuilder()
        .subroutines(advan=6)
        .pk(\"\"\"
            CL = THETA(1)*EXP(ETA(1))
            V  = THETA(2)*EXP(ETA(2))
            K10 = CL/V
        \"\"\")
        .des(\"\"\"
            DADT(1) = -K10 * A(1)
        \"\"\")
        ```

        The ODE classes also expose solver controls directly:

        ```python
        from openpkpd.pk.ode.advan6 import ADVAN6
        from openpkpd.pk.ode.advan8 import ADVAN8

        advan6 = ADVAN6(n_compartments=2, rtol=1e-7, atol=1e-9, method="RK45")
        advan8 = ADVAN8(n_compartments=2, rtol=1e-7, atol=1e-10, method="Radau")
        ```

        ### JIT acceleration

        ADVAN6 supports four solver tiers selected by the `jit=` parameter:

        | `jit=` | Backend | Speedup | Requires |
        |--------|---------|---------|----------|
        | `'scipy'` | `scipy.integrate.solve_ivp` | 1× (baseline) | — |
        | `'numpy'` | Pure-NumPy Dormand-Prince RK45 | ~1.3–1.8× | — |
        | `'numba'` | Numba `@njit` DES + NumPy RK45 | ~1.3–2× | `numba` |
        | `'llc'` | Numba RK45 + Numba DES — zero Python overhead | **10–30×** | `numba` |
        | `'auto'` | `'llc'` if numba installed, else `'numpy'` | best available | — |

        ```python
        # Install the JIT extra once:
        #   pip install "openpkpd[jit]"   (or: uv add "openpkpd[jit]")

        advan_fast = ADVAN6(n_compartments=2, jit="auto")   # picks best available
        advan_llc  = ADVAN6(n_compartments=2, jit="llc")    # explicit Numba native
        ```

        Benchmarks on 500-subject populations (measured speedup vs. scipy baseline):

        | Model | `jit='numpy'` | `jit='numba'` | `jit='llc'` |
        |-------|:---:|:---:|:---:|
        | 1-cmt IV | 1.3× | 1.2× | **11.6×** |
        | 2-cmt IV | 1.1× | 1.5× | **19.4×** |
        | 1-cmt oral | 1.2× | 1.3× | **30.3×** |
        | MM nonlinear | 1.8× | 1.5× | **9.1×** |

        The LLC tier compiles the entire RK45 loop **and** the DES right-hand side
        to native LLVM code, so no Python boundary is crossed during integration.

        > **Note**: `jit='scipy'` is the default to preserve exact numeric
        > reproducibility with existing workflows. Set `jit='auto'` (or `jit='llc'`)
        > when you need maximum throughput for population fitting or simulation.

        ### Stiff ODEs

        A PK ODE is *stiff* when its rate constants span widely separated time
        scales (e.g. fast distribution K12 >> slow elimination K10), which forces
        explicit solvers to take extremely small steps. Common PK examples include
        PBPK systems, receptor-binding models, and target-mediated drug disposition.

        **Automatic safety net**: if the explicit-RK45 tiers (`numpy`/`numba`/`llc`)
        hit the step limit, they automatically fall back to `scipy.solve_ivp` using
        `self.method` and emit a `UserWarning`:

        ```
        UserWarning: ODE step-limit exceeded (likely stiff ODE): …
        Falling back to scipy solve_ivp (method='RK45').
        Consider setting method='Radau' or method='BDF' on your ADVAN6/ADVAN8 instance.
        ```

        **What to do if you see this warning**:

        ```python
        # Option 1 — switch to ADVAN8 (LSODA, automatic stiff/nonstiff switching)
        advan8 = ADVAN8(n_compartments=2)        # default method='LSODA'

        # Option 2 — use ADVAN6 with an implicit solver
        advan6_stiff = ADVAN6(n_compartments=2, method="Radau", jit="scipy")

        # Option 3 — increase max_steps for mildly stiff models
        advan6 = ADVAN6(n_compartments=2, jit="numpy", max_steps=500_000)
        ```

        | Scenario | Recommended setup |
        |----------|-------------------|
        | Standard PK (nonstiff) | `ADVAN6(jit='auto')` |
        | Mildly stiff, want JIT speed | `ADVAN6(jit='auto', method='LSODA')` — JIT tries first, falls back |
        | Known stiff (PBPK, receptor binding) | `ADVAN8()` or `ADVAN6(method='Radau', jit='scipy')` |
        | Unknown — want a safe default | `ADVAN6(jit='scipy')` — scipy handles stiffness detection automatically |

        ### Direct ADVAN6 usage
        """
    )
    return


@app.cell
def _(DoseEvent, np, plt):
    from openpkpd.pk.ode.advan6 import ADVAN6
    from openpkpd.pk.ode.advan8 import ADVAN8

    advan6 = ADVAN6(n_compartments=2, rtol=1e-7, atol=1e-9, method="RK45")
    advan8 = ADVAN8(n_compartments=2, rtol=1e-7, atol=1e-10, method="Radau")
    t_ode = np.linspace(0.01, 24, 150)

    # Define the DES callable: 2-cmt IV model
    # Signature: (t, A_list, pk_params, theta, eta) -> list[dA/dt]
    def des_2cmt(t, a, params, theta, eta):
        k10 = params["CL"] / params["V1"]
        k12 = params["Q"] / params["V1"]
        k21 = params["Q"] / params["V2"]
        return [
            -k10 * a[0] - k12 * a[0] + k21 * a[1],
            k12 * a[0] - k21 * a[1],
        ]

    sol6 = advan6.solve(
        pk_params={"CL": 3.5, "V1": 20.0, "Q": 2.0, "V2": 40.0},
        dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)],
        obs_times=t_ode,
        des_callable=des_2cmt,
    )
    sol8 = advan8.solve(
        pk_params={"CL": 3.5, "V1": 20.0, "Q": 2.0, "V2": 40.0},
        dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)],
        obs_times=t_ode,
        des_callable=des_2cmt,
    )

    # Compare with ADVAN3 analytical for the same params
    from openpkpd.pk.analytical.advan3 import ADVAN3 as _ADVAN3

    _sol_an = _ADVAN3().solve(
        pk_params={"CL": 3.5, "V1": 20.0, "Q": 2.0, "V2": 40.0},
        dose_events=[DoseEvent(time=0.0, amount=100.0, compartment=1, rate=0.0)],
        obs_times=t_ode,
    )

    fig6, ax6 = plt.subplots(figsize=(8, 4))
    ax6.semilogy(t_ode, sol6.ipred, label="ADVAN6 RK45", lw=2)
    ax6.semilogy(t_ode, sol8.ipred, label="ADVAN8 Radau", lw=2, ls=":")
    ax6.semilogy(t_ode, _sol_an.ipred, label="ADVAN3 Analytical", lw=2, ls="--")
    ax6.set_xlabel("Time (h)")
    ax6.set_ylabel("C (mg/L)")
    ax6.set_title("ODE solver options vs ADVAN3 analytical — 2-compartment IV")
    ax6.legend()
    ax6.set_ylim(1e-2, None)
    fig6.tight_layout()
    print("ODE solver options")
    print(f"  ADVAN6 method={advan6.method} rtol={advan6.rtol} atol={advan6.atol}")
    print(f"  ADVAN8 method={advan8.method} rtol={advan8.rtol} atol={advan8.atol}")
    fig6
    return (
        ADVAN6,
        ADVAN8,
        _ADVAN3,
        _sol_an,
        advan6,
        advan8,
        ax6,
        des_2cmt,
        fig6,
        sol6,
        sol8,
        t_ode,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Transit Compartment Absorption

        For delayed oral absorption with a chain of $n$ transit compartments,
        use `TransitAbsorption` (ADVAN6 TRANS7 equivalent):

        $$\frac{dA_{\text{tr},k}}{dt} = k_{\text{tr}} A_{\text{tr},k-1} - k_{\text{tr}} A_{\text{tr},k}, \quad k=1\ldots n$$

        ```python
        from openpkpd.pk.absorption.transit import TransitAbsorption

        transit = TransitAbsorption(n_transit=4, method="ode")
        sol = transit.solve(pk_params, dose_events, obs_times)
        ```

        ## 5. Parallel Absorption

        Drugs with dual-peak profiles (e.g., enterohepatic recirculation or
        two absorption sites) can be modelled with `ParallelAbsorption`:

        ```python
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        model = ParallelAbsorption(n_paths=2)
        ```

        ## 6. Delay Differential Equations (DDE)

        For models with distributed delays (e.g., myelosuppression, tumour growth),
        use the DDE solver:

        ```python
        from openpkpd.pk.delay.dde import DDESolver

        solver = DDESolver()
        sol = solver.solve(rhs_with_delay, history_fn, t_span, y0, lags=[tau])
        ```

        ## 7. Michaelis-Menten Elimination (ADVAN10)

        ADVAN10 implements nonlinear (saturable) elimination:

        $$\frac{dC}{dt} = -\frac{V_{\max} \cdot C}{K_m + C}$$

        ```python
        from openpkpd.pk.ode.advan10 import ADVAN10

        advan10 = ADVAN10()
        sol = advan10.solve(
            pk_params={"VMAX": 100.0, "KM": 5.0, "V": 30.0},
            dose_events=dose_iv,
            obs_times=t_obs,
        )
        ```

        ## Quick Reference: ADVAN / TRANS Selection

        | Scenario | ADVAN | TRANS |
        |----------|-------|-------|
        | 1-cmt IV bolus | 1 | 2 (CL, V) |
        | 1-cmt oral | 2 | 2 (CL, V, KA) |
        | 2-cmt IV bolus | 3 | 4 (CL, V1, Q, V2) |
        | 2-cmt oral | 4 | 4 (CL, V1, Q, V2, KA) |
        | 3-cmt IV bolus | 11 | 4 (CL, V1, Q2, V2, Q3, V3) |
        | 3-cmt oral | 12 | 4 (+ KA) |
        | MM elimination | 10 | — (VMAX, KM, V) |
        | Transit absorption | 6 | 7 (MTT, N) |
        | Parallel absorption | 6 | 8 |
        | Custom ODE | 6/8 | — (user DES) |
        | Custom ODE + adjoint | 13 | — (user DES, JAX) |

        Practical guidance:

        - Start with **ADVAN6 / RK45** for smooth non-stiff systems.
        - Add `jit='auto'` (or `jit='llc'` with `openpkpd[jit]` installed) for
          **10–30× speedup** during population fitting and simulation.
        - Prefer **ADVAN8 / LSODA or Radau** when the state scales differ
          sharply or the solver needs tighter tolerances for stable integration.
        - If you see a `UserWarning: ODE step-limit exceeded`, your model is
          likely stiff — switch to `ADVAN8()` or `ADVAN6(method='Radau', jit='scipy')`.
        - Tighten `rtol` / `atol` when you need closer agreement between
          numerical and analytical reference paths.
        """
    )
    return


if __name__ == "__main__":
    app.run()
