"""
Example 11: Time-to-Event (TTE) Survival Analysis.

Demonstrates:
  1. Simulate TTE data from a Weibull hazard model with concentration-
     dependent hazard (drug effect accelerates time-to-event).
  2. Fit ConstantHazardModel and WeibullHazardModel to the simulated data.
  3. Compare AIC between models to select the best-fitting model.
  4. Plot Kaplan-Meier empirical survival vs fitted parametric survival
     curves (matplotlib optional).

No file I/O required — all data are generated inline.
"""

from __future__ import annotations

import os

import numpy as np

from openpkpd.models.tte import (
    ConstantHazardModel,
    TTEData,
    TTEResult,
    WeibullHazardModel,
)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

TRUE_SCALE = 15.0    # Weibull scale (lambda)
TRUE_SHAPE = 1.8     # Weibull shape (p > 1 => increasing hazard)
TRUE_BETA = 0.08     # log-linear concentration effect on cumulative hazard
N_SUBJECTS = 120     # number of simulated subjects
MAX_FOLLOW_UP = 40.0 # administrative censoring time
SEED = 2024


# ---------------------------------------------------------------------------
# PK simulation
# ---------------------------------------------------------------------------

def _simulate_pk(
    n_subjects: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a simple one-compartment IV bolus PK profile for each subject.

    Returns a per-subject average concentration (AUC/T) as a scalar
    exposure metric.  This is a deliberately simplified approach so
    that the example remains self-contained.

    PK model:
        C(t) = Dose/V * exp(-CL/V * t)
        CL ~ LogNormal(log(3.0), 0.3)
        V  ~ LogNormal(log(30.0), 0.3)
        Dose = 100 mg (fixed)

    Args:
        n_subjects: Number of subjects to simulate.
        rng: NumPy random generator.

    Returns:
        Tuple of (pk_times, concentrations) where:
          pk_times:       shape (n_subjects, n_pk_times)
          concentrations: shape (n_subjects, n_pk_times)
    """
    dose = 100.0
    pk_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    CL = np.exp(np.log(3.0) + rng.normal(0, 0.3, size=n_subjects))
    V = np.exp(np.log(30.0) + rng.normal(0, 0.3, size=n_subjects))
    # C_i(t) = (D/V_i) * exp(-CL_i/V_i * t)
    concentrations = (dose / V[:, np.newaxis]) * np.exp(
        -(CL / V)[:, np.newaxis] * pk_times[np.newaxis, :]
    )
    pk_times_mat = np.tile(pk_times, (n_subjects, 1))
    return pk_times_mat, concentrations


# ---------------------------------------------------------------------------
# Survival-time simulation via inverse-CDF method
# ---------------------------------------------------------------------------

def _simulate_tte(
    pk_times_mat: np.ndarray,
    concentrations: np.ndarray,
    rng: np.random.Generator,
) -> list[TTEData]:
    """Simulate Weibull survival times with concentration-dependent hazard.

    The individual's average concentration over the first 12 hours is used
    as a time-constant exposure metric.  The cumulative hazard integrating
    the concentration-modified Weibull hazard is:

        H(t | C_avg) = (t / scale)^shape * exp(beta * C_avg)

    Survival time is obtained by the inverse-CDF method:
        U ~ Uniform(0, 1)
        t = scale * (-log(U) / exp(beta * C_avg))^(1/shape)

    Administrative censoring at MAX_FOLLOW_UP is applied.

    Args:
        pk_times_mat: PK times per subject, shape (n, n_pk).
        concentrations: PK concentrations per subject, shape (n, n_pk).
        rng: NumPy random generator.

    Returns:
        List of TTEData, one per subject.
    """
    n = concentrations.shape[0]

    # Average concentration as scalar exposure per subject
    c_avg = concentrations.mean(axis=1)  # shape (n,)

    # Inverse-CDF simulation
    U = rng.uniform(size=n)
    # t = scale * (-log(U) / exp(beta * C_avg))^(1/shape)
    log_t = (
        np.log(TRUE_SCALE)
        + (1.0 / TRUE_SHAPE) * (np.log(-np.log(U)) - TRUE_BETA * c_avg)
    )
    t_event = np.exp(log_t)

    data: list[TTEData] = []
    for i in range(n):
        t_obs = min(float(t_event[i]), MAX_FOLLOW_UP)
        indicator = 1 if float(t_event[i]) <= MAX_FOLLOW_UP else 0
        data.append(
            TTEData(
                subject_id=i + 1,
                event_times=np.array([t_obs]),
                event_indicator=np.array([indicator]),
                concentration_times=pk_times_mat[i, :],
                concentrations=concentrations[i, :],
            )
        )
    return data


# ---------------------------------------------------------------------------
# Kaplan-Meier estimator
# ---------------------------------------------------------------------------

def _kaplan_meier(
    data: list[TTEData],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Kaplan-Meier empirical survival function.

    Args:
        data: List of TTEData records.

    Returns:
        Tuple of (times, survival) arrays for step-function plotting.
    """
    times = np.array([float(d.event_times[0]) for d in data])
    events = np.array([int(d.event_indicator[0]) for d in data])

    order = np.argsort(times)
    times = times[order]
    events = events[order]

    n = len(times)
    km_times = [0.0]
    km_surv = [1.0]
    n_risk = n
    S = 1.0

    for i in range(n):
        if events[i] == 1:
            S *= (n_risk - 1.0) / n_risk
            km_times.append(float(times[i]))
            km_surv.append(S)
        n_risk -= 1

    return np.array(km_times), np.array(km_surv)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_result(name: str, result: TTEResult) -> None:
    """Print a formatted summary of a TTEResult.

    Args:
        name: Human-readable model name.
        result: Fitted TTEResult.
    """
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  OFV       : {result.ofv:.4f}")
    print(f"  AIC       : {result.aic:.4f}")
    print(f"  Converged : {result.converged}")
    print(f"  Params    : {np.round(result.hazard_params, 4)}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_survival(
    km_times: np.ndarray,
    km_surv: np.ndarray,
    result_const: TTEResult,
    result_weib: TTEResult,
    out_dir: str,
) -> None:
    """Plot KM estimate vs fitted parametric survival curves.

    Args:
        km_times: Kaplan-Meier time points.
        km_surv: Kaplan-Meier survival values.
        result_const: Fitted ConstantHazardModel result.
        result_weib: Fitted WeibullHazardModel result.
        out_dir: Output directory for saving the figure (or empty string to skip).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_grid = np.linspace(0.0, MAX_FOLLOW_UP, 300)
    s_const = np.array([result_const.survival_function(t) for t in t_grid])
    s_weib = np.array([result_weib.survival_function(t) for t in t_grid])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Kaplan-Meier step function
    ax.step(
        km_times, km_surv,
        where="post",
        color="black",
        linewidth=1.5,
        label="Kaplan-Meier",
    )

    # Constant hazard (exponential) fit
    ax.plot(
        t_grid, s_const,
        color="steelblue",
        linestyle="--",
        linewidth=2,
        label=(
            f"Constant hazard  "
            f"[lambda={result_const.hazard_params[0]:.3f}, "
            f"AIC={result_const.aic:.1f}]"
        ),
    )

    # Weibull fit
    ax.plot(
        t_grid, s_weib,
        color="tomato",
        linestyle="-",
        linewidth=2,
        label=(
            f"Weibull  "
            f"[scale={result_weib.hazard_params[0]:.2f}, "
            f"shape={result_weib.hazard_params[1]:.2f}, "
            f"AIC={result_weib.aic:.1f}]"
        ),
    )

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="S(t)=0.5")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Survival probability S(t)", fontsize=12)
    ax.set_title(
        "TTE Survival Analysis — KM vs Parametric Fits\n"
        f"(True: Weibull scale={TRUE_SCALE}, shape={TRUE_SHAPE}, beta={TRUE_BETA})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, MAX_FOLLOW_UP)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if out_dir:
        path = os.path.join(out_dir, "11_tte_survival.png")
        fig.savefig(path, dpi=150)
        print(f"\nFigure saved to: {path}")
    else:
        print(
            "\nFigure created (set env var OPENPKPD_EXAMPLE_OUTPUT to save to disk)."
        )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full TTE example pipeline."""
    print("=" * 55)
    print("  Example 11: Time-to-Event Survival Analysis")
    print("=" * 55)
    print(
        f"\nTrue model: Weibull(scale={TRUE_SCALE}, shape={TRUE_SHAPE})"
        f" with concentration effect beta={TRUE_BETA}"
    )
    print(f"N={N_SUBJECTS} subjects, max follow-up={MAX_FOLLOW_UP} time units")

    # -----------------------------------------------------------------------
    # 1. Simulate data
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    pk_times_mat, concentrations = _simulate_pk(N_SUBJECTS, rng)
    data = _simulate_tte(pk_times_mat, concentrations, rng)

    n_events = sum(int(d.event_indicator[0]) for d in data)
    n_censored = N_SUBJECTS - n_events
    event_times = np.array([float(d.event_times[0]) for d in data])
    print(
        f"\nSimulated {n_events} events, {n_censored} censored "
        f"(censoring rate {100*n_censored/N_SUBJECTS:.1f}%)"
    )
    print(f"Median observation time: {np.median(event_times):.2f}")

    # -----------------------------------------------------------------------
    # 2. Kaplan-Meier estimate
    # -----------------------------------------------------------------------
    km_times, km_surv = _kaplan_meier(data)
    # Empirical median survival time
    idx_50 = np.searchsorted(-km_surv, -0.5)
    if idx_50 < len(km_times):
        print(f"KM median survival time: {km_times[idx_50]:.2f}")

    # -----------------------------------------------------------------------
    # 3. Fit constant hazard model
    # -----------------------------------------------------------------------
    print("\n--- Fitting ConstantHazardModel (exponential survival) ---")
    const_model = ConstantHazardModel()
    # MLE for exponential: lambda = n_events / sum(t)
    lam_init = n_events / event_times.sum()
    result_const = const_model.fit(data, init_params=np.array([lam_init]))
    _print_result("ConstantHazardModel", result_const)

    # -----------------------------------------------------------------------
    # 4. Fit Weibull hazard model
    # -----------------------------------------------------------------------
    print("\n--- Fitting WeibullHazardModel ---")
    weib_model = WeibullHazardModel()
    # Initial: scale from mean event time, shape=1.5
    scale_init = float(event_times[np.array([int(d.event_indicator[0]) for d in data]) == 1].mean())
    result_weib = weib_model.fit(
        data, init_params=np.array([scale_init, 1.5])
    )
    _print_result("WeibullHazardModel", result_weib)

    # -----------------------------------------------------------------------
    # 5. AIC comparison
    # -----------------------------------------------------------------------
    print("\n--- AIC Comparison ---")
    print(f"  ConstantHazard AIC : {result_const.aic:.4f}")
    print(f"  Weibull AIC        : {result_weib.aic:.4f}")
    delta_aic = result_const.aic - result_weib.aic
    if delta_aic > 2:
        print(
            f"  => Weibull model preferred (deltaAIC={delta_aic:.2f})"
        )
    elif delta_aic < -2:
        print(
            f"  => Constant hazard model preferred (deltaAIC={delta_aic:.2f})"
        )
    else:
        print(f"  => Models comparable (deltaAIC={delta_aic:.2f})")

    print(f"\n  True Weibull scale: {TRUE_SCALE:.2f}")
    print(f"  Fitted scale      : {result_weib.hazard_params[0]:.3f}")
    print(f"  True Weibull shape: {TRUE_SHAPE:.2f}")
    print(f"  Fitted shape      : {result_weib.hazard_params[1]:.3f}")

    # -----------------------------------------------------------------------
    # 6. Plot
    # -----------------------------------------------------------------------
    out_dir = os.environ.get("OPENPKPD_EXAMPLE_OUTPUT", "")
    try:
        _plot_survival(km_times, km_surv, result_const, result_weib, out_dir)
    except ImportError:
        print("\nmatplotlib not installed — skipping survival plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
