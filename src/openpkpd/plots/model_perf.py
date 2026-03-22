"""
Model performance plots: OFV history, VPC, parameter uncertainty, and residual trends.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _make_fig_ax, _style
from openpkpd.simulation.vpc import _quantile_label

if TYPE_CHECKING:
    from openpkpd.estimation.base import EstimationResult
    from openpkpd.model.population import PopulationModel


def ofv_history(
    result: EstimationResult,
    *,
    log_scale: bool = False,
    ax=None,
    title: str = "OFV History",
):
    """
    Line plot of OFV across outer iterations.

    Args:
        result:    EstimationResult with ofv_history list.
        log_scale: Use log scale on y-axis.
        ax:        Existing axes.
        title:     Plot title.
    """
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        history = result.ofv_history
        if not history:
            ax.text(
                0.5,
                0.5,
                "No OFV history available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
        else:
            ax.plot(range(1, len(history) + 1), history, color=_IBM_COLORS[0], linewidth=1.5)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("OFV")
            if log_scale:
                ax.set_yscale("log")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def vpc(
    diag_df: pd.DataFrame,
    population_model: PopulationModel | None = None,
    result: EstimationResult | None = None,
    *,
    vpc_result: Any | None = None,
    n_sim: int = 200,
    percentiles: tuple[float, float, float] = (5.0, 50.0, 95.0),
    stratify_col: str | None = None,
    ax=None,
    title: str = "Visual Predictive Check",
):
    """
    VPC: overlay observed vs simulated percentile bands.

    Simulated percentile bands are shown as shaded areas; observed
    percentiles are plotted as lines over observed data scatter.

    Can be called in two ways:

    1. With a VPCResult object (from VPCEngine.compute()):
          vpc(diag_df, vpc_result=my_vpc_result)

    2. With raw PopulationModel + EstimationResult (runs simulation internally):
          vpc(diag_df, population_model, result, n_sim=200)

    Args:
        diag_df:          Diagnostic DataFrame (from compute_diagnostics()).
        population_model: Assembled PopulationModel (required if vpc_result is None).
        result:           EstimationResult (required if vpc_result is None).
        vpc_result:       Pre-computed VPCResult from VPCEngine (optional).
                          If provided, population_model and result are ignored.
        n_sim:            Number of Monte Carlo simulations (used when vpc_result
                          is None). Must be > 0.
        percentiles:      Three percentile values (low, mid, high) as (5, 50, 95).
        stratify_col:     Column to stratify on (single plot if None).
        ax:               Existing matplotlib Axes to draw on.
        title:            Plot title.

    Returns:
        matplotlib.figure.Figure.

    Raises:
        ValueError: If n_sim <= 0 or if neither vpc_result nor population_model
                    is provided.
    """
    _check_matplotlib()

    # ── Path 1: use pre-computed VPCResult ───────────────────────────────────
    if vpc_result is not None:
        obs_pct_df = getattr(vpc_result, "obs_percentiles", None)
        sim_pct_df = getattr(vpc_result, "sim_percentiles", None)
        observed_df = getattr(vpc_result, "observed_df", diag_df)
        quantiles = tuple(
            getattr(vpc_result, "quantiles", tuple(float(p) / 100.0 for p in percentiles))
        )
        q_lo_col, q_mid_col, q_hi_col = [_quantile_label(q) for q in quantiles]
        q_lo_pct, q_mid_pct, q_hi_pct = [100.0 * float(q) for q in quantiles]

        with _style():
            fig, ax_out = _make_fig_ax(ax)

            obs_plot_df = observed_df if observed_df is not None else diag_df
            obs_times_raw = obs_plot_df["TIME"].values
            obs_dv_raw = obs_plot_df["DV"].values
            ax_out.scatter(
                obs_times_raw,
                obs_dv_raw,
                s=10,
                color="gray",
                alpha=0.4,
                edgecolors="none",
                zorder=1,
                label="Obs",
            )

            if sim_pct_df is not None and len(sim_pct_df) > 0:
                t_sim = sim_pct_df["bin_mid"].values
                sort_idx = np.argsort(t_sim)
                t_sorted = t_sim[sort_idx]

                lo_lo_col = f"{q_lo_col}_lo"
                hi_hi_col = f"{q_hi_col}_hi"
                mid_mid_col = f"{q_mid_col}_mid"

                # PI band (low–high)
                if lo_lo_col in sim_pct_df.columns and hi_hi_col in sim_pct_df.columns:
                    ax_out.fill_between(
                        t_sorted,
                        sim_pct_df[lo_lo_col].values[sort_idx],
                        sim_pct_df[hi_hi_col].values[sort_idx],
                        alpha=0.15,
                        color=_IBM_COLORS[0],
                        label=f"Sim {q_lo_pct:g}-{q_hi_pct:g}th PI",
                    )

                # Simulated median
                if mid_mid_col in sim_pct_df.columns:
                    ax_out.plot(
                        t_sorted,
                        sim_pct_df[mid_mid_col].values[sort_idx],
                        "--",
                        color=_IBM_COLORS[0],
                        linewidth=1.2,
                        label=f"Sim {q_mid_pct:g}th %ile",
                    )

            if obs_pct_df is not None and len(obs_pct_df) > 0:
                t_obs = obs_pct_df["bin_mid"].values
                sort_obs = np.argsort(t_obs)
                t_obs_s = t_obs[sort_obs]

                colors_obs = [_IBM_COLORS[2], "black", _IBM_COLORS[2]]
                styles_obs = [":", "-", ":"]
                col_names = [q_lo_col, q_mid_col, q_hi_col]
                for col, col_style, ls in zip(col_names, colors_obs, styles_obs, strict=False):
                    if col in obs_pct_df.columns:
                        ax_out.plot(
                            t_obs_s,
                            obs_pct_df[col].values[sort_obs],
                            color=col_style,
                            linestyle=ls,
                            linewidth=1.5,
                            label=f"Obs {col}",
                        )

            ax_out.set_xlabel("Time")
            ax_out.set_ylabel("Concentration")
            ax_out.set_title(title)
            ax_out.legend(fontsize=8, ncol=2)
            fig.tight_layout()
        return fig

    # ── Path 2: run Monte Carlo simulation internally ─────────────────────────
    if population_model is None or result is None:
        raise ValueError("Either vpc_result or both population_model and result must be provided.")

    if n_sim <= 0:
        raise ValueError("n_sim must be > 0")

    theta = result.theta_final
    omega = result.omega_final
    sigma = result.sigma_final
    n_eta = omega.shape[0]
    trans = population_model.trans

    rng = np.random.default_rng(42)

    # Simulate replicate datasets
    all_times: list[np.ndarray] = []
    all_concs: list[np.ndarray] = []

    for _ in range(n_sim):
        concs: list[float] = []
        times_flat: list[float] = []

        for sid in population_model.subject_ids():
            indiv = population_model.individual_model(sid)

            # Sample new eta from Omega
            try:
                L = np.linalg.cholesky(omega)
                eta_sim = L @ rng.standard_normal(n_eta)
            except Exception:
                eta_sim = np.zeros(n_eta)

            try:
                ipred, obs_mask, _ = indiv.evaluate(theta, eta_sim, sigma, trans=trans)
                obs_times_indiv = indiv.subject_events.obs_times[obs_mask]
                ipred_obs = ipred[obs_mask]

                # Add residual variability (proportional error)
                sigma_val = float(sigma[0, 0]) if sigma.size > 0 else 0.0
                eps = rng.standard_normal(len(ipred_obs)) * np.sqrt(
                    np.maximum(ipred_obs**2 * sigma_val, 1e-10)
                )
                y_sim = np.maximum(ipred_obs + eps, 0.0)

                times_flat.extend(obs_times_indiv.tolist())
                concs.extend(y_sim.tolist())
            except Exception:
                pass

        all_times.append(np.array(times_flat))
        all_concs.append(np.array(concs))

    # Bin observed data by time
    obs_times = diag_df["TIME"].values
    obs_dv = diag_df["DV"].values
    unique_times = np.unique(obs_times)

    obs_pct: dict[str, list[float]] = {f"p{int(p)}": [] for p in percentiles}
    sim_pct_lo_vals: list[float] = []
    sim_pct_hi_vals: list[float] = []
    sim_pct_med_vals: list[float] = []
    time_bins: list[float] = []

    for t in unique_times:
        obs_at_t = obs_dv[obs_times == t]
        if len(obs_at_t) == 0:
            continue
        time_bins.append(float(t))

        for p in percentiles:
            key = f"p{int(p)}"
            obs_pct[key].append(float(np.percentile(obs_at_t, p)))

        sim_vals: list[float] = []
        for i in range(n_sim):
            mask = np.isclose(all_times[i], t)
            sim_vals.extend(all_concs[i][mask].tolist())

        if sim_vals:
            lo = float(np.percentile(sim_vals, percentiles[0]))
            mid = float(np.percentile(sim_vals, percentiles[1]))
            hi = float(np.percentile(sim_vals, percentiles[2]))
        else:
            lo = mid = hi = float("nan")
        sim_pct_lo_vals.append(lo)
        sim_pct_med_vals.append(mid)
        sim_pct_hi_vals.append(hi)

    with _style():
        fig, ax_out = _make_fig_ax(ax)
        t_arr = np.array(time_bins)
        sort_idx = np.argsort(t_arr)
        t_sorted = t_arr[sort_idx]
        lo_s = np.array(sim_pct_lo_vals)[sort_idx]
        hi_s = np.array(sim_pct_hi_vals)[sort_idx]
        med_s = np.array(sim_pct_med_vals)[sort_idx]

        # Simulated PI band
        ax_out.fill_between(
            t_sorted,
            lo_s,
            hi_s,
            alpha=0.2,
            color=_IBM_COLORS[0],
            label=f"Sim {int(percentiles[0])}-{int(percentiles[2])}th %ile",
        )
        ax_out.plot(
            t_sorted,
            med_s,
            "--",
            color=_IBM_COLORS[0],
            linewidth=1.0,
            label=f"Sim {int(percentiles[1])}th %ile",
        )

        # Observed percentile lines
        colors_obs = [_IBM_COLORS[2], "black", _IBM_COLORS[2]]
        styles_obs = [":", "-", ":"]
        for p, col, ls in zip(percentiles, colors_obs, styles_obs, strict=False):
            key = f"p{int(p)}"
            p_arr = np.array(obs_pct[key])[sort_idx]
            ax_out.plot(
                t_sorted,
                p_arr,
                color=col,
                linestyle=ls,
                linewidth=1.5,
                label=f"Obs {int(p)}th %ile",
            )

        ax_out.scatter(
            obs_times,
            obs_dv,
            s=10,
            color="gray",
            alpha=0.4,
            edgecolors="none",
            zorder=1,
            label="Obs",
        )

        ax_out.set_xlabel("Time")
        ax_out.set_ylabel("Concentration")
        ax_out.set_title(title)
        ax_out.legend(fontsize=8, ncol=2)
        fig.tight_layout()
    return fig


def parameter_uncertainty_plot(
    result: Any,
    figsize: tuple[float, float] = (12, 8),
    title: str = "Parameter Uncertainty",
) -> Any:
    """
    Forest-plot style visualization of THETA estimates with 95% confidence intervals.

    Displays each THETA parameter estimate as a point with horizontal error bars
    representing ± 1.96 * SE (approximately 95% CI). Parameter values are shown
    on the x-axis and THETA labels on the y-axis.

    The SE values are extracted from the EstimationResult's covariance matrix
    (sqrt of diagonal of the inverse Hessian). If no covariance information is
    available, only the point estimates are plotted without error bars.

    Args:
        result:  EstimationResult with theta_final. Optionally has a covariance
                 matrix accessible as result.covariance_matrix (shape: n_theta x n_theta)
                 or result.standard_errors (shape: n_theta,).
        figsize: Figure size (width, height) in inches.
        title:   Plot title.

    Returns:
        matplotlib.figure.Figure.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    theta = result.theta_final
    n_theta = len(theta)

    # Try to extract standard errors from the result object
    se_values: np.ndarray | None = None

    if hasattr(result, "standard_errors") and result.standard_errors is not None:
        se_arr = np.asarray(result.standard_errors)
        if len(se_arr) >= n_theta:
            se_values = se_arr[:n_theta]
    elif hasattr(result, "covariance_matrix") and result.covariance_matrix is not None:
        cov = np.asarray(result.covariance_matrix)
        if cov.shape[0] >= n_theta:
            se_values = np.sqrt(np.maximum(np.diag(cov)[:n_theta], 0.0))

    # Build parameter labels
    labels = [f"THETA({i + 1})" for i in range(n_theta)]

    with _style():
        fig, ax = plt.subplots(figsize=figsize)

        y_positions = np.arange(n_theta)

        if se_values is not None:
            ci_lo = theta - 1.96 * se_values
            ci_hi = theta + 1.96 * se_values
            # Horizontal error bars
            ax.barh(
                y_positions,
                theta,
                xerr=np.column_stack([theta - ci_lo, ci_hi - theta]).T,
                color=_IBM_COLORS[0],
                alpha=0.7,
                error_kw={"ecolor": "#333333", "capsize": 4, "elinewidth": 1.2},
                height=0.5,
            )
        else:
            # No SE: just plot point estimates as bars
            ax.barh(
                y_positions,
                theta,
                color=_IBM_COLORS[0],
                alpha=0.7,
                height=0.5,
            )
            warnings.warn(
                "No covariance information found in result; "
                "plotting point estimates only without confidence intervals.",
                UserWarning,
                stacklevel=2,
            )

        ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":", zorder=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Parameter Estimate")
        ax.set_title(title)

        if se_values is not None:
            ax.text(
                0.98,
                0.02,
                "Error bars: ±1.96 SE (approx. 95% CI)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="gray",
            )

        fig.tight_layout()
    return fig


def model_comparison_plot(
    results: list[EstimationResult],
    labels: list[str],
    *,
    metric: str = "AIC",
    figsize: tuple[float, float] = (10, 0),
    title: str | None = None,
) -> Any:
    """
    Horizontal bar chart comparing multiple models by OFV, AIC, or BIC.

    Models are sorted best-to-worst (lowest metric first). Delta vs the
    best model is annotated on each bar. Non-converged models are shown
    with a hatched pattern.

    Args:
        results: List of EstimationResult objects.
        labels:  Model name strings, same length as results.
        metric:  ``"OFV"``, ``"AIC"``, or ``"BIC"`` (case-insensitive).
        figsize: Figure size. Height 0 = auto-size by number of models.
        title:   Plot title (default: f"Model Comparison — {metric}").

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    if len(results) != len(labels):
        raise ValueError("results and labels must have the same length.")

    metric_upper = metric.upper()
    if metric_upper not in ("OFV", "AIC", "BIC"):
        raise ValueError(f"metric must be 'OFV', 'AIC', or 'BIC'; got {metric!r}")

    def _get_metric(r: Any) -> float:
        if metric_upper == "OFV":
            return float(r.ofv)
        elif metric_upper == "AIC":
            return float(r.aic)
        else:
            return float(r.bic)

    values = [_get_metric(r) for r in results]
    converged = [bool(r.converged) for r in results]

    # Sort best (lowest) first
    order = sorted(range(len(values)), key=lambda i: values[i])
    sorted_labels = [labels[i] for i in order]
    sorted_values = [values[i] for i in order]
    sorted_conv = [converged[i] for i in order]

    best = sorted_values[0]
    deltas = [v - best for v in sorted_values]

    n = len(results)
    height = figsize[1] if figsize[1] > 0 else max(3.0, 0.55 * n + 1.5)

    with _style():
        fig, ax = plt.subplots(figsize=(figsize[0], height))

        y_pos = np.arange(n)[::-1]
        for _i, (y, val, delta, conv) in enumerate(
            zip(y_pos, sorted_values, deltas, sorted_conv, strict=False)
        ):
            color = _IBM_COLORS[0] if conv else _IBM_COLORS[2]
            hatch = "" if conv else "///"
            ax.barh(
                y,
                val,
                color=color,
                alpha=0.75,
                height=0.6,
                hatch=hatch,
                edgecolor="white" if conv else "#333333",
            )
            delta_str = f"Δ={delta:.1f}" if delta > 0 else "best"
            ax.text(val + abs(best) * 0.002, y, delta_str, va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels[::-1][::-1])
        ax.set_xlabel(metric_upper)
        ax.set_title(title or f"Model Comparison — {metric_upper}")

        from matplotlib.patches import Patch

        legend_els = [
            Patch(facecolor=_IBM_COLORS[0], alpha=0.75, label="Converged"),
            Patch(
                facecolor=_IBM_COLORS[2],
                alpha=0.75,
                hatch="///",
                edgecolor="#333333",
                label="Not converged",
            ),
        ]
        ax.legend(handles=legend_els, fontsize=8, loc="lower right")
        fig.tight_layout()
    return fig


def residual_trends_plot(
    diagnostic_df: Any,
    figsize: tuple[float, float] = (12, 8),
    title: str = "Residual Trends",
) -> Any:
    """
    Plot CWRES trends over time, predicted concentrations, and occasion.

    Produces a multi-panel figure showing conditional weighted residuals (CWRES)
    as a function of:
      - Panel 1: TIME — temporal trend of residuals
      - Panel 2: PRED — predicted concentration trend
      - Panel 3: IPRED — individual predicted concentration trend

    Panels include zero-line and ±1.96 reference bands. A LOWESS smoother
    line is overlaid if scipy is available.

    The CWRES column must be present in diagnostic_df (available from
    compute_diagnostics()).

    Args:
        diagnostic_df: DataFrame with columns CWRES, TIME, PRED, IPRED.
                       Additional columns (e.g., OCCASION) are used if present.
        figsize:       Figure size (width, height) in inches.
        title:         Overall figure title.

    Returns:
        matplotlib.figure.Figure with 3 panels.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError:  If CWRES column is not found in diagnostic_df.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    df = diagnostic_df
    if "CWRES" not in df.columns:
        raise ValueError(
            "CWRES column not found in diagnostic_df. "
            "Run compute_diagnostics() first to generate CWRES values."
        )

    cwres = df["CWRES"].values
    panel_data = [
        ("TIME", "Time", _IBM_COLORS[0]),
        ("PRED", "Population Prediction", _IBM_COLORS[1]),
        ("IPRED", "Individual Prediction", _IBM_COLORS[2]),
    ]

    # Filter to available columns
    available_panels = [
        (col, label, color) for col, label, color in panel_data if col in df.columns
    ]
    n_panels = len(available_panels)

    if n_panels == 0:
        raise ValueError("None of TIME, PRED, IPRED columns found in diagnostic_df.")

    with _style():
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        ax_row = axes[0]

        for ax, (col, xlabel, color) in zip(ax_row, available_panels, strict=False):
            x_vals = df[col].values

            # Filter valid (finite) pairs
            valid = np.isfinite(x_vals) & np.isfinite(cwres)
            x_v = x_vals[valid]
            y_v = cwres[valid]

            ax.scatter(x_v, y_v, alpha=0.5, edgecolors="none", color=color, s=16, zorder=2)

            # Reference lines
            ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
            ax.axhline(1.96, color="gray", linewidth=0.8, linestyle=":", zorder=1)
            ax.axhline(-1.96, color="gray", linewidth=0.8, linestyle=":", zorder=1)

            # LOWESS smoother overlay
            try:
                if len(x_v) >= 4:
                    sort_idx = np.argsort(x_v)
                    x_sorted = x_v[sort_idx]
                    y_sorted = y_v[sort_idx]
                    # Use a simple running median as a lightweight smoother
                    window = max(3, len(x_sorted) // 8)
                    smoothed: list[float] = []
                    for i in range(len(x_sorted)):
                        lo_idx = max(0, i - window // 2)
                        hi_idx = min(len(x_sorted), i + window // 2 + 1)
                        smoothed.append(float(np.median(y_sorted[lo_idx:hi_idx])))
                    ax.plot(
                        x_sorted,
                        smoothed,
                        color="#E74C3C",
                        linewidth=1.4,
                        linestyle="-",
                        alpha=0.8,
                        label="Smoother",
                        zorder=3,
                    )
                    ax.legend(fontsize=7)
            except ImportError:
                pass

            ax.set_xlabel(xlabel)
            ax.set_ylabel("CWRES")
            ax.set_title(f"CWRES vs {col}")

        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
    return fig


def likelihood_profile_plot(
    param_values: Any,
    ofv_values: Any,
    *,
    param_name: str = "THETA",
    final_estimate: float | None = None,
    ci_threshold: float = 3.84,
    ax=None,
    title: str | None = None,
) -> Any:
    """
    Plot OFV vs a single profiled parameter value (profile likelihood).

    The region where ΔOFV < ``ci_threshold`` (default 3.84 = χ²_{1, α=0.05})
    defines the approximate 95% profile likelihood CI. Computation of the
    profile (re-fitting with each parameter fixed) is the caller's responsibility.

    Args:
        param_values:   1-D array of parameter values at which OFV was evaluated.
        ofv_values:     Corresponding OFV values.
        param_name:     Parameter label for the x-axis.
        final_estimate: MLE estimate (plotted as vertical line).
        ci_threshold:   ΔOFV threshold for CI boundary (default 3.84).
        ax:             Existing axes.
        title:          Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()

    param_values = np.asarray(param_values, dtype=float)
    ofv_values = np.asarray(ofv_values, dtype=float)
    sort_idx = np.argsort(param_values)
    x = param_values[sort_idx]
    y = ofv_values[sort_idx]
    ofv_min = float(np.nanmin(y))
    threshold = ofv_min + ci_threshold

    with _style():
        fig, ax = _make_fig_ax(ax)

        ax.plot(x, y, color=_IBM_COLORS[0], linewidth=1.8, zorder=2)
        ax.axhline(
            threshold,
            color="gray",
            linewidth=1.0,
            linestyle="--",
            label=f"OFV_min + {ci_threshold:.2f} (95% CI)",
        )

        # Shade the CI region
        ci_mask = y <= threshold
        if np.any(ci_mask):
            ax.fill_between(
                x, ofv_min, y, where=ci_mask, alpha=0.15, color=_IBM_COLORS[0], zorder=1
            )

        if final_estimate is not None:
            ax.axvline(
                final_estimate,
                color=_IBM_COLORS[2],
                linewidth=1.2,
                linestyle=":",
                label=f"MLE = {final_estimate:.4g}",
            )

        ax.set_xlabel(param_name)
        ax.set_ylabel("OFV")
        ax.set_title(title or f"Likelihood Profile — {param_name}")
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig
