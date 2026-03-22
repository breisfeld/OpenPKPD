"""
Simulation-based diagnostic plots: VPC, NPDE, and prediction interval plots.

These functions produce publication-quality figures for simulation-based model
evaluation. They accept either VPCResult objects (from openpkpd.simulation.vpc)
or raw DataFrames / NumPy arrays as inputs.

matplotlib is required for all functions. Install with:
    uv pip install matplotlib
    uv sync --extra plots
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openpkpd.simulation.vpc import _quantile_label

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _style


def _require_matplotlib() -> None:
    """Alias for _check_matplotlib for consistent naming within this module."""
    _check_matplotlib()


def vpc_plot(
    vpc_result: Any,
    title: str = "Visual Predictive Check",
    obs_color: str = "#333333",
    sim_pi_color: str = "#3498DB",
    sim_median_color: str = "#E74C3C",
    figsize: tuple[float, float] = (10, 6),
    log_y: bool = False,
    ax: Any | None = None,
) -> Any:
    """
    VPC plot: observed data overlaid with simulated prediction intervals.

    Produces a standard VPC figure with:
      - Grey dots: individual observed DV values
      - Blue shaded regions: 90% PI bands for simulated 5th and 95th percentiles
      - Red shaded region: 90% PI band for simulated median (50th percentile)
      - Solid colored lines: medians of simulated percentile bands
      - Dashed black lines: observed 5th, 50th, 95th percentiles (binned by time)

    Accepts either a VPCResult object (from openpkpd.simulation.vpc.VPCEngine)
    or any object with observed_df, simulated_df, obs_percentiles, and
    sim_percentiles attributes.

    Args:
        vpc_result:        VPCResult from VPCEngine.compute(), or object with
                           observed_df, obs_percentiles, sim_percentiles attributes.
        title:             Plot title string.
        obs_color:         Color for observed data points and observed percentile lines.
        sim_pi_color:      Color for simulated PI shaded regions.
        sim_median_color:  Color for simulated median shaded region.
        figsize:           Figure size (width, height) in inches.
        log_y:             If True, use log scale on the y-axis.
        ax:                Existing matplotlib Axes to draw on (creates new figure if None).

    Returns:
        matplotlib.figure.Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _require_matplotlib()

    with _style():
        if ax is not None:
            fig, ax_out = ax.get_figure(), ax
        else:
            fig, ax_out = plt.subplots(figsize=figsize)

        # Extract data from vpc_result
        observed_df = getattr(vpc_result, "observed_df", None)
        obs_pct = getattr(vpc_result, "obs_percentiles", None)
        sim_pct = getattr(vpc_result, "sim_percentiles", None)
        quantiles = tuple(getattr(vpc_result, "quantiles", (0.05, 0.50, 0.95)))
        q_lo_col, q_mid_col, q_hi_col = [_quantile_label(q) for q in quantiles]

        # Plot observed raw data points
        if observed_df is not None and len(observed_df) > 0:
            obs_dv = observed_df["DV"].values
            obs_times = observed_df["TIME"].values
            valid_mask = np.isfinite(obs_dv) & np.isfinite(obs_times)
            if valid_mask.any():
                ax_out.scatter(
                    obs_times[valid_mask],
                    obs_dv[valid_mask],
                    s=12,
                    color=obs_color,
                    alpha=0.35,
                    edgecolors="none",
                    zorder=2,
                    label="Observed",
                )

        # Plot simulated percentile bands from sim_percentiles DataFrame
        if sim_pct is not None and len(sim_pct) > 0:
            t_sim = sim_pct["bin_mid"].values
            sort_idx = np.argsort(t_sim)
            t_sorted = t_sim[sort_idx]

            # 5th percentile band (PI of the p5 across replicates)
            lo_lo_col = f"{q_lo_col}_lo"
            lo_hi_col = f"{q_lo_col}_hi"
            lo_mid_col = f"{q_lo_col}_mid"
            if lo_lo_col in sim_pct.columns and lo_hi_col in sim_pct.columns:
                p5_lo = sim_pct[lo_lo_col].values[sort_idx]
                p5_hi = sim_pct[lo_hi_col].values[sort_idx]
                p5_mid = sim_pct[lo_mid_col].values[sort_idx]
                ax_out.fill_between(
                    t_sorted,
                    p5_lo,
                    p5_hi,
                    alpha=0.20,
                    color=sim_pi_color,
                    zorder=1,
                )
                ax_out.plot(
                    t_sorted,
                    p5_mid,
                    color=sim_pi_color,
                    linewidth=1.0,
                    linestyle="-",
                    zorder=3,
                )

            # 95th percentile band
            hi_lo_col = f"{q_hi_col}_lo"
            hi_hi_col = f"{q_hi_col}_hi"
            hi_mid_col = f"{q_hi_col}_mid"
            if hi_lo_col in sim_pct.columns and hi_hi_col in sim_pct.columns:
                p95_lo = sim_pct[hi_lo_col].values[sort_idx]
                p95_hi = sim_pct[hi_hi_col].values[sort_idx]
                p95_mid = sim_pct[hi_mid_col].values[sort_idx]
                ax_out.fill_between(
                    t_sorted,
                    p95_lo,
                    p95_hi,
                    alpha=0.20,
                    color=sim_pi_color,
                    zorder=1,
                )
                ax_out.plot(
                    t_sorted,
                    p95_mid,
                    color=sim_pi_color,
                    linewidth=1.0,
                    linestyle="-",
                    zorder=3,
                )

            # 50th percentile band (median)
            mid_lo_col = f"{q_mid_col}_lo"
            mid_hi_col = f"{q_mid_col}_hi"
            mid_mid_col = f"{q_mid_col}_mid"
            if mid_lo_col in sim_pct.columns and mid_hi_col in sim_pct.columns:
                p50_lo = sim_pct[mid_lo_col].values[sort_idx]
                p50_hi = sim_pct[mid_hi_col].values[sort_idx]
                p50_mid = sim_pct[mid_mid_col].values[sort_idx]
                ax_out.fill_between(
                    t_sorted,
                    p50_lo,
                    p50_hi,
                    alpha=0.25,
                    color=sim_median_color,
                    zorder=1,
                )
                ax_out.plot(
                    t_sorted,
                    p50_mid,
                    color=sim_median_color,
                    linewidth=1.5,
                    linestyle="-",
                    zorder=3,
                )

        # Plot observed percentile lines from obs_percentiles DataFrame
        if obs_pct is not None and len(obs_pct) > 0:
            t_obs = obs_pct["bin_mid"].values
            sort_obs = np.argsort(t_obs)
            t_obs_s = t_obs[sort_obs]

            pct_styles = [
                (q_lo_col, "--", 1.0),
                (q_mid_col, "-", 1.8),
                (q_hi_col, "--", 1.0),
            ]
            for col, ls, lw in pct_styles:
                if col in obs_pct.columns:
                    vals = obs_pct[col].values[sort_obs]
                    ax_out.plot(
                        t_obs_s,
                        vals,
                        color=obs_color,
                        linestyle=ls,
                        linewidth=lw,
                        zorder=4,
                    )

        # Legend patches
        legend_elements = [
            mpatches.Patch(color=obs_color, alpha=0.7, label="Observed"),
            mpatches.Patch(color=sim_pi_color, alpha=0.3, label="Sim 5th/95th PI"),
            mpatches.Patch(color=sim_median_color, alpha=0.3, label="Sim Median PI"),
        ]
        ax_out.legend(handles=legend_elements, fontsize=8, loc="upper right")

        ax_out.set_xlabel("Time")
        ax_out.set_ylabel("Concentration")
        ax_out.set_title(title)

        if log_y:
            ax_out.set_yscale("log")

        fig.tight_layout()

    return fig


def npde_plot(
    diagnostic_df: Any,
    title: str = "NPDE Diagnostics",
    figsize: tuple[float, float] = (12, 8),
) -> Any:
    """
    Four-panel NPDE diagnostic plot.

    Produces a 2x2 figure panel showing:
      1. NPDE vs TIME — scatter with zero line and +/-1.96 bands
      2. NPDE vs PRED — scatter with zero line and +/-1.96 bands
      3. NPDE histogram — bar chart with standard normal N(0,1) overlay
      4. NPDE Q-Q plot — quantile-quantile plot against standard normal

    The NPDE column must be present in diagnostic_df. Use compute_npde()
    from openpkpd.plots.diagnostics to compute NPDEs from a fitted model.

    Args:
        diagnostic_df: DataFrame with at minimum columns: TIME, PRED, NPDE.
                       Additional columns CWRES, DV, IPRED are used if present.
        title:         Overall figure title.
        figsize:       Figure size (width, height) in inches.

    Returns:
        matplotlib.figure.Figure with 4 axes.

    Raises:
        ImportError:  If matplotlib is not installed.
        ValueError:   If the NPDE column is not found in diagnostic_df.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.stats import norm, probplot

    if "NPDE" not in diagnostic_df.columns:
        raise ValueError(
            "NPDE column not found in diagnostic_df. Run compute_npde() first to add NPDE values."
        )

    npde_df = diagnostic_df[diagnostic_df["NPDE"].notna()].copy()
    npde_vals = npde_df["NPDE"].to_numpy(dtype=float)
    time_vals = npde_df["TIME"].to_numpy(dtype=float) if "TIME" in npde_df.columns else None
    pred_vals = npde_df["PRED"].to_numpy(dtype=float) if "PRED" in npde_df.columns else None

    with _style():
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax1, ax2, ax3, ax4 = axes.flatten()

        # ── Panel 1: NPDE vs TIME ──────────────────────────────────────────
        if time_vals is not None and len(npde_vals) > 0 and len(time_vals) == len(npde_vals):
            ax1.scatter(
                time_vals,
                npde_vals,
                alpha=0.5,
                edgecolors="none",
                color=_IBM_COLORS[0],
                s=18,
                zorder=2,
            )
            _x_lo, _x_hi = float(np.nanmin(time_vals)), float(np.nanmax(time_vals))
        else:
            _x_lo, _x_hi = 0.0, 1.0
        ax1.axhline(0.0, color="black", linewidth=0.9, zorder=1)
        ax1.axhline(1.96, color="gray", linewidth=0.8, linestyle=":", zorder=1)
        ax1.axhline(-1.96, color="gray", linewidth=0.8, linestyle=":", zorder=1)
        ax1.set_xlabel("TIME")
        ax1.set_ylabel("NPDE")
        ax1.set_title("NPDE vs TIME")

        # ── Panel 2: NPDE vs PRED ─────────────────────────────────────────
        if pred_vals is not None and len(npde_vals) > 0 and len(pred_vals) == len(npde_vals):
            ax2.scatter(
                pred_vals,
                npde_vals,
                alpha=0.5,
                edgecolors="none",
                color=_IBM_COLORS[1],
                s=18,
                zorder=2,
            )
        ax2.axhline(0.0, color="black", linewidth=0.9, zorder=1)
        ax2.axhline(1.96, color="gray", linewidth=0.8, linestyle=":", zorder=1)
        ax2.axhline(-1.96, color="gray", linewidth=0.8, linestyle=":", zorder=1)
        ax2.set_xlabel("PRED")
        ax2.set_ylabel("NPDE")
        ax2.set_title("NPDE vs PRED")

        # ── Panel 3: NPDE histogram with N(0,1) overlay ───────────────────
        if len(npde_vals) > 0:
            n_bins = max(10, int(np.sqrt(len(npde_vals))))
            ax3.hist(
                npde_vals,
                bins=n_bins,
                density=True,
                alpha=0.65,
                color=_IBM_COLORS[2],
                edgecolor="white",
                linewidth=0.5,
            )
            x_norm = np.linspace(
                float(np.nanmin(npde_vals)) - 0.5, float(np.nanmax(npde_vals)) + 0.5, 200
            )
            ax3.plot(x_norm, norm.pdf(x_norm), "k-", linewidth=1.5, label="N(0,1)")
        ax3.set_xlabel("NPDE")
        ax3.set_ylabel("Density")
        ax3.set_title("NPDE Histogram")
        if len(npde_vals) > 0:
            ax3.legend(fontsize=8)

        # ── Panel 4: Q-Q plot ─────────────────────────────────────────────
        if len(npde_vals) > 0:
            (osm, osr), (slope, intercept, _r) = probplot(npde_vals, dist="norm")
            ax4.scatter(
                osm, osr, alpha=0.55, edgecolors="none", color=_IBM_COLORS[3], s=18, zorder=2
            )
            q_lo = float(np.min(osm))
            q_hi = float(np.max(osm))
            ax4.plot(
                [q_lo, q_hi],
                [slope * q_lo + intercept, slope * q_hi + intercept],
                "r--",
                linewidth=1.2,
                label="Normal ref.",
            )
        ax4.set_xlabel("Theoretical Quantiles")
        ax4.set_ylabel("Sample Quantiles")
        ax4.set_title("NPDE Q-Q Plot")
        if len(npde_vals) > 0:
            ax4.legend(fontsize=8)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()

    return fig


def simulation_panel(
    simulated_df: Any,
    observed_df: Any | None = None,
    n_subjects: int = 9,
    figsize: tuple[float, float] = (15, 12),
    title: str = "Simulated Profiles",
) -> Any:
    """
    Panel of simulated individual concentration-time profiles.

    Shows up to n_subjects individual simulated profiles in a grid layout.
    Each panel displays all replicate profiles for that subject as thin
    translucent lines. If observed_df is provided, observed data points
    are overlaid as solid markers.

    The simulated_df must contain columns: ID, TIME, DV, REP.
    The observed_df (optional) must contain columns: ID, TIME, DV.

    Args:
        simulated_df: DataFrame with simulated data (columns: ID, TIME, DV, REP).
                      REP=0 is treated as the observed reference; REP>=1 are
                      simulation replicates.
        observed_df:  Optional DataFrame with observed data (columns: ID, TIME, DV).
                      If None, uses REP=0 rows from simulated_df if present.
        n_subjects:   Maximum number of subjects to display (default 9).
        figsize:      Figure size (width, height) in inches.
        title:        Overall figure title.

    Returns:
        matplotlib.figure.Figure with a grid of individual profile panels.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    # Separate observed from simulated replicates
    if observed_df is None and "REP" in simulated_df.columns:
        observed_df = simulated_df[simulated_df["REP"] == 0].copy()
        sim_only = simulated_df[simulated_df["REP"] >= 1].copy()
    else:
        sim_only = simulated_df.copy()
        if "REP" in sim_only.columns:
            sim_only = sim_only[sim_only["REP"] >= 1]

    # Select subjects to display
    all_subject_ids = sorted(sim_only["ID"].unique())
    if not all_subject_ids and observed_df is not None and len(observed_df) > 0:
        all_subject_ids = sorted(observed_df["ID"].unique())
    display_ids = all_subject_ids[:n_subjects]

    # Determine grid layout (roughly square)
    n_display = len(display_ids)
    n_cols = max(1, min(3, n_display))
    n_rows = max(1, (n_display + n_cols - 1) // n_cols)

    with _style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        flat_axes = axes.flatten()

        for panel_idx, sid in enumerate(display_ids):
            ax = flat_axes[panel_idx]
            subj_sim = sim_only[sim_only["ID"] == sid]

            # Plot simulated replicates as thin translucent lines
            reps = subj_sim["REP"].unique() if "REP" in subj_sim.columns else [1]
            for rep in reps:
                rep_data = (
                    subj_sim[subj_sim["REP"] == rep] if "REP" in subj_sim.columns else subj_sim
                )
                rep_sorted = rep_data.sort_values("TIME")
                ax.plot(
                    rep_sorted["TIME"].values,
                    rep_sorted["DV"].values,
                    color=_IBM_COLORS[0],
                    alpha=0.08,
                    linewidth=0.6,
                    zorder=1,
                )

            # Overlay observed data
            if observed_df is not None and len(observed_df) > 0:
                obs_subj = observed_df[observed_df["ID"] == sid]
                if len(obs_subj) > 0:
                    obs_sorted = obs_subj.sort_values("TIME")
                    ax.plot(
                        obs_sorted["TIME"].values,
                        obs_sorted["DV"].values,
                        "o-",
                        color="#333333",
                        markersize=4,
                        linewidth=1.2,
                        zorder=3,
                        label="Observed",
                    )

            ax.set_title(f"Subject {sid}", fontsize=9)
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Concentration", fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for idx in range(n_display, len(flat_axes)):
            flat_axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

    return fig


def prediction_interval_plot(
    times: np.ndarray,
    obs_conc: np.ndarray,
    sim_pi_lo: np.ndarray,
    sim_median: np.ndarray,
    sim_pi_hi: np.ndarray,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Prediction Interval Plot",
    ax: Any | None = None,
) -> Any:
    """
    Simple prediction interval plot with observed data overlay.

    Produces a single-panel figure showing:
      - Blue shaded band: simulation prediction interval (sim_pi_lo to sim_pi_hi)
      - Red dashed line: simulation median (sim_median)
      - Black dots: observed concentrations (obs_conc)

    All input arrays must have the same length as times.

    Args:
        times:       1-D array of time points (x-axis).
        obs_conc:    1-D array of observed concentrations at each time.
        sim_pi_lo:   1-D array of lower prediction interval bounds (e.g., 5th pctile).
        sim_median:  1-D array of simulated median concentrations (50th pctile).
        sim_pi_hi:   1-D array of upper prediction interval bounds (e.g., 95th pctile).
        figsize:     Figure size (width, height) in inches.
        title:       Plot title string.
        ax:          Existing matplotlib Axes to draw on (creates new figure if None).

    Returns:
        matplotlib.figure.Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError:  If input arrays have incompatible lengths.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=float)
    obs_conc = np.asarray(obs_conc, dtype=float)
    sim_pi_lo = np.asarray(sim_pi_lo, dtype=float)
    sim_median = np.asarray(sim_median, dtype=float)
    sim_pi_hi = np.asarray(sim_pi_hi, dtype=float)

    if not (len(obs_conc) == len(sim_pi_lo) == len(sim_median) == len(sim_pi_hi) == len(times)):
        raise ValueError(
            "All input arrays must have the same length as 'times'. "
            f"Got: times={len(times)}, obs={len(obs_conc)}, "
            f"lo={len(sim_pi_lo)}, med={len(sim_median)}, hi={len(sim_pi_hi)}"
        )

    sort_idx = np.argsort(times)
    t_s = times[sort_idx]
    lo_s = sim_pi_lo[sort_idx]
    med_s = sim_median[sort_idx]
    hi_s = sim_pi_hi[sort_idx]
    obs_s = obs_conc[sort_idx]

    with _style():
        if ax is not None:
            fig_out, ax_out = ax.get_figure(), ax
        else:
            fig_out, ax_out = plt.subplots(figsize=figsize)

        # Prediction interval shaded band
        ax_out.fill_between(
            t_s,
            lo_s,
            hi_s,
            alpha=0.25,
            color=_IBM_COLORS[0],
            label="Prediction Interval",
            zorder=1,
        )

        # Simulated median line
        ax_out.plot(
            t_s,
            med_s,
            color=_IBM_COLORS[2],
            linewidth=1.8,
            linestyle="--",
            label="Sim Median",
            zorder=2,
        )

        # Observed data points
        valid = np.isfinite(obs_s)
        ax_out.scatter(
            t_s[valid],
            obs_s[valid],
            color="#222222",
            s=30,
            zorder=3,
            label="Observed",
            edgecolors="none",
            alpha=0.85,
        )

        ax_out.set_xlabel("Time")
        ax_out.set_ylabel("Concentration")
        ax_out.set_title(title)
        ax_out.legend(fontsize=9)
        fig_out.tight_layout()

    return fig_out


def pcvpc_plot(
    diag_df: Any,
    vpc_result: Any,
    *,
    percentiles: tuple[float, float, float] = (5.0, 50.0, 95.0),
    log_y: bool = False,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Prediction-Corrected VPC",
) -> Any:
    """
    Prediction-corrected VPC (pcVPC).

    Observed and simulated concentrations are normalized by the population
    prediction (PRED) at each time point, removing the influence of dose
    and design heterogeneity.

    Correction: DV_pc = DV * median(PRED_sim_bin) / PRED_obs

    Args:
        diag_df:     Diagnostic DataFrame with columns DV, PRED, TIME.
        vpc_result:  VPCResult from ``VPCEngine``.
        percentiles: Percentile triplet (low, median, high).
        log_y:       Log scale on y-axis.
        figsize:     Figure size.
        title:       Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    getattr(vpc_result, "obs_percentiles", None)
    sim_pct = getattr(vpc_result, "sim_percentiles", None)
    observed_df = getattr(vpc_result, "observed_df", diag_df)
    quantiles = tuple(getattr(vpc_result, "quantiles", tuple(p / 100 for p in percentiles)))
    q_lo_col, q_mid_col, q_hi_col = [_quantile_label(q) for q in quantiles]

    # Prediction-correct observed DV
    obs_dv = observed_df["DV"].values.astype(float)
    obs_pred = (
        observed_df["PRED"].values.astype(float)
        if "PRED" in observed_df.columns
        else np.ones_like(obs_dv)
    )
    obs_times = observed_df["TIME"].values.astype(float)

    # Median PRED per time bin (from sim_percentiles bin_mid) for normalization
    median_pred_at_t: dict[float, float] = {}
    if sim_pct is not None and len(sim_pct) > 0 and "bin_mid" in sim_pct.columns:
        mid_mid_col = f"{q_mid_col}_mid"
        if mid_mid_col in sim_pct.columns:
            for _, row in sim_pct.iterrows():
                median_pred_at_t[float(row["bin_mid"])] = float(row[mid_mid_col])

    def _pc_correct(dv: np.ndarray, pred: np.ndarray, times: np.ndarray) -> np.ndarray:
        dv_pc = dv.copy()
        for i, (t, p) in enumerate(zip(times, pred, strict=False)):
            if p > 0 and median_pred_at_t:
                closest_t = min(median_pred_at_t.keys(), key=lambda bt: abs(bt - t))
                med_pred = median_pred_at_t[closest_t]
                dv_pc[i] = dv[i] * med_pred / p
        return dv_pc

    dv_pc = _pc_correct(obs_dv, obs_pred, obs_times)

    with _style():
        if True:
            fig, ax_out = plt.subplots(figsize=figsize)

        valid = np.isfinite(dv_pc) & np.isfinite(obs_times)
        ax_out.scatter(
            obs_times[valid],
            dv_pc[valid],
            s=12,
            color="#333333",
            alpha=0.35,
            edgecolors="none",
            zorder=2,
            label="Obs (pc)",
        )

        # Simulated PI bands (already prediction-corrected in a proper pcVPC engine;
        # here we re-use raw sim_percentiles as an approximation when no pcVPC engine)
        if sim_pct is not None and len(sim_pct) > 0:
            t_sim = sim_pct["bin_mid"].values
            sort_idx = np.argsort(t_sim)
            t_sorted = t_sim[sort_idx]
            lo_lo_col = f"{q_lo_col}_lo"
            hi_hi_col = f"{q_hi_col}_hi"
            mid_mid_col = f"{q_mid_col}_mid"
            if lo_lo_col in sim_pct.columns and hi_hi_col in sim_pct.columns:
                ax_out.fill_between(
                    t_sorted,
                    sim_pct[lo_lo_col].values[sort_idx],
                    sim_pct[hi_hi_col].values[sort_idx],
                    alpha=0.15,
                    color=_IBM_COLORS[0],
                    label=f"Sim {int(percentiles[0])}-{int(percentiles[2])}th PI",
                )
            if mid_mid_col in sim_pct.columns:
                ax_out.plot(
                    t_sorted,
                    sim_pct[mid_mid_col].values[sort_idx],
                    "--",
                    color=_IBM_COLORS[0],
                    linewidth=1.2,
                    label=f"Sim {int(percentiles[1])}th %ile",
                )

        if log_y:
            ax_out.set_yscale("log")
        ax_out.set_xlabel("Time")
        ax_out.set_ylabel("Pred-Corrected Concentration")
        ax_out.set_title(title)
        ax_out.legend(fontsize=8, ncol=2)
        fig.tight_layout()

    return fig


def stratified_vpc_plot(
    diag_df: Any,
    vpc_result: Any,
    stratify_col: str,
    *,
    percentiles: tuple[float, float, float] = (5.0, 50.0, 95.0),
    log_y: bool = False,
    n_cols: int = 2,
    title: str = "Stratified VPC",
) -> Any:
    """
    Multi-panel VPC stratified by a covariate column.

    One VPC panel is drawn per unique value of ``stratify_col``. Each panel
    uses the same ``vpc_plot`` style (observed scatter + simulated PI bands).
    Subsets of ``diag_df`` are used for the observed data in each panel;
    the simulated percentiles from ``vpc_result`` are applied to the matching
    stratum rows.

    Args:
        diag_df:      Diagnostic DataFrame with ``stratify_col`` present.
        vpc_result:   VPCResult from ``VPCEngine``.
        stratify_col: Column name to stratify on.
        percentiles:  Percentile triplet.
        log_y:        Log scale on y-axis.
        n_cols:       Number of columns.
        title:        Overall figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if stratify_col not in diag_df.columns:
        raise ValueError(f"stratify_col {stratify_col!r} not found in diag_df.")

    strata = sorted(diag_df[stratify_col].dropna().unique())
    n_strata = len(strata)
    n_rows = (n_strata + n_cols - 1) // n_cols

    with _style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        flat_axes = axes.flatten()

        for idx, stratum in enumerate(strata):
            ax = flat_axes[idx]
            sub_df = diag_df[diag_df[stratify_col] == stratum]

            obs_times = sub_df["TIME"].values.astype(float)
            obs_dv = sub_df["DV"].values.astype(float)
            valid = np.isfinite(obs_dv) & np.isfinite(obs_times)
            ax.scatter(
                obs_times[valid],
                obs_dv[valid],
                s=10,
                color="#333333",
                alpha=0.35,
                edgecolors="none",
                zorder=2,
                label="Obs",
            )

            # Overlay vpc_result simulated bands (shared across strata as approximation)
            sim_pct = getattr(vpc_result, "sim_percentiles", None)
            quantiles = tuple(getattr(vpc_result, "quantiles", tuple(p / 100 for p in percentiles)))
            q_lo_col, q_mid_col, q_hi_col = [_quantile_label(q) for q in quantiles]

            if sim_pct is not None and len(sim_pct) > 0:
                t_sim = sim_pct["bin_mid"].values
                sort_i = np.argsort(t_sim)
                t_s = t_sim[sort_i]
                mid_mid_col = f"{q_mid_col}_mid"
                lo_lo_col = f"{q_lo_col}_lo"
                hi_hi_col = f"{q_hi_col}_hi"
                if lo_lo_col in sim_pct.columns and hi_hi_col in sim_pct.columns:
                    ax.fill_between(
                        t_s,
                        sim_pct[lo_lo_col].values[sort_i],
                        sim_pct[hi_hi_col].values[sort_i],
                        alpha=0.15,
                        color=_IBM_COLORS[0],
                        label=f"Sim {int(percentiles[0])}-{int(percentiles[2])}th PI",
                    )
                if mid_mid_col in sim_pct.columns:
                    ax.plot(
                        t_s,
                        sim_pct[mid_mid_col].values[sort_i],
                        "--",
                        color=_IBM_COLORS[0],
                        linewidth=1.0,
                        label=f"Sim {int(percentiles[1])}th %ile",
                    )

            if log_y:
                ax.set_yscale("log")
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.set_title(f"{stratify_col} = {stratum}")
            ax.legend(fontsize=7)

        for idx in range(n_strata, len(flat_axes)):
            flat_axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig
