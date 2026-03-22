"""
PK concentration-time plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _make_fig_ax, _style


def concentration_time(
    diag_df: pd.DataFrame,
    *,
    log_y: bool = False,
    individual: bool = True,
    mean_overlay: bool = True,
    highlight_ids: list[int] | None = None,
    ax=None,
    title: str = "Concentration-Time Profile",
):
    """
    Individual and/or mean concentration-time plot.

    Args:
        diag_df:       Diagnostic DataFrame from compute_diagnostics().
        log_y:         Use log scale on y-axis.
        individual:    Plot individual IPRED lines.
        mean_overlay:  Overlay the mean IPRED profile.
        highlight_ids: Subject IDs to highlight with distinct color.
        ax:            Existing axes to plot into.
        title:         Plot title.
    """
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        ids = diag_df["ID"].unique()
        highlight_ids_set: set = set(highlight_ids or [])

        for sid in ids:
            sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")
            color = _IBM_COLORS[2] if sid in highlight_ids_set else "steelblue"
            alpha = 0.9 if sid in highlight_ids_set else 0.4
            zorder = 3 if sid in highlight_ids_set else 2

            ax.scatter(
                sub["TIME"],
                sub["DV"],
                s=20,
                alpha=alpha,
                color=color,
                edgecolors="none",
                zorder=zorder,
            )
            if individual:
                ax.plot(
                    sub["TIME"],
                    sub["IPRED"],
                    color=color,
                    alpha=alpha,
                    linewidth=1.2,
                    zorder=zorder,
                )

        if mean_overlay:
            mean_df = diag_df.groupby("TIME")[["DV", "IPRED"]].mean().reset_index()
            mean_df = mean_df.sort_values("TIME")
            ax.plot(
                mean_df["TIME"],
                mean_df["IPRED"],
                color="black",
                linewidth=2.0,
                label="Mean IPRED",
                zorder=4,
            )

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(title)
        if mean_overlay:
            ax.legend(fontsize=9)
        fig.tight_layout()
    return fig


def spaghetti_plot(
    diag_df: pd.DataFrame,
    *,
    log_y: bool = False,
    mean_overlay: bool = True,
    alpha: float = 0.35,
    ax=None,
    title: str = "Spaghetti Plot",
):
    """
    Overlaid individual IPRED profiles.

    Args:
        diag_df:      Diagnostic DataFrame.
        log_y:        Log scale on y-axis.
        mean_overlay: Overlay mean profile in bold.
        alpha:        Transparency for individual lines.
        ax:           Existing axes to plot into.
        title:        Plot title.
    """
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        for sid in diag_df["ID"].unique():
            sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")
            ax.plot(sub["TIME"], sub["IPRED"], color="steelblue", alpha=alpha, linewidth=1.0)

        if mean_overlay:
            mean_df = diag_df.groupby("TIME")["IPRED"].mean().reset_index()
            mean_df = mean_df.sort_values("TIME")
            ax.plot(
                mean_df["TIME"], mean_df["IPRED"], color="black", linewidth=2.0, label="Mean IPRED"
            )
            ax.legend(fontsize=9)

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def individual_fit_grid(
    diag_df: pd.DataFrame,
    *,
    subject_ids: list[int] | None = None,
    n_cols: int = 4,
    log_y: bool = False,
    title: str = "Individual Fits",
):
    """
    Grid of per-subject panels showing observed DV, IPRED, and PRED vs time.

    Args:
        diag_df:     Diagnostic DataFrame with ID, TIME, DV, IPRED, PRED columns.
        subject_ids: Subset of subject IDs to plot (default: all).
        n_cols:      Number of columns in the grid.
        log_y:       Use log scale on y-axis.
        title:       Overall figure title.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    ids = subject_ids if subject_ids is not None else sorted(diag_df["ID"].unique().tolist())
    n_subj = len(ids)
    n_rows = max(1, (n_subj + n_cols - 1) // n_cols)

    with _style():
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows), squeeze=False
        )
        flat_axes = axes.flatten()

        for idx, sid in enumerate(ids):
            ax = flat_axes[idx]
            sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")
            ax.scatter(
                sub["TIME"],
                sub["DV"],
                s=18,
                color="#333333",
                edgecolors="none",
                alpha=0.85,
                zorder=3,
                label="Obs",
            )
            ax.plot(
                sub["TIME"],
                sub["IPRED"],
                color=_IBM_COLORS[0],
                linewidth=1.4,
                zorder=2,
                label="IPRED",
            )
            if "PRED" in sub.columns:
                ax.plot(
                    sub["TIME"],
                    sub["PRED"],
                    color=_IBM_COLORS[2],
                    linewidth=1.0,
                    linestyle="--",
                    zorder=1,
                    label="PRED",
                )
            if log_y:
                ax.set_yscale("log")
            ax.set_title(f"ID {sid}", fontsize=9)
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Conc", fontsize=8)
            ax.tick_params(labelsize=7)

        # Legend on first panel only
        if len(ids) > 0:
            flat_axes[0].legend(fontsize=7, loc="upper right")

        for idx in range(n_subj, len(flat_axes)):
            flat_axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
    return fig


def mean_profile(
    diag_df: pd.DataFrame,
    *,
    log_y: bool = False,
    sd_band: bool = True,
    ax=None,
    title: str = "Mean Concentration Profile",
):
    """
    Mean IPRED profile with optional ± SD band.

    Args:
        diag_df:   Diagnostic DataFrame.
        log_y:     Log scale on y-axis.
        sd_band:   Show mean ± 1 SD shaded band.
        ax:        Existing axes to plot into.
        title:     Plot title.
    """
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        grp = diag_df.groupby("TIME")["IPRED"]
        mean_df = grp.mean().reset_index()
        mean_df = mean_df.sort_values("TIME")
        t = mean_df["TIME"].values
        m = mean_df["IPRED"].values

        ax.plot(t, m, color=_IBM_COLORS[0], linewidth=2.0, label="Mean IPRED")

        if sd_band:
            std_df = grp.std().reset_index().sort_values("TIME")
            s = std_df["IPRED"].values
            ax.fill_between(
                t, np.maximum(m - s, 0), m + s, alpha=0.25, color=_IBM_COLORS[0], label="±1 SD"
            )

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig
