"""
PD effect plots.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _make_fig_ax, _style


def effect_time(
    diag_df: pd.DataFrame,
    effect_col: str,
    *,
    individual: bool = True,
    mean_overlay: bool = True,
    ax=None,
    title: str = "Effect-Time Profile",
):
    """
    Effect vs time plot.

    Args:
        diag_df:      Diagnostic DataFrame with an effect column.
        effect_col:   Column name for the effect variable.
        individual:   Plot individual profiles.
        mean_overlay: Overlay mean profile.
        ax:           Existing axes.
        title:        Plot title.
    """
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        for sid in diag_df["ID"].unique():
            sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")
            ax.scatter(
                sub["TIME"], sub[effect_col], s=20, alpha=0.5, color="steelblue", edgecolors="none"
            )
            if individual:
                ax.plot(sub["TIME"], sub[effect_col], color="steelblue", alpha=0.4, linewidth=1.0)

        if mean_overlay:
            mean_df = diag_df.groupby("TIME")[effect_col].mean().reset_index()
            mean_df = mean_df.sort_values("TIME")
            ax.plot(
                mean_df["TIME"], mean_df[effect_col], color="black", linewidth=2.0, label="Mean"
            )
            ax.legend(fontsize=9)

        ax.set_xlabel("Time")
        ax.set_ylabel(effect_col)
        ax.set_title(title)
        fig.tight_layout()
    return fig


def emax_curve(
    diag_df: pd.DataFrame,
    conc_col: str,
    effect_col: str,
    *,
    emax: float | None = None,
    ec50: float | None = None,
    gamma: float = 1.0,
    e0: float = 0.0,
    ax=None,
    title: str = "Emax Curve",
):
    """
    Observed E vs C scatter with optional Emax model overlay.

    Args:
        diag_df:    Diagnostic DataFrame.
        conc_col:   Column for concentration.
        effect_col: Column for effect.
        emax:       Emax parameter (skip overlay if None).
        ec50:       EC50 parameter (skip overlay if None).
        gamma:      Hill exponent (default 1.0).
        e0:         Baseline effect.
        ax:         Existing axes.
        title:      Plot title.
    """
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        x = diag_df[conc_col].values
        y = diag_df[effect_col].values
        ax.scatter(x, y, alpha=0.6, edgecolors="none", zorder=2, label="Observed")

        if emax is not None and ec50 is not None:
            c_range = np.linspace(0, float(np.nanmax(x)) * 1.1, 200)
            c_g = c_range**gamma
            ec50_g = ec50**gamma
            eff = e0 + emax * c_g / (ec50_g + c_g)
            ax.plot(
                c_range,
                eff,
                color=_IBM_COLORS[2],
                linewidth=2.0,
                label=f"Emax={emax:.2g}, EC50={ec50:.2g}",
            )
            ax.legend(fontsize=9)

        ax.set_xlabel(conc_col)
        ax.set_ylabel(effect_col)
        ax.set_title(title)
        fig.tight_layout()
    return fig


def hysteresis_loop(
    diag_df: pd.DataFrame,
    conc_col: str,
    effect_col: str,
    *,
    color_by_time: bool = True,
    ax=None,
    title: str = "Hysteresis Loop",
):
    """
    C–E loop colored by time (one loop per subject, then mean).

    Args:
        diag_df:       Diagnostic DataFrame.
        conc_col:      Concentration column.
        effect_col:    Effect column.
        color_by_time: Color scatter points by time value.
        ax:            Existing axes.
        title:         Plot title.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    with _style():
        fig, ax = _make_fig_ax(ax)
        for sid in diag_df["ID"].unique():
            sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")
            x = sub[conc_col].values
            y = sub[effect_col].values
            t = sub["TIME"].values
            if color_by_time and len(t) > 0:
                sc = ax.scatter(
                    x, y, c=t, cmap="viridis", s=20, alpha=0.7, edgecolors="none", zorder=2
                )
            else:
                ax.scatter(x, y, s=20, alpha=0.5, edgecolors="none", zorder=2)
            ax.plot(x, y, alpha=0.3, linewidth=0.8, color="gray", zorder=1)

        if color_by_time:
            with contextlib.suppress(Exception):
                plt.colorbar(sc, ax=ax, label="Time")

        ax.set_xlabel(conc_col)
        ax.set_ylabel(effect_col)
        ax.set_title(title)
        fig.tight_layout()
    return fig


def indirect_response_plot(
    diag_df: pd.DataFrame,
    conc_col: str,
    response_col: str,
    *,
    individual: bool = True,
    mean_overlay: bool = True,
    title: str = "Indirect Response",
):
    """
    Dual-panel (PK / PD) time course for indirect response models.

    Top panel: drug concentration vs time.
    Bottom panel: response R(t) vs time with observed DV and IPRED.

    Args:
        diag_df:      Diagnostic DataFrame with TIME, DV, IPRED, conc_col, response_col.
        conc_col:     Column name for drug concentration (top panel).
        response_col: Column name for pharmacodynamic response (bottom panel).
        individual:   Plot per-subject profiles.
        mean_overlay: Overlay mean profiles.
        title:        Figure title.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    with _style():
        fig, (ax_pk, ax_pd) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ids = diag_df["ID"].unique()

        for sid in ids:
            sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")
            if individual:
                ax_pk.plot(
                    sub["TIME"], sub[conc_col], color=_IBM_COLORS[0], alpha=0.4, linewidth=1.0
                )
                ax_pd.scatter(
                    sub["TIME"], sub["DV"], s=15, alpha=0.5, color=_IBM_COLORS[1], edgecolors="none"
                )
                if response_col in sub.columns:
                    ax_pd.plot(
                        sub["TIME"],
                        sub[response_col],
                        color=_IBM_COLORS[1],
                        alpha=0.4,
                        linewidth=1.0,
                    )

        if mean_overlay:
            mean_pk = diag_df.groupby("TIME")[conc_col].mean().reset_index().sort_values("TIME")
            ax_pk.plot(
                mean_pk["TIME"],
                mean_pk[conc_col],
                color=_IBM_COLORS[0],
                linewidth=2.0,
                label="Mean Conc",
            )
            ax_pk.legend(fontsize=9)

            if response_col in diag_df.columns:
                mean_pd = (
                    diag_df.groupby("TIME")[response_col].mean().reset_index().sort_values("TIME")
                )
                ax_pd.plot(
                    mean_pd["TIME"],
                    mean_pd[response_col],
                    color=_IBM_COLORS[1],
                    linewidth=2.0,
                    label="Mean Response",
                )
                ax_pd.legend(fontsize=9)

        ax_pk.set_ylabel("Concentration")
        ax_pk.set_title(title)
        ax_pd.set_xlabel("Time")
        ax_pd.set_ylabel(response_col)
        fig.tight_layout()
    return fig


def effect_compartment_plot(
    diag_df: pd.DataFrame,
    cp_col: str,
    ce_col: str,
    *,
    individual: bool = False,
    mean_overlay: bool = True,
    ax=None,
    title: str = "Effect Compartment",
):
    """
    Overlay central (Cp) and effect compartment (Ce) concentrations vs time.

    Illustrates the temporal lag between Cp and Ce caused by Ke0.

    Args:
        diag_df:      Diagnostic DataFrame with TIME, cp_col, ce_col.
        cp_col:       Column name for central compartment concentration.
        ce_col:       Column name for effect compartment concentration.
        individual:   Plot per-subject thin lines.
        mean_overlay: Overlay mean Cp and Ce profiles.
        ax:           Existing axes.
        title:        Plot title.
    """
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)

        if individual:
            for sid in diag_df["ID"].unique():
                sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")
                ax.plot(sub["TIME"], sub[cp_col], color=_IBM_COLORS[0], alpha=0.3, linewidth=0.8)
                ax.plot(
                    sub["TIME"],
                    sub[ce_col],
                    color=_IBM_COLORS[2],
                    alpha=0.3,
                    linewidth=0.8,
                    linestyle="--",
                )

        if mean_overlay:
            mean_cp = diag_df.groupby("TIME")[cp_col].mean().reset_index().sort_values("TIME")
            mean_ce = diag_df.groupby("TIME")[ce_col].mean().reset_index().sort_values("TIME")
            ax.plot(
                mean_cp["TIME"],
                mean_cp[cp_col],
                color=_IBM_COLORS[0],
                linewidth=2.0,
                label=f"{cp_col} (Central)",
            )
            ax.plot(
                mean_ce["TIME"],
                mean_ce[ce_col],
                color=_IBM_COLORS[2],
                linewidth=2.0,
                linestyle="--",
                label=f"{ce_col} (Effect)",
            )
            ax.legend(fontsize=9)

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def pd_individual(
    diag_df: pd.DataFrame,
    conc_col: str,
    effect_col: str,
    *,
    subject_ids: list[int] | None = None,
    n_cols: int = 3,
    title: str = "Individual PK+PD Profiles",
):
    """
    Per-subject dual-panel (PK and PD) plot.

    Args:
        diag_df:     Diagnostic DataFrame.
        conc_col:    Concentration column.
        effect_col:  Effect column.
        subject_ids: Subset of IDs (default: all).
        n_cols:      Number of columns in the grid.
        title:       Overall figure title.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    ids = subject_ids or sorted(diag_df["ID"].unique().tolist())
    n_subj = len(ids)
    n_rows = (n_subj + n_cols - 1) // n_cols

    with _style():
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4 * n_cols, 3 * n_rows * 2))
        if axes.ndim == 1:
            axes = axes.reshape(-1, 1)

        for idx, sid in enumerate(ids):
            row = (idx // n_cols) * 2
            col = idx % n_cols
            sub = diag_df[diag_df["ID"] == sid].sort_values("TIME")

            # PK panel
            ax_pk = axes[row, col]
            ax_pk.plot(sub["TIME"], sub[conc_col], color=_IBM_COLORS[0])
            ax_pk.scatter(
                sub["TIME"], sub["DV"], s=15, color=_IBM_COLORS[0], edgecolors="none", alpha=0.8
            )
            ax_pk.set_ylabel("Conc")
            ax_pk.set_title(f"Subject {sid}")

            # PD panel
            ax_pd = axes[row + 1, col]
            ax_pd.plot(sub["TIME"], sub[effect_col], color=_IBM_COLORS[2])
            ax_pd.set_xlabel("Time")
            ax_pd.set_ylabel(effect_col)

        # Hide empty panels
        for idx in range(n_subj, n_rows * n_cols):
            row = (idx // n_cols) * 2
            col = idx % n_cols
            axes[row, col].set_visible(False)
            axes[row + 1, col].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig
