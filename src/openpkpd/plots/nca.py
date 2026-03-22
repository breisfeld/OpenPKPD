"""
NCA visualization plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _make_fig_ax, _style

if TYPE_CHECKING:
    from openpkpd.nca.nca import NCAParameters

_DEFAULT_NCA_PARAMS = ["c0", "cmax", "auc_last", "auc_inf", "t_half", "cl_f", "vz_f"]

_NCA_LABELS: dict[str, str] = {
    "c0": "C0",
    "cmax": "Cmax",
    "auc_last": "AUC_last",
    "auc_inf": "AUC_inf",
    "t_half": "t½",
    "cl_f": "CL/F",
    "vz_f": "Vz/F",
    "mrt": "MRT",
    "lambda_z": "λz",
    "tmax": "Tmax",
}


def _resolve_nca_params(nca_df: pd.DataFrame, params: list[str] | None) -> list[str]:
    """Resolve plot parameter columns, skipping empty defaults like all-NaN C0."""
    if params is not None:
        return [p for p in params if p in nca_df.columns]
    return [p for p in _DEFAULT_NCA_PARAMS if p in nca_df.columns and not nca_df[p].dropna().empty]


def nca_distributions(
    nca_df: pd.DataFrame,
    *,
    params: list[str] | None = None,
    n_cols: int = 3,
    title: str = "NCA Parameter Distributions",
) -> Any:
    """
    Grid of histograms for key NCA parameters across subjects.

    Args:
        nca_df:  DataFrame from ``NCAEngine.compute_dataset()`` (one row per subject).
        params:  Column names to plot. Default: C0 (when available), Cmax,
                 AUC_last, AUC_inf, t½, CL/F, Vz/F.
        n_cols:  Number of columns in the grid.
        title:   Figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    cols = _resolve_nca_params(nca_df, params)
    if not cols:
        raise ValueError("No recognizable NCA parameter columns found in nca_df.")

    n = len(cols)
    n_rows = (n + n_cols - 1) // n_cols

    with _style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes_flat = np.array(axes).flatten()

        for k, col in enumerate(cols):
            ax = axes_flat[k]
            vals = nca_df[col].dropna().values
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            ax.hist(
                vals,
                bins=max(5, int(np.sqrt(len(vals)))),
                color=_IBM_COLORS[0],
                alpha=0.75,
                edgecolor="white",
            )
            median_val = float(np.median(vals))
            ax.axvline(
                median_val,
                color=_IBM_COLORS[2],
                linewidth=1.2,
                linestyle="--",
                label=f"Median={median_val:.3g}",
            )
            ax.set_xlabel(_NCA_LABELS.get(col, col))
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

        for k in range(n, len(axes_flat)):
            axes_flat[k].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def nca_boxplot(
    nca_df: pd.DataFrame,
    params: list[str] | None = None,
    *,
    group_col: str | None = None,
    title: str = "NCA Parameters",
) -> Any:
    """
    Grouped boxplots of NCA parameters, optionally stratified by a column.

    Args:
        nca_df:    DataFrame from ``NCAEngine.compute_dataset()``.
        params:    NCA parameter column names to plot.
        group_col: Optional column for grouping (e.g. dose group, sex).
        title:     Figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    cols = _resolve_nca_params(nca_df, params)
    if not cols:
        raise ValueError("No recognizable NCA parameter columns found in nca_df.")

    n = len(cols)

    with _style():
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
        ax_row = axes[0]

        for ax, col in zip(ax_row, cols, strict=False):
            if group_col and group_col in nca_df.columns:
                groups = sorted(nca_df[group_col].dropna().unique())
                data = [nca_df[nca_df[group_col] == g][col].dropna().values for g in groups]
                ax.boxplot(
                    data,
                    tick_labels=[str(g) for g in groups],
                    patch_artist=True,
                    boxprops={"facecolor": _IBM_COLORS[0], "alpha": 0.6},
                )
                ax.set_xlabel(str(group_col))
            else:
                vals = nca_df[col].dropna().values
                ax.boxplot(
                    [vals],
                    tick_labels=["All"],
                    patch_artist=True,
                    boxprops={"facecolor": _IBM_COLORS[0], "alpha": 0.6},
                )
            ax.set_ylabel(_NCA_LABELS.get(col, col))
            ax.set_title(_NCA_LABELS.get(col, col))

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def nca_profile_plot(
    times: np.ndarray,
    conc: np.ndarray,
    nca_params: NCAParameters,
    *,
    log_y: bool = True,
    ax=None,
    title: str | None = None,
) -> Any:
    """
    Individual C-t profile annotated with key NCA landmarks.

    Shows observed concentrations with:
    - Cmax horizontal dashed line and label
    - Tmax vertical line
    - Terminal regression line (lambda_z fit) over the terminal points
    - AUC shading (trapezoidal fill under the curve)

    Args:
        times:      Observed time array.
        conc:       Observed concentration array.
        nca_params: ``NCAParameters`` from ``NCAEngine.compute_subject()``.
        log_y:      Use log scale on y-axis (default True; semi-log is standard).
        ax:         Existing axes.
        title:      Plot title (defaults to "Subject {nca_params.subject_id}").
    """
    _check_matplotlib()

    times = np.asarray(times, dtype=float)
    conc = np.asarray(conc, dtype=float)
    sort_idx = np.argsort(times)
    t = times[sort_idx]
    c = conc[sort_idx]

    with _style():
        fig, ax = _make_fig_ax(ax)

        # Observed data
        valid = np.isfinite(c) & (c > 0 if log_y else np.isfinite(c))
        ax.scatter(
            t[valid],
            c[valid],
            color=_IBM_COLORS[0],
            s=30,
            edgecolors="none",
            zorder=3,
            label="Observed",
        )
        ax.plot(t[valid], c[valid], color=_IBM_COLORS[0], linewidth=1.2, alpha=0.6, zorder=2)

        # AUC shading
        pos = c > 0
        if np.any(pos):
            ax.fill_between(
                t,
                0 if not log_y else np.nanmin(c[c > 0]) * 0.01,
                c,
                where=pos & np.isfinite(c),
                alpha=0.10,
                color=_IBM_COLORS[0],
            )

        # Cmax line
        if np.isfinite(nca_params.cmax):
            ax.axhline(nca_params.cmax, color=_IBM_COLORS[2], linewidth=0.9, linestyle="--")
            ax.text(
                float(np.nanmax(t)) * 0.02,
                nca_params.cmax * 1.05,
                f"Cmax={nca_params.cmax:.3g}",
                fontsize=8,
                color=_IBM_COLORS[2],
                va="bottom",
            )

        # Tmax line
        if np.isfinite(nca_params.tmax):
            ax.axvline(nca_params.tmax, color=_IBM_COLORS[3], linewidth=0.9, linestyle=":")
            ax.text(
                nca_params.tmax * 1.02,
                ax.get_ylim()[1] if not log_y else float(np.nanmax(c[c > 0])) * 0.9,
                f"Tmax={nca_params.tmax:.3g}",
                fontsize=8,
                color=_IBM_COLORS[3],
                va="top",
            )

        # Terminal regression line
        if (
            np.isfinite(nca_params.lambda_z)
            and nca_params.lambda_z > 0
            and nca_params.n_points_lambda >= 3
            and np.isfinite(nca_params.tmax)
        ):
            # Use last n_points_lambda post-Tmax observations
            post_tmax = t >= nca_params.tmax
            t_term = t[post_tmax]
            c_term = c[post_tmax]
            pos_term = c_term > 0
            t_term = t_term[pos_term]
            c_term = c_term[pos_term]
            if len(t_term) >= nca_params.n_points_lambda:
                t_reg = t_term[-nca_params.n_points_lambda :]
                c_reg = c_term[-nca_params.n_points_lambda :]
                # Fitted line: C(t) = C0 * exp(-lambda_z * t)
                log_c_reg = np.log(c_reg)
                intercept = float(np.mean(log_c_reg + nca_params.lambda_z * t_reg))
                t_fit = np.linspace(float(t_reg[0]), float(t[-1]), 100)
                c_fit = np.exp(intercept - nca_params.lambda_z * t_fit)
                ax.plot(
                    t_fit,
                    c_fit,
                    color=_IBM_COLORS[4],
                    linewidth=1.2,
                    linestyle="--",
                    label=f"λz={nca_params.lambda_z:.3g} (R²={nca_params.r_squared:.3f})",
                )

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title(title or f"Subject {nca_params.subject_id}")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return fig


def dose_proportionality_plot(
    nca_df: pd.DataFrame,
    *,
    metric: str = "auc_inf",
    dose_col: str = "dose",
    fit_power_model: bool = True,
    ax=None,
    title: str = "Dose Proportionality",
) -> Any:
    """
    Log-log plot of a PK exposure metric vs dose.

    A slope-1 reference line (proportional) and optionally a fitted power
    model are overlaid.

    Args:
        nca_df:          DataFrame with dose and exposure metric columns.
        metric:          NCA metric column to plot on y-axis (default ``auc_inf``).
        dose_col:        Column name for dose (default ``dose``).
        fit_power_model: Overlay power model fit (log Y = a + b*log(dose)).
        ax:              Existing axes.
        title:           Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    from scipy.stats import linregress

    df = nca_df[[dose_col, metric]].dropna()
    if len(df) < 2:
        raise ValueError(f"Insufficient data for dose proportionality plot (n={len(df)}).")

    doses = df[dose_col].values.astype(float)
    exposure = df[metric].values.astype(float)
    pos = (doses > 0) & (exposure > 0)
    doses = doses[pos]
    exposure = exposure[pos]

    log_d = np.log(doses)
    log_e = np.log(exposure)

    with _style():
        fig, ax = _make_fig_ax(ax)
        ax.scatter(
            doses,
            exposure,
            color=_IBM_COLORS[0],
            s=40,
            edgecolors="none",
            alpha=0.8,
            zorder=3,
            label="Observed",
        )

        # Slope-1 reference line through geometric mean at median dose
        med_dose = float(np.exp(np.median(log_d)))
        med_exp = float(np.exp(np.median(log_e)))
        d_range = np.array([float(doses.min()) * 0.8, float(doses.max()) * 1.2])
        ax.plot(
            d_range,
            med_exp * (d_range / med_dose),
            color="gray",
            linewidth=1.0,
            linestyle="--",
            label="Slope=1 (proportional)",
        )

        if fit_power_model and len(log_d) >= 3:
            slope, intercept, r, _, _ = linregress(log_d, log_e)
            d_fit = np.linspace(float(doses.min()) * 0.8, float(doses.max()) * 1.2, 100)
            e_fit = np.exp(intercept + slope * np.log(d_fit))
            ax.plot(
                d_fit,
                e_fit,
                color=_IBM_COLORS[2],
                linewidth=1.4,
                label=f"Power model: slope={slope:.2f} (R²={r**2:.3f})",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Dose ({dose_col})")
        ax.set_ylabel(_NCA_LABELS.get(metric, metric))
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig
