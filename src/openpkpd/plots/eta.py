"""
ETA diagnostic plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _make_fig_ax, _style


def _get_eta_cols(diag_df: pd.DataFrame) -> list[str]:
    """Return sorted list of ETAn columns present in the DataFrame."""
    cols = sorted(
        [c for c in diag_df.columns if c.startswith("ETA") and c[3:].isdigit()],
        key=lambda c: int(c[3:]),
    )
    return cols


def eta_histograms(
    diag_df: pd.DataFrame,
    omega: np.ndarray,
    *,
    n_cols: int = 3,
    bins: int = 20,
    overlay_normal: bool = True,
    title: str = "ETA Distributions",
):
    """
    Histogram of each ETA with optional N(0, omega_kk) overlay.

    Args:
        diag_df:        Diagnostic DataFrame with ETAn columns.
        omega:          Final Omega matrix (n_eta × n_eta).
        n_cols:         Number of columns in the grid.
        bins:           Number of histogram bins.
        overlay_normal: Overlay N(0, omega_kk) density curve.
        title:          Figure title.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    eta_cols = _get_eta_cols(diag_df)
    if not eta_cols:
        raise ValueError(
            "No ETAn columns found in diag_df. "
            "Run compute_diagnostics() with a FOCE result that has post_hoc_etas."
        )

    # Get per-subject ETAs (deduplicated)
    eta_df = diag_df.groupby("ID")[eta_cols].first().reset_index()

    n = len(eta_cols)
    n_rows = (n + n_cols - 1) // n_cols

    with _style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes_flat = np.array(axes).flatten()

        for k, col in enumerate(eta_cols):
            ax = axes_flat[k]
            vals = eta_df[col].dropna().values
            ax.hist(
                vals, bins=bins, density=True, color=_IBM_COLORS[0], alpha=0.7, edgecolor="white"
            )

            if overlay_normal and k < omega.shape[0]:
                var_kk = float(omega[k, k])
                if var_kk > 0:
                    sd = np.sqrt(var_kk)
                    x = np.linspace(vals.min() - 2 * sd, vals.max() + 2 * sd, 200)
                    ax.plot(
                        x,
                        norm.pdf(x, 0, sd),
                        color=_IBM_COLORS[2],
                        linewidth=1.5,
                        label=f"N(0,{var_kk:.3f})",
                    )
                    ax.legend(fontsize=8)

            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        # Hide extra panels
        for k in range(n, len(axes_flat)):
            axes_flat[k].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def eta_pairs(
    diag_df: pd.DataFrame,
    *,
    alpha: float = 0.6,
    title: str = "ETA Pairs",
):
    """
    Pairs scatter matrix of all ETA columns.

    Args:
        diag_df: Diagnostic DataFrame with ETAn columns.
        alpha:   Scatter point transparency.
        title:   Figure title.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    eta_cols = _get_eta_cols(diag_df)
    if not eta_cols:
        raise ValueError("No ETAn columns found in diag_df.")

    eta_df = diag_df.groupby("ID")[eta_cols].first().reset_index()
    n = len(eta_cols)

    with _style():
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
        if n == 1:
            axes = np.array([[axes]])

        for i, col_i in enumerate(eta_cols):
            for j, col_j in enumerate(eta_cols):
                ax = axes[i, j]
                if i == j:
                    ax.hist(
                        eta_df[col_i].dropna().values,
                        bins=15,
                        color=_IBM_COLORS[0],
                        alpha=0.7,
                        edgecolor="white",
                    )
                    ax.set_xlabel(col_i if i == n - 1 else "")
                else:
                    ax.scatter(
                        eta_df[col_j],
                        eta_df[col_i],
                        s=20,
                        alpha=alpha,
                        color=_IBM_COLORS[0],
                        edgecolors="none",
                    )
                    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
                    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
                if j == 0:
                    ax.set_ylabel(col_i)
                if i == n - 1:
                    ax.set_xlabel(col_j)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def eta_shrinkage_plot(
    result: Any,
    *,
    eta_labels: list[str] | None = None,
    ax=None,
    title: str = "ETA Shrinkage",
):
    """
    Bar chart of ETA shrinkage (%) per random effect with 30% warning threshold.

    Args:
        result:     EstimationResult with eta_shrinkage array.
        eta_labels: Override axis labels (default: ETA1, ETA2, ...).
        ax:         Existing axes.
        title:      Plot title.
    """
    _check_matplotlib()

    shrinkage = np.asarray(result.eta_shrinkage) * 100.0  # convert to %
    n = len(shrinkage)
    labels = eta_labels or [f"ETA{i + 1}" for i in range(n)]

    with _style():
        fig, ax = _make_fig_ax(ax)
        y_pos = np.arange(n)
        colors = [_IBM_COLORS[2] if s > 30 else _IBM_COLORS[0] for s in shrinkage]
        ax.barh(y_pos, shrinkage, color=colors, alpha=0.8, height=0.6)
        ax.axvline(30, color="gray", linewidth=1.0, linestyle="--", label="30% threshold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Shrinkage (%)")
        ax.set_xlim(0, max(100, float(np.nanmax(shrinkage)) * 1.1))
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig


def eps_shrinkage_plot(
    result: Any,
    *,
    eps_labels: list[str] | None = None,
    ax=None,
    title: str = "EPS Shrinkage",
):
    """
    Bar chart of EPS (residual) shrinkage (%) per sigma term.

    Args:
        result:     EstimationResult with eps_shrinkage array.
        eps_labels: Override axis labels (default: EPS1, EPS2, ...).
        ax:         Existing axes.
        title:      Plot title.
    """
    _check_matplotlib()

    shrinkage = np.asarray(result.eps_shrinkage) * 100.0
    n = len(shrinkage)
    labels = eps_labels or [f"EPS{i + 1}" for i in range(n)]

    with _style():
        fig, ax = _make_fig_ax(ax)
        y_pos = np.arange(n)
        colors = [_IBM_COLORS[2] if s > 30 else _IBM_COLORS[0] for s in shrinkage]
        ax.barh(y_pos, shrinkage, color=colors, alpha=0.8, height=0.6)
        ax.axvline(30, color="gray", linewidth=1.0, linestyle="--", label="30% threshold")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Shrinkage (%)")
        ax.set_xlim(0, max(100, float(np.nanmax(shrinkage)) * 1.1))
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig


def omega_heatmap(
    result: Any,
    *,
    eta_labels: list[str] | None = None,
    ax=None,
    title: str = "Omega Correlation Matrix",
):
    """
    Heatmap of the IIV correlation matrix derived from omega_final.

    Converts the covariance matrix to a correlation matrix and displays
    it as a diverging heatmap (blue–white–red, range −1 to +1).

    Args:
        result:     EstimationResult with omega_final (n_eta × n_eta).
        eta_labels: Override axis labels (default: ETA1, ETA2, ...).
        ax:         Existing axes.
        title:      Plot title.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    omega = np.asarray(result.omega_final)
    n = omega.shape[0]
    labels = eta_labels or [f"ETA{i + 1}" for i in range(n)]

    # Covariance → correlation
    diag = np.sqrt(np.maximum(np.diag(omega), 1e-30))
    corr = omega / np.outer(diag, diag)
    corr = np.clip(corr, -1.0, 1.0)

    with _style():
        fig, ax = _make_fig_ax(ax, figsize=(max(4, n), max(3.5, n - 0.5)))
        cmap = plt.colormaps["RdBu_r"]
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(n):
            for j in range(n):
                val = corr[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

        ax.set_title(title)
        fig.tight_layout()
    return fig


def iiv_cv_plot(
    result: Any,
    *,
    eta_labels: list[str] | None = None,
    ax=None,
    title: str = "IIV (% CV)",
):
    """
    Horizontal bar chart of inter-individual variability expressed as % CV.

    % CV = sqrt(exp(omega_kk) - 1) * 100  (log-normal ETA assumption).

    Args:
        result:     EstimationResult with omega_final.
        eta_labels: Override axis labels (default: ETA1, ETA2, ...).
        ax:         Existing axes.
        title:      Plot title.
    """
    _check_matplotlib()

    omega = np.asarray(result.omega_final)
    n = omega.shape[0]
    labels = eta_labels or [f"ETA{i + 1}" for i in range(n)]
    cv_pct = np.sqrt(np.maximum(np.exp(np.diag(omega)) - 1.0, 0.0)) * 100.0

    with _style():
        fig, ax = _make_fig_ax(ax)
        y_pos = np.arange(n)
        ax.barh(y_pos, cv_pct, color=_IBM_COLORS[0], alpha=0.8, height=0.6)
        for i, v in enumerate(cv_pct):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("% CV")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def eta_vs_covariate(
    diag_df: pd.DataFrame,
    covariate: str,
    eta_col: str,
    *,
    categorical: bool = False,
    ax=None,
    title: str | None = None,
):
    """
    ETA vs covariate scatter (continuous) or boxplot (categorical).

    Args:
        diag_df:     Diagnostic DataFrame.
        covariate:   Covariate column name.
        eta_col:     ETA column name (e.g. "ETA1").
        categorical: If True, produce a boxplot instead of scatter.
        ax:          Existing axes.
        title:       Plot title.
    """
    _check_matplotlib()

    t = title or f"{eta_col} vs {covariate}"
    eta_df = diag_df.groupby("ID")[[covariate, eta_col]].first().reset_index()

    with _style():
        fig, ax = _make_fig_ax(ax)

        if categorical:
            groups = sorted(eta_df[covariate].unique())
            data = [eta_df[eta_df[covariate] == g][eta_col].dropna().values for g in groups]
            ax.boxplot(data, labels=[str(g) for g in groups])
        else:
            ax.scatter(
                eta_df[covariate],
                eta_df[eta_col],
                alpha=0.7,
                color=_IBM_COLORS[0],
                edgecolors="none",
            )
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        ax.set_xlabel(covariate)
        ax.set_ylabel(eta_col)
        ax.set_title(t)
        fig.tight_layout()
    return fig
