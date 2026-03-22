"""
Goodness-of-fit diagnostic plots.

All functions accept an optional ``ax`` argument (reuse existing axes)
and return a ``matplotlib.figure.Figure``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpkpd.plots._core import (
    _check_matplotlib,
    _identity_line,
    _make_fig_ax,
    _style,
    _zero_bands,
)


def dv_vs_ipred(
    diag_df: pd.DataFrame,
    *,
    log_scale: bool = False,
    ax=None,
    title: str = "DV vs IPRED",
):
    """Scatter plot of DV vs IPRED with identity line."""
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        x = diag_df["IPRED"].values
        y = diag_df["DV"].values
        ax.scatter(x, y, alpha=0.6, edgecolors="none", zorder=2, label="Obs")
        _identity_line(ax, float(np.nanmin([x, y])), float(np.nanmax([x, y])))
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("IPRED")
        ax.set_ylabel("DV")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def dv_vs_pred(
    diag_df: pd.DataFrame,
    *,
    log_scale: bool = False,
    ax=None,
    title: str = "DV vs PRED",
):
    """Scatter plot of DV vs PRED with identity line."""
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        x = diag_df["PRED"].values
        y = diag_df["DV"].values
        ax.scatter(x, y, alpha=0.6, edgecolors="none", zorder=2, label="Obs")
        _identity_line(ax, float(np.nanmin([x, y])), float(np.nanmax([x, y])))
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("PRED")
        ax.set_ylabel("DV")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def cwres_vs_time(
    diag_df: pd.DataFrame,
    *,
    ax=None,
    title: str = "CWRES vs TIME",
):
    """CWRES scatter vs TIME with zero line and ±2 bands."""
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        x = diag_df["TIME"].values
        y = diag_df["CWRES"].values
        ax.scatter(x, y, alpha=0.6, edgecolors="none", zorder=2)
        _zero_bands(ax, float(np.nanmin(x)), float(np.nanmax(x)))
        ax.set_xlabel("TIME")
        ax.set_ylabel("CWRES")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def cwres_vs_pred(
    diag_df: pd.DataFrame,
    *,
    ax=None,
    title: str = "CWRES vs PRED",
):
    """CWRES scatter vs PRED with zero line and ±2 bands."""
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        x = diag_df["PRED"].values
        y = diag_df["CWRES"].values
        ax.scatter(x, y, alpha=0.6, edgecolors="none", zorder=2)
        _zero_bands(ax, float(np.nanmin(x)), float(np.nanmax(x)))
        ax.set_xlabel("PRED")
        ax.set_ylabel("CWRES")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def cwres_qq(
    diag_df: pd.DataFrame,
    *,
    ax=None,
    title: str = "CWRES Q-Q Plot",
):
    """Q-Q normal probability plot of CWRES."""
    _check_matplotlib()
    from scipy.stats import probplot

    with _style():
        fig, ax = _make_fig_ax(ax)
        cwres = diag_df["CWRES"].dropna().values
        (osm, osr), (slope, intercept, r) = probplot(cwres, dist="norm")
        ax.scatter(osm, osr, alpha=0.6, edgecolors="none", zorder=2)
        lo, hi = float(np.min(osm)), float(np.max(osm))
        ax.plot(
            [lo, hi],
            [slope * lo + intercept, slope * hi + intercept],
            "r--",
            linewidth=1.0,
            label="Normal",
        )
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def abs_iwres_vs_ipred(
    diag_df: pd.DataFrame,
    *,
    ax=None,
    title: str = "|IWRES| vs IPRED",
):
    """Absolute IWRES scatter versus IPRED."""
    _check_matplotlib()

    with _style():
        fig, ax = _make_fig_ax(ax)
        x = diag_df["IPRED"].values
        y = np.abs(diag_df["IWRES"].values)
        ax.scatter(x, y, alpha=0.6, edgecolors="none", zorder=2)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", label="|IWRES|=1")
        ax.set_xlabel("IPRED")
        ax.set_ylabel("|IWRES|")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def cwres_histogram(
    diag_df: pd.DataFrame,
    *,
    bins: int = 30,
    ax=None,
    title: str = "CWRES Histogram",
):
    """CWRES histogram with N(0,1) overlay and ±1.96 reference lines."""
    _check_matplotlib()
    from scipy.stats import norm

    with _style():
        fig, ax = _make_fig_ax(ax)
        cwres = diag_df["CWRES"].dropna().values
        ax.hist(
            cwres,
            bins=bins,
            density=True,
            color="#648FFF",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        x = np.linspace(float(np.nanmin(cwres)) - 0.5, float(np.nanmax(cwres)) + 0.5, 200)
        ax.plot(x, norm.pdf(x, 0, 1), "k-", linewidth=1.5, label="N(0,1)")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(1.96, color="gray", linewidth=0.8, linestyle=":")
        ax.axvline(-1.96, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("CWRES")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig


def diagnostic_panel(
    diag_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (18, 9),
    title: str = "Diagnostic Panel",
):
    """2×4 combined panel of all 7 GOF plots (includes CWRES histogram)."""
    _check_matplotlib()
    import matplotlib.pyplot as plt

    with _style():
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()

        dv_vs_ipred(diag_df, ax=axes[0], title="DV vs IPRED")
        dv_vs_pred(diag_df, ax=axes[1], title="DV vs PRED")
        cwres_vs_time(diag_df, ax=axes[2], title="CWRES vs TIME")
        cwres_vs_pred(diag_df, ax=axes[3], title="CWRES vs PRED")
        cwres_qq(diag_df, ax=axes[4], title="CWRES Q-Q")
        cwres_histogram(diag_df, ax=axes[5], title="CWRES Histogram")
        abs_iwres_vs_ipred(diag_df, ax=axes[6], title="|IWRES| vs IPRED")
        axes[7].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig
