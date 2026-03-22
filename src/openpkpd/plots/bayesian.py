"""
Bayesian (NUTS) MCMC diagnostic and summary plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _make_fig_ax, _style


def _parse_samples(
    samples: np.ndarray | dict[str, np.ndarray],
    param_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Return (samples_2d, labels) from either an array or a name→samples dict."""
    if isinstance(samples, dict):
        keys = list(samples.keys())
        arr = np.column_stack(
            [
                np.asarray(samples[k]).ravel()
                if np.asarray(samples[k]).ndim == 1
                else np.asarray(samples[k])
                for k in keys
            ]
        )
        labels = param_names or keys
    else:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        n = arr.shape[1]
        labels = param_names or [f"param_{i + 1}" for i in range(n)]
    return arr, labels


def mcmc_trace_plot(
    samples: np.ndarray | dict[str, np.ndarray],
    *,
    param_names: list[str] | None = None,
    n_cols: int = 2,
    burnin: int = 0,
    title: str = "MCMC Traces",
) -> Any:
    """
    Trace plots showing sampled parameter values vs iteration.

    Args:
        samples:      2-D array (n_samples × n_params) or dict of {name: samples}.
        param_names:  Override parameter labels.
        n_cols:       Number of columns in the grid.
        burnin:       Number of burn-in iterations to shade (shown in lighter color).
        title:        Figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    arr, labels = _parse_samples(samples, param_names)
    n_samples, n_params = arr.shape
    n_rows = (n_params + n_cols - 1) // n_cols

    with _style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 2.5 * n_rows))
        axes_flat = np.array(axes).flatten()

        iters = np.arange(n_samples)

        for k in range(n_params):
            ax = axes_flat[k]
            vals = arr[:, k]
            if burnin > 0:
                ax.plot(iters[:burnin], vals[:burnin], color="lightgray", linewidth=0.6, zorder=1)
                ax.axvspan(0, burnin, alpha=0.10, color="gray", zorder=0, label="Burn-in")
            ax.plot(iters[burnin:], vals[burnin:], color=_IBM_COLORS[0], linewidth=0.6, zorder=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(labels[k])
            ax.set_title(labels[k])
            if burnin > 0:
                ax.legend(fontsize=7, loc="upper right")

        for k in range(n_params, len(axes_flat)):
            axes_flat[k].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def posterior_density_plot(
    samples: np.ndarray | dict[str, np.ndarray],
    *,
    param_names: list[str] | None = None,
    point_estimate: np.ndarray | None = None,
    ci: float = 0.95,
    n_cols: int = 3,
    title: str = "Posterior Densities",
) -> Any:
    """
    KDE posterior density per parameter with credible interval shading.

    Args:
        samples:         2-D array or dict of MCMC samples.
        param_names:     Parameter labels.
        point_estimate:  1-D array of point estimates (MAP or posterior mean)
                         to mark with a vertical line.
        ci:              Credible interval level to shade (default 0.95).
        n_cols:          Grid columns.
        title:           Figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    arr, labels = _parse_samples(samples, param_names)
    n_params = arr.shape[1]
    n_rows = (n_params + n_cols - 1) // n_cols
    alpha = (1 - ci) / 2

    with _style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes_flat = np.array(axes).flatten()

        for k in range(n_params):
            ax = axes_flat[k]
            vals = arr[:, k]
            valid = vals[np.isfinite(vals)]
            if len(valid) < 3:
                ax.set_visible(False)
                continue

            ci_lo = float(np.percentile(valid, 100 * alpha))
            ci_hi = float(np.percentile(valid, 100 * (1 - alpha)))

            kde = gaussian_kde(valid)
            x_grid = np.linspace(float(valid.min()), float(valid.max()), 200)
            density = kde(x_grid)

            ax.plot(x_grid, density, color=_IBM_COLORS[0], linewidth=1.8)
            # Shade CI
            ci_mask = (x_grid >= ci_lo) & (x_grid <= ci_hi)
            ax.fill_between(
                x_grid,
                density,
                where=ci_mask,
                alpha=0.25,
                color=_IBM_COLORS[0],
                label=f"{int(ci * 100)}% CI",
            )

            if point_estimate is not None and k < len(point_estimate):
                ax.axvline(
                    float(point_estimate[k]),
                    color=_IBM_COLORS[2],
                    linewidth=1.4,
                    linestyle="--",
                    label=f"Est={float(point_estimate[k]):.3g}",
                )

            ax.set_xlabel(labels[k])
            ax.set_ylabel("Density")
            ax.set_title(labels[k])
            ax.legend(fontsize=7)

        for k in range(n_params, len(axes_flat)):
            axes_flat[k].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def posterior_forest_plot(
    samples: np.ndarray | dict[str, np.ndarray],
    *,
    param_names: list[str] | None = None,
    ci: float = 0.95,
    title: str = "Posterior Credible Intervals",
) -> Any:
    """
    Forest plot of posterior medians with credible interval error bars.

    Args:
        samples:     2-D array or dict of MCMC samples.
        param_names: Parameter labels.
        ci:          Credible interval level (default 0.95).
        title:       Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()

    arr, labels = _parse_samples(samples, param_names)
    n_params = arr.shape[1]
    alpha = (1 - ci) / 2

    medians = np.nanmedian(arr, axis=0)
    ci_lo = np.nanpercentile(arr, 100 * alpha, axis=0)
    ci_hi = np.nanpercentile(arr, 100 * (1 - alpha), axis=0)

    height = max(3.0, 0.5 * n_params + 1.5)

    with _style():
        fig, ax = _make_fig_ax(None, figsize=(10, height))
        y_pos = np.arange(n_params)

        ax.errorbar(
            medians,
            y_pos,
            xerr=[medians - ci_lo, ci_hi - medians],
            fmt="o",
            color=_IBM_COLORS[0],
            markersize=7,
            capsize=4,
            elinewidth=1.2,
            capthick=1.2,
        )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Parameter Value")
        ax.set_title(f"{title} ({int(ci * 100)}% CrI)")
        fig.tight_layout()
    return fig
