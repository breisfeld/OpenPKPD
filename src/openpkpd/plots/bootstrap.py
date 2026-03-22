"""
Bootstrap parameter distribution and confidence interval plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _style

if TYPE_CHECKING:
    from openpkpd.estimation.base import EstimationResult


def _extract_param_array(
    results: list[EstimationResult],
    param: str,
) -> np.ndarray:
    """
    Extract a 2-D array (n_replicates × n_params) from a list of EstimationResult.

    ``param`` must be one of ``"theta"``, ``"omega_diag"``, or ``"sigma_diag"``.
    """
    arrays = []
    for r in results:
        if param == "theta":
            arrays.append(np.asarray(r.theta_final).ravel())
        elif param == "omega_diag":
            arrays.append(np.diag(np.asarray(r.omega_final)))
        elif param == "sigma_diag":
            arrays.append(np.diag(np.asarray(r.sigma_final)))
        else:
            raise ValueError(f"param must be 'theta', 'omega_diag', or 'sigma_diag'; got {param!r}")
    return np.vstack(arrays)


def _param_labels(param: str, n: int) -> list[str]:
    prefix = {"theta": "THETA", "omega_diag": "OMEGA", "sigma_diag": "SIGMA"}[param]
    return [f"{prefix}({i + 1})" for i in range(n)]


def bootstrap_parameter_distributions(
    bootstrap_results: list[EstimationResult],
    final_result: EstimationResult,
    *,
    param: str = "theta",
    n_cols: int = 3,
    ci: float = 0.95,
    title: str = "Bootstrap Distributions",
) -> Any:
    """
    Grid of histograms of bootstrap parameter estimates.

    Each panel shows the bootstrap distribution for one parameter with:
    - The final model estimate as a vertical line
    - Shaded bootstrap CI region

    Args:
        bootstrap_results: List of ``EstimationResult`` from bootstrap replicates.
        final_result:      Original model ``EstimationResult``.
        param:             ``"theta"``, ``"omega_diag"``, or ``"sigma_diag"``.
        n_cols:            Grid columns.
        ci:                Confidence interval level (default 0.95).
        title:             Figure title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    samples = _extract_param_array(bootstrap_results, param)  # (n_boot, n_params)
    n_params = samples.shape[1]
    labels = _param_labels(param, n_params)
    final_vals = _extract_param_array([final_result], param)[0]

    alpha = (1 - ci) / 2
    ci_lo = np.nanpercentile(samples, 100 * alpha, axis=0)
    ci_hi = np.nanpercentile(samples, 100 * (1 - alpha), axis=0)

    n_rows = (n_params + n_cols - 1) // n_cols

    with _style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes_flat = np.array(axes).flatten()

        for k in range(n_params):
            ax = axes_flat[k]
            vals = samples[:, k]
            valid = vals[np.isfinite(vals)]
            if len(valid) == 0:
                ax.set_visible(False)
                continue

            n_bins = max(10, int(np.sqrt(len(valid))))
            ax.hist(
                valid,
                bins=n_bins,
                color=_IBM_COLORS[0],
                alpha=0.70,
                edgecolor="white",
                density=True,
            )

            # Shade CI region
            ax.axvspan(
                ci_lo[k], ci_hi[k], alpha=0.20, color=_IBM_COLORS[0], label=f"{int(ci * 100)}% CI"
            )

            # Final estimate
            ax.axvline(
                final_vals[k],
                color=_IBM_COLORS[2],
                linewidth=1.4,
                linestyle="--",
                label=f"Final={final_vals[k]:.3g}",
            )

            ax.set_xlabel(labels[k])
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        for k in range(n_params, len(axes_flat)):
            axes_flat[k].set_visible(False)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def bootstrap_ci_plot(
    bootstrap_results: list[EstimationResult],
    final_result: EstimationResult,
    *,
    param: str = "theta",
    ci: float = 0.95,
    title: str = "Bootstrap Confidence Intervals",
) -> Any:
    """
    Forest-plot style bootstrap CI per parameter.

    Each row shows the final model point estimate with the bootstrap CI
    as horizontal error bars.

    Args:
        bootstrap_results: List of ``EstimationResult`` from bootstrap replicates.
        final_result:      Original model ``EstimationResult``.
        param:             ``"theta"``, ``"omega_diag"``, or ``"sigma_diag"``.
        ci:                Confidence interval level (default 0.95).
        title:             Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    samples = _extract_param_array(bootstrap_results, param)
    n_params = samples.shape[1]
    labels = _param_labels(param, n_params)
    final_vals = _extract_param_array([final_result], param)[0]

    alpha = (1 - ci) / 2
    ci_lo = np.nanpercentile(samples, 100 * alpha, axis=0)
    ci_hi = np.nanpercentile(samples, 100 * (1 - alpha), axis=0)

    height = max(3.0, 0.5 * n_params + 1.5)

    with _style():
        fig, ax = plt.subplots(figsize=(10, height))
        y_pos = np.arange(n_params)

        ax.errorbar(
            final_vals,
            y_pos,
            xerr=[final_vals - ci_lo, ci_hi - final_vals],
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
        ax.set_xlabel("Parameter Estimate")
        ax.set_title(f"{title} ({int(ci * 100)}% CI, n={len(bootstrap_results)} replicates)")
        fig.tight_layout()
    return fig
