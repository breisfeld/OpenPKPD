"""
Categorical, count, and Markov model diagnostic plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _make_fig_ax, _style


def categorical_probability_plot(
    x_values: np.ndarray,
    observed_categories: np.ndarray,
    predicted_probs: np.ndarray,
    *,
    n_categories: int | None = None,
    x_label: str = "Time",
    title: str = "Categorical Response",
) -> Any:
    """
    Observed proportions vs model-predicted probability curves per category.

    Args:
        x_values:            1-D array of time or concentration values (x-axis).
        observed_categories: 1-D integer array of observed category labels (0-based).
        predicted_probs:     2-D array (n_obs × n_categories) of predicted probabilities.
        n_categories:        Number of categories (inferred from predicted_probs if None).
        x_label:             x-axis label (default "Time").
        title:               Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    x = np.asarray(x_values, dtype=float)
    obs = np.asarray(observed_categories, dtype=int)
    probs = np.asarray(predicted_probs, dtype=float)
    n_cat = n_categories or probs.shape[1]

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    obs = obs[sort_idx]
    probs = probs[sort_idx]

    # Bin observed proportions by unique x values
    unique_x = np.unique(x)
    obs_proportions = np.zeros((len(unique_x), n_cat))
    for i, xv in enumerate(unique_x):
        mask = x == xv
        for k in range(n_cat):
            obs_proportions[i, k] = np.mean(obs[mask] == k)

    with _style():
        fig, ax = plt.subplots(figsize=(9, 5))

        colors = _IBM_COLORS[:n_cat] + ["#999999"] * max(0, n_cat - len(_IBM_COLORS))

        for k in range(n_cat):
            color = colors[k % len(colors)]
            # Observed binned proportions as step
            ax.step(
                unique_x,
                obs_proportions[:, k],
                where="mid",
                color=color,
                linewidth=1.0,
                linestyle=":",
                label=f"Obs Cat {k}",
            )
            # Predicted smooth curve
            ax.plot(x, probs[:, k], color=color, linewidth=1.6, label=f"Pred Cat {k}")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
    return fig


def cumulative_probability_plot(
    x_values: np.ndarray,
    observed_categories: np.ndarray,
    predicted_cum_probs: np.ndarray,
    *,
    x_label: str = "Time",
    title: str = "Cumulative Probability",
) -> Any:
    """
    P(Y ≤ k) cumulative probability curves vs observed cumulative proportions.

    Args:
        x_values:             1-D array of x values (time or concentration).
        observed_categories:  1-D integer observed category array (0-based).
        predicted_cum_probs:  2-D array (n_obs × n_thresholds) of cumulative
                              predicted probabilities P(Y ≤ k).
        x_label:              x-axis label.
        title:                Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()

    x = np.asarray(x_values, dtype=float)
    obs = np.asarray(observed_categories, dtype=int)
    cum_probs = np.asarray(predicted_cum_probs, dtype=float)
    n_thresh = cum_probs.shape[1]

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    obs = obs[sort_idx]
    cum_probs = cum_probs[sort_idx]

    unique_x = np.unique(x)
    obs_cum = np.zeros((len(unique_x), n_thresh))
    for i, xv in enumerate(unique_x):
        mask = x == xv
        for k in range(n_thresh):
            obs_cum[i, k] = np.mean(obs[mask] <= k)

    with _style():
        fig, ax = _make_fig_ax(None)
        colors = _IBM_COLORS[:n_thresh] + ["#999999"] * max(0, n_thresh - len(_IBM_COLORS))

        for k in range(n_thresh):
            color = colors[k % len(colors)]
            ax.step(
                unique_x,
                obs_cum[:, k],
                where="mid",
                color=color,
                linewidth=1.0,
                linestyle=":",
                label=f"Obs P(Y≤{k})",
            )
            ax.plot(x, cum_probs[:, k], color=color, linewidth=1.6, label=f"Pred P(Y≤{k})")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative Probability")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
    return fig


def count_frequency_plot(
    observed_counts: np.ndarray,
    predicted_mean: float,
    *,
    model: str = "poisson",
    overdispersion: float | None = None,
    max_count: int | None = None,
    ax=None,
    title: str = "Count Model Fit",
) -> Any:
    """
    Observed vs predicted frequency histogram for count models.

    Args:
        observed_counts:  1-D integer array of observed event counts per subject/interval.
        predicted_mean:   Predicted mean count (lambda for Poisson, mu for NB).
        model:            ``"poisson"`` or ``"negbinom"``.
        overdispersion:   Overdispersion parameter for negative binomial (r in NB).
                          Required when ``model="negbinom"``.
        max_count:        Truncate x-axis at this count (default: 99th percentile + 1).
        ax:               Existing axes.
        title:            Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    from scipy.stats import nbinom, poisson

    obs = np.asarray(observed_counts, dtype=int)
    if max_count is None:
        max_count = int(np.percentile(obs, 99)) + 2

    counts = np.arange(max_count + 1)
    obs_freq = np.array([np.mean(obs == k) for k in counts])

    if model == "poisson":
        pred_pmf = poisson.pmf(counts, mu=predicted_mean)
    elif model == "negbinom":
        if overdispersion is None:
            raise ValueError("overdispersion parameter required for 'negbinom' model.")
        r = overdispersion
        p = r / (r + predicted_mean)
        pred_pmf = nbinom.pmf(counts, r, p)
    else:
        raise ValueError(f"model must be 'poisson' or 'negbinom'; got {model!r}")

    with _style():
        fig, ax = _make_fig_ax(ax)
        width = 0.4
        ax.bar(
            counts - width / 2,
            obs_freq,
            width=width,
            color=_IBM_COLORS[0],
            alpha=0.75,
            label="Observed",
            edgecolor="white",
        )
        ax.bar(
            counts + width / 2,
            pred_pmf,
            width=width,
            color=_IBM_COLORS[2],
            alpha=0.75,
            label="Predicted",
            edgecolor="white",
        )
        ax.set_xlabel("Count")
        ax.set_ylabel("Proportion / Probability")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig


def markov_transition_heatmap(
    transition_matrix: np.ndarray,
    *,
    state_labels: list[str] | None = None,
    ax=None,
    title: str = "Transition Probabilities",
) -> Any:
    """
    Heatmap of a Markov state transition probability matrix.

    Args:
        transition_matrix: Square ndarray (n_states × n_states).
                           Rows represent "from" states; columns "to" states.
                           Row sums should be 1.
        state_labels:      Override default ``["State 0", "State 1", ...]`` labels.
        ax:                Existing axes.
        title:             Plot title.

    Returns:
        matplotlib.figure.Figure.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    P = np.asarray(transition_matrix, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("transition_matrix must be square.")
    n = P.shape[0]
    labels = state_labels or [f"State {i}" for i in range(n)]

    with _style():
        fig, ax = _make_fig_ax(ax, figsize=(max(4, n), max(3.5, n - 0.5)))
        im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Probability")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")

        for i in range(n):
            for j in range(n):
                val = P[i, j]
                text_color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

        ax.set_title(title)
        fig.tight_layout()
    return fig
