"""
Covariate analysis plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openpkpd.plots._core import _IBM_COLORS, _check_matplotlib, _style


def covariate_forest_plot(
    effects: list[dict[str, Any]],
    *,
    ref_line: float = 1.0,
    zone_inner: float = 0.20,
    zone_outer: float = 0.50,
    show_zones: bool = True,
    figsize: tuple[float, float] = (10, 0),
    title: str = "Covariate Effects",
) -> Any:
    """
    Forest plot of covariate effects expressed as fold-change on PK/PD parameters.

    Each row represents one covariate-parameter pair with a point estimate
    (fold-change) and optional confidence interval. A vertical reference line
    at ``ref_line`` (default 1.0 = no effect) is included. Optional shaded
    zones at ±20% and ±50% highlight clinically relevant thresholds.

    Args:
        effects:    List of dicts, each with keys:
                      - ``label`` (str): row label, e.g. "WT on CL (5th–95th pct)"
                      - ``fold_change`` (float): point estimate
                      - ``ci_lo`` (float, optional): lower CI bound
                      - ``ci_hi`` (float, optional): upper CI bound
        ref_line:   x-position of the "no effect" reference line (default 1.0).
        zone_inner: Half-width of the inner (±20%) shaded zone, as a fraction
                    of ref_line.  Set to 0 to hide.
        zone_outer: Half-width of the outer (±50%) shaded zone, as a fraction
                    of ref_line.  Set to 0 to hide.
        show_zones: Whether to draw the ±zone shaded regions.
        figsize:    Figure size (width, height). Height 0 = auto-size by n rows.
        title:      Plot title.

    Returns:
        matplotlib.figure.Figure.

    Example::

        effects = [
            {"label": "WT on CL (5th–95th pct)", "fold_change": 0.72,
             "ci_lo": 0.61, "ci_hi": 0.85},
            {"label": "AGE on V (5th–95th pct)", "fold_change": 1.10,
             "ci_lo": 0.95, "ci_hi": 1.28},
            {"label": "SEX (F vs M) on CL", "fold_change": 0.88},
        ]
        fig = covariate_forest_plot(effects)
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    n = len(effects)
    if n == 0:
        raise ValueError("effects list is empty.")

    height = figsize[1] if figsize[1] > 0 else max(3.0, 0.55 * n + 1.5)
    fig_size = (figsize[0], height)

    with _style():
        fig, ax = plt.subplots(figsize=fig_size)

        y_pos = np.arange(n)[::-1]  # top-to-bottom order

        # Shaded zones relative to ref_line
        if show_zones and zone_outer > 0:
            ax.axvspan(
                ref_line * (1 - zone_outer), ref_line * (1 + zone_outer), color="#EEEEEE", zorder=0
            )
        if show_zones and zone_inner > 0:
            ax.axvspan(
                ref_line * (1 - zone_inner), ref_line * (1 + zone_inner), color="#DDDDDD", zorder=0
            )

        # Reference line
        ax.axvline(ref_line, color="black", linewidth=1.0, linestyle="--", zorder=1)

        for _i, (eff, y) in enumerate(zip(effects, y_pos, strict=False)):
            fc = float(eff["fold_change"])
            ci_lo = eff.get("ci_lo")
            ci_hi = eff.get("ci_hi")

            color = _IBM_COLORS[0]

            if ci_lo is not None and ci_hi is not None:
                xerr_lo = fc - float(ci_lo)
                xerr_hi = float(ci_hi) - fc
                ax.errorbar(
                    fc,
                    y,
                    xerr=[[xerr_lo], [xerr_hi]],
                    fmt="o",
                    color=color,
                    markersize=7,
                    capsize=4,
                    elinewidth=1.2,
                    capthick=1.2,
                    zorder=3,
                )
                ax.text(
                    float(ci_hi) + 0.01 * ref_line,
                    y,
                    f"{fc:.2f} [{float(ci_lo):.2f}, {float(ci_hi):.2f}]",
                    va="center",
                    fontsize=8,
                    color="#333333",
                )
            else:
                ax.plot(fc, y, "o", color=color, markersize=7, zorder=3)
                ax.text(
                    fc + 0.01 * ref_line, y, f"{fc:.2f}", va="center", fontsize=8, color="#333333"
                )

        labels = [eff["label"] for eff in effects]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1][::-1])  # same order as y_pos
        ax.set_xlabel("Fold-Change vs Reference")
        ax.set_title(title)

        # Axis ticks symmetric around ref_line
        all_vals = [float(e["fold_change"]) for e in effects]
        for e in effects:
            if "ci_lo" in e and e["ci_lo"] is not None:
                all_vals.append(float(e["ci_lo"]))
            if "ci_hi" in e and e["ci_hi"] is not None:
                all_vals.append(float(e["ci_hi"]))
        margin = max(abs(v - ref_line) for v in all_vals) * 1.25
        ax.set_xlim(ref_line - margin, ref_line + margin)

        fig.tight_layout()
    return fig
