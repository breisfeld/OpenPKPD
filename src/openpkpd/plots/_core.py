"""
Style defaults and context manager used by all plot modules.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

# IBM colorblind-friendly palette
_IBM_COLORS = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

PYNONMEM_STYLE: dict = {
    "figure.figsize": (6, 5),
    "figure.dpi": 100,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}


def _check_matplotlib() -> None:
    """Raise ImportError with install hint if matplotlib is not available."""
    import importlib.util
    import sys

    if "matplotlib" in sys.modules:
        _mpl_found = sys.modules["matplotlib"] is not None
    else:
        try:
            _mpl_found = importlib.util.find_spec("matplotlib") is not None
        except (ValueError, ModuleNotFoundError):
            _mpl_found = False
    if not _mpl_found:
        raise ImportError(
            "matplotlib is required for plotting. Install it with:\n"
            "  uv pip install matplotlib\n"
            "or: uv sync --extra plots"
        )


@contextmanager
def _style() -> Generator[None, None, None]:
    """Apply openpkpd style with colorblind-friendly palette."""
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from cycler import cycler

    style = dict(PYNONMEM_STYLE)
    style["axes.prop_cycle"] = cycler(color=_IBM_COLORS)
    with plt.rc_context(style):
        yield


def _make_fig_ax(ax=None, figsize=None):
    """Create or reuse an axes object. Returns (fig, ax)."""
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _identity_line(ax, data_min: float, data_max: float) -> None:
    """Draw a unity (y=x) reference line."""
    lo = min(data_min, data_min)
    hi = max(data_max, data_max)
    margin = (hi - lo) * 0.05
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        "k--",
        linewidth=1.0,
        zorder=1,
        label="Identity",
    )


def _zero_bands(ax, x_min: float, x_max: float) -> None:
    """Draw zero line and ±2 horizontal bands."""
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.axhline(2, color="gray", linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(-2, color="gray", linewidth=0.8, linestyle=":", zorder=1)
