"""Live OFV convergence plot widget for the Fit workflow (P1-A).

``build_convergence_plot_widget`` creates a ``QGroupBox`` containing a
matplotlib canvas that updates in real-time as OFV events arrive from the
background fit job.

Design
------
- Hidden by default; shown when a fit run starts.
- ``add_ofv_point(iteration, ofv)`` appends one data point and redraws
  (called from the Qt main thread after draining the event queue).
- ``reset()`` clears the plot for a new run.
- ``finalize(ofv_history)`` replaces the incremental points with the
  authoritative OFV history from the completed ``EstimationResult`` (covers
  any points that may have been missed or batched).
- Uses a thin matplotlib figure with a minimal visual footprint so it can
  sit above the log output without dominating the layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def build_convergence_plot_widget(qt_modules):
    """Return ``(widget, add_ofv_point, reset, finalize)`` for a live OFV plot.

    Parameters
    ----------
    qt_modules:
        ``(QtCore, QtGui, QtWidgets)`` tuple from ``load_qt_modules()``.

    Returns
    -------
    widget:
        A ``QGroupBox`` containing the matplotlib canvas.
        Hidden by default.
    add_ofv_point:
        ``(iteration: int, ofv: float) -> None`` — append one point and redraw.
    reset:
        ``() -> None`` — clear the plot for a new run.
    finalize:
        ``(ofv_history: list[float]) -> None`` — replace data with the
        authoritative history from the completed estimation result.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    qt_core, qt_gui, qt_widgets = qt_modules

    # --- GroupBox shell ---
    group = qt_widgets.QGroupBox("OFV convergence")
    group.setObjectName("fit-convergence-group")
    group_layout = qt_widgets.QVBoxLayout(group)
    group_layout.setContentsMargins(8, 8, 8, 4)
    group_layout.setSpacing(4)

    # --- Matplotlib figure ---
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=90)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("OFV", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    (line,) = ax.plot([], [], color="#2563eb", linewidth=1.4, antialiased=True)
    fig.tight_layout(pad=0.6)

    canvas = FigureCanvasQTAgg(fig)
    canvas.setObjectName("fit-convergence-canvas")
    canvas.setMinimumHeight(140)
    canvas.setMaximumHeight(220)
    group_layout.addWidget(canvas)

    # Status label (shows current OFV or final message)
    status_label = qt_widgets.QLabel("")
    status_label.setObjectName("fit-convergence-status")
    status_label.setAlignment(qt_core.Qt.AlignmentFlag.AlignRight)
    font = status_label.font()
    font.setPointSize(8)
    status_label.setFont(font)
    group_layout.addWidget(status_label)

    # Internal state
    _iters: list[int] = []
    _ofvs: list[float] = []

    def _redraw() -> None:
        if not _iters:
            return
        line.set_data(_iters, _ofvs)
        ax.set_xlim(0, max(_iters) + 1)
        _valid = [v for v in _ofvs if v < 1e6]
        if _valid:
            lo, hi = min(_valid), max(_valid)
            pad = max(abs(hi - lo) * 0.1, 1.0)
            ax.set_ylim(lo - pad, hi + pad)
        canvas.draw_idle()

    def add_ofv_point(iteration: int, ofv: float) -> None:
        """Append one OFV point and redraw the canvas."""
        _iters.append(iteration)
        _ofvs.append(ofv)
        if ofv < 1e6:
            status_label.setText(f"Iter {iteration} — OFV = {ofv:.4f}")
        _redraw()

    def reset() -> None:
        """Clear the plot for a new run."""
        _iters.clear()
        _ofvs.clear()
        line.set_data([], [])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        status_label.setText("")
        canvas.draw_idle()

    def finalize(ofv_history: list[float]) -> None:
        """Replace incremental data with the authoritative OFV history."""
        _iters.clear()
        _ofvs.clear()
        for i, ofv in enumerate(ofv_history, 1):
            _iters.append(i)
            _ofvs.append(ofv)
        if ofv_history:
            final_ofv = ofv_history[-1]
            status_label.setText(
                f"Converged — final OFV = {final_ofv:.4f}" if final_ofv < 1e6
                else "Fit complete"
            )
        _redraw()

    group.setVisible(False)
    return group, add_ofv_point, reset, finalize
