"""Compartment model diagram widget (P1-E).

Provides ``ModelDiagramWidget`` — a small SVG view that updates in real-time
as the user selects different structural models in the builder.

Diagrams are inline SVG strings keyed by (advan, trans) tuples.  All SVGs
use a 220 × 120 viewport so the widget stays compact inside the model panel.

Colour palette (matches app theme):
  Box fill    #eff6ff  (light blue)
  Box stroke  #3b82f6  (blue-500)
  Arrow       #64748b  (slate-500)
  Label       #1e40af  (blue-800)
  Small label #64748b  (slate-500)
"""

from __future__ import annotations

W = 220   # SVG viewport width
H = 120   # SVG viewport height

# ---------------------------------------------------------------------------
# Low-level SVG primitives
# ---------------------------------------------------------------------------

_BOX_FILL = "#eff6ff"
_BOX_STROKE = "#3b82f6"
_ARROW_COLOR = "#64748b"
_LABEL_COLOR = "#1e40af"
_SMALL_COLOR = "#64748b"


def _box(x: float, y: float, w: float, h: float, label: str, sublabel: str = "") -> str:
    cx = x + w / 2
    cy = y + h / 2
    sub = ""
    if sublabel:
        sub = (
            f'<text x="{cx:.1f}" y="{cy + 9:.1f}" '
            f'font-size="8" fill="{_SMALL_COLOR}" text-anchor="middle">'
            f'{sublabel}</text>'
        )
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="6" fill="{_BOX_FILL}" stroke="{_BOX_STROKE}" stroke-width="1.5"/>'
        f'<text x="{cx:.1f}" y="{cy - (5 if sublabel else 0):.1f}" '
        f'font-size="10" fill="{_LABEL_COLOR}" text-anchor="middle" '
        f'font-weight="bold">{label}</text>'
        f'{sub}'
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> str:
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    lbl = ""
    if label:
        # Offset label slightly above the midpoint
        offset_y = -6 if y1 == y2 else 0
        offset_x = 6 if x1 == x2 else 0
        lbl = (
            f'<text x="{mx + offset_x:.1f}" y="{my + offset_y:.1f}" '
            f'font-size="8" fill="{_SMALL_COLOR}" text-anchor="middle">{label}</text>'
        )
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{_ARROW_COLOR}" stroke-width="1.5" '
        f'marker-end="url(#arr)"/>'
        f'{lbl}'
    )


def _double_arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> str:
    """Bidirectional arrow (both marker-start and marker-end)."""
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    lbl = ""
    if label:
        offset_y = -6 if y1 == y2 else 0
        lbl = (
            f'<text x="{mx:.1f}" y="{my + offset_y:.1f}" '
            f'font-size="8" fill="{_SMALL_COLOR}" text-anchor="middle">{label}</text>'
        )
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{_ARROW_COLOR}" stroke-width="1.5" '
        f'marker-start="url(#arr-rev)" marker-end="url(#arr)"/>'
        f'{lbl}'
    )


def _elim_arrow(x1: float, y1: float, label: str = "CL") -> str:
    """Short downward elimination arrow."""
    return _arrow(x1, y1, x1, y1 + 28, label)


_DEFS = (
    '<defs>'
    '<marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
    f'<path d="M0,0 L0,6 L8,3 z" fill="{_ARROW_COLOR}"/>'
    '</marker>'
    '<marker id="arr-rev" markerWidth="8" markerHeight="8" refX="2" refY="3" orient="auto-start-reverse">'
    f'<path d="M0,0 L0,6 L8,3 z" fill="{_ARROW_COLOR}"/>'
    '</marker>'
    '</defs>'
)


def _svg(body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
        f'{_DEFS}'
        f'<rect width="{W}" height="{H}" fill="white"/>'
        f'{body}'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Diagram builders
# ---------------------------------------------------------------------------

def _diag_1cmt_iv() -> str:
    """1-compartment IV bolus (ADVAN1)."""
    bw, bh = 90, 44
    bx, by = (W - bw) / 2, 20
    body = (
        _box(bx, by, bw, bh, "Central", "V, CL")
        + _elim_arrow(bx + bw / 2, by + bh, "CL")
    )
    return _svg(body)


def _diag_1cmt_oral() -> str:
    """1-compartment oral (ADVAN2)."""
    bw, bh = 70, 38
    gap = 30
    depot_x = 10
    central_x = depot_x + bw + gap
    y = 30
    body = (
        _box(depot_x, y, bw, bh, "Depot", "Ka")
        + _arrow(depot_x + bw, y + bh / 2, central_x, y + bh / 2, "Ka")
        + _box(central_x, y, bw, bh, "Central", "V, CL")
        + _elim_arrow(central_x + bw / 2, y + bh, "CL")
    )
    return _svg(body)


def _diag_1cmt_mm() -> str:
    """1-compartment IV, Michaelis-Menten elimination (ADVAN1/TRANS2)."""
    bw, bh = 90, 44
    bx, by = (W - bw) / 2, 20
    cx = bx + bw / 2
    body = (
        _box(bx, by, bw, bh, "Central", "V")
        + _arrow(cx, by + bh, cx, by + bh + 28, "Vm/Km")
    )
    return _svg(body)


def _diag_2cmt_iv() -> str:
    """2-compartment IV bolus (ADVAN3)."""
    bw, bh = 70, 38
    gap = 20
    c_x = 15
    p_x = c_x + bw + gap
    y = 25
    body = (
        _box(c_x, y, bw, bh, "Central", "V1, CL")
        + _double_arrow(c_x + bw, y + bh / 2, p_x, y + bh / 2, "Q")
        + _box(p_x, y, bw, bh, "Periph.", "V2")
        + _elim_arrow(c_x + bw / 2, y + bh, "CL")
    )
    return _svg(body)


def _diag_2cmt_oral() -> str:
    """2-compartment oral (ADVAN4)."""
    bw, bh = 58, 36
    gap_ab = 18
    gap_cp = 16
    d_x = 4
    c_x = d_x + bw + gap_ab
    p_x = c_x + bw + gap_cp
    y = 28
    body = (
        _box(d_x, y, bw, bh, "Depot", "Ka")
        + _arrow(d_x + bw, y + bh / 2, c_x, y + bh / 2, "Ka")
        + _box(c_x, y, bw, bh, "Central", "V2")
        + _double_arrow(c_x + bw, y + bh / 2, p_x, y + bh / 2, "Q")
        + _box(p_x, y, bw, bh, "Periph.", "V3")
        + _elim_arrow(c_x + bw / 2, y + bh, "CL")
    )
    return _svg(body)


def _diag_ncmt_general() -> str:
    """N-compartment general linear (ADVAN5)."""
    bw, bh = 90, 44
    bx, by = (W - bw) / 2, 20
    cx = bx + bw / 2
    body = (
        _box(bx, by, bw, bh, "N-cmt General", "Linear (ADVAN5)")
        + _arrow(cx, by + bh, cx, by + bh + 28, "k(n,0)")
    )
    return _svg(body)


def _diag_3cmt_iv() -> str:
    """3-compartment IV bolus (ADVAN11)."""
    bw, bh = 52, 34
    gap = 12
    c_x = 10
    p1_x = c_x + bw + gap
    p2_x = p1_x + bw + gap
    y = 20
    p_y = y + bh + 14  # second peripheral below
    body = (
        _box(c_x, y, bw, bh, "Central", "V1")
        + _double_arrow(c_x + bw, y + bh / 2, p1_x, y + bh / 2, "Q2")
        + _box(p1_x, y, bw, bh, "Periph.1", "V2")
        + _double_arrow(c_x + bw / 2, y + bh, c_x + bw / 2, p_y, "Q3")
        + _box(c_x, p_y, bw, bh, "Periph.2", "V3")
        + _elim_arrow(p2_x + bw / 2 - bw, y + bh / 2, "")
        + _arrow(c_x + bw / 2 + 52, y + bh / 2, c_x + bw / 2 + 52 + 20, y + bh / 2, "CL")
    )
    # Simpler layout: central top, two peripherals arranged horizontally
    c_x2 = (W - bw) / 2
    p1_x2 = 8
    p2_x2 = W - bw - 8
    body2 = (
        _box(c_x2, 8, bw, bh, "Central", "V1, CL")
        + _elim_arrow(c_x2 + bw / 2, 8 + bh, "CL")
        + _double_arrow(c_x2, 8 + bh / 2, p1_x2 + bw, 8 + bh / 2, "Q2")
        + _box(p1_x2, 8, bw, bh, "Periph.1", "V2")
        + _double_arrow(c_x2 + bw, 8 + bh / 2, p2_x2, 8 + bh / 2, "Q3")
        + _box(p2_x2, 8, bw, bh, "Periph.2", "V3")
    )
    return _svg(body2)


def _diag_3cmt_oral() -> str:
    """3-compartment oral (ADVAN12)."""
    bw, bh = 46, 32
    gap = 10
    d_x = 2
    c_x = d_x + bw + gap
    p1_x = 2
    p2_x = W - bw - 2
    y_top = 10
    y_bot = y_top + bh + 14
    body = (
        _box(d_x, y_top, bw, bh, "Depot", "Ka")
        + _arrow(d_x + bw, y_top + bh / 2, c_x, y_top + bh / 2, "Ka")
        + _box(c_x, y_top, bw, bh, "Central", "V2")
        + _elim_arrow(c_x + bw / 2, y_top + bh, "CL")
        + _double_arrow(c_x, y_top + bh / 2, p1_x + bw, y_top + bh / 2, "Q2")
        + _box(p1_x, y_bot, bw, bh, "Periph.1", "V3")
        + _double_arrow(c_x + bw, y_top + bh / 2, p2_x, y_top + bh / 2, "Q3")
        + _box(p2_x, y_top, bw, bh, "Periph.2", "V4")
    )
    # cleaner layout
    bw2, bh2 = 48, 32
    cx2 = (W - bw2) / 2
    body2 = (
        _box(2, 12, bw2, bh2, "Depot", "Ka")
        + _arrow(2 + bw2, 12 + bh2 / 2, cx2, 12 + bh2 / 2, "Ka")
        + _box(cx2, 12, bw2, bh2, "Central", "V2")
        + _elim_arrow(cx2 + bw2 / 2, 12 + bh2, "CL")
        + _double_arrow(cx2, 12 + bh2 / 2, 2 + bw2, 12 + bh2 / 2 + 30, "Q2")
        + _box(2, 12 + bh2 + 14, bw2, bh2, "Periph.1", "V3")
        + _double_arrow(cx2 + bw2, 12 + bh2 / 2, W - bw2 - 2, 12 + bh2 / 2, "Q3")
        + _box(W - bw2 - 2, 12, bw2, bh2, "Periph.2", "V4")
    )
    return _svg(body2)


def _diag_ode() -> str:
    """User-defined ODE (ADVAN6 / ADVAN8)."""
    bw, bh = 120, 50
    bx, by = (W - bw) / 2, (H - bh) / 2 - 8
    cx = bx + bw / 2
    body = (
        _box(bx, by, bw, bh, "User ODE", "ADVAN6/8 + $DES")
        + _arrow(cx, by + bh, cx, by + bh + 24, "dA/dt")
    )
    return _svg(body)


def _diag_custom() -> str:
    """Custom / unknown model."""
    bw, bh = 100, 50
    bx, by = (W - bw) / 2, (H - bh) / 2 - 8
    body = _box(bx, by, bw, bh, "Custom", "ADVAN / TRANS")
    return _svg(body)


# ---------------------------------------------------------------------------
# Registry: (advan, trans) → SVG string function
# ---------------------------------------------------------------------------

_DIAGRAM_MAP: dict[tuple[int, int], str] = {}


def _register() -> None:
    global _DIAGRAM_MAP
    _DIAGRAM_MAP = {
        (1, 1): _diag_1cmt_iv(),
        (2, 2): _diag_1cmt_oral(),
        (1, 2): _diag_1cmt_mm(),
        (3, 4): _diag_2cmt_iv(),
        (4, 4): _diag_2cmt_oral(),
        (5, 1): _diag_ncmt_general(),
        (11, 4): _diag_3cmt_iv(),
        (12, 4): _diag_3cmt_oral(),
        (6, 1): _diag_ode(),
        (8, 1): _diag_ode(),
    }


_register()


def get_diagram_svg(advan: int, trans: int) -> str:
    """Return the SVG string for the given ADVAN/TRANS combination.

    Falls back to the generic custom diagram for unrecognised combinations.
    """
    return _DIAGRAM_MAP.get((advan, trans), _diag_custom())


# ---------------------------------------------------------------------------
# Qt widget
# ---------------------------------------------------------------------------


def build_model_diagram_widget(advan: int = 2, trans: int = 2):
    """Return a ``QSvgWidget`` pre-loaded with the diagram for (advan, trans).

    The widget exposes an ``update_diagram(advan, trans)`` method so the model
    workflow can call it whenever the selection changes.
    """
    from PySide6 import QtSvgWidgets

    widget = QtSvgWidgets.QSvgWidget()
    widget.setObjectName("model-diagram-svg")
    widget.setFixedSize(W, H)
    widget.setToolTip(
        "Structural model diagram.\n"
        "Updates automatically when you change the model selection above."
    )

    def update_diagram(new_advan: int, new_trans: int) -> None:
        svg_bytes = get_diagram_svg(new_advan, new_trans).encode("utf-8")
        widget.load(svg_bytes)

    widget.update_diagram = update_diagram  # type: ignore[attr-defined]
    update_diagram(advan, trans)
    return widget
