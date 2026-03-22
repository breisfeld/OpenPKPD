"""Shared helpers for Overview readiness cards."""

from __future__ import annotations

from dataclasses import dataclass

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.widgets.semantic_state import install_semantic_state_styles


@dataclass(frozen=True, slots=True)
class OverviewReadinessCardSpec:
    """Declarative configuration for one Overview readiness card."""

    key: str
    title: str
    icon_text: str


def build_overview_readiness_card(root, *, spec: OverviewReadinessCardSpec):
    """Build a readiness card with shared typography and object names."""
    _qt_core, _, qt_widgets = load_qt_modules()

    card = qt_widgets.QFrame(root)
    card.setObjectName(f"overview-readiness-card-{spec.key}")
    card.setFrameShape(qt_widgets.QFrame.Shape.StyledPanel)
    install_semantic_state_styles(card)

    layout = qt_widgets.QVBoxLayout(card)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(4)

    heading_row = qt_widgets.QHBoxLayout()
    heading_row.setContentsMargins(0, 0, 0, 0)
    heading_row.setSpacing(6)

    icon_label = qt_widgets.QLabel(spec.icon_text, card)
    icon_label.setObjectName(f"overview-readiness-{spec.key}-icon")
    icon_font = icon_label.font()
    icon_font.setPointSize(icon_font.pointSize() + 3)
    icon_font.setBold(True)
    icon_label.setFont(icon_font)

    heading_label = qt_widgets.QLabel(spec.title, card)
    heading_label.setObjectName(f"overview-readiness-{spec.key}-heading")
    heading_font = heading_label.font()
    heading_font.setBold(True)
    heading_label.setFont(heading_font)

    state_label = qt_widgets.QLabel(card)
    state_label.setObjectName(f"overview-readiness-{spec.key}-state")
    state_font = state_label.font()
    state_font.setPointSize(state_font.pointSize() + 1)
    state_font.setBold(True)
    state_label.setFont(state_font)

    summary_label = qt_widgets.QLabel(card)
    summary_label.setObjectName(f"overview-readiness-{spec.key}-summary")
    summary_label.setWordWrap(True)

    heading_row.addWidget(icon_label)
    heading_row.addWidget(heading_label)
    heading_row.addStretch(1)
    layout.addLayout(heading_row)
    layout.addWidget(state_label)
    layout.addWidget(summary_label)
    return card, state_label, summary_label
