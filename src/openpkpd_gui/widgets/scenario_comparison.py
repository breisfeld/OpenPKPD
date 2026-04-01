"""Side-by-side scenario comparison table widget (P1-D).

``build_scenario_comparison_widget`` creates a ``QTableWidget`` showing key fit
metrics (OFV, convergence, method, run/output counts) for all sibling scenarios
in the active project, suitable for quick model comparison.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openpkpd_gui.domain.workspace import Workspace

# ---------------------------------------------------------------------------
# OFV extraction from summary_text
# ---------------------------------------------------------------------------

_OFV_RE = re.compile(r"OFV=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_CONVERGED_RE = re.compile(r"converged=(True|False)")
_METHOD_RE = re.compile(r"•\s*([A-Z0-9_]+)\s*•\s*converged=")


def _parse_summary(summary_text: str) -> dict[str, str]:
    """Extract OFV, convergence, and method from a fit run's summary_text."""
    result: dict[str, str] = {"ofv": "—", "converged": "—", "method": "—"}
    m = _OFV_RE.search(summary_text)
    if m:
        result["ofv"] = m.group(1)
    m = _CONVERGED_RE.search(summary_text)
    if m:
        result["converged"] = "Yes" if m.group(1) == "True" else "No"
    m = _METHOD_RE.search(summary_text)
    if m:
        result["method"] = m.group(1)
    return result


# ---------------------------------------------------------------------------
# Row data builder
# ---------------------------------------------------------------------------

def build_comparison_rows(workspace: "Workspace") -> list[dict[str, str]]:
    """Return one row dict per sibling scenario (excluding the current scenario).

    Each row contains:
        name, relation, ofv, converged, method, runs, fit_runs, artifacts
    Rows are sorted by OFV ascending (best first); scenarios without a
    successful fit are sorted last.
    """
    from openpkpd_gui.domain.run_record import RunStatus

    project = workspace.active_project
    current = project.active_scenario

    rows: list[dict[str, str]] = []
    for scenario in project.scenarios:
        if scenario.scenario_id == current.scenario_id:
            continue
        fit_runs = [run for run in scenario.runs if run.workflow == "fit"]
        successful_fits = [run for run in fit_runs if run.status == RunStatus.SUCCEEDED]
        latest_fit = successful_fits[-1] if successful_fits else (fit_runs[-1] if fit_runs else None)

        if latest_fit is not None and latest_fit.status == RunStatus.SUCCEEDED:
            parsed = _parse_summary(latest_fit.summary_text)
        else:
            parsed = {"ofv": "—", "converged": "—", "method": "—"}

        relation = "sibling"
        if scenario.parent_scenario_id == current.scenario_id:
            relation = "child"
        elif current.parent_scenario_id == scenario.scenario_id:
            relation = "parent"
        elif scenario.parent_scenario_id:
            relation = "branched"

        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "relation": relation,
                "ofv": parsed["ofv"],
                "converged": parsed["converged"],
                "method": parsed["method"],
                "fit_runs": str(len(fit_runs)),
                "runs": str(len(scenario.runs)),
                "artifacts": str(len(scenario.artifacts)),
            }
        )

    def _sort_key(row: dict[str, str]) -> tuple[int, float, str]:
        try:
            ofv_val = float(row["ofv"])
            return (0, ofv_val, row["name"])
        except ValueError:
            return (1, 0.0, row["name"])

    rows.sort(key=_sort_key)
    return rows


# ---------------------------------------------------------------------------
# Qt widget
# ---------------------------------------------------------------------------

_COLUMNS = [
    ("Scenario", "name"),
    ("Relation", "relation"),
    ("OFV", "ofv"),
    ("Converged", "converged"),
    ("Method", "method"),
    ("Fit runs", "fit_runs"),
    ("Outputs", "artifacts"),
]

_CONVERGED_YES_COLOR = "#dcfce7"   # light green
_CONVERGED_NO_COLOR = "#fee2e2"    # light red


def build_scenario_comparison_widget(workspace: "Workspace", qt_modules):
    """Return ``(widget, refresh)`` for the scenario comparison table.

    Parameters
    ----------
    workspace:
        The active ``Workspace`` instance.
    qt_modules:
        ``(QtCore, QtGui, QtWidgets)`` tuple from ``load_qt_modules()``.

    Returns
    -------
    widget:
        A ``QGroupBox`` containing the comparison table.
    refresh:
        ``(workspace: Workspace) -> None`` — call after the workspace changes.
    """
    qt_core, qt_gui, qt_widgets = qt_modules

    group = qt_widgets.QGroupBox("Scenario comparison")
    group.setObjectName("results-scenario-comparison-group")
    group.setToolTip("Side-by-side fit metrics for sibling scenarios in this project.")
    group_layout = qt_widgets.QVBoxLayout(group)
    group_layout.setSpacing(4)
    group_layout.setContentsMargins(8, 8, 8, 8)

    desc = qt_widgets.QLabel(
        "Sibling scenarios in the current project — sorted by OFV (best first)."
    )
    desc.setObjectName("results-scenario-comparison-desc")
    desc.setWordWrap(True)
    group_layout.addWidget(desc)

    no_peers_label = qt_widgets.QLabel(
        "No sibling scenarios yet. Duplicate or branch this scenario to compare models."
    )
    no_peers_label.setObjectName("results-scenario-comparison-empty")
    no_peers_label.setWordWrap(True)
    no_peers_label.setVisible(False)
    group_layout.addWidget(no_peers_label)

    table = qt_widgets.QTableWidget()
    table.setObjectName("results-scenario-comparison-table")
    table.setColumnCount(len(_COLUMNS))
    table.setHorizontalHeaderLabels([col[0] for col in _COLUMNS])
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(qt_widgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(qt_widgets.QAbstractItemView.SelectRows)
    hdr = table.horizontalHeader()
    hdr.setSectionResizeMode(0, qt_widgets.QHeaderView.ResizeMode.Stretch)
    for i in range(1, len(_COLUMNS)):
        hdr.setSectionResizeMode(i, qt_widgets.QHeaderView.ResizeMode.ResizeToContents)
    group_layout.addWidget(table)

    def _populate(ws: "Workspace") -> None:
        rows = build_comparison_rows(ws)
        has_rows = bool(rows)
        no_peers_label.setVisible(not has_rows)
        table.setVisible(has_rows)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, (_, key) in enumerate(_COLUMNS):
                item = qt_widgets.QTableWidgetItem(row[key])
                item.setFlags(item.flags() & ~qt_core.Qt.ItemFlag.ItemIsEditable)
                # Highlight convergence column
                if key == "converged":
                    if row["converged"] == "Yes":
                        item.setBackground(qt_gui.QColor(_CONVERGED_YES_COLOR))
                    elif row["converged"] == "No":
                        item.setBackground(qt_gui.QColor(_CONVERGED_NO_COLOR))
                # Store scenario_id in UserRole for programmatic access
                if key == "name":
                    item.setData(qt_core.Qt.ItemDataRole.UserRole, row["scenario_id"])
                table.setItem(r, c, item)
        table.resizeRowsToContents()

    def refresh(ws: "Workspace") -> None:
        _populate(ws)

    _populate(workspace)
    return group, refresh
