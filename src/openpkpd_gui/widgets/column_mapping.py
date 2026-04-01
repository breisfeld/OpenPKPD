"""Column mapping widget for the Data workflow (P1-B).

``build_column_mapping_widget`` creates a compact panel that lets users
remap arbitrary CSV column names to NONMEM standard names before loading.

Design
------
- Shown only when a CSV is imported whose headers don't already satisfy the
  required NONMEM columns (ID, TIME, DV).
- Displays one row per CSV column: [source name | NONMEM target dropdown].
- Required targets (ID, TIME, DV) are flagged; duplicates highlighted in red.
- "Auto-detect" heuristically fills targets from common aliases.
- Returns ``get_input_columns() -> list[str]`` — ordered NONMEM names, one
  per source column.  Columns kept as-is return their original name unchanged.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# NONMEM column constants (no Qt dependency here)
# ---------------------------------------------------------------------------

REQUIRED_NONMEM = ("ID", "TIME", "DV")
OPTIONAL_NONMEM = ("AMT", "RATE", "EVID", "MDV", "CMT", "ADDL", "II", "SS", "BLQ", "LLOQ")
ALL_NONMEM = REQUIRED_NONMEM + OPTIONAL_NONMEM

_KEEP_AS_IS = "(keep as-is)"

# ---------------------------------------------------------------------------
# Auto-detect heuristic
# ---------------------------------------------------------------------------

# Aliases: lower-case source pattern → NONMEM target
_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(subj(ect)?|patient|pat|pid|subject_?id)\b"), "ID"),
    (re.compile(r"\b(time|time_?h(r)?|t|tim|hours?|elapsed)\b"), "TIME"),
    (re.compile(r"\b(dv|conc(entration)?|obs(erved)?|y|response|cp)\b"), "DV"),
    (re.compile(r"\b(amt|amount|dose|dosing|dose_?mg|mg)\b"), "AMT"),
    (re.compile(r"\b(rate|infusion_?rate|inf_?rate)\b"), "RATE"),
    (re.compile(r"\b(evid|event_?id|event)\b"), "EVID"),
    (re.compile(r"\b(mdv|miss(ing)?_?dv|missing)\b"), "MDV"),
    (re.compile(r"\b(cmt|compartment|comp)\b"), "CMT"),
    (re.compile(r"\b(addl|additional|add_?dose)\b"), "ADDL"),
    (re.compile(r"\b(ii|inter_?dose|interdose|tau)\b"), "II"),
    (re.compile(r"\b(ss|steady_?state|stead_state)\b"), "SS"),
    (re.compile(r"\b(blq|below_?lloq|blloq|censored)\b"), "BLQ"),
    (re.compile(r"\b(lloq|lloq_?val|limit_?of_?quant)\b"), "LLOQ"),
]


def _normalize_col(col: str) -> str:
    """Normalize a column name for alias matching.

    Replaces underscores, hyphens, and other non-alphanumeric characters with
    spaces so that word-boundary patterns match across separator characters
    (e.g. ``PLASMA_CONC`` → ``PLASMA CONC``).
    """
    return re.sub(r"[^a-z0-9]", " ", col.strip().lower())


def auto_detect_mapping(csv_columns: list[str]) -> list[str]:
    """Return an ordered list of NONMEM target names for *csv_columns*.

    Each entry is either a NONMEM standard name (if a match was found) or
    the original source column name (keep-as-is).  Exact case-insensitive
    matches take priority; alias patterns are tried next.
    """
    # Track which NONMEM targets are already assigned to avoid duplicates.
    used: set[str] = set()
    results: list[str] = []

    nonmem_upper = {n.upper(): n for n in ALL_NONMEM}

    for col in csv_columns:
        target = col  # default: keep as-is

        # 1. Exact case-insensitive match against NONMEM standard names
        col_up = col.strip().upper()
        if col_up in nonmem_upper and nonmem_upper[col_up] not in used:
            target = nonmem_upper[col_up]
        else:
            # 2. Pattern/alias match against normalised form
            col_norm = _normalize_col(col)
            for pattern, nm_name in _ALIASES:
                if nm_name in used:
                    continue
                if pattern.search(col_norm):
                    target = nm_name
                    break

        if target in ALL_NONMEM:
            used.add(target)
        results.append(target)

    return results


def needs_mapping(csv_columns: list[str]) -> bool:
    """Return True when *csv_columns* doesn't already contain all required NONMEM names."""
    upper = {c.strip().upper() for c in csv_columns}
    return not all(req.upper() in upper for req in REQUIRED_NONMEM)


# ---------------------------------------------------------------------------
# Qt widget
# ---------------------------------------------------------------------------

_RED_STYLE = "background-color: #fee2e2; color: #991b1b;"
_AMBER_STYLE = "background-color: #fef9c3; color: #854d0e;"


def build_column_mapping_widget(csv_columns: list[str], qt_modules):
    """Return ``(widget, get_input_columns)`` for *csv_columns*.

    Parameters
    ----------
    csv_columns:
        Ordered list of column names as read from the CSV header row.
    qt_modules:
        ``(QtCore, QtGui, QtWidgets)`` tuple from ``load_qt_modules()``.

    Returns
    -------
    widget:
        A ``QGroupBox`` to insert into the data workflow layout.
        Hidden by default; call ``widget.setVisible(True)`` when needed.
    get_input_columns:
        Callable ``() -> list[str]`` that returns the current ordered mapping
        (one NONMEM name per source column, or original name if kept as-is).
    refresh:
        Callable ``(new_csv_columns: list[str]) -> None`` to reload the
        widget with a different set of source columns.
    """
    qt_core, qt_gui, qt_widgets = qt_modules

    # --- GroupBox shell ---
    group = qt_widgets.QGroupBox("Column mapping")
    group.setObjectName("data-column-mapping-group")
    group.setToolTip(
        "Your dataset columns don't match standard NONMEM names.\n"
        "Map each source column to the correct NONMEM name, then click "
        "'Apply mapping' to reload with the new names."
    )
    group_layout = qt_widgets.QVBoxLayout(group)
    group_layout.setSpacing(6)

    # --- Header description ---
    desc = qt_widgets.QLabel(
        "Some required NONMEM columns (ID, TIME, DV) were not found under "
        "their standard names. Map your source columns below, then click "
        "<b>Apply mapping</b>."
    )
    desc.setObjectName("data-column-mapping-desc")
    desc.setWordWrap(True)
    group_layout.addWidget(desc)

    # --- Auto-detect button + Apply button ---
    btn_row = qt_widgets.QHBoxLayout()
    auto_btn = qt_widgets.QPushButton("Auto-detect")
    auto_btn.setObjectName("data-column-mapping-auto-btn")
    auto_btn.setToolTip("Fill in likely NONMEM names using column name heuristics.")
    apply_btn = qt_widgets.QPushButton("Apply mapping")
    apply_btn.setObjectName("data-column-mapping-apply-btn")
    apply_btn.setProperty("primaryAction", True)
    apply_btn.setToolTip("Reload the dataset using the mapping defined below.")
    btn_row.addWidget(auto_btn)
    btn_row.addStretch(1)
    btn_row.addWidget(apply_btn)
    group_layout.addLayout(btn_row)

    # --- Mapping table ---
    table = qt_widgets.QTableWidget()
    table.setObjectName("data-column-mapping-table")
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["Source column", "NONMEM name", "Status"])
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(qt_widgets.QAbstractItemView.NoEditTriggers)
    table.horizontalHeader().setStretchLastSection(False)
    table.horizontalHeader().setSectionResizeMode(
        0, qt_widgets.QHeaderView.ResizeMode.Stretch
    )
    table.horizontalHeader().setSectionResizeMode(
        1, qt_widgets.QHeaderView.ResizeMode.Stretch
    )
    table.horizontalHeader().setSectionResizeMode(
        2, qt_widgets.QHeaderView.ResizeMode.ResizeToContents
    )
    group_layout.addWidget(table)

    # Track current columns and their combo widgets
    _state: dict[str, object] = {"columns": list(csv_columns)}
    _combos: list[qt_widgets.QComboBox] = []

    combo_options = [_KEEP_AS_IS] + list(ALL_NONMEM)

    def _status_for(target: str) -> str:
        if target in REQUIRED_NONMEM:
            return "Required ✓"
        if target in OPTIONAL_NONMEM:
            return "Optional"
        return ""

    def _refresh_status() -> None:
        """Highlight duplicate assignments and missing required columns."""
        seen: dict[str, int] = {}
        for row, combo in enumerate(_combos):
            target = combo.currentText()
            if target != _KEEP_AS_IS and target in ALL_NONMEM:
                seen[target] = seen.get(target, 0) + 1

        for row, combo in enumerate(_combos):
            target = combo.currentText()
            status_item = table.item(row, 2)
            if target in REQUIRED_NONMEM:
                label = "Required ✓"
            elif target in OPTIONAL_NONMEM:
                label = "Optional"
            else:
                label = ""
            if status_item:
                status_item.setText(label)

            # Red background for duplicates
            is_dup = target != _KEEP_AS_IS and target in ALL_NONMEM and seen.get(target, 0) > 1
            for col in range(3):
                item = table.item(row, col)
                if item:
                    if is_dup:
                        item.setBackground(qt_gui.QColor("#fee2e2"))
                        item.setForeground(qt_gui.QColor("#991b1b"))
                    else:
                        item.setBackground(qt_gui.QColor(0, 0, 0, 0))
                        item.setForeground(qt_gui.QColor())

    def _populate(columns: list[str], targets: list[str]) -> None:
        """Fill the table with *columns* mapped to *targets*."""
        _combos.clear()
        table.setRowCount(len(columns))
        for row, (src, tgt) in enumerate(zip(columns, targets)):
            src_item = qt_widgets.QTableWidgetItem(src)
            table.setItem(row, 0, src_item)

            combo = qt_widgets.QComboBox()
            combo.addItems(combo_options)
            # Select correct entry
            idx = combo_options.index(tgt) if tgt in combo_options else 0
            combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(lambda _i: _refresh_status())
            table.setCellWidget(row, 1, combo)
            _combos.append(combo)

            status_item = qt_widgets.QTableWidgetItem(_status_for(tgt))
            table.setItem(row, 2, status_item)

        _refresh_status()

    def _auto_detect() -> None:
        cols = list(_state["columns"])
        targets = auto_detect_mapping(cols)
        for row, (combo, tgt) in enumerate(zip(_combos, targets)):
            idx = combo_options.index(tgt) if tgt in combo_options else 0
            combo.setCurrentIndex(idx)
        _refresh_status()

    def get_input_columns() -> list[str]:
        """Return ordered NONMEM names for the current mapping."""
        result: list[str] = []
        cols = list(_state["columns"])
        for i, combo in enumerate(_combos):
            tgt = combo.currentText()
            if tgt == _KEEP_AS_IS:
                result.append(cols[i])
            else:
                result.append(tgt)
        return result

    def refresh(new_csv_columns: list[str]) -> None:
        """Reload the widget for a new set of source columns."""
        _state["columns"] = list(new_csv_columns)
        targets = auto_detect_mapping(new_csv_columns)
        _populate(new_csv_columns, targets)

    # Initial population
    initial_targets = auto_detect_mapping(csv_columns)
    _populate(csv_columns, initial_targets)

    # Wire buttons — apply_btn signal is connected by the caller (data workflow)
    auto_btn.clicked.connect(_auto_detect)

    group.setVisible(False)  # hidden until the data workflow decides to show it
    group._apply_btn = apply_btn  # type: ignore[attr-defined]
    group._refresh = refresh  # type: ignore[attr-defined]

    return group, get_input_columns, refresh
