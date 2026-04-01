"""Covariate modelling workflow — SCM candidate configuration and background run."""

from __future__ import annotations

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
)
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.scm_service import (
    SCMCandidate,
    SCMPreparationResult,
    SCMRunResult,
    SCMService,
    generate_scm_step_plot,
)
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.responsive_layout import (
    install_responsive_box_layouts,
    install_responsive_splitters,
)
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page
from openpkpd_gui.widgets.table_headers import configure_resizable_table_columns

_EFFECT_CHOICES = ["power", "linear", "exp", "categorical"]
_COVARIATE_RESPONSIVE_LAYOUT_BREAKPOINT = 1000
_STANDARD_PK_COLUMNS = frozenset(
    {
        "ID",
        "TIME",
        "DV",
        "MDV",
        "EVID",
        "AMT",
        "RATE",
        "CMT",
        "ADDL",
        "II",
        "SS",
        "BLQ",
        "LLOQ",
        "OCC",
        "DOSE",
        "TAD",
        "IPRED",
        "PRED",
        "RES",
        "WRES",
        "CWRES",
    }
)


def format_scm_step_summary(step_row: dict) -> str:
    """Return a compact display string for one SCM step row."""
    status = "ACCEPTED" if step_row.get("accepted") else "rejected"
    return (
        f"[{step_row.get('type', '?').upper()}] {step_row.get('rel', '?')}  "
        f"ΔOFV={step_row.get('delta_ofv', 0.0):+.3f}  "
        f"p={step_row.get('p_value', 1.0):.4f}  {status}"
    )


def format_scm_result_summary(result: SCMRunResult | None, run: RunRecord | None) -> str:
    """Return a one-line status string for the latest SCM run."""
    if result is not None:
        return result.summary_text
    if run is None:
        return "No SCM runs yet."
    if run.status == RunStatus.SUCCEEDED and run.summary_text:
        return f"Latest run — Succeeded • {run.summary_text}"
    if run.status == RunStatus.FAILED and run.error_text:
        return f"Latest run — Failed • {run.error_text}"
    if run.status == RunStatus.CANCELLED:
        return "Latest run — Cancelled"
    return f"Latest run — {run.status.value.title()}"


def recommend_covariate_next_action(
    project: Workspace,
    preparation: SCMPreparationResult,
    latest_run: RunRecord | None,
    candidate_count: int,
) -> tuple[str, str, str] | None:
    """Return the primary CTA for blocked or empty Covariate states."""
    if latest_run is not None and latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        return None
    if project.active_dataset is None:
        return (
            "Open Data",
            "data",
            "Load a dataset in the Data workflow before running SCM.",
        )
    if project.active_model_spec is None:
        return (
            "Open Model",
            "model",
            "Configure a model in the Model workflow before running SCM.",
        )
    if not preparation.ready:
        message = (
            preparation.validation.issues[0].message
            if preparation.validation.issues
            else "Resolve the model setup issues before running SCM."
        )
        return ("Open Model", "model", message)
    if candidate_count == 0:
        return (
            "Add candidate",
            "__add_candidate__",
            "Add at least one candidate before running SCM.",
        )
    return None


def covariate_blocking_message(project: Workspace, preparation: SCMPreparationResult) -> str | None:
    """Return the highest-priority message blocking SCM launch, if any."""
    if project.active_dataset is None:
        return "Load a dataset in the Data workflow before running SCM."
    if project.active_model_spec is None:
        return "Configure a model in the Model workflow before running SCM."
    if not preparation.ready:
        if getattr(preparation, "has_builder", None) is False:
            return "SCM requires a builder-mode model. Switch to Model Builder mode."
        if preparation.validation.issues:
            return preparation.validation.issues[0].message
        return "Resolve the model setup issues before running SCM."
    return None


def build_covariate_workflow(
    project: Workspace,
    scm_service: SCMService | None = None,
    project_service: ProjectService | None = None,
    job_runner: JobRunner | None = None,
):
    """Build the Covariate Modelling (SCM) workflow page."""
    qt_core, _, qt_widgets = load_qt_modules()
    scm_service = scm_service or SCMService()
    project_service = project_service or ProjectService()
    job_runner = job_runner or JobRunner(max_workers=1)

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="covariate-workflow"
    )

    # ── Title ──────────────────────────────────────────────────────────────────
    title_label = qt_widgets.QLabel("Covariate modelling")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_covariate():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_covariate"))

    hint_widget, _ = build_dismissible_hint(
        "Define covariate candidates, set p-value thresholds, then launch a "
        "stepwise covariate search (SCM) against the saved base model.",
        dismissed="hint_covariate" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_covariate,
    )

    candidates_header = qt_widgets.QLabel("Candidates")
    candidates_header_font = candidates_header.font()
    candidates_header_font.setBold(True)
    candidates_header.setFont(candidates_header_font)

    candidates_table = qt_widgets.QTableWidget(0, 4)
    candidates_table.setObjectName("covariate-candidates-table")
    candidates_table.setHorizontalHeaderLabels(["Parameter", "Covariate", "Effect", "Reference"])
    configure_resizable_table_columns(candidates_table, qt_widgets)
    candidates_table.setSelectionBehavior(qt_widgets.QAbstractItemView.SelectionBehavior.SelectRows)
    cand_row_widget = qt_widgets.QWidget(root)
    cand_row_widget.setObjectName("covariate-candidate-controls-row")
    cand_row = qt_widgets.QHBoxLayout(cand_row_widget)
    cand_row.setContentsMargins(0, 0, 0, 0)
    cand_row.setSpacing(8)
    add_row_button = qt_widgets.QPushButton("Add candidate")
    add_row_button.setObjectName("covariate-add-candidate-button")
    remove_row_button = qt_widgets.QPushButton("Remove selected")
    remove_row_button.setObjectName("covariate-remove-candidate-button")
    cand_row.addWidget(add_row_button)
    cand_row.addWidget(remove_row_button)
    cand_row.addStretch(1)

    # ── Search parameters ──────────────────────────────────────────────────────
    params_header = qt_widgets.QLabel("Search parameters")
    params_header_font = params_header.font()
    params_header_font.setBold(True)
    params_header.setFont(params_header_font)

    params_widget = qt_widgets.QWidget(root)
    params_form = qt_widgets.QFormLayout(params_widget)
    params_form.setContentsMargins(0, 0, 0, 0)
    params_form.setSpacing(8)

    forward_spin = qt_widgets.QDoubleSpinBox()
    forward_spin.setObjectName("covariate-forward-pvalue")
    forward_spin.setRange(0.001, 0.5)
    forward_spin.setSingleStep(0.01)
    forward_spin.setDecimals(3)
    forward_spin.setValue(0.05)
    params_form.addRow("Forward p-value:", forward_spin)

    backward_spin = qt_widgets.QDoubleSpinBox()
    backward_spin.setObjectName("covariate-backward-pvalue")
    backward_spin.setRange(0.0001, 0.5)
    backward_spin.setSingleStep(0.001)
    backward_spin.setDecimals(4)
    backward_spin.setValue(0.001)
    params_form.addRow("Backward p-value:", backward_spin)

    njobs_spin = qt_widgets.QSpinBox()
    njobs_spin.setObjectName("covariate-njobs")
    njobs_spin.setRange(-1, 64)
    njobs_spin.setValue(-1)
    njobs_spin.setSpecialValueText("auto")
    params_form.addRow("Parallel jobs:", njobs_spin)

    # ── Status + results ───────────────────────────────────────────────────────
    status_label = qt_widgets.QLabel(
        format_scm_result_summary(None, scm_service.latest_run(project))
    )
    status_label.setObjectName("covariate-status-label")
    status_label.setWordWrap(True)

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("covariate-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)

    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("covariate-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)

    results_table = qt_widgets.QTableWidget(0, 5)
    results_table.setObjectName("covariate-results-table")
    results_table.setHorizontalHeaderLabels(["Type", "Relationship", "ΔOFV", "p-value", "Status"])
    configure_resizable_table_columns(results_table, qt_widgets)
    results_table.setEditTriggers(qt_widgets.QAbstractItemView.EditTrigger.NoEditTriggers)

    content_row_widget = qt_widgets.QSplitter(root)
    content_row_widget.setObjectName("covariate-content-row")
    content_row_widget.setChildrenCollapsible(False)
    content_row_widget.setHandleWidth(8)

    configuration_panel = qt_widgets.QWidget(content_row_widget)
    configuration_panel.setObjectName("covariate-configuration-panel")
    configuration_layout = qt_widgets.QVBoxLayout(configuration_panel)
    configuration_layout.setContentsMargins(12, 12, 12, 12)
    configuration_layout.setSpacing(8)
    configuration_layout.addWidget(next_action_label)
    configuration_layout.addWidget(next_action_button)
    configuration_layout.addWidget(candidates_header)
    configuration_layout.addWidget(candidates_table, 1)
    configuration_layout.addWidget(cand_row_widget)
    configuration_layout.addWidget(params_header)
    configuration_layout.addWidget(params_widget)

    results_panel = qt_widgets.QWidget(content_row_widget)
    results_panel.setObjectName("covariate-results-panel")
    results_layout = qt_widgets.QVBoxLayout(results_panel)
    results_layout.setContentsMargins(12, 12, 12, 12)
    results_layout.setSpacing(8)
    results_layout.addWidget(status_label)
    results_layout.addWidget(results_table, 1)

    content_row_widget.addWidget(configuration_panel)
    content_row_widget.addWidget(results_panel)
    content_row_widget.setStretchFactor(0, 3)
    content_row_widget.setStretchFactor(1, 2)
    content_row_widget.setSizes([600, 400])

    # ── Action row ─────────────────────────────────────────────────────────────
    action_row_widget = qt_widgets.QWidget(root)
    action_row_widget.setObjectName("covariate-action-row")
    action_row = qt_widgets.QHBoxLayout(action_row_widget)
    action_row.setContentsMargins(0, 0, 0, 0)
    action_row.setSpacing(8)
    cancel_button = qt_widgets.QPushButton("Cancel")
    cancel_button.setObjectName("covariate-cancel-button")
    cancel_button.setEnabled(False)
    run_button = qt_widgets.QPushButton("Run SCM")
    run_button.setObjectName("covariate-run-button")
    run_button.setProperty("primaryAction", True)

    run_progress = qt_widgets.QProgressBar()
    run_progress.setObjectName("covariate-run-progress")
    run_progress.setRange(0, 0)
    run_progress.setFixedHeight(20)
    run_progress.setVisible(False)

    action_row.addStretch(1)
    action_row.addWidget(run_progress)
    action_row.addWidget(cancel_button)
    action_row.addWidget(run_button)

    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(content_row_widget, 1)
    layout.addWidget(action_row_widget)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=_COVARIATE_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        layouts=(cand_row, action_row),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=_COVARIATE_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(content_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    # ── State ──────────────────────────────────────────────────────────────────
    future = None
    next_action_target = [""]
    poll_timer = qt_core.QTimer(root)
    poll_timer.setInterval(100)
    last_result: list[SCMRunResult | None] = [None]

    preparation = scm_service.prepare(project)
    run_button.setEnabled(preparation.ready)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _collect_candidates() -> list[SCMCandidate]:
        rows = candidates_table.rowCount()
        result = []
        for row in range(rows):
            param_widget = candidates_table.cellWidget(row, 0)
            cov_widget = candidates_table.cellWidget(row, 1)
            effect_widget = candidates_table.cellWidget(row, 2)
            ref_item = candidates_table.item(row, 3)
            param = (param_widget.currentText().strip() if param_widget else "") or ""
            cov = (cov_widget.currentText().strip() if cov_widget else "") or ""
            if not param or not cov:
                continue
            effect = effect_widget.currentText() if effect_widget else "power"
            try:
                ref = float(ref_item.text()) if ref_item else 70.0
            except ValueError:
                ref = 70.0
            result.append(
                SCMCandidate(parameter=param, covariate=cov, effect=effect, reference=ref)
            )
        return result

    def _render_results(rows: list[dict]) -> None:
        results_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            results_table.setItem(row_idx, 0, qt_widgets.QTableWidgetItem(row.get("type", "")))
            results_table.setItem(row_idx, 1, qt_widgets.QTableWidgetItem(row.get("rel", "")))
            results_table.setItem(
                row_idx,
                2,
                qt_widgets.QTableWidgetItem(f"{row.get('delta_ofv', 0.0):+.3f}"),
            )
            results_table.setItem(
                row_idx,
                3,
                qt_widgets.QTableWidgetItem(f"{row.get('p_value', 1.0):.4f}"),
            )
            results_table.setItem(
                row_idx,
                4,
                qt_widgets.QTableWidgetItem("ACCEPTED" if row.get("accepted") else "rejected"),
            )

    def _notify_project_state_changed() -> None:
        callback = getattr(root, "_project_state_changed", None)
        if callable(callback):
            callback()

    def _refresh_next_action() -> None:
        action = recommend_covariate_next_action(
            project,
            preparation,
            scm_service.latest_run(project),
            candidates_table.rowCount(),
        )
        if action is None:
            next_action_target[0] = ""
            next_action_label.clear()
            next_action_label.setVisible(False)
            next_action_button.setText("")
            next_action_button.setToolTip("")
            next_action_button.setVisible(False)
            return
        button_text, target, summary = action
        next_action_target[0] = target
        next_action_label.setText(summary)
        next_action_label.setVisible(True)
        next_action_button.setText(button_text)
        next_action_button.setToolTip(summary)
        next_action_button.setVisible(True)

    def _navigate_to_next_action() -> None:
        target = next_action_target[0]
        if not target:
            return
        if target == "__add_candidate__":
            _add_candidate_row()
            return
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(target)

    def _refresh_workflow() -> None:
        nonlocal preparation, future
        preparation = scm_service.prepare(project)
        blocking_message = covariate_blocking_message(project, preparation)
        run_button.setEnabled(preparation.ready and future is None)
        run_button.setToolTip(
            blocking_message
            or (
                "Add at least one candidate before running SCM."
                if candidates_table.rowCount() == 0
                else "Submit the stepwise covariate search."
            )
        )
        latest_run = scm_service.latest_run(project)
        if blocking_message is not None and latest_run is None and last_result[0] is None:
            status_label.setText(blocking_message)
        else:
            status_label.setText(format_scm_result_summary(last_result[0], latest_run))
        _refresh_next_action()

    def _available_parameters() -> list[str]:
        model_spec = project.active_scenario.active_model_spec
        if model_spec is not None:
            labels = [
                str(row.get("label", "")).strip()
                for row in (model_spec.theta_rows or [])
                if str(row.get("label", "")).strip()
            ]
            if labels:
                return labels
        return ["CL", "V", "KA", "Q", "V2", "V3", "F1", "ALAG1"]

    def _available_covariates() -> list[str]:
        dataset = project.active_scenario.active_dataset
        if dataset is not None and dataset.columns:
            return [col for col in dataset.columns if col.upper() not in _STANDARD_PK_COLUMNS]
        return ["WT", "AGE", "SEX", "CRCL", "ALB"]

    def _make_combo(items: list[str], default: str, *, editable: bool) -> object:
        combo = qt_widgets.QComboBox()
        combo.setEditable(editable)
        available_items = [item for item in items if str(item).strip()]
        if default and default not in available_items:
            available_items = [default, *available_items]
        combo.addItems(available_items)
        if default in available_items:
            combo.setCurrentText(default)
        elif available_items:
            combo.setCurrentText(available_items[0])
        return combo

    def _insert_candidate_row(
        param: str = "", cov: str = "", effect: str = "power", reference: float = 70.0
    ) -> None:
        """Insert one row into the candidates table with given values."""
        row = candidates_table.rowCount()
        candidates_table.insertRow(row)
        params = _available_parameters()
        covariates = _available_covariates()
        param_combo = _make_combo(
            params,
            param or (params[0] if params else "CL"),
            editable=False,
        )
        cov_combo = _make_combo(
            covariates,
            cov or (covariates[0] if covariates else "WT"),
            editable=False,
        )
        candidates_table.setCellWidget(row, 0, param_combo)
        candidates_table.setCellWidget(row, 1, cov_combo)
        effect_combo = qt_widgets.QComboBox()
        effect_combo.addItems(_EFFECT_CHOICES)
        if effect in _EFFECT_CHOICES:
            effect_combo.setCurrentText(effect)
        candidates_table.setCellWidget(row, 2, effect_combo)
        candidates_table.setItem(row, 3, qt_widgets.QTableWidgetItem(str(reference)))
        param_combo.currentTextChanged.connect(lambda *_args: _save_candidates_to_project())
        cov_combo.currentTextChanged.connect(lambda *_args: _save_candidates_to_project())
        effect_combo.currentTextChanged.connect(lambda *_args: _save_candidates_to_project())

    def _save_candidates_to_project() -> None:
        """Persist the current candidate rows into scenario metadata and trigger a save."""
        candidates = _collect_candidates()
        project.active_scenario.metadata["scm_candidates"] = [
            {
                "parameter": c.parameter,
                "covariate": c.covariate,
                "effect": c.effect,
                "reference": c.reference,
            }
            for c in candidates
        ]
        _notify_project_state_changed()

    def _add_candidate_row() -> None:
        _insert_candidate_row()
        _save_candidates_to_project()
        _refresh_next_action()

    def _remove_candidate_row() -> None:
        selected = candidates_table.selectedItems()
        rows = sorted({item.row() for item in selected}, reverse=True)
        for row in rows:
            candidates_table.removeRow(row)
        _save_candidates_to_project()
        _refresh_next_action()

    # Restore persisted candidates from project metadata
    for _saved in project.active_scenario.metadata.get("scm_candidates", []):
        _insert_candidate_row(
            param=str(_saved.get("parameter", "")),
            cov=str(_saved.get("covariate", "")),
            effect=str(_saved.get("effect", "power")),
            reference=float(_saved.get("reference", 70.0)),
        )

    def _save_scm_forest_plot(scm_result: SCMRunResult, run_id: str) -> None:
        """Generate and register a SCM step significance plot artifact (P3-B)."""
        from pathlib import Path

        from openpkpd_gui.app.settings import default_workspace_root_path

        base = project.root_path
        output_dir = (
            Path(base).resolve() / "artifacts" if base else default_workspace_root_path() / "artifacts"
        )
        artifact = generate_scm_step_plot(scm_result, run_id=run_id, output_dir=output_dir)
        if artifact is not None:
            project.active_scenario.add_artifact(artifact)

    def _poll_future() -> None:
        nonlocal future
        if future is None or not future.done():
            return
        outcome = future.result()
        run = scm_service.latest_run(project)
        if run is not None:
            scm_result = scm_service.apply_job_outcome(run, outcome)
            last_result[0] = scm_result
            if scm_result is not None:
                _render_results(scm_result.step_rows)
                _save_scm_forest_plot(scm_result, run.run_id)
            _notify_project_state_changed()
        future = None
        poll_timer.stop()
        run_progress.setVisible(False)
        cancel_button.setEnabled(False)
        _refresh_workflow()

    def _start_run() -> None:
        nonlocal future
        candidates = _collect_candidates()
        if not candidates:
            status_label.setText("Add at least one candidate before running SCM.")
            return
        _refresh_workflow()
        if not preparation.ready:
            status_label.setText(
                covariate_blocking_message(project, preparation)
                or "Resolve the model setup issues before running SCM."
            )
            return
        run = RunRecord(workflow="covariate")
        run.mark_running()
        run.add_log("SCM run submitted.")
        project_service.add_run(project, run)
        status_label.setText("SCM running…")
        results_table.setRowCount(0)
        _refresh_next_action()
        _notify_project_state_changed()
        future = job_runner.submit(
            scm_service.create_job(
                project,
                candidates=candidates,
                forward_pvalue=forward_spin.value(),
                backward_pvalue=backward_spin.value(),
                n_jobs=njobs_spin.value(),
                preparation=preparation,
            )
        )
        run_button.setEnabled(False)
        cancel_button.setEnabled(True)
        run_progress.setVisible(True)
        poll_timer.start()

    def _cancel_run() -> None:
        nonlocal future
        if future is None:
            return
        future.cancel()
        future = None
        poll_timer.stop()
        run_progress.setVisible(False)
        cancel_button.setEnabled(False)
        run = scm_service.latest_run(project)
        if run is not None:
            run.mark_cancelled()
            _notify_project_state_changed()
        _refresh_workflow()

    candidates_table.cellChanged.connect(lambda *_: _save_candidates_to_project())
    add_row_button.clicked.connect(_add_candidate_row)
    remove_row_button.clicked.connect(_remove_candidate_row)
    cancel_button.clicked.connect(_cancel_run)
    run_button.clicked.connect(_start_run)
    next_action_button.clicked.connect(_navigate_to_next_action)
    poll_timer.timeout.connect(_poll_future)
    root.destroyed.connect(lambda *_args: job_runner.shutdown(wait=False))
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh_workflow  # type: ignore[attr-defined]

    _refresh_workflow()
    _apply_responsive_layout()
    return root
