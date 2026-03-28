"""Results workflow widget for browsing run history and outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
)
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.workflow_state_service import describe_fit_input_changes
from openpkpd_gui.widgets.collapsible_section import build_collapsible_section
from openpkpd_gui.widgets.combined_header import build_combined_header
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.output_browser import (
    build_output_preview_panel,
    export_output_file,
    open_output_file,
)
from openpkpd_gui.widgets.responsive_layout import (
    install_responsive_box_layouts,
    install_responsive_splitters,
)
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page

RESULTS_RESPONSIVE_LAYOUT_BREAKPOINT = 1100
ANALYSIS_FILTER_ALL = "All analyses"

_ANALYSIS_TYPE_ORDER = {"Fit": 0, "NCA": 1, "Covariate": 2, "Advanced": 3, "Other": 4}
_RUN_WORKFLOW_LABELS = {
    "fit": "Fit",
    "npde": "NPDE",
    "nca": "NCA",
    "covariate": "Covariate",
    "vpc": "VPC",
    "bootstrap": "Bootstrap",
    "design": "Design",
}
_RUN_ANALYSIS_TYPES = {
    "fit": "Fit",
    "npde": "Fit",
    "nca": "NCA",
    "covariate": "Covariate",
    "vpc": "Advanced",
    "bootstrap": "Advanced",
    "design": "Advanced",
}
_ARTIFACT_ROLE_ANALYSIS_TYPES = {
    "nca_summary": "NCA",
    "vpc_summary": "Advanced",
    "bootstrap_summary": "Advanced",
    "bootstrap_ci_table": "Advanced",
    "bootstrap_samples": "Advanced",
    "design_summary": "Advanced",
    "design_metrics": "Advanced",
    "design_schedule": "Advanced",
    "design_fim": "Advanced",
    "design_expected_se": "Advanced",
    "npde_table": "Fit",
}
_ARTIFACT_PLOT_ANALYSIS_TYPES = {
    "vpc": "Advanced",
    "simulation_panel": "Advanced",
    "prediction_interval_plot": "Advanced",
    "npde_plot": "Fit",
}


def _sorted_analysis_types(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (_ANALYSIS_TYPE_ORDER.get(value, 99), value))


def run_workflow_label(run: RunRecord) -> str:
    return _RUN_WORKFLOW_LABELS.get(run.workflow, run.workflow.replace("_", " ").title())


def run_analysis_type(run: RunRecord) -> str:
    return _RUN_ANALYSIS_TYPES.get(run.workflow, run_workflow_label(run))


def scenario_runs_by_id(workspace: Workspace) -> dict[str, RunRecord]:
    return {run.run_id: run for run in workspace.active_scenario.runs}


def review_runs(
    workspace: Workspace, analysis_filter: str = ANALYSIS_FILTER_ALL
) -> list[RunRecord]:
    """Return selected-scenario reviewable runs in reverse chronological order."""
    return [
        run
        for run in reversed(workspace.active_scenario.runs)
        if run.workflow in _RUN_ANALYSIS_TYPES
        and (analysis_filter == ANALYSIS_FILTER_ALL or run_analysis_type(run) == analysis_filter)
    ]


def review_analysis_type_options(workspace: Workspace) -> list[str]:
    """Return analysis filter options derived from scenario runs and artifacts."""
    runs_by_id = scenario_runs_by_id(workspace)
    analysis_types = {run_analysis_type(run) for run in review_runs(workspace)}
    analysis_types.update(
        artifact_analysis_type(artifact, runs_by_id)
        for artifact in workspace.active_scenario.artifacts
    )
    analysis_types.discard("")
    return [ANALYSIS_FILTER_ALL, *_sorted_analysis_types(analysis_types)]


def artifact_analysis_type(
    artifact: ArtifactRecord, runs_by_id: dict[str, RunRecord] | None = None
) -> str:
    """Infer the primary analysis type responsible for an artifact."""
    explicit = artifact.metadata.get("analysis_type")
    if explicit:
        return str(explicit)

    for metadata_key, analysis_type in (
        ("vpc_run_id", "Advanced"),
        ("bootstrap_run_id", "Advanced"),
        ("design_run_id", "Advanced"),
        ("npde_run_id", "Fit"),
    ):
        if artifact.metadata.get(metadata_key):
            return analysis_type

    role = artifact_role(artifact)
    if role in _ARTIFACT_ROLE_ANALYSIS_TYPES:
        return _ARTIFACT_ROLE_ANALYSIS_TYPES[role]

    plot_type = artifact_plot_type(artifact)
    if plot_type in _ARTIFACT_PLOT_ANALYSIS_TYPES:
        return _ARTIFACT_PLOT_ANALYSIS_TYPES[plot_type]

    if (
        artifact.source_run_id
        and runs_by_id is not None
        and (run := runs_by_id.get(artifact.source_run_id))
    ):
        return run_analysis_type(run)

    if artifact.metadata.get("fit_run_id"):
        return "Fit"
    return "Other"


def artifact_analysis_type_options(
    artifacts: list[ArtifactRecord],
    runs_by_id: dict[str, RunRecord] | None = None,
) -> list[str]:
    """Return analysis filter options for the provided artifacts."""
    analysis_types = {artifact_analysis_type(artifact, runs_by_id) for artifact in artifacts}
    analysis_types.discard("")
    return [ANALYSIS_FILTER_ALL, *_sorted_analysis_types(analysis_types)]


def format_results_overview(workspace: Workspace) -> str:
    """Return a compact overview of current selected-scenario run/artifact state."""
    runs = review_runs(workspace)
    runs_by_id = scenario_runs_by_id(workspace)
    scenario = workspace.active_scenario
    analysis_types = _sorted_analysis_types(
        {
            *{run_analysis_type(run) for run in runs},
            *{artifact_analysis_type(artifact, runs_by_id) for artifact in scenario.artifacts},
        }
    )
    analysis_text = ", ".join(analysis_types) if analysis_types else "none yet"
    if not runs:
        return (
            f"Scenario {scenario.name} • 0 review runs • {len(scenario.artifacts)} outputs "
            f"• analyses {analysis_text} • latest status No runs yet"
        )
    latest = runs[0]
    counts = Counter(run.status for run in runs)
    status_text = " / ".join(
        f"{counts[status]} {status.value}" for status in RunStatus if counts.get(status, 0)
    )
    return (
        f"Scenario {scenario.name} • {len(runs)} review runs • {len(scenario.artifacts)} outputs • "
        f"analyses {analysis_text} • latest {run_analysis_type(latest)} {latest.status.value.title()} • {status_text}"
    )


def format_results_comparison_summary(workspace: Workspace) -> str:
    """Summarize sibling scenarios in the active project for quick review/comparison."""
    project = workspace.active_project
    current = project.active_scenario
    peers = [scenario for scenario in project.scenarios if scenario.scenario_id != current.scenario_id]
    if not peers:
        return "No sibling scenarios yet. Duplicate or branch this scenario to compare runs, outputs, and decisions."

    summaries: list[str] = []
    for scenario in peers[:3]:
        fit_runs = [run for run in scenario.runs if run.workflow == "fit"]
        latest_fit = fit_runs[-1] if fit_runs else None
        if latest_fit is None:
            fit_text = "no fit yet"
        else:
            fit_text = f"latest fit {latest_fit.status.value}"
        relation = ""
        if scenario.parent_scenario_id == current.scenario_id:
            relation = "derived from current"
        elif current.parent_scenario_id == scenario.scenario_id:
            relation = "parent of current"
        elif scenario.parent_scenario_id:
            relation = "branched"
        relation_text = f" • {relation}" if relation else ""
        summaries.append(
            f"{scenario.name}: {fit_text} • {len(scenario.runs)} runs • {len(scenario.artifacts)} outputs{relation_text}"
        )
    if len(peers) > 3:
        summaries.append(f"+{len(peers) - 3} more scenarios")
    return "Comparison snapshot: " + " | ".join(summaries)


def format_results_comparison_action(workspace: Workspace) -> str:
    """Recommend the next most useful sibling scenario to inspect."""
    project = workspace.active_project
    current = project.active_scenario
    peers = [scenario for scenario in project.scenarios if scenario.scenario_id != current.scenario_id]
    if not peers:
        return (
            "Comparison focus: create a sibling scenario after major fit or covariate changes "
            "to compare outputs, diagnostics, and conclusions."
        )

    def _peer_score(scenario) -> tuple[int, int, int]:
        successful_runs = sum(1 for run in scenario.runs if run.status == RunStatus.SUCCEEDED)
        fit_runs = sum(1 for run in scenario.runs if run.workflow == "fit")
        return successful_runs, fit_runs, len(scenario.artifacts)

    best = max(peers, key=_peer_score)
    successful_runs, fit_runs, output_count = _peer_score(best)
    relation = "sibling"
    if best.parent_scenario_id == current.scenario_id:
        relation = "child scenario"
    elif current.parent_scenario_id == best.scenario_id:
        relation = "parent scenario"

    if successful_runs == 0:
        return (
            f"Comparison focus: {best.name} is the busiest {relation} so far, but it has no "
            "successful runs yet. Fit it before using it as a review baseline."
        )
    return (
        f"Comparison focus: inspect {best.name} next. It is the richest {relation} for review "
        f"with {successful_runs} successful runs, {fit_runs} fit runs, and {output_count} outputs."
    )


def select_results_comparison_target(workspace: Workspace) -> str | None:
    """Return the most review-worthy sibling scenario id, if any."""
    project = workspace.active_project
    current = project.active_scenario
    peers = [scenario for scenario in project.scenarios if scenario.scenario_id != current.scenario_id]
    if not peers:
        return None

    def _peer_score(scenario) -> tuple[int, int, int]:
        successful_runs = sum(1 for run in scenario.runs if run.status == RunStatus.SUCCEEDED)
        fit_runs = sum(1 for run in scenario.runs if run.workflow == "fit")
        return successful_runs, fit_runs, len(scenario.artifacts)

    return max(peers, key=_peer_score).scenario_id


def format_results_stale_warning(workspace: Workspace) -> str:
    """Return a stale-results warning when saved inputs outpace the latest successful fit."""
    subject = describe_fit_input_changes(workspace.active_scenario)
    if subject is None:
        return ""
    return f"{subject} since the latest successful fit. Results may be stale until you rerun the analysis."


def recommend_results_next_action(project: Workspace) -> tuple[str, str, str] | None:
    """Return the primary CTA for an empty or blocked Results page."""
    if project.active_dataset is None:
        return (
            "Open Data",
            "data",
            "Load a dataset before expecting review runs, reports, or saved outputs for this scenario.",
        )
    if project.active_model_spec is None:
        return (
            "Open Model",
            "model",
            "Configure a model in the Model workflow first.",
        )
    if not review_runs(project):
        return (
            "Open Fit",
            "fit",
            "Run a fit to populate Results with run logs, reports, and saved outputs.",
        )
    return None


def format_run_label(run: RunRecord, index: int | None = None) -> str:
    """Render one run row for the Results workflow."""
    detail = run.summary_text or run.error_text or (run.started_at or "No summary yet")
    analysis_label = run_analysis_type(run)
    workflow_label = run_workflow_label(run)
    prefix = (
        analysis_label
        if workflow_label == analysis_label
        else f"{analysis_label} / {workflow_label}"
    )
    num = f"#{index}" if index is not None else run.run_id[:8]
    return f"{prefix} • {run.status.value.title()} • {num} • {detail}"


def format_artifact_label(
    artifact: ArtifactRecord,
    runs_by_id: dict[str, RunRecord] | None = None,
) -> str:
    """Render one artifact row for the Results workflow (no path — shown in tooltip)."""
    kind_badge = f"[{artifact.kind}]" if artifact.kind else ""
    return f"{artifact.label}  {kind_badge}"


def format_artifact_tooltip(
    artifact: ArtifactRecord,
    runs_by_id: dict[str, RunRecord] | None = None,
) -> str:
    """Return tooltip text for an artifact list item."""
    path_text = artifact.path or "in-memory"
    role = artifact_role(artifact)
    analysis = artifact_analysis_type(artifact, runs_by_id)
    return f"{analysis} · {artifact.kind} · {role}\n{path_text}"


def format_run_metadata(run: RunRecord | None) -> str:
    """Render metadata for the currently selected run."""
    if run is None:
        return "Started n/a • Finished n/a • 0 log lines • 0 linked artifacts"
    started = run.started_at or "not started"
    finished = run.finished_at or "not finished"
    return (
        f"Analysis {run_analysis_type(run)} • Workflow {run_workflow_label(run)} • "
        f"Started {started} • Finished {finished} • "
        f"{len(run.log_lines)} log lines • {len(run.artifact_ids)} linked outputs"
    )


def format_run_details(run: RunRecord | None) -> str:
    """Render a detailed summary of the currently selected run."""
    if run is None:
        return "Select an analysis run to inspect its summary and logs."
    analysis_label = run_analysis_type(run)
    workflow_label = run_workflow_label(run)
    run_label = (
        analysis_label
        if workflow_label == analysis_label
        else f"{analysis_label} / {workflow_label}"
    )
    if run.status == RunStatus.SUCCEEDED:
        detail = run.summary_text or f"{workflow_label} completed successfully."
    elif run.status == RunStatus.FAILED:
        detail = run.error_text or f"{workflow_label} failed."
    else:
        detail = run.summary_text or f"{workflow_label} is still in progress."
    return f"{run_label} run {run.run_id[:8]} • {run.status.value.title()} • {detail}"


def artifact_kind_options(artifacts: list[ArtifactRecord]) -> list[str]:
    """Return filter options for artifact kinds."""
    return ["All kinds", *sorted({artifact.kind for artifact in artifacts if artifact.kind})]


def artifact_role(artifact: ArtifactRecord) -> str:
    """Return the most useful logical role for an artifact."""
    role = str(artifact.metadata.get("artifact_role") or artifact.kind or "").strip()
    return role or "unclassified"


def artifact_plot_type(artifact: ArtifactRecord) -> str | None:
    """Return the plot type recorded for an artifact, if any."""
    plot_type = artifact.metadata.get("plot_type")
    return str(plot_type) if plot_type else None


def artifact_role_options(artifacts: list[ArtifactRecord]) -> list[str]:
    """Return filter options for artifact roles."""
    return ["All roles", *sorted({artifact_role(artifact) for artifact in artifacts})]


def artifact_plot_type_options(artifacts: list[ArtifactRecord]) -> list[str]:
    """Return filter options for known plot types."""
    return [
        "All plot types",
        *sorted(
            {plot_type for artifact in artifacts if (plot_type := artifact_plot_type(artifact))}
        ),
    ]


_PLOT_TYPE_LABELS: dict[str, str] = {
    "ofv_history": "Convergence — OFV history",
    "parameter_uncertainty": "Convergence — Parameter uncertainty",
    "gof_panel": "GOF panel",
    "dv_vs_ipred": "GOF — DV vs IPRED",
    "dv_vs_pred": "GOF — DV vs PRED",
    "cwres_vs_time": "Residuals — CWRES vs TIME",
    "cwres_vs_pred": "Residuals — CWRES vs PRED",
    "cwres_qq": "Residuals — CWRES Q-Q",
    "abs_iwres_vs_ipred": "Residuals — |IWRES| vs IPRED",
    "residual_trends": "Residuals — Trends",
    "spaghetti_plot": "Profiles — Individual",
    "mean_profile": "Profiles — Mean",
    "eta_histograms": "ETA — Histograms",
    "eta_pairs": "ETA — Pairs",
    "nca_distributions": "NCA — Distributions",
    "nca_boxplot": "NCA — Boxplot",
    "vpc": "VPC",
    "npde_plot": "NPDE",
}
_ROLE_LABELS: dict[str, str] = {
    "report": "HTML Report",
    "diagnostics_table": "Diagnostics table (CSV)",
    "npde_table": "NPDE table (CSV)",
    "nca_summary": "NCA summary (CSV)",
    "vpc_summary": "VPC summary (CSV)",
    "bootstrap_summary": "Bootstrap summary (CSV)",
}
ARTIFACT_TYPE_FILTER_ALL = "All types"


def artifact_friendly_type(artifact: ArtifactRecord) -> str:
    """Return a plain-language type label for use in the artifact type filter combo."""
    plot_type = artifact_plot_type(artifact)
    role = artifact_role(artifact)
    kind = artifact.kind or ""
    if plot_type and plot_type in _PLOT_TYPE_LABELS:
        return _PLOT_TYPE_LABELS[plot_type]
    if role in _ROLE_LABELS:
        return _ROLE_LABELS[role]
    if kind == "report":
        return "HTML Report"
    if kind == "table":
        return f"Table — {role}"
    if kind == "plot":
        return f"Plot — {plot_type or role}"
    return kind or "Other"


def artifact_friendly_type_options(artifacts: list[ArtifactRecord]) -> list[str]:
    """Return sorted unique type labels for all artifacts, with an 'All types' sentinel first."""
    return [ARTIFACT_TYPE_FILTER_ALL, *sorted({artifact_friendly_type(a) for a in artifacts})]


def filter_artifacts(
    artifacts: list[ArtifactRecord],
    kind_filter: str = "All kinds",
    role_filter: str = "All roles",
    plot_type_filter: str = "All plot types",
    analysis_filter: str = ANALYSIS_FILTER_ALL,
    runs_by_id: dict[str, RunRecord] | None = None,
    type_filter: str = ARTIFACT_TYPE_FILTER_ALL,
) -> list[ArtifactRecord]:
    """Filter artifacts by analysis type, kind, role, plot type, and friendly type label."""
    filtered = list(artifacts)
    if analysis_filter != ANALYSIS_FILTER_ALL:
        filtered = [
            artifact
            for artifact in filtered
            if artifact_analysis_type(artifact, runs_by_id) == analysis_filter
        ]
    if kind_filter != "All kinds":
        filtered = [artifact for artifact in filtered if artifact.kind == kind_filter]
    if role_filter != "All roles":
        filtered = [artifact for artifact in filtered if artifact_role(artifact) == role_filter]
    if plot_type_filter != "All plot types":
        filtered = [
            artifact for artifact in filtered if artifact_plot_type(artifact) == plot_type_filter
        ]
    if type_filter != ARTIFACT_TYPE_FILTER_ALL:
        filtered = [
            artifact for artifact in filtered if artifact_friendly_type(artifact) == type_filter
        ]
    return filtered


def latest_artifact(
    artifacts: list[ArtifactRecord],
    *,
    kind: str | None = None,
    role: str | None = None,
    plot_type: str | None = None,
) -> ArtifactRecord | None:
    """Return the newest artifact matching the requested criteria."""
    for artifact in reversed(artifacts):
        if kind is not None and artifact.kind != kind:
            continue
        if role is not None and artifact_role(artifact) != role:
            continue
        if plot_type is not None and artifact_plot_type(artifact) != plot_type:
            continue
        return artifact
    return None


FIT_REVIEW_PLOT_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("convergence", "Convergence", ("ofv_history", "parameter_uncertainty")),
    ("gof", "GOF", ("gof_panel", "dv_vs_ipred", "dv_vs_pred")),
    (
        "residuals",
        "Residuals",
        ("residual_trends", "cwres_vs_time", "cwres_vs_pred", "cwres_qq", "abs_iwres_vs_ipred"),
    ),
    ("profiles", "Profiles", ("spaghetti_plot", "mean_profile")),
    ("eta", "ETA", ("eta_histograms", "eta_pairs")),
)

FIT_REVIEW_TABLE_ROLES: tuple[tuple[str, str], ...] = (
    ("diagnostics_table", "Diagnostics table"),
    ("npde_table", "NPDE table"),
)


def latest_artifact_for_plot_group(
    artifacts: list[ArtifactRecord],
    group_key: str,
) -> ArtifactRecord | None:
    """Return the newest artifact for one curated fit-review plot group."""
    plot_types = next(
        (plot_types for key, _label, plot_types in FIT_REVIEW_PLOT_GROUPS if key == group_key),
        (),
    )
    if not plot_types:
        return None
    plot_type_set = set(plot_types)
    for artifact in reversed(artifacts):
        if artifact_plot_type(artifact) in plot_type_set:
            return artifact
    return None


def format_fit_review_summary(artifacts: list[ArtifactRecord]) -> str:
    """Summarize which curated fit-review shortcuts are currently available."""
    available_groups = [
        label
        for key, label, _plot_types in FIT_REVIEW_PLOT_GROUPS
        if latest_artifact_for_plot_group(artifacts, key) is not None
    ]
    available_tables = [
        label
        for role, label in FIT_REVIEW_TABLE_ROLES
        if latest_artifact(artifacts, role=role) is not None
    ]
    available = [*available_groups, *available_tables]
    if not available:
        return (
            "Fit review shortcuts will appear after report, plot, or table outputs are generated."
        )
    return f"Quick review available: {', '.join(available)}."


def format_artifact_summary(artifacts: list[ArtifactRecord]) -> str:
    """Render an artifact count summary grouped by kind."""
    if not artifacts:
        return "0 outputs"
    counts = Counter(artifact.kind for artifact in artifacts)
    detail = " • ".join(f"{kind}: {count}" for kind, count in sorted(counts.items()))
    return f"{len(artifacts)} outputs • {detail}"


def artifact_preview_kind(artifact: ArtifactRecord | None) -> str:
    """Return the most useful preview mode for an artifact."""
    if artifact is None:
        return "none"
    media_type = str(artifact.metadata.get("media_type") or "").lower()
    suffix = Path(artifact.path or "").suffix.lower()
    if media_type == "text/html" or suffix in {".html", ".htm"}:
        return "html"
    if media_type.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
        return "image"
    if suffix == ".csv":
        return "table"
    if media_type.startswith("text/") or suffix in {".txt", ".log", ".json", ".md"}:
        return "text"
    return "none"


def format_artifact_metadata(
    artifact: ArtifactRecord | None,
    runs_by_id: dict[str, RunRecord] | None = None,
) -> str:
    """Render metadata for the currently selected artifact."""
    if artifact is None:
        return "No output selected."
    path_text = artifact.path or "not backed by a file"
    media_type = artifact.metadata.get("media_type") or "unknown"
    analysis_type = artifact_analysis_type(artifact, runs_by_id)
    role_text = artifact_role(artifact)
    plot_type = artifact_plot_type(artifact)
    plot_text = f" • Plot {plot_type}" if plot_type else ""
    method = artifact.metadata.get("estimation_method")
    method_text = f" • Method {method}" if method else ""
    return (
        f"Analysis {analysis_type} • Kind {artifact.kind} • Role {role_text}{plot_text}{method_text} "
        f"• Media {media_type} • Created {artifact.created_at}"
        f"\nPath: {path_text}"
    )


def build_results_workflow(
    project: Workspace,
    artifact_service: ArtifactService | None = None,
):
    """Build the first real Results workflow page."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    artifact_service = artifact_service or ArtifactService()

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="results-workflow"
    )

    combined_header, refresh_combined_header = build_combined_header(
        root,
        project,
        workflow_id="results",
        workflow_label="Results",
        status_workflow_ids=("data", "model", "fit", "results", "diagnostics"),
    )

    title_label = qt_widgets.QLabel("Results workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_results():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_results"))

    hint_widget, _ = build_dismissible_hint(
        "Review completed and in-progress analysis runs, inspect their logs, and browse "
        "saved outputs registered on the selected scenario by analysis type.",
        dismissed="hint_results" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_results,
    )

    overview_label = qt_widgets.QLabel(format_results_overview(project))
    overview_label.setObjectName("results-overview-label")
    overview_label.setWordWrap(True)

    comparison_label = qt_widgets.QLabel(format_results_comparison_summary(project))
    comparison_label.setObjectName("results-comparison-label")
    comparison_label.setWordWrap(True)

    comparison_action_label = qt_widgets.QLabel(format_results_comparison_action(project))
    comparison_action_label.setObjectName("results-comparison-action-label")
    comparison_action_label.setWordWrap(True)

    comparison_action_button = qt_widgets.QPushButton("Open comparison scenario")
    comparison_action_button.setObjectName("results-comparison-action-button")
    comparison_action_button.setMinimumHeight(32)

    stale_warning_label = qt_widgets.QLabel(format_results_stale_warning(project))
    stale_warning_label.setObjectName("results-stale-warning-label")
    stale_warning_label.setWordWrap(True)
    stale_warning_label.setVisible(bool(stale_warning_label.text()))

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("results-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)

    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("results-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)

    content_row_widget = qt_widgets.QSplitter(root)
    content_row_widget.setObjectName("results-content-row")
    content_row_widget.setChildrenCollapsible(False)
    content_row_widget.setHandleWidth(8)
    runs_list = qt_widgets.QListWidget()
    runs_list.setObjectName("results-runs-list")
    runs_list.setProperty("persistListSelection", True)
    artifacts_list = qt_widgets.QListWidget()
    artifacts_list.setObjectName("results-artifacts-list")
    artifacts_list.setProperty("persistListSelection", True)

    detail_column = qt_widgets.QVBoxLayout()
    detail_column.setContentsMargins(12, 12, 12, 12)
    detail_column.setSpacing(8)
    detail_label = qt_widgets.QLabel(format_run_details(None))
    detail_label.setObjectName("results-run-detail-label")
    detail_label.setWordWrap(True)
    metadata_label = qt_widgets.QLabel(format_run_metadata(None))
    metadata_label.setObjectName("results-run-metadata-label")
    metadata_label.setWordWrap(True)
    log_output = qt_widgets.QPlainTextEdit()
    log_output.setObjectName("results-log-output")
    log_output.setReadOnly(True)
    log_output.setPlaceholderText("Selected run logs will appear here.")
    detail_column.addWidget(detail_label)
    detail_column.addWidget(metadata_label)
    log_section, _, log_layout, _log_toggle = build_collapsible_section(
        root,
        title="Run log",
        object_name="results-run-log-section",
        expanded=True,
    )
    log_layout.addWidget(log_output)
    detail_column.addWidget(log_section)

    artifact_column = qt_widgets.QVBoxLayout()
    artifact_column.setContentsMargins(12, 12, 12, 12)
    artifact_column.setSpacing(8)
    artifact_scope_label = qt_widgets.QLabel("Outputs for selected scenario")
    artifact_scope_label.setObjectName("results-artifact-scope-label")
    artifact_scope_label.setWordWrap(True)
    artifact_filter_row_widget = qt_widgets.QWidget(root)
    artifact_filter_row_widget.setObjectName("results-artifact-filter-row")
    artifact_filter_row = qt_widgets.QHBoxLayout(artifact_filter_row_widget)
    artifact_filter_row.setContentsMargins(0, 0, 0, 0)
    artifact_filter_row.setSpacing(8)
    artifact_filter_row.addWidget(qt_widgets.QLabel("Analysis"))
    analysis_type_combo = qt_widgets.QComboBox()
    analysis_type_combo.setObjectName("results-analysis-filter")
    analysis_type_combo.setProperty("persistComboSelection", True)
    artifact_filter_row.addWidget(analysis_type_combo, 1)
    artifact_filter_row.addStretch(1)

    kind_button_group = qt_widgets.QButtonGroup(root)
    kind_button_group.setExclusive(True)
    _kind_button_specs = [("", "All"), ("report", "Report"), ("plot", "Plot"), ("table", "Table")]
    kind_buttons: dict[str, object] = {}
    for _kind_id, _kind_label in _kind_button_specs:
        _btn = qt_widgets.QPushButton(_kind_label)
        _btn.setCheckable(True)
        _btn.setFlat(True)
        _btn.setObjectName(f"results-kind-filter-{_kind_id or 'all'}")
        _btn.setProperty("persistButtonGroupName", "results-kind-filter")
        kind_button_group.addButton(_btn)
        kind_buttons[_kind_id] = _btn
        artifact_filter_row.addWidget(_btn)
    kind_buttons[""].setChecked(True)
    artifact_summary_label = qt_widgets.QLabel("0 outputs")
    artifact_summary_label.setObjectName("results-artifact-summary-label")
    artifact_summary_label.setWordWrap(True)
    report_summary_label = qt_widgets.QLabel("No report available yet.")
    report_summary_label.setObjectName("results-report-summary-label")
    report_summary_label.setWordWrap(True)
    preview_panel_obj = build_output_preview_panel(root, object_prefix="results-artifact")
    artifact_preview_title = preview_panel_obj.title_label
    artifact_preview_metadata = preview_panel_obj.metadata_label
    artifact_preview_stack = preview_panel_obj.stack
    # ── Primary "Open" button ────────────────────────────────────────────
    open_artifact_button = qt_widgets.QPushButton("Open")
    open_artifact_button.setObjectName("results-artifact-open-button")
    open_artifact_button.setToolTip("Open the selected output in the system default viewer")
    open_artifact_button.setEnabled(False)

    # ── Review ▾ dropdown — quick access to named plot groups / tables ──
    review_button = qt_widgets.QToolButton(root)
    review_button.setObjectName("results-review-menu-button")
    review_button.setText("Review ▾")
    review_button.setMinimumHeight(32)
    review_button.setPopupMode(qt_widgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    review_menu = qt_widgets.QMenu(review_button)
    review_menu.setObjectName("results-review-menu")

    open_convergence_action = qt_gui.QAction("Convergence", review_menu)
    open_convergence_action.setObjectName("results-open-convergence-button")
    open_convergence_action.setEnabled(False)
    open_gof_action = qt_gui.QAction("GOF review", review_menu)
    open_gof_action.setObjectName("results-open-gof-review-button")
    open_gof_action.setEnabled(False)
    open_residual_action = qt_gui.QAction("Residuals", review_menu)
    open_residual_action.setObjectName("results-open-residual-review-button")
    open_residual_action.setEnabled(False)
    open_eta_action = qt_gui.QAction("ETA", review_menu)
    open_eta_action.setObjectName("results-open-eta-review-button")
    open_eta_action.setEnabled(False)
    open_profile_action = qt_gui.QAction("Profiles", review_menu)
    open_profile_action.setObjectName("results-open-profile-review-button")
    open_profile_action.setEnabled(False)
    review_menu.addAction(open_convergence_action)
    review_menu.addAction(open_gof_action)
    review_menu.addAction(open_residual_action)
    review_menu.addAction(open_eta_action)
    review_menu.addAction(open_profile_action)
    review_menu.addSeparator()
    open_latest_report_action = qt_gui.QAction("Latest report", review_menu)
    open_latest_report_action.setObjectName("results-open-latest-report-button")
    open_latest_report_action.setEnabled(False)
    open_latest_plot_action = qt_gui.QAction("Latest plot", review_menu)
    open_latest_plot_action.setObjectName("results-open-latest-plot-button")
    open_latest_plot_action.setEnabled(False)
    review_menu.addAction(open_latest_report_action)
    review_menu.addAction(open_latest_plot_action)
    review_menu.addSeparator()
    open_diagnostics_action = qt_gui.QAction("Diagnostics CSV", review_menu)
    open_diagnostics_action.setObjectName("results-open-diagnostics-table-button")
    open_diagnostics_action.setEnabled(False)
    open_npde_table_action = qt_gui.QAction("NPDE CSV", review_menu)
    open_npde_table_action.setObjectName("results-open-npde-table-button")
    open_npde_table_action.setEnabled(False)
    review_menu.addAction(open_diagnostics_action)
    review_menu.addAction(open_npde_table_action)
    review_button.setMenu(review_menu)
    review_button.setEnabled(False)

    # ── Export ▾ dropdown — save / copy actions ──────────────────────────
    export_button = qt_widgets.QToolButton(root)
    export_button.setObjectName("results-export-menu-button")
    export_button.setText("Export ▾")
    export_button.setMinimumHeight(32)
    export_button.setPopupMode(qt_widgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    export_menu = qt_widgets.QMenu(export_button)
    export_menu.setObjectName("results-export-menu")

    export_artifact_action = qt_gui.QAction("Save output copy…", export_menu)
    export_artifact_action.setObjectName("results-artifact-export-button")
    export_artifact_action.setEnabled(False)
    save_latest_plot_copy_action = qt_gui.QAction("Save latest plot copy…", export_menu)
    save_latest_plot_copy_action.setObjectName("results-save-latest-plot-copy-button")
    save_latest_plot_copy_action.setEnabled(False)
    export_latest_report_pdf_action = qt_gui.QAction("Export report PDF…", export_menu)
    export_latest_report_pdf_action.setObjectName("results-export-latest-report-pdf-button")
    export_latest_report_pdf_action.setEnabled(False)
    export_menu.addAction(export_artifact_action)
    export_menu.addAction(save_latest_plot_copy_action)
    export_menu.addAction(export_latest_report_pdf_action)
    export_menu.addSeparator()
    open_artifact_folder_action = qt_gui.QAction("Open folder", export_menu)
    open_artifact_folder_action.setObjectName("results-artifact-open-folder-button")
    open_artifact_folder_action.setEnabled(False)
    export_menu.addAction(open_artifact_folder_action)
    export_button.setMenu(export_menu)
    export_button.setEnabled(False)

    # ── Unified action row ───────────────────────────────────────────────
    artifact_action_row_widget = qt_widgets.QWidget(root)
    artifact_action_row_widget.setObjectName("results-artifact-action-row")
    artifact_action_row = qt_widgets.QHBoxLayout(artifact_action_row_widget)
    artifact_action_row.setContentsMargins(0, 0, 0, 0)
    artifact_action_row.setSpacing(8)
    artifact_action_row.addWidget(open_artifact_button)
    artifact_action_row.addWidget(review_button)
    artifact_action_row.addWidget(export_button)
    artifact_action_row.addStretch(1)
    artifacts_hint_label = qt_widgets.QLabel("Click to preview  ·  Double-click to open")
    artifacts_hint_label.setObjectName("results-artifacts-hint-label")
    font = artifacts_hint_label.font()
    font.setItalic(True)
    artifacts_hint_label.setFont(font)

    plot_type_row_widget = qt_widgets.QWidget(root)
    plot_type_row_widget.setObjectName("results-plot-type-filter-row")
    _plot_type_row = qt_widgets.QHBoxLayout(plot_type_row_widget)
    _plot_type_row.setContentsMargins(0, 0, 0, 0)
    _plot_type_row.setSpacing(8)
    _plot_type_row.addWidget(qt_widgets.QLabel("Plot type"))
    plot_type_combo = qt_widgets.QComboBox()
    plot_type_combo.setObjectName("results-plot-type-filter")
    plot_type_combo.setProperty("persistComboSelection", True)
    _plot_type_row.addWidget(plot_type_combo, 1)
    plot_type_row_widget.setVisible(False)

    artifact_column.addWidget(artifact_scope_label)
    artifact_column.addWidget(artifact_filter_row_widget)
    artifact_column.addWidget(plot_type_row_widget)
    artifact_column.addWidget(artifact_summary_label)
    artifact_column.addWidget(report_summary_label)
    artifact_column.addWidget(artifacts_hint_label)
    artifact_column.addWidget(artifacts_list, 1)
    artifact_preview_section, _, artifact_preview_layout, _artifact_preview_toggle = (
        build_collapsible_section(
            root,
            title="Output preview",
            object_name="results-artifact-preview-section",
            expanded=True,
        )
    )
    artifact_preview_layout.addWidget(artifact_preview_title)
    artifact_preview_layout.addWidget(artifact_preview_metadata)
    artifact_preview_layout.addWidget(artifact_preview_stack, 1)
    artifact_column.addWidget(artifact_preview_section)
    artifact_column.addWidget(artifact_action_row_widget)

    content_row_widget.addWidget(runs_list)
    detail_widget = qt_widgets.QWidget(content_row_widget)
    detail_widget.setObjectName("results-detail-panel")
    detail_widget.setLayout(detail_column)
    content_row_widget.addWidget(detail_widget)
    artifact_widget = qt_widgets.QWidget(content_row_widget)
    artifact_widget.setObjectName("results-artifact-panel")
    artifact_widget.setLayout(artifact_column)
    content_row_widget.addWidget(artifact_widget)
    content_row_widget.setStretchFactor(0, 2)
    content_row_widget.setStretchFactor(1, 3)
    content_row_widget.setStretchFactor(2, 2)
    content_row_widget.setSizes([250, 450, 300])

    next_action_target = [""]

    layout.addWidget(combined_header)
    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(overview_label)
    layout.addWidget(comparison_label)
    layout.addWidget(comparison_action_label)
    layout.addWidget(comparison_action_button)
    layout.addWidget(stale_warning_label)
    layout.addWidget(next_action_label)
    layout.addWidget(next_action_button)
    layout.addWidget(content_row_widget, 1)
    action_row = qt_widgets.QHBoxLayout()
    action_row.addStretch(1)
    layout.addLayout(action_row)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=RESULTS_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        layouts=(artifact_filter_row, artifact_action_row),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=RESULTS_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(content_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    current_runs: list[RunRecord] = []
    current_run: RunRecord | None = None
    current_artifacts: list[ArtifactRecord] = []
    filtered_artifacts: list[ArtifactRecord] = []
    current_artifact: ArtifactRecord | None = None

    def _analysis_filter_text() -> str:
        return analysis_type_combo.currentText() or ANALYSIS_FILTER_ALL

    def _runs_by_id() -> dict[str, RunRecord]:
        return scenario_runs_by_id(project)

    def _refresh_next_action() -> None:
        action = recommend_results_next_action(project)
        if action is None:
            next_action_target[0] = ""
            next_action_label.clear()
            next_action_label.setVisible(False)
            next_action_button.setText("")
            next_action_button.setToolTip("")
            next_action_button.setVisible(False)
            return
        button_text, workflow_id, summary = action
        next_action_target[0] = workflow_id
        next_action_label.setText(summary)
        next_action_label.setVisible(True)
        next_action_button.setText(button_text)
        next_action_button.setToolTip(summary)
        next_action_button.setVisible(True)

    def _activate_comparison_target() -> None:
        target = select_results_comparison_target(project)
        if target is None:
            return
        project.set_active_scenario(target)
        _refresh()

    def _navigate_to_next_action() -> None:
        workflow_id = next_action_target[0]
        if not workflow_id:
            return
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(workflow_id)

    def _sync_review_button() -> None:
        review_button.setEnabled(
            any(
                action.isEnabled()
                for action in (
                    open_convergence_action,
                    open_gof_action,
                    open_residual_action,
                    open_eta_action,
                    open_profile_action,
                    open_latest_report_action,
                    open_latest_plot_action,
                    open_diagnostics_action,
                    open_npde_table_action,
                )
            )
        )

    def _sync_export_button() -> None:
        export_button.setEnabled(
            any(
                action.isEnabled()
                for action in (
                    export_artifact_action,
                    save_latest_plot_copy_action,
                    export_latest_report_pdf_action,
                    open_artifact_folder_action,
                )
            )
        )

    def _artifacts_for_run(run: RunRecord | None) -> list[ArtifactRecord]:
        if run is None:
            return list(project.active_scenario.artifacts)
        artifacts = [
            artifact
            for artifact in project.active_scenario.artifacts
            if artifact.artifact_id in run.artifact_ids
        ]
        if not artifacts:
            artifacts = artifact_service.list_for_run(project, run.run_id)
        if not artifacts:
            artifacts = [
                artifact
                for artifact in project.active_scenario.artifacts
                if artifact.artifact_id in run.artifact_ids
            ]
        analysis_type = run_analysis_type(run)
        runs_by_id = _runs_by_id()
        filtered_candidates = [
            artifact
            for artifact in artifacts
            if artifact_analysis_type(artifact, runs_by_id) == analysis_type
        ]
        return filtered_candidates or artifacts

    def _refresh_analysis_filter_options() -> None:
        current_analysis = _analysis_filter_text()
        options = review_analysis_type_options(project)
        analysis_type_combo.blockSignals(True)
        analysis_type_combo.clear()
        analysis_type_combo.addItems(options)
        analysis_type_combo.setCurrentText(
            current_analysis if current_analysis in options else ANALYSIS_FILTER_ALL
        )
        analysis_type_combo.blockSignals(False)

    def _kind_filter() -> str:
        return next((k for k, btn in kind_buttons.items() if btn.isChecked()), "")

    def _refresh_plot_type_options(artifacts: list) -> None:
        current_type = plot_type_combo.currentText() or "All plot types"
        plot_arts = [a for a in artifacts if a.kind == "plot"]
        options = artifact_plot_type_options(plot_arts)
        plot_type_combo.blockSignals(True)
        plot_type_combo.clear()
        plot_type_combo.addItems(options)
        plot_type_combo.setCurrentText(
            current_type if current_type in options else "All plot types"
        )
        plot_type_combo.blockSignals(False)

    def _on_kind_changed() -> None:
        kind = _kind_filter()
        plot_type_row_widget.setVisible(kind == "plot")
        _render_artifacts(current_run)

    def _refresh_report_summary() -> None:
        latest_report = latest_artifact(current_artifacts, kind="report")
        latest_plot = latest_artifact(current_artifacts, role="plot")
        has_report = bool(latest_report and latest_report.path)
        has_plot = bool(latest_plot and latest_plot.path)
        if latest_report is None:
            report_summary_label.setText("No report available yet.")
        else:
            report_summary_label.setText(f"Latest report · {latest_report.label}")
        open_latest_report_action.setEnabled(has_report)
        export_latest_report_pdf_action.setEnabled(has_report)
        open_latest_plot_action.setEnabled(has_plot)
        save_latest_plot_copy_action.setEnabled(has_plot)

    def _refresh_quick_actions() -> None:
        fit_scope_active = _analysis_filter_text() == "Fit" or (
            current_run is not None and run_analysis_type(current_run) == "Fit"
        )
        if not fit_scope_active:
            open_convergence_action.setEnabled(False)
            open_gof_action.setEnabled(False)
            open_residual_action.setEnabled(False)
            open_eta_action.setEnabled(False)
            open_profile_action.setEnabled(False)
            open_diagnostics_action.setEnabled(False)
            open_npde_table_action.setEnabled(False)
        else:
            open_convergence_action.setEnabled(
                bool(
                    (artifact := latest_artifact_for_plot_group(current_artifacts, "convergence"))
                    and artifact.path
                )
            )
            open_gof_action.setEnabled(
                bool(
                    (artifact := latest_artifact_for_plot_group(current_artifacts, "gof"))
                    and artifact.path
                )
            )
            open_residual_action.setEnabled(
                bool(
                    (artifact := latest_artifact_for_plot_group(current_artifacts, "residuals"))
                    and artifact.path
                )
            )
            open_eta_action.setEnabled(
                bool(
                    (artifact := latest_artifact_for_plot_group(current_artifacts, "eta"))
                    and artifact.path
                )
            )
            open_profile_action.setEnabled(
                bool(
                    (artifact := latest_artifact_for_plot_group(current_artifacts, "profiles"))
                    and artifact.path
                )
            )
            open_diagnostics_action.setEnabled(
                bool(
                    (artifact := latest_artifact(current_artifacts, role="diagnostics_table"))
                    and artifact.path
                )
            )
            open_npde_table_action.setEnabled(
                bool(
                    (artifact := latest_artifact(current_artifacts, role="npde_table"))
                    and artifact.path
                )
            )
        _sync_review_button()
        _sync_export_button()

    def _render_artifact_preview(artifact: ArtifactRecord | None) -> None:
        nonlocal current_artifact
        current_artifact = artifact

        def _on_has_path(has_path: bool) -> None:
            open_artifact_button.setEnabled(has_path)
            export_artifact_action.setEnabled(has_path)
            open_artifact_folder_action.setEnabled(has_path)
            _sync_export_button()

        preview_panel_obj.render(artifact, _runs_by_id(), on_has_path=_on_has_path)

    def _open_selected_artifact() -> None:
        open_output_file(current_artifact)

    def _open_artifact_list_item(item) -> None:
        index = artifacts_list.row(item)
        if index < 0 or index >= len(filtered_artifacts):
            return
        open_output_file(filtered_artifacts[index])

    def _open_selected_artifact_folder() -> None:
        if current_artifact is None or not current_artifact.path:
            return
        qt_gui.QDesktopServices.openUrl(
            qt_core.QUrl.fromLocalFile(str(Path(current_artifact.path).resolve().parent))
        )

    def _open_latest_report() -> None:
        open_output_file(latest_artifact(current_artifacts, kind="report"))

    def _open_latest_plot() -> None:
        open_output_file(latest_artifact(current_artifacts, role="plot"))

    def _open_plot_group(group_key: str) -> None:
        open_output_file(latest_artifact_for_plot_group(current_artifacts, group_key))

    def _open_role_artifact(role: str) -> None:
        open_output_file(latest_artifact(current_artifacts, role=role))

    def _call_root_callback(callback_name: str) -> None:
        callback = getattr(root, callback_name, None)
        if callable(callback):
            callback()

    def _export_selected_artifact() -> None:
        export_output_file(current_artifact, title="Save output copy", parent=root)

    def _export_latest_plot() -> None:
        export_output_file(
            latest_artifact(current_artifacts, role="plot"),
            title="Save latest plot copy",
            parent=root,
        )

    def _render_artifacts(run: RunRecord | None) -> None:
        nonlocal current_artifacts, filtered_artifacts
        runs_by_id = _runs_by_id()
        analysis_filter = _analysis_filter_text()
        selected_artifact_id = (
            current_artifact.artifact_id if current_artifact is not None else None
        )
        current_artifacts = _artifacts_for_run(run)
        _refresh_plot_type_options(current_artifacts)
        kind = _kind_filter()
        plot_type_filter = plot_type_combo.currentText() or "All plot types"
        filtered_artifacts = filter_artifacts(
            current_artifacts,
            analysis_filter=analysis_filter,
            runs_by_id=runs_by_id,
            kind_filter=kind if kind else "All kinds",
            plot_type_filter=plot_type_filter,
        )
        scope_suffix = (
            "" if analysis_filter == ANALYSIS_FILTER_ALL else f" • Analysis {analysis_filter}"
        )
        artifact_scope_label.setText(
            (
                "Outputs for selected run"
                if run is not None
                else f"Outputs for scenario {project.active_scenario.name}"
            )
            + scope_suffix
        )
        artifact_summary_label.setText(format_artifact_summary(filtered_artifacts))
        _refresh_report_summary()
        _refresh_quick_actions()
        artifacts_list.clear()
        if not filtered_artifacts:
            artifacts_list.addItem("No outputs available for this filter.")
            _render_artifact_preview(None)
            return
        for artifact in filtered_artifacts:
            item = qt_widgets.QListWidgetItem(format_artifact_label(artifact, runs_by_id))
            item.setToolTip(format_artifact_tooltip(artifact, runs_by_id))
            item.setData(qt_core.Qt.ItemDataRole.UserRole, artifact.artifact_id)
            artifacts_list.addItem(item)
        selected_index = next(
            (
                index
                for index, artifact in enumerate(filtered_artifacts)
                if artifact.artifact_id == selected_artifact_id
            ),
            0,
        )
        artifacts_list.setCurrentRow(selected_index)

    def _handle_artifact_selection_changed(index: int) -> None:
        if index < 0 or index >= len(filtered_artifacts):
            _render_artifact_preview(None)
            return
        _render_artifact_preview(filtered_artifacts[index])

    def _render_run(run: RunRecord | None) -> None:
        nonlocal current_run
        current_run = run
        detail_label.setText(format_run_details(run))
        metadata_label.setText(format_run_metadata(run))
        log_output.setPlainText("\n".join(run.log_lines) if run is not None else "")
        _render_artifacts(run)

    def _refresh() -> None:
        nonlocal current_runs
        refresh_combined_header()
        _refresh_analysis_filter_options()
        current_runs = review_runs(project, _analysis_filter_text())
        overview_label.setText(format_results_overview(project))
        comparison_label.setText(format_results_comparison_summary(project))
        comparison_action_label.setText(format_results_comparison_action(project))
        comparison_action_button.setEnabled(select_results_comparison_target(project) is not None)
        stale_warning_label.setText(format_results_stale_warning(project))
        stale_warning_label.setVisible(bool(stale_warning_label.text()))
        _refresh_next_action()
        runs_list.clear()
        selected_run_id = current_run.run_id if current_run is not None else None
        if not current_runs:
            runs_list.addItem("No analysis runs yet for this filter.")
            _render_run(None)
            return
        for index, run in enumerate(current_runs, start=1):
            item = qt_widgets.QListWidgetItem(format_run_label(run, index))
            item.setToolTip(f"Run ID: {run.run_id}")
            item.setData(qt_core.Qt.ItemDataRole.UserRole, run.run_id)
            runs_list.addItem(item)
        if selected_run_id is not None:
            for index, run in enumerate(current_runs):
                if run.run_id == selected_run_id:
                    runs_list.setCurrentRow(index)
                    break
            else:
                runs_list.setCurrentRow(0)
            return
        runs_list.setCurrentRow(0)

    def _handle_selection_changed(index: int) -> None:
        if index < 0 or index >= len(current_runs):
            _render_run(None)
            return
        _render_run(current_runs[index])

    runs_list.currentRowChanged.connect(_handle_selection_changed)
    analysis_type_combo.currentTextChanged.connect(lambda _text: _refresh())
    kind_button_group.buttonClicked.connect(lambda _btn: _on_kind_changed())
    plot_type_combo.currentTextChanged.connect(lambda _text: _render_artifacts(current_run))
    artifacts_list.currentRowChanged.connect(_handle_artifact_selection_changed)
    artifacts_list.itemActivated.connect(_open_artifact_list_item)
    artifacts_list.itemDoubleClicked.connect(_open_artifact_list_item)
    open_artifact_button.clicked.connect(_open_selected_artifact)
    open_convergence_action.triggered.connect(lambda: _open_plot_group("convergence"))
    open_gof_action.triggered.connect(lambda: _open_plot_group("gof"))
    open_residual_action.triggered.connect(lambda: _open_plot_group("residuals"))
    open_eta_action.triggered.connect(lambda: _open_plot_group("eta"))
    open_profile_action.triggered.connect(lambda: _open_plot_group("profiles"))
    open_latest_report_action.triggered.connect(_open_latest_report)
    open_latest_plot_action.triggered.connect(_open_latest_plot)
    open_diagnostics_action.triggered.connect(lambda: _open_role_artifact("diagnostics_table"))
    open_npde_table_action.triggered.connect(lambda: _open_role_artifact("npde_table"))
    export_artifact_action.triggered.connect(_export_selected_artifact)
    save_latest_plot_copy_action.triggered.connect(_export_latest_plot)
    export_latest_report_pdf_action.triggered.connect(
        lambda: _call_root_callback("_project_export_latest_report_pdf")
    )
    open_artifact_folder_action.triggered.connect(_open_selected_artifact_folder)
    comparison_action_button.clicked.connect(_activate_comparison_target)
    next_action_button.clicked.connect(_navigate_to_next_action)
    root._open_latest_report = _open_latest_report  # type: ignore[attr-defined]
    root._open_latest_plot = _open_latest_plot  # type: ignore[attr-defined]
    root._export_latest_plot = _export_latest_plot  # type: ignore[attr-defined]
    root._export_selected_artifact = _export_selected_artifact  # type: ignore[attr-defined]
    root._open_selected_artifact = _open_selected_artifact  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh  # type: ignore[attr-defined]
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_context_header = refresh_combined_header  # type: ignore[attr-defined]

    _refresh()
    _apply_responsive_layout()
    return root
