"""Advanced GUI workflow page for post-fit VPC, bootstrap, and design workflows."""

from __future__ import annotations

import shutil
from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    default_workspace_root_path,
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
    with_last_file_dialog_dir,
)
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.bootstrap_service import (
    BootstrapConfig,
    BootstrapService,
)
from openpkpd_gui.services.design_service import DesignConfig, DesignService
from openpkpd_gui.services.fit_service import FitService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.vpc_service import VPCConfig, VPCService
from openpkpd_gui.widgets.collapsible_section import build_collapsible_section
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.responsive_layout import (
    install_responsive_box_layouts,
    install_responsive_splitters,
)
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page
from openpkpd_gui.workflows.results_workflow import (
    artifact_plot_type,
    artifact_preview_kind,
    artifact_role,
    format_artifact_label,
    format_artifact_metadata,
)

ADVANCED_RESPONSIVE_LAYOUT_BREAKPOINT = 1000
VPC_PLOT_TYPES = VPCService.VPC_PLOT_TYPES


def latest_vpc_run(workspace: Workspace) -> RunRecord | None:
    for run in reversed(workspace.active_scenario.runs):
        if run.workflow == "vpc":
            return run
    return None


def vpc_artifacts(workspace: Workspace) -> list[ArtifactRecord]:
    return [
        artifact
        for artifact in workspace.active_scenario.artifacts
        if artifact_role(artifact) == "vpc_summary"
        or artifact_plot_type(artifact) in VPC_PLOT_TYPES
    ]


def bootstrap_artifacts(workspace: Workspace) -> list[ArtifactRecord]:
    return [
        artifact
        for artifact in workspace.active_scenario.artifacts
        if artifact_role(artifact)
        in {"bootstrap_summary", "bootstrap_ci_table", "bootstrap_samples"}
    ]


def design_artifacts(workspace: Workspace) -> list[ArtifactRecord]:
    return [
        artifact
        for artifact in workspace.active_scenario.artifacts
        if artifact_role(artifact)
        in {
            "design_summary",
            "design_metrics",
            "design_schedule",
            "design_fim",
            "design_expected_se",
        }
    ]


def latest_vpc_artifact(
    artifacts: list[ArtifactRecord], *, role: str | None = None
) -> ArtifactRecord | None:
    for artifact in reversed(artifacts):
        if role is not None and artifact_role(artifact) != role:
            continue
        if role is None and artifact_plot_type(artifact) != "vpc":
            continue
        return artifact
    return None


def latest_bootstrap_run(workspace: Workspace) -> RunRecord | None:
    for run in reversed(workspace.active_scenario.runs):
        if run.workflow == "bootstrap":
            return run
    return None


def latest_bootstrap_artifact(
    artifacts: list[ArtifactRecord], *, role: str | None = None
) -> ArtifactRecord | None:
    for artifact in reversed(artifacts):
        if role is not None and artifact_role(artifact) != role:
            continue
        return artifact
    return None


def latest_design_run(workspace: Workspace) -> RunRecord | None:
    for run in reversed(workspace.active_scenario.runs):
        if run.workflow == "design":
            return run
    return None


def latest_design_artifact(
    artifacts: list[ArtifactRecord], *, role: str | None = None
) -> ArtifactRecord | None:
    for artifact in reversed(artifacts):
        if role is not None and artifact_role(artifact) != role:
            continue
        return artifact
    return None


def advanced_artifacts(workspace: Workspace) -> list[ArtifactRecord]:
    return [
        artifact
        for artifact in workspace.active_scenario.artifacts
        if artifact_role(artifact)
        in {
            "vpc_summary",
            "bootstrap_summary",
            "bootstrap_ci_table",
            "bootstrap_samples",
            "design_summary",
            "design_metrics",
            "design_schedule",
            "design_fim",
            "design_expected_se",
        }
        or artifact_plot_type(artifact) in VPC_PLOT_TYPES
    ]


def filter_advanced_artifacts(artifacts: list[ArtifactRecord], scope: str) -> list[ArtifactRecord]:
    if scope == "all":
        return list(artifacts)
    if scope == "vpc":
        return [
            artifact
            for artifact in artifacts
            if artifact_role(artifact) == "vpc_summary"
            or artifact_plot_type(artifact) in VPC_PLOT_TYPES
        ]
    if scope == "bootstrap":
        return [
            artifact
            for artifact in artifacts
            if artifact_role(artifact)
            in {"bootstrap_summary", "bootstrap_ci_table", "bootstrap_samples"}
        ]
    if scope == "design":
        return [
            artifact
            for artifact in artifacts
            if artifact_role(artifact)
            in {
                "design_summary",
                "design_metrics",
                "design_schedule",
                "design_fim",
                "design_expected_se",
            }
        ]
    return list(artifacts)


def artifact_scope_empty_message(scope: str) -> str:
    if scope == "vpc":
        return "No VPC artifacts are available yet."
    if scope == "bootstrap":
        return "No bootstrap artifacts are available yet."
    if scope == "design":
        return "No design artifacts are available yet."
    return "No Advanced artifacts are available yet."


def format_artifact_scope_summary(scope: str, visible_count: int, total_count: int) -> str:
    scope_label = {
        "all": "all advanced artifacts",
        "vpc": "VPC artifacts",
        "bootstrap": "bootstrap artifacts",
        "design": "design artifacts",
    }.get(scope, "artifacts")
    return f"Showing {visible_count} of {total_count} {scope_label}."


def format_advanced_overview(workspace: Workspace) -> str:
    scenario = workspace.active_scenario
    fit_runs = [run for run in scenario.runs if run.workflow == "fit"]
    return (
        f"Scenario {scenario.name} • {len(fit_runs)} fit runs • {len(vpc_artifacts(workspace))} VPC artifacts • "
        f"{len(bootstrap_artifacts(workspace))} bootstrap artifacts • "
        f"{len(design_artifacts(workspace))} design artifacts • "
        f"dataset {'ready' if scenario.active_dataset else 'missing'}"
    )


def format_vpc_generation_status(
    workspace: Workspace,
    *,
    fit_context_available: bool,
    vpc_available: bool,
    generation_running: bool = False,
) -> str:
    fit_run = next(
        (run for run in reversed(workspace.active_scenario.runs) if run.workflow == "fit"), None
    )
    if generation_running:
        return "A VPC run is currently running in the background. Wait for it to finish or refresh this page."
    if fit_run is None or fit_run.status != RunStatus.SUCCEEDED or not fit_context_available:
        return "Generate VPC from the latest reusable successful fit for this scenario."
    if vpc_available:
        return "VPC artifacts are already available. Adjust the controls below to generate an updated VPC run."
    return "Generate VPC on demand from the latest successful fit using the controls below."


def format_bootstrap_generation_status(
    workspace: Workspace,
    *,
    fit_context_available: bool,
    bootstrap_available: bool,
    generation_running: bool = False,
) -> str:
    fit_run = next(
        (run for run in reversed(workspace.active_scenario.runs) if run.workflow == "fit"), None
    )
    if generation_running:
        return "A bootstrap run is currently running in the background. Wait for it to finish or refresh this page."
    if fit_run is None or fit_run.status != RunStatus.SUCCEEDED or not fit_context_available:
        return "Generate bootstrap summaries from the latest reusable successful fit for this scenario."
    if bootstrap_available:
        return "Bootstrap artifacts are already available. Adjust the controls below to generate an updated bootstrap run."
    return "Generate bootstrap summaries on demand from the latest successful fit using the controls below."


def format_design_generation_status(
    workspace: Workspace,
    *,
    fit_context_available: bool,
    design_available: bool,
    generation_running: bool = False,
) -> str:
    fit_run = next(
        (run for run in reversed(workspace.active_scenario.runs) if run.workflow == "fit"), None
    )
    if generation_running:
        return "A design optimization is currently running in the background. Wait for it to finish or refresh this page."
    if fit_run is None or fit_run.status != RunStatus.SUCCEEDED or not fit_context_available:
        return "Generate optimal design summaries from the latest reusable successful fit for this scenario."
    if design_available:
        return "Design artifacts are already available. Adjust the controls below to generate an updated design run."
    return "Generate optimal design summaries on demand from the latest successful fit using the controls below."


def recommend_vpc_next_action(
    *,
    fit_context_available: bool,
    latest_run: RunRecord | None,
    artifacts: list[ArtifactRecord],
) -> tuple[str, str, str] | None:
    """Return the primary CTA for the VPC tab."""
    if latest_run is not None and latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        return None
    if (plot := latest_vpc_artifact(artifacts)) is not None and plot.path:
        return (
            "Open latest VPC plot",
            "__open_vpc_plot__",
            "Latest VPC outputs are already available. Open the newest plot or adjust the controls below to generate an updated run.",
        )
    if (summary := latest_vpc_artifact(artifacts, role="vpc_summary")) is not None and summary.path:
        return (
            "Open latest VPC summary",
            "__open_vpc_summary__",
            "Latest VPC outputs are already available. Open the newest summary or adjust the controls below to generate an updated run.",
        )
    if artifacts:
        return (
            "Open Results",
            "results",
            "Latest VPC outputs are already available. Review them in Results or adjust the controls below to generate an updated run.",
        )
    if not fit_context_available:
        return (
            "Open Fit",
            "fit",
            "Complete a successful fit for this scenario before generating VPC outputs here.",
        )
    return None


def recommend_bootstrap_next_action(
    *,
    fit_context_available: bool,
    latest_run: RunRecord | None,
    artifacts: list[ArtifactRecord],
) -> tuple[str, str, str] | None:
    """Return the primary CTA for the Bootstrap tab."""
    if latest_run is not None and latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        return None
    if (
        summary := latest_bootstrap_artifact(artifacts, role="bootstrap_summary")
    ) is not None and summary.path:
        return (
            "Open latest bootstrap summary",
            "__open_bootstrap_summary__",
            "Latest bootstrap outputs are already available. Open the newest summary or adjust the controls below to generate an updated run.",
        )
    if (
        ci_table := latest_bootstrap_artifact(artifacts, role="bootstrap_ci_table")
    ) is not None and ci_table.path:
        return (
            "Open latest bootstrap CI table",
            "__open_bootstrap_ci__",
            "Latest bootstrap outputs are already available. Open the newest confidence interval table or adjust the controls below to generate an updated run.",
        )
    if (
        samples := latest_bootstrap_artifact(artifacts, role="bootstrap_samples")
    ) is not None and samples.path:
        return (
            "Open latest bootstrap samples",
            "__open_bootstrap_samples__",
            "Latest bootstrap outputs are already available. Open the newest samples or adjust the controls below to generate an updated run.",
        )
    if artifacts:
        return (
            "Open Results",
            "results",
            "Latest bootstrap outputs are already available. Review them in Results or adjust the controls below to generate an updated run.",
        )
    if not fit_context_available:
        return (
            "Open Fit",
            "fit",
            "Complete a successful fit for this scenario before generating bootstrap outputs here.",
        )
    return None


def recommend_design_next_action(
    *,
    fit_context_available: bool,
    latest_run: RunRecord | None,
    artifacts: list[ArtifactRecord],
) -> tuple[str, str, str] | None:
    """Return the primary CTA for the Design tab."""
    if latest_run is not None and latest_run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
        return None
    if (
        summary := latest_design_artifact(artifacts, role="design_summary")
    ) is not None and summary.path:
        return (
            "Open latest design summary",
            "__open_design_summary__",
            "Latest design outputs are already available. Open the newest summary or adjust the controls below to generate an updated run.",
        )
    if (
        schedule := latest_design_artifact(artifacts, role="design_schedule")
    ) is not None and schedule.path:
        return (
            "Open latest design schedule",
            "__open_design_schedule__",
            "Latest design outputs are already available. Open the newest schedule or adjust the controls below to generate an updated run.",
        )
    if artifacts:
        return (
            "Open Results",
            "results",
            "Latest design outputs are already available. Review them in Results or adjust the controls below to generate an updated run.",
        )
    if not fit_context_available:
        return (
            "Open Fit",
            "fit",
            "Complete a successful fit for this scenario before generating design outputs here.",
        )
    return None


def build_advanced_workflow(
    project: Workspace,
    artifact_service: ArtifactService | None = None,
    fit_service: FitService | None = None,
    vpc_service: VPCService | None = None,
    bootstrap_service: BootstrapService | None = None,
    design_service: DesignService | None = None,
    project_service: ProjectService | None = None,
    job_runner: JobRunner | None = None,
    preferences: list | None = None,
):
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    artifact_service = artifact_service or ArtifactService()
    fit_service = fit_service or FitService()
    vpc_service = vpc_service or VPCService()
    bootstrap_service = bootstrap_service or BootstrapService()
    design_service = design_service or DesignService()
    project_service = project_service or ProjectService()
    job_runner = job_runner or JobRunner(max_workers=1)

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="advanced-workflow"
    )

    title_label = qt_widgets.QLabel("Advanced workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)
    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_advanced():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_advanced"))

    hint_widget, _ = build_dismissible_hint(
        "Use this page to run post-fit VPC diagnostics, bootstrap summaries, and optimal-design workflows.",
        dismissed="hint_advanced" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_advanced,
    )
    overview_label = qt_widgets.QLabel(format_advanced_overview(project))
    overview_label.setWordWrap(True)
    tab_widget = qt_widgets.QTabWidget()
    tab_widget.setObjectName("advanced-tab-widget")
    vpc_tab = qt_widgets.QWidget()
    vpc_tab.setObjectName("advanced-vpc-tab")
    vpc_layout = qt_widgets.QVBoxLayout(vpc_tab)
    bootstrap_tab = qt_widgets.QWidget()
    bootstrap_tab.setObjectName("advanced-bootstrap-tab")
    bootstrap_layout = qt_widgets.QVBoxLayout(bootstrap_tab)
    design_tab = qt_widgets.QWidget()
    design_tab.setObjectName("advanced-design-tab")
    design_layout = qt_widgets.QVBoxLayout(design_tab)
    artifacts_tab = qt_widgets.QWidget()
    artifacts_tab.setObjectName("advanced-artifacts-tab")
    artifacts_layout = qt_widgets.QVBoxLayout(artifacts_tab)
    vpc_heading = qt_widgets.QLabel("Visual predictive checks")
    vpc_heading_font = vpc_heading.font()
    vpc_heading_font.setBold(True)
    vpc_heading.setFont(vpc_heading_font)
    status_label = qt_widgets.QLabel(
        format_vpc_generation_status(project, fit_context_available=False, vpc_available=False)
    )
    status_label.setObjectName("advanced-vpc-status-label")
    status_label.setWordWrap(True)
    vpc_next_action_label = qt_widgets.QLabel("")
    vpc_next_action_label.setObjectName("advanced-vpc-next-action-label")
    vpc_next_action_label.setWordWrap(True)
    vpc_next_action_label.setVisible(False)
    vpc_next_action_button = qt_widgets.QPushButton("")
    vpc_next_action_button.setObjectName("advanced-vpc-next-action-button")
    vpc_next_action_button.setProperty("primaryAction", True)
    vpc_next_action_button.setMinimumHeight(36)
    vpc_next_action_button.setVisible(False)
    latest_run_label = qt_widgets.QLabel("No VPC run has been started yet.")
    latest_run_label.setObjectName("advanced-vpc-latest-run-label")
    latest_run_label.setWordWrap(True)

    controls_widget = qt_widgets.QWidget(vpc_tab)
    controls_widget.setObjectName("advanced-vpc-controls-row")
    controls = qt_widgets.QHBoxLayout(controls_widget)
    controls.setContentsMargins(0, 0, 0, 0)
    controls.setSpacing(8)
    controls.addWidget(qt_widgets.QLabel("Replicates"))
    replicates_spin = qt_widgets.QSpinBox()
    replicates_spin.setObjectName("advanced-vpc-replicates-spinbox")
    replicates_spin.setRange(10, 5000)
    replicates_spin.setValue(200)
    controls.addWidget(replicates_spin)
    controls.addWidget(qt_widgets.QLabel("Bins"))
    bins_spin = qt_widgets.QSpinBox()
    bins_spin.setObjectName("advanced-vpc-bins-spinbox")
    bins_spin.setRange(2, 50)
    bins_spin.setValue(10)
    controls.addWidget(bins_spin)
    controls.addWidget(qt_widgets.QLabel("Seed"))
    seed_spin = qt_widgets.QSpinBox()
    seed_spin.setObjectName("advanced-vpc-seed-spinbox")
    seed_spin.setRange(1, 999999)
    seed_spin.setValue(42)
    controls.addWidget(seed_spin)
    pc_checkbox = qt_widgets.QCheckBox("pcVPC")
    pc_checkbox.setObjectName("advanced-vpc-pc-checkbox")
    pc_checkbox.setToolTip(
        "Prediction-corrected VPC: normalise DV and predictions by median PRED within each bin."
    )
    controls.addWidget(pc_checkbox)
    controls.addWidget(qt_widgets.QLabel("Stratify by:"))
    stratify_combo = qt_widgets.QComboBox()
    stratify_combo.setObjectName("advanced-vpc-stratify-combo")
    stratify_combo.setToolTip(
        "Optional covariate column to stratify the VPC (e.g. DOSE, SEX). "
        "Produces one VPC panel per stratum."
    )
    stratify_combo.setMinimumWidth(100)
    stratify_combo.addItem("None", userData=None)
    controls.addWidget(stratify_combo)
    controls.addStretch(1)
    vpc_settings_section, _, vpc_settings_layout, _vpc_settings_toggle = build_collapsible_section(
        vpc_tab,
        title="Generation settings",
        object_name="advanced-vpc-settings-section",
        expanded=True,
        framed=True,
    )
    vpc_settings_layout.addWidget(controls_widget)

    bootstrap_heading = qt_widgets.QLabel("Bootstrap confidence intervals")
    bootstrap_heading_font = bootstrap_heading.font()
    bootstrap_heading_font.setBold(True)
    bootstrap_heading.setFont(bootstrap_heading_font)
    bootstrap_status_label = qt_widgets.QLabel(
        format_bootstrap_generation_status(
            project, fit_context_available=False, bootstrap_available=False
        )
    )
    bootstrap_status_label.setObjectName("advanced-bootstrap-status-label")
    bootstrap_status_label.setWordWrap(True)
    bootstrap_next_action_label = qt_widgets.QLabel("")
    bootstrap_next_action_label.setObjectName("advanced-bootstrap-next-action-label")
    bootstrap_next_action_label.setWordWrap(True)
    bootstrap_next_action_label.setVisible(False)
    bootstrap_next_action_button = qt_widgets.QPushButton("")
    bootstrap_next_action_button.setObjectName("advanced-bootstrap-next-action-button")
    bootstrap_next_action_button.setProperty("primaryAction", True)
    bootstrap_next_action_button.setMinimumHeight(36)
    bootstrap_next_action_button.setVisible(False)
    bootstrap_run_label = qt_widgets.QLabel("No bootstrap run has been started yet.")
    bootstrap_run_label.setObjectName("advanced-bootstrap-latest-run-label")
    bootstrap_run_label.setWordWrap(True)
    bootstrap_controls_widget = qt_widgets.QWidget(bootstrap_tab)
    bootstrap_controls_widget.setObjectName("advanced-bootstrap-controls-row")
    bootstrap_controls = qt_widgets.QHBoxLayout(bootstrap_controls_widget)
    bootstrap_controls.setContentsMargins(0, 0, 0, 0)
    bootstrap_controls.setSpacing(8)
    bootstrap_controls.addWidget(qt_widgets.QLabel("Replicates"))
    bootstrap_replicates_spin = qt_widgets.QSpinBox()
    bootstrap_replicates_spin.setObjectName("advanced-bootstrap-replicates-spinbox")
    bootstrap_replicates_spin.setRange(10, 2000)
    bootstrap_replicates_spin.setValue(100)
    bootstrap_controls.addWidget(bootstrap_replicates_spin)
    bootstrap_controls.addWidget(qt_widgets.QLabel("Jobs"))
    bootstrap_jobs_spin = qt_widgets.QSpinBox()
    bootstrap_jobs_spin.setObjectName("advanced-bootstrap-jobs-spinbox")
    bootstrap_jobs_spin.setRange(1, 64)
    bootstrap_jobs_spin.setValue(1)
    bootstrap_controls.addWidget(bootstrap_jobs_spin)
    bootstrap_controls.addWidget(qt_widgets.QLabel("Seed"))
    bootstrap_seed_spin = qt_widgets.QSpinBox()
    bootstrap_seed_spin.setObjectName("advanced-bootstrap-seed-spinbox")
    bootstrap_seed_spin.setRange(1, 999999)
    bootstrap_seed_spin.setValue(42)
    bootstrap_controls.addWidget(bootstrap_seed_spin)
    bootstrap_controls.addWidget(qt_widgets.QLabel("CI level"))
    bootstrap_ci_spin = qt_widgets.QDoubleSpinBox()
    bootstrap_ci_spin.setObjectName("advanced-bootstrap-ci-spinbox")
    bootstrap_ci_spin.setDecimals(2)
    bootstrap_ci_spin.setSingleStep(0.01)
    bootstrap_ci_spin.setRange(0.5, 0.99)
    bootstrap_ci_spin.setValue(0.95)
    bootstrap_controls.addWidget(bootstrap_ci_spin)
    bootstrap_controls.addStretch(1)
    bootstrap_settings_section, _, bootstrap_settings_layout, _bootstrap_settings_toggle = (
        build_collapsible_section(
            bootstrap_tab,
            title="Generation settings",
            object_name="advanced-bootstrap-settings-section",
            expanded=True,
            framed=True,
        )
    )
    bootstrap_settings_layout.addWidget(bootstrap_controls_widget)

    design_heading = qt_widgets.QLabel("Optimal design")
    design_heading_font = design_heading.font()
    design_heading_font.setBold(True)
    design_heading.setFont(design_heading_font)
    design_status_label = qt_widgets.QLabel(
        format_design_generation_status(
            project, fit_context_available=False, design_available=False
        )
    )
    design_status_label.setObjectName("advanced-design-status-label")
    design_status_label.setWordWrap(True)
    design_next_action_label = qt_widgets.QLabel("")
    design_next_action_label.setObjectName("advanced-design-next-action-label")
    design_next_action_label.setWordWrap(True)
    design_next_action_label.setVisible(False)
    design_next_action_button = qt_widgets.QPushButton("")
    design_next_action_button.setObjectName("advanced-design-next-action-button")
    design_next_action_button.setProperty("primaryAction", True)
    design_next_action_button.setMinimumHeight(36)
    design_next_action_button.setVisible(False)
    design_run_label = qt_widgets.QLabel("No design run has been started yet.")
    design_run_label.setObjectName("advanced-design-latest-run-label")
    design_run_label.setWordWrap(True)
    design_controls_widget = qt_widgets.QWidget(design_tab)
    design_controls_widget.setObjectName("advanced-design-controls-row")
    design_controls = qt_widgets.QHBoxLayout(design_controls_widget)
    design_controls.setContentsMargins(0, 0, 0, 0)
    design_controls.setSpacing(8)
    design_controls.addWidget(qt_widgets.QLabel("Samples"))
    design_samples_spin = qt_widgets.QSpinBox()
    design_samples_spin.setObjectName("advanced-design-samples-spinbox")
    design_samples_spin.setRange(2, 24)
    design_samples_spin.setValue(6)
    design_controls.addWidget(design_samples_spin)
    design_controls.addWidget(qt_widgets.QLabel("Subjects"))
    design_subjects_spin = qt_widgets.QSpinBox()
    design_subjects_spin.setObjectName("advanced-design-subjects-spinbox")
    design_subjects_spin.setRange(1, 10000)
    design_subjects_spin.setValue(10)
    design_controls.addWidget(design_subjects_spin)
    design_controls.addWidget(qt_widgets.QLabel("Min time"))
    design_t_min_spin = qt_widgets.QDoubleSpinBox()
    design_t_min_spin.setObjectName("advanced-design-tmin-spinbox")
    design_t_min_spin.setDecimals(2)
    design_t_min_spin.setRange(0.0, 1000.0)
    design_t_min_spin.setValue(0.0)
    design_controls.addWidget(design_t_min_spin)
    design_controls.addWidget(qt_widgets.QLabel("Max time"))
    design_t_max_spin = qt_widgets.QDoubleSpinBox()
    design_t_max_spin.setObjectName("advanced-design-tmax-spinbox")
    design_t_max_spin.setDecimals(2)
    design_t_max_spin.setRange(0.01, 1000.0)
    design_t_max_spin.setValue(24.0)
    design_controls.addWidget(design_t_max_spin)
    design_controls.addWidget(qt_widgets.QLabel("Criterion"))
    design_criterion_combo = qt_widgets.QComboBox()
    design_criterion_combo.setObjectName("advanced-design-criterion-combo")
    design_criterion_combo.addItem("D-optimal", "D")
    design_criterion_combo.addItem("A-optimal", "A")
    design_criterion_combo.addItem("E-optimal", "E")
    design_controls.addWidget(design_criterion_combo)
    design_controls.addWidget(qt_widgets.QLabel("Method"))
    design_method_combo = qt_widgets.QComboBox()
    design_method_combo.setObjectName("advanced-design-method-combo")
    design_method_combo.addItem("Differential evolution", "differential_evolution")
    design_method_combo.addItem("L-BFGS-B", "L-BFGS-B")
    design_controls.addWidget(design_method_combo)
    design_controls.addWidget(qt_widgets.QLabel("Restarts"))
    design_restarts_spin = qt_widgets.QSpinBox()
    design_restarts_spin.setObjectName("advanced-design-restarts-spinbox")
    design_restarts_spin.setRange(1, 50)
    design_restarts_spin.setValue(10)
    design_controls.addWidget(design_restarts_spin)
    design_controls.addStretch(1)
    design_settings_section, _, design_settings_layout, _design_settings_toggle = (
        build_collapsible_section(
            design_tab,
            title="Generation settings",
            object_name="advanced-design-settings-section",
            expanded=True,
            framed=True,
        )
    )
    design_settings_layout.addWidget(design_controls_widget)

    artifacts_heading = qt_widgets.QLabel("Artifacts hub")
    artifacts_heading_font = artifacts_heading.font()
    artifacts_heading_font.setBold(True)
    artifacts_heading.setFont(artifacts_heading_font)
    artifacts_hint = qt_widgets.QLabel(
        "Browse and preview artifacts generated by the VPC, bootstrap, and design workflows from one place."
    )
    artifacts_hint.setWordWrap(True)
    artifact_scope_row_widget = qt_widgets.QWidget(artifacts_tab)
    artifact_scope_row_widget.setObjectName("advanced-artifact-scope-row")
    artifact_scope_row = qt_widgets.QHBoxLayout(artifact_scope_row_widget)
    artifact_scope_row.setContentsMargins(0, 0, 0, 0)
    artifact_scope_row.setSpacing(8)
    artifact_scope_row.addWidget(qt_widgets.QLabel("Artifact scope"))
    artifact_scope_combo = qt_widgets.QComboBox()
    artifact_scope_combo.setObjectName("advanced-artifact-scope-combo")
    artifact_scope_combo.setProperty("persistComboSelection", True)
    artifact_scope_combo.addItem("All advanced artifacts", "all")
    artifact_scope_combo.addItem("VPC only", "vpc")
    artifact_scope_combo.addItem("Bootstrap only", "bootstrap")
    artifact_scope_combo.addItem("Design only", "design")
    artifact_scope_row.addWidget(artifact_scope_combo)
    artifact_scope_row.addStretch(1)
    artifact_scope_summary = qt_widgets.QLabel(format_artifact_scope_summary("all", 0, 0))
    artifact_scope_summary.setObjectName("advanced-artifact-scope-summary")
    artifact_scope_summary.setWordWrap(True)

    artifacts_list = qt_widgets.QListWidget()
    artifacts_list.setObjectName("advanced-artifacts-list")
    artifacts_list.setProperty("persistListSelection", True)
    preview_title = qt_widgets.QLabel("Select an Advanced artifact to preview.")
    preview_title.setObjectName("advanced-preview-title")
    preview_title.setWordWrap(True)
    preview_metadata = qt_widgets.QLabel(format_artifact_metadata(None))
    preview_metadata.setObjectName("advanced-preview-metadata")
    preview_metadata.setWordWrap(True)
    preview_stack = qt_widgets.QStackedWidget()
    placeholder = qt_widgets.QLabel("Preview supported for HTML, text, and image artifacts.")
    placeholder.setWordWrap(True)
    placeholder.setAlignment(qt_core.Qt.AlignmentFlag.AlignTop)
    browser = qt_widgets.QTextBrowser()
    browser.setObjectName("advanced-preview-browser")
    image = qt_widgets.QLabel()
    image.setObjectName("advanced-preview-image")
    image.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    scroll = qt_widgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(image)
    preview_stack.addWidget(placeholder)
    preview_stack.addWidget(browser)
    preview_stack.addWidget(scroll)

    artifact_content_row_widget = qt_widgets.QSplitter(artifacts_tab)
    artifact_content_row_widget.setObjectName("advanced-artifact-content-row")
    artifact_content_row_widget.setChildrenCollapsible(False)
    artifact_content_row_widget.setHandleWidth(8)

    artifact_list_panel = qt_widgets.QWidget(artifact_content_row_widget)
    artifact_list_panel.setObjectName("advanced-artifact-list-panel")
    artifact_list_column = qt_widgets.QVBoxLayout(artifact_list_panel)
    artifact_list_column.setContentsMargins(12, 12, 12, 12)
    artifact_list_column.setSpacing(8)
    artifact_list_column.addWidget(artifact_scope_summary)
    artifact_list_column.addWidget(artifacts_list, 1)

    artifact_preview_panel = qt_widgets.QWidget(artifact_content_row_widget)
    artifact_preview_panel.setObjectName("advanced-artifact-preview-panel")
    artifact_preview_column = qt_widgets.QVBoxLayout(artifact_preview_panel)
    artifact_preview_column.setContentsMargins(12, 12, 12, 12)
    artifact_preview_column.setSpacing(8)
    artifact_preview_section, _, artifact_preview_layout, _artifact_preview_toggle = (
        build_collapsible_section(
            artifact_preview_panel,
            title="Artifact preview",
            object_name="advanced-artifact-preview-section",
            expanded=True,
            framed=True,
        )
    )
    artifact_preview_layout.addWidget(preview_title)
    artifact_preview_layout.addWidget(preview_metadata)
    artifact_preview_layout.addWidget(preview_stack, 1)
    artifact_preview_column.addWidget(artifact_preview_section, 1)

    artifact_content_row_widget.addWidget(artifact_list_panel)
    artifact_content_row_widget.addWidget(artifact_preview_panel)
    artifact_content_row_widget.setStretchFactor(0, 2)
    artifact_content_row_widget.setStretchFactor(1, 3)
    artifact_content_row_widget.setSizes([400, 600])

    log_output = qt_widgets.QPlainTextEdit()
    log_output.setObjectName("advanced-vpc-log-output")
    log_output.setReadOnly(True)
    log_output.setPlaceholderText("VPC run logs will appear here.")
    vpc_log_section, _, vpc_log_layout, _vpc_log_toggle = build_collapsible_section(
        vpc_tab,
        title="Run log",
        object_name="advanced-vpc-log-section",
        expanded=False,
        framed=True,
    )
    vpc_log_layout.addWidget(log_output)
    bootstrap_log_output = qt_widgets.QPlainTextEdit()
    bootstrap_log_output.setObjectName("advanced-bootstrap-log-output")
    bootstrap_log_output.setReadOnly(True)
    bootstrap_log_output.setPlaceholderText("Bootstrap run logs will appear here.")
    bootstrap_log_section, _, bootstrap_log_layout, _bootstrap_log_toggle = (
        build_collapsible_section(
            bootstrap_tab,
            title="Run log",
            object_name="advanced-bootstrap-log-section",
            expanded=False,
            framed=True,
        )
    )
    bootstrap_log_layout.addWidget(bootstrap_log_output)
    design_log_output = qt_widgets.QPlainTextEdit()
    design_log_output.setObjectName("advanced-design-log-output")
    design_log_output.setReadOnly(True)
    design_log_output.setPlaceholderText("Design run logs will appear here.")
    design_log_section, _, design_log_layout, _design_log_toggle = build_collapsible_section(
        design_tab,
        title="Run log",
        object_name="advanced-design-log-section",
        expanded=False,
        framed=True,
    )
    design_log_layout.addWidget(design_log_output)

    generate_button = qt_widgets.QPushButton("Generate VPC")
    generate_button.setObjectName("advanced-generate-vpc-button")
    generate_button.setProperty("primaryAction", True)
    vpc_cancel_button = qt_widgets.QPushButton("Cancel")
    vpc_cancel_button.setObjectName("advanced-cancel-vpc-button")
    vpc_cancel_button.setEnabled(False)
    open_plot_button = qt_widgets.QPushButton("Open latest VPC plot")
    open_plot_button.setObjectName("advanced-open-vpc-plot-button")
    open_plot_button.setEnabled(False)
    open_summary_button = qt_widgets.QPushButton("Open latest VPC summary")
    open_summary_button.setObjectName("advanced-open-vpc-summary-button")
    open_summary_button.setEnabled(False)
    bootstrap_generate_button = qt_widgets.QPushButton("Generate bootstrap")
    bootstrap_generate_button.setObjectName("advanced-generate-bootstrap-button")
    bootstrap_generate_button.setProperty("primaryAction", True)
    bootstrap_cancel_button = qt_widgets.QPushButton("Cancel")
    bootstrap_cancel_button.setObjectName("advanced-cancel-bootstrap-button")
    bootstrap_cancel_button.setEnabled(False)
    bootstrap_open_summary_button = qt_widgets.QPushButton("Open latest bootstrap summary")
    bootstrap_open_summary_button.setObjectName("advanced-open-bootstrap-summary-button")
    bootstrap_open_summary_button.setEnabled(False)
    bootstrap_open_ci_button = qt_widgets.QPushButton("Open latest bootstrap CI table")
    bootstrap_open_ci_button.setObjectName("advanced-open-bootstrap-ci-button")
    bootstrap_open_ci_button.setEnabled(False)
    bootstrap_open_samples_button = qt_widgets.QPushButton("Open latest bootstrap samples")
    bootstrap_open_samples_button.setObjectName("advanced-open-bootstrap-samples-button")
    bootstrap_open_samples_button.setEnabled(False)
    design_generate_button = qt_widgets.QPushButton("Generate design")
    design_generate_button.setObjectName("advanced-generate-design-button")
    design_generate_button.setProperty("primaryAction", True)
    design_cancel_button = qt_widgets.QPushButton("Cancel")
    design_cancel_button.setObjectName("advanced-cancel-design-button")
    design_cancel_button.setEnabled(False)
    design_open_summary_button = qt_widgets.QPushButton("Open latest design summary")
    design_open_summary_button.setObjectName("advanced-open-design-summary-button")
    design_open_summary_button.setEnabled(False)
    design_open_schedule_button = qt_widgets.QPushButton("Open latest design schedule")
    design_open_schedule_button.setObjectName("advanced-open-design-schedule-button")
    design_open_schedule_button.setEnabled(False)
    export_button = qt_widgets.QPushButton("Save selected artifact copy…")
    export_button.setObjectName("advanced-export-artifact-button")
    export_button.setEnabled(False)
    open_button = qt_widgets.QPushButton("Open selected artifact")
    open_button.setObjectName("advanced-open-artifact-button")
    open_button.setEnabled(False)
    open_folder_button = qt_widgets.QPushButton("Open artifact folder")
    open_folder_button.setObjectName("advanced-open-folder-button")
    open_folder_button.setEnabled(False)
    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(overview_label)
    vpc_layout.addWidget(vpc_heading)
    vpc_layout.addWidget(status_label)
    vpc_layout.addWidget(vpc_next_action_label)
    vpc_layout.addWidget(vpc_next_action_button)
    vpc_layout.addWidget(latest_run_label)
    vpc_layout.addWidget(vpc_settings_section)
    vpc_layout.addWidget(vpc_log_section, 1)
    vpc_actions_widget = qt_widgets.QWidget(vpc_tab)
    vpc_actions_widget.setObjectName("advanced-vpc-actions-row")
    vpc_actions = qt_widgets.QHBoxLayout(vpc_actions_widget)
    vpc_actions.setContentsMargins(0, 0, 0, 0)
    vpc_actions.setSpacing(8)
    vpc_progress = qt_widgets.QProgressBar(vpc_tab)
    vpc_progress.setObjectName("advanced-vpc-progress")
    vpc_progress.setRange(0, 0)
    vpc_progress.setFixedHeight(20)
    vpc_progress.setVisible(False)
    bootstrap_progress = qt_widgets.QProgressBar(bootstrap_tab)
    bootstrap_progress.setObjectName("advanced-bootstrap-progress")
    bootstrap_progress.setRange(0, 0)
    bootstrap_progress.setFixedHeight(20)
    bootstrap_progress.setVisible(False)
    design_progress = qt_widgets.QProgressBar(design_tab)
    design_progress.setObjectName("advanced-design-progress")
    design_progress.setRange(0, 0)
    design_progress.setFixedHeight(20)
    design_progress.setVisible(False)

    for button in [open_plot_button, open_summary_button]:
        vpc_actions.addWidget(button)
    vpc_actions.addStretch(1)
    vpc_actions.addWidget(vpc_progress)
    vpc_actions.addWidget(vpc_cancel_button)
    vpc_actions.addWidget(generate_button)
    vpc_layout.addWidget(vpc_actions_widget)
    bootstrap_layout.addWidget(bootstrap_heading)
    bootstrap_layout.addWidget(bootstrap_status_label)
    bootstrap_layout.addWidget(bootstrap_next_action_label)
    bootstrap_layout.addWidget(bootstrap_next_action_button)
    bootstrap_layout.addWidget(bootstrap_run_label)
    bootstrap_layout.addWidget(bootstrap_settings_section)
    bootstrap_layout.addWidget(bootstrap_log_section, 1)
    bootstrap_actions_widget = qt_widgets.QWidget(bootstrap_tab)
    bootstrap_actions_widget.setObjectName("advanced-bootstrap-actions-row")
    bootstrap_actions = qt_widgets.QHBoxLayout(bootstrap_actions_widget)
    bootstrap_actions.setContentsMargins(0, 0, 0, 0)
    bootstrap_actions.setSpacing(8)
    for button in [
        bootstrap_open_summary_button,
        bootstrap_open_ci_button,
        bootstrap_open_samples_button,
    ]:
        bootstrap_actions.addWidget(button)
    bootstrap_actions.addStretch(1)
    bootstrap_actions.addWidget(bootstrap_progress)
    bootstrap_actions.addWidget(bootstrap_cancel_button)
    bootstrap_actions.addWidget(bootstrap_generate_button)
    bootstrap_layout.addWidget(bootstrap_actions_widget)
    design_layout.addWidget(design_heading)
    design_layout.addWidget(design_status_label)
    design_layout.addWidget(design_next_action_label)
    design_layout.addWidget(design_next_action_button)
    design_layout.addWidget(design_run_label)
    design_layout.addWidget(design_settings_section)
    design_layout.addWidget(design_log_section, 1)
    design_actions_widget = qt_widgets.QWidget(design_tab)
    design_actions_widget.setObjectName("advanced-design-actions-row")
    design_actions = qt_widgets.QHBoxLayout(design_actions_widget)
    design_actions.setContentsMargins(0, 0, 0, 0)
    design_actions.setSpacing(8)
    for button in [design_open_summary_button, design_open_schedule_button]:
        design_actions.addWidget(button)
    design_actions.addStretch(1)
    design_actions.addWidget(design_progress)
    design_actions.addWidget(design_cancel_button)
    design_actions.addWidget(design_generate_button)
    design_layout.addWidget(design_actions_widget)
    artifacts_layout.addWidget(artifacts_heading)
    artifacts_layout.addWidget(artifacts_hint)
    artifacts_layout.addWidget(artifact_scope_row_widget)
    artifacts_layout.addWidget(artifact_content_row_widget, 1)
    actions_widget = qt_widgets.QWidget(artifacts_tab)
    actions_widget.setObjectName("advanced-artifact-actions-row")
    actions = qt_widgets.QHBoxLayout(actions_widget)
    actions.setContentsMargins(0, 0, 0, 0)
    actions.setSpacing(8)
    for button in [export_button, open_button, open_folder_button]:
        actions.addWidget(button)
    actions.addStretch(1)
    artifacts_layout.addWidget(actions_widget)
    tab_widget.addTab(vpc_tab, "VPC")
    tab_widget.addTab(bootstrap_tab, "Bootstrap")
    tab_widget.addTab(design_tab, "Design")
    tab_widget.addTab(artifacts_tab, "Artifacts")
    layout.addWidget(tab_widget, 1)

    _apply_responsive_box_layout = install_responsive_box_layouts(
        root,
        breakpoint=ADVANCED_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        layouts=(
            controls,
            vpc_actions,
            bootstrap_controls,
            bootstrap_actions,
            design_controls,
            design_actions,
            artifact_scope_row,
            actions,
        ),
    )

    _apply_responsive_splitter = install_responsive_splitters(
        root,
        breakpoint=ADVANCED_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(artifact_content_row_widget,),
    )

    def _apply_responsive_layout(width: int | None = None) -> None:
        _apply_responsive_box_layout(width)
        _apply_responsive_splitter(width)

    all_advanced_artifacts: list[ArtifactRecord] = []
    current_artifacts: list[ArtifactRecord] = []
    current_vpc_artifacts: list[ArtifactRecord] = []
    current_bootstrap_artifacts: list[ArtifactRecord] = []
    current_design_artifacts: list[ArtifactRecord] = []
    current_artifact: ArtifactRecord | None = None
    current_fit_context_available = False
    vpc_future = None
    bootstrap_future = None
    design_future = None
    poll_timer = qt_core.QTimer(root)
    poll_timer.setInterval(100)

    def _notify_project_state_changed() -> None:
        callback = getattr(root, "_project_state_changed", None)
        if callable(callback):
            callback()

    def _apply_next_action(
        label_widget,
        button_widget,
        action: tuple[str, str, str] | None,
    ) -> None:
        if action is None:
            label_widget.clear()
            label_widget.setVisible(False)
            button_widget.setText("")
            button_widget.setToolTip("")
            button_widget.setProperty("actionTarget", "")
            button_widget.setVisible(False)
            return
        button_text, target, summary = action
        label_widget.setText(summary)
        label_widget.setVisible(True)
        button_widget.setText(button_text)
        button_widget.setToolTip(summary)
        button_widget.setProperty("actionTarget", target)
        button_widget.setVisible(True)

    def _refresh_next_actions() -> None:
        _apply_next_action(
            vpc_next_action_label,
            vpc_next_action_button,
            recommend_vpc_next_action(
                fit_context_available=current_fit_context_available,
                latest_run=latest_vpc_run(project),
                artifacts=current_vpc_artifacts,
            ),
        )
        _apply_next_action(
            bootstrap_next_action_label,
            bootstrap_next_action_button,
            recommend_bootstrap_next_action(
                fit_context_available=current_fit_context_available,
                latest_run=latest_bootstrap_run(project),
                artifacts=current_bootstrap_artifacts,
            ),
        )
        _apply_next_action(
            design_next_action_label,
            design_next_action_button,
            recommend_design_next_action(
                fit_context_available=current_fit_context_available,
                latest_run=latest_design_run(project),
                artifacts=current_design_artifacts,
            ),
        )

    def _navigate_to_workflow_target(target: str) -> None:
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(target)

    def _handle_vpc_next_action() -> None:
        target = str(vpc_next_action_button.property("actionTarget") or "")
        if target == "fit" or target == "results":
            _navigate_to_workflow_target(target)
            return
        if target == "__open_vpc_plot__":
            _open_artifact(latest_vpc_artifact(current_vpc_artifacts))
            return
        if target == "__open_vpc_summary__":
            _open_artifact(latest_vpc_artifact(current_vpc_artifacts, role="vpc_summary"))

    def _handle_bootstrap_next_action() -> None:
        target = str(bootstrap_next_action_button.property("actionTarget") or "")
        if target == "fit" or target == "results":
            _navigate_to_workflow_target(target)
            return
        if target == "__open_bootstrap_summary__":
            _open_artifact(
                latest_bootstrap_artifact(current_bootstrap_artifacts, role="bootstrap_summary")
            )
            return
        if target == "__open_bootstrap_ci__":
            _open_artifact(
                latest_bootstrap_artifact(current_bootstrap_artifacts, role="bootstrap_ci_table")
            )
            return
        if target == "__open_bootstrap_samples__":
            _open_artifact(
                latest_bootstrap_artifact(current_bootstrap_artifacts, role="bootstrap_samples")
            )

    def _handle_design_next_action() -> None:
        target = str(design_next_action_button.property("actionTarget") or "")
        if target == "fit" or target == "results":
            _navigate_to_workflow_target(target)
            return
        if target == "__open_design_summary__":
            _open_artifact(latest_design_artifact(current_design_artifacts, role="design_summary"))
            return
        if target == "__open_design_schedule__":
            _open_artifact(latest_design_artifact(current_design_artifacts, role="design_schedule"))

    def _render_preview(artifact: ArtifactRecord | None) -> None:
        nonlocal current_artifact
        current_artifact = artifact
        preview_title.setText(
            artifact.label if artifact is not None else "Select an Advanced artifact to preview."
        )
        preview_metadata.setText(format_artifact_metadata(artifact))
        export_button.setEnabled(bool(artifact and artifact.path))
        open_button.setEnabled(bool(artifact and artifact.path))
        open_folder_button.setEnabled(bool(artifact and artifact.path))
        if artifact is None or not artifact.path:
            browser.clear()
            image.clear()
            preview_stack.setCurrentWidget(placeholder)
            return
        path = Path(artifact.path)
        if not path.exists():
            placeholder.setText("Artifact file is not available on disk.")
            preview_stack.setCurrentWidget(placeholder)
            return
        kind = artifact_preview_kind(artifact)
        if kind == "html":
            browser.document().setBaseUrl(qt_core.QUrl.fromLocalFile(str(path.parent.resolve())))
            browser.setHtml(path.read_text(encoding="utf-8", errors="replace"))
            preview_stack.setCurrentWidget(browser)
            return
        if kind == "text":
            browser.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
            preview_stack.setCurrentWidget(browser)
            return
        if kind == "image":
            pixmap = qt_gui.QPixmap(str(path))
            if not pixmap.isNull():
                image.setPixmap(pixmap)
                preview_stack.setCurrentWidget(scroll)
                return
        placeholder.setText(
            "Preview is not available for this artifact type in the Advanced workflow."
        )
        preview_stack.setCurrentWidget(placeholder)

    def _refresh_quick_actions() -> None:
        open_plot_button.setEnabled(
            bool((artifact := latest_vpc_artifact(current_vpc_artifacts)) and artifact.path)
        )
        open_summary_button.setEnabled(
            bool(
                (artifact := latest_vpc_artifact(current_vpc_artifacts, role="vpc_summary"))
                and artifact.path
            )
        )
        bootstrap_open_summary_button.setEnabled(
            bool(
                (
                    artifact := latest_bootstrap_artifact(
                        current_bootstrap_artifacts, role="bootstrap_summary"
                    )
                )
                and artifact.path
            )
        )
        bootstrap_open_ci_button.setEnabled(
            bool(
                (
                    artifact := latest_bootstrap_artifact(
                        current_bootstrap_artifacts, role="bootstrap_ci_table"
                    )
                )
                and artifact.path
            )
        )
        bootstrap_open_samples_button.setEnabled(
            bool(
                (
                    artifact := latest_bootstrap_artifact(
                        current_bootstrap_artifacts, role="bootstrap_samples"
                    )
                )
                and artifact.path
            )
        )
        design_open_summary_button.setEnabled(
            bool(
                (
                    artifact := latest_design_artifact(
                        current_design_artifacts, role="design_summary"
                    )
                )
                and artifact.path
            )
        )
        design_open_schedule_button.setEnabled(
            bool(
                (
                    artifact := latest_design_artifact(
                        current_design_artifacts, role="design_schedule"
                    )
                )
                and artifact.path
            )
        )
        generate_button.setEnabled(vpc_future is None and current_fit_context_available)
        bootstrap_generate_button.setEnabled(
            bootstrap_future is None and current_fit_context_available
        )
        design_generate_button.setEnabled(design_future is None and current_fit_context_available)

    def _render_artifacts(artifacts: list[ArtifactRecord]) -> None:
        current = current_artifact.artifact_id if current_artifact is not None else None
        artifacts_list.clear()
        if not artifacts:
            artifacts_list.addItem(
                artifact_scope_empty_message(str(artifact_scope_combo.currentData() or "all"))
            )
            _refresh_quick_actions()
            _render_preview(None)
            return
        for artifact in artifacts:
            item = qt_widgets.QListWidgetItem(format_artifact_label(artifact))
            item.setData(qt_core.Qt.ItemDataRole.UserRole, artifact.artifact_id)
            artifacts_list.addItem(item)
        _refresh_quick_actions()
        selected_index = next(
            (i for i, artifact in enumerate(artifacts) if artifact.artifact_id == current), 0
        )
        artifacts_list.setCurrentRow(selected_index)

    def _apply_artifact_scope() -> None:
        nonlocal current_artifacts
        scope = str(artifact_scope_combo.currentData() or "all")
        current_artifacts = filter_advanced_artifacts(all_advanced_artifacts, scope)
        artifact_scope_summary.setText(
            format_artifact_scope_summary(
                scope, len(current_artifacts), len(all_advanced_artifacts)
            )
        )
        _render_artifacts(current_artifacts)

    # --- VPC stratification helpers ---

    _STRATIFY_SKIP = frozenset({"ID", "TIME", "AMT", "DV", "EVID", "MDV", "REP"})

    def _stratify_columns() -> list[str]:
        """Return candidate covariate columns from the active dataset."""
        dataset_asset = project.active_dataset
        if dataset_asset is None:
            return []
        return [col for col in dataset_asset.columns if col not in _STRATIFY_SKIP]

    def _refresh_stratify_combo() -> None:
        """Repopulate the stratify-by combo from the active dataset columns."""
        current = stratify_combo.currentData()
        stratify_combo.blockSignals(True)
        stratify_combo.clear()
        stratify_combo.addItem("None", userData=None)
        for col in _stratify_columns():
            stratify_combo.addItem(col, userData=col)
        # Restore previous selection if still available
        idx = stratify_combo.findData(current)
        stratify_combo.setCurrentIndex(max(0, idx))
        stratify_combo.blockSignals(False)

    def _refresh() -> None:
        nonlocal \
            all_advanced_artifacts, \
            current_vpc_artifacts, \
            current_bootstrap_artifacts, \
            current_design_artifacts, \
            current_fit_context_available
        current_vpc_artifacts = vpc_artifacts(project)
        current_bootstrap_artifacts = bootstrap_artifacts(project)
        current_design_artifacts = design_artifacts(project)
        all_advanced_artifacts = advanced_artifacts(project)
        current_fit_context_available = fit_service.latest_fit_context(project) is not None
        _refresh_stratify_combo()
        overview_label.setText(format_advanced_overview(project))
        status_label.setText(
            format_vpc_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                vpc_available=bool(current_vpc_artifacts),
                generation_running=vpc_future is not None,
            )
        )
        run = latest_vpc_run(project)
        latest_run_label.setText(
            run.summary_text if run and run.summary_text else "No VPC run has been started yet."
        )
        log_output.setPlainText("\n".join(run.log_lines) if run else "")
        bootstrap_status_label.setText(
            format_bootstrap_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                bootstrap_available=bool(current_bootstrap_artifacts),
                generation_running=bootstrap_future is not None,
            )
        )
        bootstrap_run = latest_bootstrap_run(project)
        bootstrap_run_label.setText(
            bootstrap_run.summary_text
            if bootstrap_run and bootstrap_run.summary_text
            else "No bootstrap run has been started yet."
        )
        bootstrap_log_output.setPlainText(
            "\n".join(bootstrap_run.log_lines) if bootstrap_run else ""
        )
        design_status_label.setText(
            format_design_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                design_available=bool(current_design_artifacts),
                generation_running=design_future is not None,
            )
        )
        design_run = latest_design_run(project)
        design_run_label.setText(
            design_run.summary_text
            if design_run and design_run.summary_text
            else "No design run has been started yet."
        )
        design_log_output.setPlainText("\n".join(design_run.log_lines) if design_run else "")
        _refresh_next_actions()
        _apply_artifact_scope()

    def _poll_future() -> None:
        nonlocal vpc_future, bootstrap_future, design_future
        state_changed = False
        if vpc_future is not None and vpc_future.done():
            outcome = vpc_future.result()
            run = vpc_service.latest_run(project)
            if run is not None:
                artifacts = vpc_service.apply_job_outcome(run, outcome)
                for artifact in artifacts:
                    artifact_service.register(project, artifact)
                    run.add_log(f"[artifact] {artifact.kind}: {artifact.label}")
                state_changed = True
            vpc_future = None
            vpc_progress.setVisible(False)
            vpc_cancel_button.setEnabled(False)
        if bootstrap_future is not None and bootstrap_future.done():
            outcome = bootstrap_future.result()
            run = bootstrap_service.latest_run(project)
            if run is not None:
                artifacts = bootstrap_service.apply_job_outcome(run, outcome)
                for artifact in artifacts:
                    artifact_service.register(project, artifact)
                    run.add_log(f"[artifact] {artifact.kind}: {artifact.label}")
                state_changed = True
            bootstrap_future = None
            bootstrap_progress.setVisible(False)
            bootstrap_cancel_button.setEnabled(False)
        if design_future is not None and design_future.done():
            outcome = design_future.result()
            run = design_service.latest_run(project)
            if run is not None:
                artifacts = design_service.apply_job_outcome(run, outcome)
                for artifact in artifacts:
                    artifact_service.register(project, artifact)
                    run.add_log(f"[artifact] {artifact.kind}: {artifact.label}")
                state_changed = True
            design_future = None
            design_progress.setVisible(False)
            design_cancel_button.setEnabled(False)
        if state_changed:
            _notify_project_state_changed()
        if vpc_future is None and bootstrap_future is None and design_future is None:
            poll_timer.stop()
        _refresh()

    def _start_vpc_generation() -> None:
        nonlocal vpc_future
        _refresh()
        if not current_fit_context_available:
            return
        run = RunRecord(workflow="vpc")
        config = VPCConfig(
            n_replicates=replicates_spin.value(),
            n_bins=bins_spin.value(),
            seed=seed_spin.value(),
            prediction_corrected=pc_checkbox.isChecked(),
            n_parallel=preferences[0].n_parallel if preferences else 0,
            stratify_by=stratify_combo.currentData() or None,
        )
        try:
            job = vpc_service.create_job(
                project, fit_service=fit_service, config=config, run_id=run.run_id
            )
        except ValueError:
            _refresh()
            return
        run.mark_running()
        run.add_log("VPC generation submitted.")
        project_service.add_run(project, run)
        _notify_project_state_changed()
        vpc_future = job_runner.submit(job)
        generate_button.setEnabled(False)
        vpc_cancel_button.setEnabled(True)
        vpc_progress.setVisible(True)
        status_label.setText(
            format_vpc_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                vpc_available=bool(current_vpc_artifacts),
                generation_running=True,
            )
        )
        _refresh_next_actions()
        poll_timer.start()

    def _start_bootstrap_generation() -> None:
        nonlocal bootstrap_future
        _refresh()
        if not current_fit_context_available:
            return
        run = RunRecord(workflow="bootstrap")
        config = BootstrapConfig(
            n_boot=bootstrap_replicates_spin.value(),
            seed=bootstrap_seed_spin.value(),
            n_jobs=bootstrap_jobs_spin.value(),
            ci_level=bootstrap_ci_spin.value(),
        )
        try:
            job = bootstrap_service.create_job(
                project, fit_service=fit_service, config=config, run_id=run.run_id
            )
        except ValueError:
            _refresh()
            return
        run.mark_running()
        run.add_log("Bootstrap generation submitted.")
        project_service.add_run(project, run)
        _notify_project_state_changed()
        bootstrap_future = job_runner.submit(job)
        bootstrap_generate_button.setEnabled(False)
        bootstrap_cancel_button.setEnabled(True)
        bootstrap_progress.setVisible(True)
        bootstrap_status_label.setText(
            format_bootstrap_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                bootstrap_available=bool(current_bootstrap_artifacts),
                generation_running=True,
            )
        )
        _refresh_next_actions()
        poll_timer.start()

    def _start_design_generation() -> None:
        nonlocal design_future
        _refresh()
        if not current_fit_context_available:
            return
        if design_t_max_spin.value() <= design_t_min_spin.value():
            design_status_label.setText(
                "Max time must be greater than min time before running a design optimization."
            )
            return
        run = RunRecord(workflow="design")
        config = DesignConfig(
            n_samples=design_samples_spin.value(),
            t_min=design_t_min_spin.value(),
            t_max=design_t_max_spin.value(),
            n_subjects=design_subjects_spin.value(),
            criterion=str(design_criterion_combo.currentData() or "D"),
            method=str(design_method_combo.currentData() or "differential_evolution"),
            n_starts=design_restarts_spin.value(),
        )
        try:
            job = design_service.create_job(
                project, fit_service=fit_service, config=config, run_id=run.run_id
            )
        except ValueError:
            _refresh()
            return
        run.mark_running()
        run.add_log("Design optimization submitted.")
        project_service.add_run(project, run)
        _notify_project_state_changed()
        design_future = job_runner.submit(job)
        design_generate_button.setEnabled(False)
        design_cancel_button.setEnabled(True)
        design_progress.setVisible(True)
        design_status_label.setText(
            format_design_generation_status(
                project,
                fit_context_available=current_fit_context_available,
                design_available=bool(current_design_artifacts),
                generation_running=True,
            )
        )
        _refresh_next_actions()
        poll_timer.start()

    def _handle_selection_changed(index: int) -> None:
        if index < 0 or index >= len(current_artifacts):
            _render_preview(None)
            return
        _render_preview(current_artifacts[index])

    def _handle_artifact_scope_changed(_index: int) -> None:
        _apply_artifact_scope()

    def _open_artifact(artifact: ArtifactRecord | None) -> None:
        if artifact is None or not artifact.path:
            return
        qt_gui.QDesktopServices.openUrl(qt_core.QUrl.fromLocalFile(artifact.path))

    def _export_selected_artifact() -> None:
        if current_artifact is None or not current_artifact.path:
            return
        source_path = Path(current_artifact.path)
        if not source_path.exists():
            return
        preferences = load_gui_preferences()
        start_dir = Path(preferences.last_file_dialog_dir or default_workspace_root_path())
        destination_path, _ = qt_widgets.QFileDialog.getSaveFileName(
            root,
            "Save Advanced artifact copy",
            str(start_dir / source_path.name),
            "All files (*)",
        )
        if destination_path:
            shutil.copy2(source_path, destination_path)
            save_gui_preferences(with_last_file_dialog_dir(preferences, destination_path))

    def _open_selected_folder() -> None:
        if current_artifact is None or not current_artifact.path:
            return
        qt_gui.QDesktopServices.openUrl(
            qt_core.QUrl.fromLocalFile(str(Path(current_artifact.path).resolve().parent))
        )

    def _cancel_vpc() -> None:
        nonlocal vpc_future
        if vpc_future is None:
            return
        vpc_future.cancel()
        vpc_future = None
        vpc_progress.setVisible(False)
        vpc_cancel_button.setEnabled(False)
        run = vpc_service.latest_run(project)
        if run is not None:
            run.mark_cancelled()
            _notify_project_state_changed()
        if bootstrap_future is None and design_future is None:
            poll_timer.stop()
        _refresh()

    def _cancel_bootstrap() -> None:
        nonlocal bootstrap_future
        if bootstrap_future is None:
            return
        bootstrap_future.cancel()
        bootstrap_future = None
        bootstrap_progress.setVisible(False)
        bootstrap_cancel_button.setEnabled(False)
        run = bootstrap_service.latest_run(project)
        if run is not None:
            run.mark_cancelled()
            _notify_project_state_changed()
        if vpc_future is None and design_future is None:
            poll_timer.stop()
        _refresh()

    def _cancel_design() -> None:
        nonlocal design_future
        if design_future is None:
            return
        design_future.cancel()
        design_future = None
        design_progress.setVisible(False)
        design_cancel_button.setEnabled(False)
        run = design_service.latest_run(project)
        if run is not None:
            run.mark_cancelled()
            _notify_project_state_changed()
        if vpc_future is None and bootstrap_future is None:
            poll_timer.stop()
        _refresh()

    artifacts_list.currentRowChanged.connect(_handle_selection_changed)
    artifacts_list.itemDoubleClicked.connect(
        lambda item: (
            _open_artifact(current_artifacts[artifacts_list.row(item)])
            if 0 <= artifacts_list.row(item) < len(current_artifacts)
            else None
        )
    )
    artifact_scope_combo.currentIndexChanged.connect(_handle_artifact_scope_changed)
    generate_button.clicked.connect(_start_vpc_generation)
    vpc_next_action_button.clicked.connect(_handle_vpc_next_action)
    vpc_cancel_button.clicked.connect(_cancel_vpc)
    bootstrap_generate_button.clicked.connect(_start_bootstrap_generation)
    bootstrap_next_action_button.clicked.connect(_handle_bootstrap_next_action)
    bootstrap_cancel_button.clicked.connect(_cancel_bootstrap)
    design_generate_button.clicked.connect(_start_design_generation)
    design_next_action_button.clicked.connect(_handle_design_next_action)
    design_cancel_button.clicked.connect(_cancel_design)
    open_plot_button.clicked.connect(
        lambda: _open_artifact(latest_vpc_artifact(current_vpc_artifacts))
    )
    open_summary_button.clicked.connect(
        lambda: _open_artifact(latest_vpc_artifact(current_vpc_artifacts, role="vpc_summary"))
    )
    bootstrap_open_summary_button.clicked.connect(
        lambda: _open_artifact(
            latest_bootstrap_artifact(current_bootstrap_artifacts, role="bootstrap_summary")
        )
    )
    bootstrap_open_ci_button.clicked.connect(
        lambda: _open_artifact(
            latest_bootstrap_artifact(current_bootstrap_artifacts, role="bootstrap_ci_table")
        )
    )
    bootstrap_open_samples_button.clicked.connect(
        lambda: _open_artifact(
            latest_bootstrap_artifact(current_bootstrap_artifacts, role="bootstrap_samples")
        )
    )
    design_open_summary_button.clicked.connect(
        lambda: _open_artifact(
            latest_design_artifact(current_design_artifacts, role="design_summary")
        )
    )
    design_open_schedule_button.clicked.connect(
        lambda: _open_artifact(
            latest_design_artifact(current_design_artifacts, role="design_schedule")
        )
    )
    export_button.clicked.connect(_export_selected_artifact)
    open_button.clicked.connect(lambda: _open_artifact(current_artifact))
    open_folder_button.clicked.connect(_open_selected_folder)
    poll_timer.timeout.connect(_poll_future)
    root.destroyed.connect(lambda *_args: job_runner.shutdown(wait=False))
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh  # type: ignore[attr-defined]

    _refresh()
    _apply_responsive_layout()
    return root
