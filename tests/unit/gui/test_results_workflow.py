"""Pure tests for Results workflow helper functions."""

from __future__ import annotations

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.workflows.results_workflow import (
    ANALYSIS_FILTER_ALL,
    ARTIFACT_TYPE_FILTER_ALL,
    artifact_analysis_type,
    artifact_friendly_type,
    artifact_friendly_type_options,
    artifact_plot_type,
    artifact_preview_kind,
    artifact_role,
    build_results_workflow,
    filter_artifacts,
    format_artifact_metadata,
    format_results_comparison_action,
    format_results_comparison_panel,
    format_fit_review_summary,
    format_results_comparison_summary,
    format_results_stale_warning,
    latest_artifact,
    latest_artifact_for_plot_group,
    recommend_results_next_action,
    review_analysis_type_options,
    review_runs,
    select_results_comparison_target,
    scenario_runs_by_id,
)


def test_artifact_preview_kind_detects_html_and_images() -> None:
    html_artifact = ArtifactRecord(kind="report", label="Report", path="/tmp/report.html")
    image_artifact = ArtifactRecord(kind="plot", label="Plot", path="/tmp/plot.PNG")

    assert artifact_preview_kind(html_artifact) == "html"
    assert artifact_preview_kind(image_artifact) == "image"


def test_artifact_preview_kind_prefers_media_type_metadata() -> None:
    artifact = ArtifactRecord(
        kind="report",
        label="Snapshot report",
        path="/tmp/no-extension",
        metadata={"media_type": "text/html"},
    )

    assert artifact_preview_kind(artifact) == "html"


def test_format_artifact_metadata_reports_key_fields() -> None:
    artifact = ArtifactRecord(
        kind="plot",
        label="OFV history",
        path="/tmp/ofv-history.png",
        metadata={
            "media_type": "image/png",
            "artifact_role": "plot",
            "plot_type": "ofv_history",
            "estimation_method": "FOCE",
        },
    )

    metadata = format_artifact_metadata(artifact)

    assert "Kind plot" in metadata
    assert "Role plot" in metadata
    assert "Plot ofv_history" in metadata
    assert "Method FOCE" in metadata
    assert "image/png" in metadata
    assert "/tmp/ofv-history.png" in metadata


def test_artifact_role_plot_type_filtering_and_latest_selection() -> None:
    report = ArtifactRecord(
        kind="report",
        label="Run report",
        path="/tmp/report.html",
        metadata={"artifact_role": "report", "media_type": "text/html"},
    )
    gof = ArtifactRecord(
        kind="plot",
        label="GOF panel",
        path="/tmp/gof-panel.png",
        metadata={"artifact_role": "plot", "plot_type": "gof_panel", "media_type": "image/png"},
    )
    residual = ArtifactRecord(
        kind="plot",
        label="Residual trends",
        path="/tmp/residual-trends.png",
        metadata={
            "artifact_role": "plot",
            "plot_type": "residual_trends",
            "media_type": "image/png",
        },
    )

    assert artifact_role(report) == "report"
    assert artifact_plot_type(gof) == "gof_panel"
    assert filter_artifacts([report, gof, residual], "All kinds", "plot", "gof_panel") == [gof]
    assert latest_artifact([report, gof, residual], role="plot") is residual
    assert latest_artifact([report, gof, residual], plot_type="gof_panel") is gof


def test_results_helpers_classify_analysis_types_from_runs_and_artifacts() -> None:
    workspace = Workspace(name="Review helpers")
    fit_run = RunRecord(workflow="fit", run_id="fit-1", status=RunStatus.SUCCEEDED)
    nca_run = RunRecord(workflow="nca", run_id="nca-1", status=RunStatus.SUCCEEDED)
    fit_artifact = ArtifactRecord(kind="report", label="Fit report", source_run_id=fit_run.run_id)
    nca_artifact = ArtifactRecord(
        kind="table",
        label="NCA summary",
        source_run_id=nca_run.run_id,
        metadata={"artifact_role": "nca_summary"},
    )
    workspace.active_scenario.runs = [fit_run, nca_run]
    workspace.active_scenario.artifacts = [fit_artifact, nca_artifact]

    runs_by_id = scenario_runs_by_id(workspace)

    assert review_analysis_type_options(workspace) == [ANALYSIS_FILTER_ALL, "Fit", "NCA"]
    assert review_runs(workspace, "NCA") == [nca_run]
    assert artifact_analysis_type(fit_artifact, runs_by_id) == "Fit"
    assert artifact_analysis_type(nca_artifact, runs_by_id) == "NCA"
    assert filter_artifacts(
        workspace.active_scenario.artifacts,
        "All kinds",
        analysis_filter="NCA",
        runs_by_id=runs_by_id,
    ) == [nca_artifact]


def test_recommend_results_next_action_prioritizes_prerequisites_and_fit() -> None:
    workspace = Workspace(name="Review helpers")

    assert recommend_results_next_action(workspace) == (
        "Open Data",
        "data",
        "Load a dataset before expecting review runs, reports, or saved outputs for this scenario.",
    )

    workspace.active_dataset = object()  # type: ignore[assignment]
    assert recommend_results_next_action(workspace) == (
        "Open Model",
        "model",
        "Configure a model in the Model workflow first.",
    )

    workspace.active_model_spec = object()  # type: ignore[assignment]
    assert recommend_results_next_action(workspace) == (
        "Open Fit",
        "fit",
        "Run a fit to populate Results with run logs, reports, and saved outputs.",
    )

    workspace.active_scenario.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    assert recommend_results_next_action(workspace) is None


def test_format_results_comparison_summary_reports_peer_scenarios() -> None:
    workspace = Workspace(name="Review helpers")
    peer = workspace.active_scenario.snapshot_clone(name="Variant A")
    peer.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    peer.artifacts = [ArtifactRecord(kind="report", label="Report", path="/tmp/report.html")]
    workspace.active_project.add_scenario(peer, make_active=False)

    summary = format_results_comparison_summary(workspace)

    assert "Comparison snapshot:" in summary
    assert "Variant A: latest fit succeeded" in summary
    assert "1 runs" in summary
    assert "1 outputs" in summary


def test_format_results_comparison_summary_mentions_parent_relation() -> None:
    workspace = Workspace(name="Review helpers")
    parent = workspace.active_scenario
    child = parent.snapshot_clone(name="Child scenario")
    workspace.active_project.add_scenario(child)

    summary = format_results_comparison_summary(workspace)

    assert "Baseline: no fit yet" in summary
    assert "parent of current" in summary


def test_format_results_comparison_action_recommends_richest_peer() -> None:
    workspace = Workspace(name="Review helpers")
    peer_a = workspace.active_scenario.snapshot_clone(name="Variant A")
    peer_a.runs = [
        RunRecord(workflow="fit", status=RunStatus.SUCCEEDED),
        RunRecord(workflow="nca", status=RunStatus.SUCCEEDED),
    ]
    peer_a.artifacts = [
        ArtifactRecord(kind="report", label="Report", path="/tmp/a-report.html"),
        ArtifactRecord(kind="plot", label="Plot", path="/tmp/a-plot.png"),
    ]
    peer_b = workspace.active_scenario.snapshot_clone(name="Variant B")
    peer_b.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    workspace.active_project.add_scenario(peer_a, make_active=False)
    workspace.active_project.add_scenario(peer_b, make_active=False)

    action = format_results_comparison_action(workspace)

    assert "inspect Variant A next" in action
    assert "2 successful runs" in action
    assert "2 outputs" in action


def test_format_results_comparison_action_handles_unfit_peer() -> None:
    workspace = Workspace(name="Review helpers")
    peer = workspace.active_scenario.snapshot_clone(name="Variant A")
    workspace.active_project.add_scenario(peer, make_active=False)

    action = format_results_comparison_action(workspace)

    assert "Variant A" in action
    assert "no successful runs yet" in action


def test_format_results_comparison_panel_lists_peer_metrics() -> None:
    workspace = Workspace(name="Review helpers")
    peer_a = workspace.active_scenario.snapshot_clone(name="Variant A")
    peer_a.runs = [
        RunRecord(workflow="fit", status=RunStatus.SUCCEEDED),
        RunRecord(workflow="nca", status=RunStatus.SUCCEEDED),
    ]
    peer_a.artifacts = [
        ArtifactRecord(kind="report", label="Report", path="/tmp/a-report.html"),
        ArtifactRecord(kind="plot", label="Plot", path="/tmp/a-plot.png"),
    ]
    peer_b = workspace.active_scenario.snapshot_clone(name="Variant B")
    peer_b.runs = [RunRecord(workflow="fit", status=RunStatus.FAILED)]
    workspace.active_project.add_scenario(peer_a, make_active=False)
    workspace.active_project.add_scenario(peer_b, make_active=False)

    panel = format_results_comparison_panel(workspace)

    assert panel.startswith("Comparison panel:")
    assert "Variant A [child]" in panel
    assert "2 successful runs" in panel
    assert "1 successful fits" in panel
    assert "2 outputs" in panel
    assert "Variant B [child] — latest fit failed" in panel


def test_format_results_comparison_panel_handles_empty_peer_set() -> None:
    workspace = Workspace(name="Review helpers")

    panel = format_results_comparison_panel(workspace)

    assert "Comparison panel:" in panel
    assert "No sibling scenarios yet" in panel


def test_select_results_comparison_target_returns_richest_peer() -> None:
    workspace = Workspace(name="Review helpers")
    peer_a = workspace.active_scenario.snapshot_clone(name="Variant A")
    peer_a.runs = [
        RunRecord(workflow="fit", status=RunStatus.SUCCEEDED),
        RunRecord(workflow="nca", status=RunStatus.SUCCEEDED),
    ]
    peer_b = workspace.active_scenario.snapshot_clone(name="Variant B")
    peer_b.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    workspace.active_project.add_scenario(peer_a, make_active=False)
    workspace.active_project.add_scenario(peer_b, make_active=False)

    target = select_results_comparison_target(workspace)

    assert target == peer_a.scenario_id


def test_format_results_stale_warning_mentions_changed_inputs() -> None:
    workspace = Workspace(name="Review helpers")
    fit_run = RunRecord(workflow="fit")
    fit_run.mark_running()
    fit_run.mark_succeeded("ok")
    fit_run.finished_at = "2026-03-15T12:00:00+00:00"
    workspace.active_scenario.runs = [fit_run]
    workspace.active_scenario.dataset_updated_at = "2026-03-15T12:05:00+00:00"

    warning = format_results_stale_warning(workspace)

    assert warning == (
        "Dataset changed since the latest successful fit. Results may be stale until you rerun the analysis."
    )


def test_format_results_stale_warning_is_suppressed_while_rerun_is_in_progress() -> None:
    workspace = Workspace(name="Review helpers")
    successful_fit = RunRecord(workflow="fit")
    successful_fit.mark_running()
    successful_fit.mark_succeeded("ok")
    successful_fit.finished_at = "2026-03-15T12:00:00+00:00"
    rerun = RunRecord(workflow="fit")
    rerun.mark_running()
    workspace.active_scenario.runs = [successful_fit, rerun]
    workspace.active_scenario.dataset_updated_at = "2026-03-15T12:05:00+00:00"

    warning = format_results_stale_warning(workspace)

    assert warning == ""


def test_fit_review_helpers_summarize_curated_groups_and_tables() -> None:
    ofv = ArtifactRecord(
        kind="plot",
        label="OFV history",
        path="/tmp/ofv-history.png",
        metadata={"artifact_role": "plot", "plot_type": "ofv_history", "media_type": "image/png"},
    )
    gof = ArtifactRecord(
        kind="plot",
        label="GOF panel",
        path="/tmp/gof-panel.png",
        metadata={"artifact_role": "plot", "plot_type": "gof_panel", "media_type": "image/png"},
    )
    diagnostics_table = ArtifactRecord(
        kind="table",
        label="Diagnostics table",
        path="/tmp/diagnostics.csv",
        metadata={"artifact_role": "diagnostics_table", "media_type": "text/csv"},
    )

    assert format_fit_review_summary([]).startswith("Fit review shortcuts will appear")
    assert latest_artifact_for_plot_group([ofv, gof], "convergence") is ofv

    summary = format_fit_review_summary([ofv, gof, diagnostics_table])

    assert "Convergence" in summary
    assert "GOF" in summary
    assert "Diagnostics table" in summary


def test_artifact_friendly_type_maps_plot_type_to_readable_label() -> None:
    gof = ArtifactRecord(
        kind="plot",
        label="GOF",
        path="/tmp/gof.png",
        metadata={"plot_type": "gof_panel"},
    )
    assert artifact_friendly_type(gof) == "GOF panel"


def test_artifact_friendly_type_maps_role_to_readable_label() -> None:
    report = ArtifactRecord(
        kind="report",
        label="Report",
        path="/tmp/report.html",
        metadata={"artifact_role": "nca_summary"},
    )
    assert artifact_friendly_type(report) == "NCA summary (CSV)"


def test_artifact_friendly_type_falls_back_gracefully() -> None:
    unknown = ArtifactRecord(
        kind="table",
        label="Something",
        path="/tmp/x.csv",
        metadata={"artifact_role": "custom_role"},
    )
    assert artifact_friendly_type(unknown) == "Table — custom_role"

    bare = ArtifactRecord(kind="plot", label="Plot", path="/tmp/p.png")
    assert artifact_friendly_type(bare).startswith("Plot")


def test_artifact_friendly_type_options_starts_with_all_types() -> None:
    gof = ArtifactRecord(
        kind="plot", label="GOF", path="/tmp/g.png", metadata={"plot_type": "gof_panel"}
    )
    report = ArtifactRecord(
        kind="report", label="Report", path="/tmp/r.html", metadata={"artifact_role": "report"}
    )
    options = artifact_friendly_type_options([gof, report])
    assert options[0] == ARTIFACT_TYPE_FILTER_ALL
    assert "GOF panel" in options
    assert "HTML Report" in options


def test_filter_artifacts_by_type_filter() -> None:
    gof = ArtifactRecord(
        kind="plot", label="GOF", path="/tmp/g.png", metadata={"plot_type": "gof_panel"}
    )
    ofv = ArtifactRecord(
        kind="plot", label="OFV", path="/tmp/o.png", metadata={"plot_type": "ofv_history"}
    )
    result = filter_artifacts([gof, ofv], type_filter="GOF panel")
    assert result == [gof]


@pytest.mark.unit
def test_results_workflow_empty_state_button_navigates_to_fit_and_clears_after_run() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Results workflow")
    project.active_dataset = object()  # type: ignore[assignment]
    project.active_model_spec = object()  # type: ignore[assignment]
    widget = build_results_workflow(project)
    navigations: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]

    try:
        widget.show()
        app.processEvents()

        next_action_label = widget.findChild(qt_widgets.QLabel, "results-next-action-label")
        next_action_button = widget.findChild(qt_widgets.QPushButton, "results-next-action-button")
        comparison_label = widget.findChild(qt_widgets.QLabel, "results-comparison-label")
        comparison_action_label = widget.findChild(
            qt_widgets.QLabel, "results-comparison-action-label"
        )
        comparison_action_button = widget.findChild(
            qt_widgets.QPushButton, "results-comparison-action-button"
        )

        assert next_action_label is not None
        assert next_action_button is not None
        assert comparison_label is not None
        assert comparison_action_label is not None
        assert comparison_action_button is not None
        assert next_action_button.text() == "Open Fit"
        assert (
            next_action_label.text()
            == "Run a fit to populate Results with run logs, reports, and saved outputs."
        )
        assert "No sibling scenarios yet" in comparison_label.text()
        assert "create a sibling scenario" in comparison_action_label.text()
        assert comparison_action_button.isEnabled() is False

        next_action_button.click()
        app.processEvents()

        assert navigations == ["fit"]

        project.active_scenario.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert next_action_label.isHidden() is True
        assert next_action_button.isHidden() is True
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_results_workflow_comparison_button_switches_to_target_scenario() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    workspace = Workspace(name="Results workflow")
    peer = workspace.active_scenario.snapshot_clone(name="Variant A")
    peer.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    peer.artifacts = [ArtifactRecord(kind="report", label="Report", path="/tmp/report.html")]
    workspace.active_project.add_scenario(peer, make_active=False)
    widget = build_results_workflow(workspace)

    try:
        widget.show()
        app.processEvents()

        comparison_action_button = widget.findChild(
            qt_widgets.QPushButton, "results-comparison-action-button"
        )
        comparison_label = widget.findChild(qt_widgets.QLabel, "results-comparison-label")

        assert comparison_action_button is not None
        assert comparison_label is not None
        assert comparison_action_button.isEnabled() is True

        comparison_action_button.click()
        app.processEvents()

        assert workspace.active_scenario.scenario_id == peer.scenario_id
        assert "No sibling scenarios yet" not in comparison_label.text()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
