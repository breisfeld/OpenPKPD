"""Pure tests for the Advanced GUI workflow presentation helpers."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob
from openpkpd_gui.jobs.runner import JobRunner
from openpkpd_gui.services.artifact_service import ArtifactService
from openpkpd_gui.services.bootstrap_service import BootstrapRunResult, BootstrapService
from openpkpd_gui.services.design_service import DesignRunResult, DesignService
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.vpc_service import VPCRunResult, VPCService
from openpkpd_gui.workflows.advanced_workflow import (
    bootstrap_artifacts,
    build_advanced_workflow,
    design_artifacts,
    filter_advanced_artifacts,
    format_artifact_scope_summary,
    format_bootstrap_generation_status,
    format_design_generation_status,
    format_vpc_generation_status,
    latest_bootstrap_run,
    latest_design_run,
    latest_vpc_run,
    recommend_bootstrap_next_action,
    recommend_design_next_action,
    recommend_vpc_next_action,
    vpc_artifacts,
)


def test_vpc_generation_status_reports_fit_requirement_and_ready_state() -> None:
    workspace = Workspace(name="Advanced demo")
    assert "reusable successful fit" in format_vpc_generation_status(
        workspace,
        fit_context_available=False,
        vpc_available=False,
    )

    workspace.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    assert "Generate VPC on demand" in format_vpc_generation_status(
        workspace,
        fit_context_available=True,
        vpc_available=False,
    )
    assert "already available" in format_vpc_generation_status(
        workspace,
        fit_context_available=True,
        vpc_available=True,
    )


def test_vpc_artifact_helpers_filter_and_latest_run() -> None:
    fit = RunRecord(workflow="fit", run_id="fit-1")
    vpc = RunRecord(workflow="vpc", run_id="vpc-1")
    workspace = Workspace()
    workspace.runs = [fit, vpc]
    summary = ArtifactRecord(
        kind="table", label="VPC summary", metadata={"artifact_role": "vpc_summary"}
    )
    plot = ArtifactRecord(
        kind="plot", label="VPC plot", metadata={"artifact_role": "plot", "plot_type": "vpc"}
    )
    panel = ArtifactRecord(
        kind="plot",
        label="Simulation panel",
        metadata={"artifact_role": "plot", "plot_type": "simulation_panel"},
    )
    other = ArtifactRecord(
        kind="plot",
        label="Other plot",
        metadata={"artifact_role": "plot", "plot_type": "gof_panel"},
    )
    workspace.artifacts = [summary, plot, panel, other]

    assert latest_vpc_run(workspace) is vpc
    assert vpc_artifacts(workspace) == [summary, plot, panel]


def test_bootstrap_generation_status_reports_fit_requirement_and_ready_state() -> None:
    workspace = Workspace(name="Advanced demo")
    assert "reusable successful fit" in format_bootstrap_generation_status(
        workspace,
        fit_context_available=False,
        bootstrap_available=False,
    )

    workspace.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    assert "Generate bootstrap summaries" in format_bootstrap_generation_status(
        workspace,
        fit_context_available=True,
        bootstrap_available=False,
    )
    assert "already available" in format_bootstrap_generation_status(
        workspace,
        fit_context_available=True,
        bootstrap_available=True,
    )


def test_bootstrap_artifact_helpers_filter_and_latest_run() -> None:
    fit = RunRecord(workflow="fit", run_id="fit-1")
    bootstrap = RunRecord(workflow="bootstrap", run_id="boot-1")
    workspace = Workspace()
    workspace.runs = [fit, bootstrap]
    summary = ArtifactRecord(
        kind="table", label="Bootstrap summary", metadata={"artifact_role": "bootstrap_summary"}
    )
    ci = ArtifactRecord(
        kind="table", label="Bootstrap CI", metadata={"artifact_role": "bootstrap_ci_table"}
    )
    other = ArtifactRecord(
        kind="plot",
        label="Other plot",
        metadata={"artifact_role": "plot", "plot_type": "gof_panel"},
    )
    workspace.artifacts = [summary, ci, other]

    assert latest_bootstrap_run(workspace) is bootstrap
    assert bootstrap_artifacts(workspace) == [summary, ci]


def test_design_generation_status_reports_fit_requirement_and_ready_state() -> None:
    workspace = Workspace(name="Advanced demo")
    assert "reusable successful fit" in format_design_generation_status(
        workspace,
        fit_context_available=False,
        design_available=False,
    )

    workspace.runs = [RunRecord(workflow="fit", status=RunStatus.SUCCEEDED)]
    assert "Generate optimal design summaries" in format_design_generation_status(
        workspace,
        fit_context_available=True,
        design_available=False,
    )
    assert "already available" in format_design_generation_status(
        workspace,
        fit_context_available=True,
        design_available=True,
    )


def test_design_artifact_helpers_filter_and_latest_run() -> None:
    fit = RunRecord(workflow="fit", run_id="fit-1")
    design = RunRecord(workflow="design", run_id="design-1")
    workspace = Workspace()
    workspace.runs = [fit, design]
    summary = ArtifactRecord(
        kind="report", label="Design summary", metadata={"artifact_role": "design_summary"}
    )
    schedule = ArtifactRecord(
        kind="table", label="Design schedule", metadata={"artifact_role": "design_schedule"}
    )
    other = ArtifactRecord(
        kind="plot",
        label="Other plot",
        metadata={"artifact_role": "plot", "plot_type": "gof_panel"},
    )
    workspace.artifacts = [summary, schedule, other]

    assert latest_design_run(workspace) is design
    assert design_artifacts(workspace) == [summary, schedule]


def test_advanced_artifact_scope_filtering_and_summary() -> None:
    vpc_summary = ArtifactRecord(
        kind="table", label="VPC summary", metadata={"artifact_role": "vpc_summary"}
    )
    vpc_plot = ArtifactRecord(
        kind="plot", label="VPC plot", metadata={"artifact_role": "plot", "plot_type": "vpc"}
    )
    prediction_interval = ArtifactRecord(
        kind="plot",
        label="Prediction interval",
        metadata={"artifact_role": "plot", "plot_type": "prediction_interval_plot"},
    )
    bootstrap_summary = ArtifactRecord(
        kind="table", label="Bootstrap summary", metadata={"artifact_role": "bootstrap_summary"}
    )
    design_summary = ArtifactRecord(
        kind="report", label="Design summary", metadata={"artifact_role": "design_summary"}
    )
    artifacts = [vpc_summary, vpc_plot, prediction_interval, bootstrap_summary, design_summary]

    assert filter_advanced_artifacts(artifacts, "all") == artifacts
    assert filter_advanced_artifacts(artifacts, "vpc") == [
        vpc_summary,
        vpc_plot,
        prediction_interval,
    ]
    assert filter_advanced_artifacts(artifacts, "bootstrap") == [bootstrap_summary]
    assert filter_advanced_artifacts(artifacts, "design") == [design_summary]
    assert format_artifact_scope_summary("design", 1, 5) == "Showing 1 of 5 design artifacts."


def test_recommend_advanced_next_actions_cover_blocked_ready_and_available_states() -> None:
    plot = ArtifactRecord(
        kind="plot",
        label="VPC plot",
        path="/tmp/vpc.png",
        metadata={"artifact_role": "plot", "plot_type": "vpc"},
    )
    vpc_summary = ArtifactRecord(
        kind="table",
        label="VPC summary",
        path="/tmp/vpc.csv",
        metadata={"artifact_role": "vpc_summary"},
    )
    bootstrap_summary = ArtifactRecord(
        kind="table",
        label="Bootstrap summary",
        path="/tmp/bootstrap.csv",
        metadata={"artifact_role": "bootstrap_summary"},
    )
    design_summary = ArtifactRecord(
        kind="report",
        label="Design summary",
        path="/tmp/design.txt",
        metadata={"artifact_role": "design_summary"},
    )

    assert recommend_vpc_next_action(
        fit_context_available=False, latest_run=None, artifacts=[]
    ) == (
        "Open Fit",
        "fit",
        "Complete a successful fit for this scenario before generating VPC outputs here.",
    )
    assert recommend_bootstrap_next_action(
        fit_context_available=False, latest_run=None, artifacts=[]
    ) == (
        "Open Fit",
        "fit",
        "Complete a successful fit for this scenario before generating bootstrap outputs here.",
    )
    assert recommend_design_next_action(
        fit_context_available=False, latest_run=None, artifacts=[]
    ) == (
        "Open Fit",
        "fit",
        "Complete a successful fit for this scenario before generating design outputs here.",
    )

    assert (
        recommend_vpc_next_action(fit_context_available=True, latest_run=None, artifacts=[]) is None
    )
    assert (
        recommend_bootstrap_next_action(fit_context_available=True, latest_run=None, artifacts=[])
        is None
    )
    assert (
        recommend_design_next_action(fit_context_available=True, latest_run=None, artifacts=[])
        is None
    )

    assert recommend_vpc_next_action(
        fit_context_available=False, latest_run=None, artifacts=[vpc_summary, plot]
    ) == (
        "Open latest VPC plot",
        "__open_vpc_plot__",
        "Latest VPC outputs are already available. Open the newest plot or adjust the controls below to generate an updated run.",
    )
    assert recommend_bootstrap_next_action(
        fit_context_available=False, latest_run=None, artifacts=[bootstrap_summary]
    ) == (
        "Open latest bootstrap summary",
        "__open_bootstrap_summary__",
        "Latest bootstrap outputs are already available. Open the newest summary or adjust the controls below to generate an updated run.",
    )
    assert recommend_design_next_action(
        fit_context_available=False, latest_run=None, artifacts=[design_summary]
    ) == (
        "Open latest design summary",
        "__open_design_summary__",
        "Latest design outputs are already available. Open the newest summary or adjust the controls below to generate an updated run.",
    )

    running = RunRecord(workflow="vpc", status=RunStatus.RUNNING)
    assert (
        recommend_vpc_next_action(fit_context_available=False, latest_run=running, artifacts=[plot])
        is None
    )


@pytest.mark.unit
def test_advanced_workflow_generates_vpc_on_demand(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Advanced VPC", root_path=str(tmp_path))
    project_service = ProjectService()
    artifact_service = ArtifactService()
    fit_run = RunRecord(workflow="fit", run_id="fit-run-1", status=RunStatus.SUCCEEDED)
    project_service.add_run(project, fit_run)

    plot_path = tmp_path / "generated-vpc.png"
    plot_path.write_bytes(b"PNG")
    summary_path = tmp_path / "generated-vpc.csv"
    summary_path.write_text("bin_mid,p50\n1.0,2.0\n", encoding="utf-8")

    class _FakeFitService:
        def latest_fit_context(self, workspace: Workspace):
            return object() if workspace is project else None

    class _FakeVPCService(VPCService):
        def create_job(
            self, workspace: Workspace, *, fit_service, run_id: str | None = None, **_kwargs
        ):
            plot = ArtifactRecord(
                kind="plot",
                label="Generated VPC plot",
                path=str(plot_path),
                source_run_id=fit_run.run_id,
                metadata={
                    "artifact_role": "plot",
                    "plot_type": "vpc",
                    "media_type": "image/png",
                    "fit_run_id": fit_run.run_id,
                    "vpc_run_id": run_id,
                },
            )
            table = ArtifactRecord(
                kind="table",
                label="Generated VPC summary",
                path=str(summary_path),
                source_run_id=fit_run.run_id,
                metadata={
                    "artifact_role": "vpc_summary",
                    "media_type": "text/csv",
                    "fit_run_id": fit_run.run_id,
                    "vpc_run_id": run_id,
                },
            )
            return BackgroundJob(
                name="vpc:demo",
                func=lambda _ctx: VPCRunResult(
                    summary_text="Generated VPC", artifacts=[plot, table]
                ),
            )

    job_runner = JobRunner(max_workers=1)
    widget = build_advanced_workflow(
        project,
        artifact_service=artifact_service,
        fit_service=_FakeFitService(),
        vpc_service=_FakeVPCService(),
        bootstrap_service=BootstrapService(),
        design_service=DesignService(),
        project_service=project_service,
        job_runner=job_runner,
    )

    try:
        widget.show()
        app.processEvents()

        tab_widget = widget.findChild(qt_widgets.QTabWidget, "advanced-tab-widget")
        artifact_scope_combo = widget.findChild(
            qt_widgets.QComboBox, "advanced-artifact-scope-combo"
        )
        status_label = widget.findChild(qt_widgets.QLabel, "advanced-vpc-status-label")
        generate_button = widget.findChild(qt_widgets.QPushButton, "advanced-generate-vpc-button")
        open_plot_button = widget.findChild(qt_widgets.QPushButton, "advanced-open-vpc-plot-button")
        open_summary_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-open-vpc-summary-button"
        )

        assert tab_widget is not None
        assert artifact_scope_combo is not None
        assert tab_widget.count() == 4
        assert status_label is not None
        assert generate_button is not None
        assert open_plot_button is not None
        assert open_summary_button is not None
        assert generate_button.isEnabled() is True
        assert "Generate VPC on demand" in status_label.text()

        generate_button.click()
        app.processEvents()
        assert "running in the background" in status_label.text()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            if open_plot_button.isEnabled() and open_summary_button.isEnabled():
                break
            time.sleep(0.05)

        assert open_plot_button.isEnabled() is True
        assert open_summary_button.isEnabled() is True
        assert len(project.artifacts) == 2
        vpc_runs = [run for run in project.runs if run.workflow == "vpc"]
        assert len(vpc_runs) == 1
        assert vpc_runs[0].status == RunStatus.SUCCEEDED
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
        job_runner.shutdown(wait=True)


@pytest.mark.unit
def test_advanced_workflow_next_actions_transition_from_fit_to_latest_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    vpc_plot_path = tmp_path / "generated-vpc.png"
    vpc_plot_path.write_bytes(b"PNG")
    bootstrap_summary_path = tmp_path / "generated-bootstrap-summary.csv"
    bootstrap_summary_path.write_text("parameter,mean\nTHETA(1),1.2\n", encoding="utf-8")
    design_summary_path = tmp_path / "generated-design-summary.txt"
    design_summary_path.write_text("Design summary", encoding="utf-8")

    _, qt_gui, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Advanced CTA", root_path=str(tmp_path))
    project_service = ProjectService()
    artifact_service = ArtifactService()
    job_runner = JobRunner(max_workers=1)

    class _FakeFitService:
        def latest_fit_context(self, workspace: Workspace):
            run = next(
                (
                    candidate
                    for candidate in reversed(workspace.runs)
                    if candidate.workflow == "fit"
                ),
                None,
            )
            if workspace is project and run is not None and run.status == RunStatus.SUCCEEDED:
                return object()
            return None

    widget = build_advanced_workflow(
        project,
        artifact_service=artifact_service,
        fit_service=_FakeFitService(),
        vpc_service=VPCService(),
        bootstrap_service=BootstrapService(),
        design_service=DesignService(),
        project_service=project_service,
        job_runner=job_runner,
    )
    navigations: list[str] = []
    opened_urls: list[str] = []
    widget._navigate_to_workflow = lambda workflow_id: navigations.append(workflow_id)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        qt_gui.QDesktopServices, "openUrl", lambda url: opened_urls.append(url.toLocalFile())
    )

    try:
        widget.show()
        app.processEvents()

        vpc_button = widget.findChild(qt_widgets.QPushButton, "advanced-vpc-next-action-button")
        bootstrap_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-bootstrap-next-action-button"
        )
        design_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-design-next-action-button"
        )
        vpc_label = widget.findChild(qt_widgets.QLabel, "advanced-vpc-next-action-label")
        generate_button = widget.findChild(qt_widgets.QPushButton, "advanced-generate-vpc-button")

        assert vpc_button is not None
        assert bootstrap_button is not None
        assert design_button is not None
        assert vpc_label is not None
        assert generate_button is not None

        assert vpc_button.text() == "Open Fit"
        assert bootstrap_button.text() == "Open Fit"
        assert design_button.text() == "Open Fit"

        vpc_button.click()
        app.processEvents()
        assert navigations == ["fit"]

        fit_run = RunRecord(workflow="fit", status=RunStatus.SUCCEEDED, summary_text="Fit finished")
        project_service.add_run(project, fit_run)
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert vpc_button.isHidden() is True
        assert bootstrap_button.isHidden() is True
        assert design_button.isHidden() is True
        assert generate_button.isEnabled() is True

        project_service.add_artifact(
            project,
            ArtifactRecord(
                kind="plot",
                label="Generated VPC plot",
                path=str(vpc_plot_path),
                source_run_id=fit_run.run_id,
                metadata={"artifact_role": "plot", "plot_type": "vpc", "media_type": "image/png"},
            ),
        )
        project_service.add_artifact(
            project,
            ArtifactRecord(
                kind="table",
                label="Generated bootstrap summary",
                path=str(bootstrap_summary_path),
                source_run_id=fit_run.run_id,
                metadata={"artifact_role": "bootstrap_summary", "media_type": "text/csv"},
            ),
        )
        project_service.add_artifact(
            project,
            ArtifactRecord(
                kind="report",
                label="Generated design summary",
                path=str(design_summary_path),
                source_run_id=fit_run.run_id,
                metadata={"artifact_role": "design_summary", "media_type": "text/plain"},
            ),
        )
        widget._refresh_workflow()  # type: ignore[attr-defined]
        app.processEvents()

        assert vpc_button.text() == "Open latest VPC plot"
        assert bootstrap_button.text() == "Open latest bootstrap summary"
        assert design_button.text() == "Open latest design summary"
        assert "Latest VPC outputs are already available" in vpc_label.text()

        vpc_button.click()
        bootstrap_button.click()
        design_button.click()
        app.processEvents()

        assert opened_urls == [
            str(vpc_plot_path),
            str(bootstrap_summary_path),
            str(design_summary_path),
        ]
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
        job_runner.shutdown(wait=True)


@pytest.mark.unit
def test_advanced_workflow_generates_bootstrap_on_demand(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Advanced bootstrap", root_path=str(tmp_path))
    project_service = ProjectService()
    artifact_service = ArtifactService()
    fit_run = RunRecord(workflow="fit", run_id="fit-run-2", status=RunStatus.SUCCEEDED)
    project_service.add_run(project, fit_run)

    summary_path = tmp_path / "generated-bootstrap-summary.csv"
    summary_path.write_text("parameter,mean\nTHETA(1),1.2\n", encoding="utf-8")
    ci_path = tmp_path / "generated-bootstrap-ci.csv"
    ci_path.write_text("parameter,bca_lo,bca_hi\nTHETA(1),1.0,1.4\n", encoding="utf-8")
    samples_path = tmp_path / "generated-bootstrap-samples.csv"
    samples_path.write_text("THETA(1)\n1.1\n1.3\n", encoding="utf-8")

    class _FakeFitService:
        def latest_fit_context(self, workspace: Workspace):
            return object() if workspace is project else None

    class _FakeBootstrapService(BootstrapService):
        def create_job(
            self, workspace: Workspace, *, fit_service, run_id: str | None = None, **_kwargs
        ):
            artifacts = [
                ArtifactRecord(
                    kind="table",
                    label="Generated bootstrap summary",
                    path=str(summary_path),
                    source_run_id=fit_run.run_id,
                    metadata={
                        "artifact_role": "bootstrap_summary",
                        "media_type": "text/csv",
                        "bootstrap_run_id": run_id,
                    },
                ),
                ArtifactRecord(
                    kind="table",
                    label="Generated bootstrap CI",
                    path=str(ci_path),
                    source_run_id=fit_run.run_id,
                    metadata={
                        "artifact_role": "bootstrap_ci_table",
                        "media_type": "text/csv",
                        "bootstrap_run_id": run_id,
                    },
                ),
                ArtifactRecord(
                    kind="table",
                    label="Generated bootstrap samples",
                    path=str(samples_path),
                    source_run_id=fit_run.run_id,
                    metadata={
                        "artifact_role": "bootstrap_samples",
                        "media_type": "text/csv",
                        "bootstrap_run_id": run_id,
                    },
                ),
            ]
            return BackgroundJob(
                name="bootstrap:demo",
                func=lambda _ctx: BootstrapRunResult(
                    summary_text="Generated bootstrap", n_success=12, artifacts=artifacts
                ),
            )

    job_runner = JobRunner(max_workers=1)
    widget = build_advanced_workflow(
        project,
        artifact_service=artifact_service,
        fit_service=_FakeFitService(),
        vpc_service=VPCService(),
        bootstrap_service=_FakeBootstrapService(),
        design_service=DesignService(),
        project_service=project_service,
        job_runner=job_runner,
    )

    try:
        widget.show()
        app.processEvents()

        status_label = widget.findChild(qt_widgets.QLabel, "advanced-bootstrap-status-label")
        generate_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-generate-bootstrap-button"
        )
        open_summary_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-open-bootstrap-summary-button"
        )
        open_ci_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-open-bootstrap-ci-button"
        )
        open_samples_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-open-bootstrap-samples-button"
        )

        assert status_label is not None
        assert generate_button is not None
        assert open_summary_button is not None
        assert open_ci_button is not None
        assert open_samples_button is not None
        assert generate_button.isEnabled() is True
        assert "Generate bootstrap summaries" in status_label.text()

        generate_button.click()
        app.processEvents()
        assert "running in the background" in status_label.text()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            if (
                open_summary_button.isEnabled()
                and open_ci_button.isEnabled()
                and open_samples_button.isEnabled()
            ):
                break
            time.sleep(0.05)

        assert open_summary_button.isEnabled() is True
        assert open_ci_button.isEnabled() is True
        assert open_samples_button.isEnabled() is True
        assert len(project.artifacts) == 3
        bootstrap_runs = [run for run in project.runs if run.workflow == "bootstrap"]
        assert len(bootstrap_runs) == 1
        assert bootstrap_runs[0].status == RunStatus.SUCCEEDED
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
        job_runner.shutdown(wait=True)


@pytest.mark.unit
def test_advanced_workflow_generates_design_on_demand(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Advanced design", root_path=str(tmp_path))
    project_service = ProjectService()
    artifact_service = ArtifactService()
    fit_run = RunRecord(workflow="fit", run_id="fit-run-3", status=RunStatus.SUCCEEDED)
    project_service.add_run(project, fit_run)

    summary_path = tmp_path / "generated-design-summary.txt"
    summary_path.write_text("Design summary", encoding="utf-8")
    schedule_path = tmp_path / "generated-design-schedule.csv"
    schedule_path.write_text("design_kind,order,time\noptimized,1,1.0\n", encoding="utf-8")

    class _FakeFitService:
        def latest_fit_context(self, workspace: Workspace):
            return object() if workspace is project else None

    class _FakeDesignService(DesignService):
        def create_job(
            self, workspace: Workspace, *, fit_service, run_id: str | None = None, **_kwargs
        ):
            artifacts = [
                ArtifactRecord(
                    kind="report",
                    label="Generated design summary",
                    path=str(summary_path),
                    source_run_id=fit_run.run_id,
                    metadata={
                        "artifact_role": "design_summary",
                        "media_type": "text/plain",
                        "design_run_id": run_id,
                    },
                ),
                ArtifactRecord(
                    kind="table",
                    label="Generated design schedule",
                    path=str(schedule_path),
                    source_run_id=fit_run.run_id,
                    metadata={
                        "artifact_role": "design_schedule",
                        "media_type": "text/csv",
                        "design_run_id": run_id,
                    },
                ),
            ]
            return BackgroundJob(
                name="design:demo",
                func=lambda _ctx: DesignRunResult(
                    summary_text="Generated design", artifacts=artifacts
                ),
            )

    job_runner = JobRunner(max_workers=1)
    widget = build_advanced_workflow(
        project,
        artifact_service=artifact_service,
        fit_service=_FakeFitService(),
        vpc_service=VPCService(),
        bootstrap_service=BootstrapService(),
        design_service=_FakeDesignService(),
        project_service=project_service,
        job_runner=job_runner,
    )

    try:
        widget.show()
        app.processEvents()

        status_label = widget.findChild(qt_widgets.QLabel, "advanced-design-status-label")
        generate_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-generate-design-button"
        )
        open_summary_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-open-design-summary-button"
        )
        open_schedule_button = widget.findChild(
            qt_widgets.QPushButton, "advanced-open-design-schedule-button"
        )

        assert status_label is not None
        assert generate_button is not None
        assert open_summary_button is not None
        assert open_schedule_button is not None
        assert generate_button.isEnabled() is True
        assert "Generate optimal design summaries" in status_label.text()

        generate_button.click()
        app.processEvents()
        assert "running in the background" in status_label.text()

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            if open_summary_button.isEnabled() and open_schedule_button.isEnabled():
                break
            time.sleep(0.05)

        assert open_summary_button.isEnabled() is True
        assert open_schedule_button.isEnabled() is True
        assert len(project.artifacts) == 2
        design_runs = [run for run in project.runs if run.workflow == "design"]
        assert len(design_runs) == 1
        assert design_runs[0].status == RunStatus.SUCCEEDED
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
        job_runner.shutdown(wait=True)


# ---------------------------------------------------------------------------
# P3-F: VPC stratification combo and pcVPC checkbox
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_advanced_workflow_has_stratify_combo() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Stratify combo test")
    widget = build_advanced_workflow(project)
    try:
        widget.show()
        app.processEvents()
        combo = widget.findChild(qt_widgets.QComboBox, "advanced-vpc-stratify-combo")
        assert combo is not None, "advanced-vpc-stratify-combo not found"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_advanced_workflow_stratify_combo_has_none_default() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Stratify default test")
    widget = build_advanced_workflow(project)
    try:
        widget.show()
        app.processEvents()
        combo = widget.findChild(qt_widgets.QComboBox, "advanced-vpc-stratify-combo")
        assert combo is not None
        assert combo.currentData() is None
        assert combo.currentText() == "None"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_advanced_workflow_stratify_combo_populated_from_dataset() -> None:
    """Stratify combo shows dataset columns (excluding mandatory NONMEM columns) after refresh."""
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    from openpkpd_gui.domain.dataset_asset import DatasetAsset

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="Stratify columns test")
    project.active_dataset = DatasetAsset(
        display_name="theo.csv",
        columns=["ID", "TIME", "AMT", "DV", "EVID", "MDV", "DOSE", "SEX"],
    )
    widget = build_advanced_workflow(project)
    try:
        widget.show()
        app.processEvents()
        combo = widget.findChild(qt_widgets.QComboBox, "advanced-vpc-stratify-combo")
        assert combo is not None
        items = [combo.itemData(i) for i in range(combo.count())]
        assert None in items          # "None" placeholder
        assert "DOSE" in items
        assert "SEX" in items
        # Mandatory NONMEM columns must be excluded
        assert "ID" not in items
        assert "TIME" not in items
        assert "DV" not in items
        assert "MDV" not in items
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.unit
def test_advanced_workflow_has_pc_checkbox() -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    _, _, qt_widgets = load_qt_modules()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    project = Workspace(name="pcVPC checkbox test")
    widget = build_advanced_workflow(project)
    try:
        widget.show()
        app.processEvents()
        checkbox = widget.findChild(qt_widgets.QCheckBox, "advanced-vpc-pc-checkbox")
        assert checkbox is not None, "advanced-vpc-pc-checkbox not found"
        assert checkbox.isChecked() is False
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
