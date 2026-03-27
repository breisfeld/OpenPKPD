"""Prepare and run on-demand NPDE jobs from the selected workspace trial."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from openpkpd.plots.diagnostics import compute_npde
from openpkpd.plots.simulation import npde_plot
from openpkpd_gui.app.settings import default_workspace_root_path
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobOutcome, JobStatus
from openpkpd_gui.services.fit_service import FitService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NPDERunResult:
    """Minimal NPDE result surfaced back to the GUI."""

    summary_text: str
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)


class NPDEService:
    """Create background jobs that compute NPDE on demand from a cached fit."""

    def create_job(
        self,
        workspace: Workspace,
        *,
        fit_service: FitService,
        run_id: str | None = None,
        n_simulations: int = 1000,
        seed: int = 42,
        decorrelate: bool = True,
    ) -> BackgroundJob:
        context = fit_service.latest_fit_context(workspace)
        if context is None:
            raise ValueError(
                "NPDE generation requires a reusable successful fit for this scenario."
            )

        def _run(ctx) -> NPDERunResult:
            ctx.emit(f"Preparing NPDE for {context.problem_title}", progress=0.05)
            npde_df = compute_npde(
                context.population_model,
                context.estimation_result,
                n_simulations=n_simulations,
                seed=seed,
                decorrelate=decorrelate,
            )
            artifacts, warnings = self._write_results_artifacts(
                workspace,
                context.problem_title,
                npde_df,
                run_id,
                fit_run_id=context.fit_run_id,
                project_id=context.project_id,
                scenario_id=context.scenario_id,
                dataset_path=context.dataset_path,
                estimation_method=context.estimation_method,
                n_simulations=n_simulations,
                seed=seed,
                decorrelate=decorrelate,
            )
            summary_text = (
                f"{context.problem_title} • NPDE rows {len(npde_df)} • {context.estimation_method}"
            )
            return NPDERunResult(
                summary_text=summary_text,
                artifacts=artifacts,
                warning_messages=warnings,
            )

        return BackgroundJob(name=f"npde:{context.problem_title}", func=_run)

    def apply_job_outcome(self, run: RunRecord, outcome: JobOutcome) -> list[ArtifactRecord]:
        for event in outcome.events:
            run.add_log(f"[{event.kind}] {event.message}")
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, NPDERunResult):
            for warning in outcome.value.warning_messages:
                run.add_log(f"[warning] {warning}")
            artifacts: list[ArtifactRecord] = []
            for artifact in outcome.value.artifacts:
                if artifact.source_run_id is None:
                    artifact.source_run_id = run.run_id
                if artifact.artifact_id not in run.artifact_ids:
                    run.artifact_ids.append(artifact.artifact_id)
                artifacts.append(artifact)
            run.mark_succeeded(outcome.value.summary_text)
            return artifacts
        if outcome.status == JobStatus.CANCELLED:
            run.mark_cancelled(outcome.error or "Cancelled by user.")
            return []
        run.mark_failed(outcome.error or "NPDE generation failed.")
        return []

    @staticmethod
    def latest_run(workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.active_scenario.runs):
            if run.workflow == "npde":
                return run
        return None

    @staticmethod
    def _slug(text: str) -> str:
        safe = "".join(
            character if character.isalnum() else "-" for character in text.strip().lower()
        )
        return safe.strip("-") or "npde"

    def _artifact_directory(
        self,
        workspace: Workspace,
        *,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
    ) -> Path:
        base_path = workspace.root_path
        if not base_path and dataset_path:
            base_path = str(Path(dataset_path).resolve().parent)
        root = Path(base_path).resolve() if base_path else default_workspace_root_path()
        artifact_dir = (
            root / ".openpkpd_gui_artifacts" / workspace.workspace_id / project_id / scenario_id
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _write_results_artifacts(
        self,
        workspace: Workspace,
        title: str,
        npde_df,
        run_id: str | None,
        *,
        fit_run_id: str,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
        estimation_method: str,
        n_simulations: int,
        seed: int,
        decorrelate: bool,
    ) -> tuple[list[ArtifactRecord], list[str]]:
        artifact_dir = self._artifact_directory(
            workspace,
            project_id=project_id,
            scenario_id=scenario_id,
            dataset_path=dataset_path,
        )
        stem = f"{self._slug(title)}-npde-{(run_id or 'run')[:8]}"
        csv_path = artifact_dir / f"{stem}.csv"
        npde_df.to_csv(csv_path, index=False)
        metadata = {
            "estimation_method": estimation_method,
            "n_simulations": n_simulations,
            "seed": seed,
            "decorrelate": decorrelate,
            "npde_run_id": run_id,
            "fit_run_id": fit_run_id,
        }
        artifacts = [
            ArtifactRecord(
                kind="table",
                label=f"{title} NPDE table",
                path=str(csv_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "media_type": "text/csv",
                    "artifact_role": "npde_table",
                    "row_count": len(npde_df),
                },
            )
        ]
        warnings: list[str] = []

        try:
            figure = npde_plot(npde_df, title=f"{title} — NPDE diagnostics")
            plot_path = artifact_dir / f"{stem}-plot.png"
            try:
                figure.savefig(plot_path, dpi=150, bbox_inches="tight")
            finally:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:
                    logger.debug("Failed to close NPDE figure", exc_info=True)
            artifacts.append(
                ArtifactRecord(
                    kind="plot",
                    label=f"{title} NPDE plot",
                    path=str(plot_path),
                    source_run_id=fit_run_id,
                    metadata={
                        **metadata,
                        "media_type": "image/png",
                        "artifact_role": "plot",
                        "plot_type": "npde_plot",
                    },
                )
            )
        except Exception as exc:
            warning = f"Could not generate NPDE plot: {exc}"
            warnings.append(warning)
            logger.warning("%s", warning, exc_info=True)

        if warnings:
            artifacts[0].metadata = {**artifacts[0].metadata, "warnings": list(warnings)}
        return artifacts, warnings
