"""Background VPC generation for the GUI Advanced workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from openpkpd.plots.simulation import prediction_interval_plot, simulation_panel, vpc_plot
from openpkpd.simulation.engine import SimulationEngine
from openpkpd.simulation.vpc import VPCEngine
from openpkpd_gui.app.settings import default_workspace_root_path
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobOutcome, JobStatus
from openpkpd_gui.services.fit_service import FitService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VPCConfig:
    """User-selected VPC options from the Advanced workflow."""

    n_replicates: int = 200
    n_bins: int = 10
    seed: int = 42
    prediction_corrected: bool = False
    n_parallel: int = 0


@dataclass(slots=True)
class VPCRunResult:
    """Minimal VPC run result surfaced back to the GUI."""

    summary_text: str
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)


class VPCService:
    """Create background jobs that compute VPC artifacts from a cached fit."""

    VPC_PLOT_TYPES = {"vpc", "simulation_panel", "prediction_interval_plot"}

    def create_job(
        self,
        workspace: Workspace,
        *,
        fit_service: FitService,
        config: VPCConfig | None = None,
        run_id: str | None = None,
    ) -> BackgroundJob:
        context = fit_service.latest_fit_context(workspace)
        if context is None:
            raise ValueError(
                "VPC generation requires a reusable successful fit for this scenario."
            )
        config = config or VPCConfig()
        title = context.problem_title or workspace.name or "VPC"
        target_project_id = context.project_id
        target_scenario_id = context.scenario_id
        target_dataset_path = context.dataset_path
        target_fit_run_id = context.fit_run_id
        target_method = context.estimation_method

        def _run(ctx) -> VPCRunResult:
            ctx.emit(f"Preparing VPC for {title}", progress=0.1)
            sim_engine = SimulationEngine(
                context.population_model,
                context.estimation_result,
                seed=config.seed,
                n_parallel=config.n_parallel,
            )
            ctx.emit(
                f"Running {config.n_replicates} simulation replicates across {config.n_bins} bins",
                progress=0.45,
            )
            vpc_result = VPCEngine(sim_engine).compute(
                n_replicates=config.n_replicates,
                n_bins=config.n_bins,
                prediction_corrected=config.prediction_corrected,
            )
            ctx.emit("Writing VPC artifacts", progress=0.8)
            artifacts, warnings = self._write_artifacts(
                workspace,
                title,
                vpc_result,
                config,
                run_id=run_id,
                fit_run_id=target_fit_run_id,
                estimation_method=target_method,
                project_id=target_project_id,
                scenario_id=target_scenario_id,
                dataset_path=target_dataset_path,
            )
            kind = "pcVPC" if config.prediction_corrected else "VPC"
            summary = f"{title} • {kind} • {config.n_replicates} sims • {config.n_bins} bins • seed={config.seed}"
            return VPCRunResult(
                summary_text=summary,
                artifacts=artifacts,
                warning_messages=warnings,
            )

        return BackgroundJob(name=f"vpc:{title}", func=_run)

    def apply_job_outcome(self, run: RunRecord, outcome: JobOutcome) -> list[ArtifactRecord]:
        for event in outcome.events:
            run.add_log(f"[{event.kind}] {event.message}")
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, VPCRunResult):
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
        run.mark_failed(outcome.error or "VPC generation failed.")
        return []

    @staticmethod
    def latest_run(workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.active_scenario.runs):
            if run.workflow == "vpc":
                return run
        return None

    @staticmethod
    def _slug(text: str) -> str:
        safe = "".join(
            character if character.isalnum() else "-" for character in text.strip().lower()
        )
        return safe.strip("-") or "vpc"

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

    def _write_artifacts(
        self,
        workspace: Workspace,
        title: str,
        vpc_result,
        config: VPCConfig,
        *,
        run_id: str | None,
        fit_run_id: str,
        estimation_method: str,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
    ) -> tuple[list[ArtifactRecord], list[str]]:
        artifact_dir = self._artifact_directory(
            workspace,
            project_id=project_id,
            scenario_id=scenario_id,
            dataset_path=dataset_path,
        )
        stem = f"{self._slug(title)}-vpc-{(run_id or 'run')[:8]}"
        summary_df = pd.merge(
            vpc_result.obs_percentiles,
            vpc_result.sim_percentiles,
            on="bin_mid",
            how="outer",
        )
        summary_path = artifact_dir / f"{stem}-summary.csv"
        summary_df.to_csv(summary_path, index=False)
        metadata = {
            "estimation_method": estimation_method,
            "n_replicates": config.n_replicates,
            "n_bins": config.n_bins,
            "seed": config.seed,
            "prediction_corrected": config.prediction_corrected,
            "fit_run_id": fit_run_id,
            "vpc_run_id": run_id,
        }
        artifacts = [
            ArtifactRecord(
                kind="table",
                label=f"{title} VPC summary",
                path=str(summary_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "vpc_summary",
                    "media_type": "text/csv",
                    "row_count": len(summary_df),
                },
            )
        ]
        warnings: list[str] = []

        def _append_plot(figure, *, suffix: str, label: str, plot_type: str) -> None:
            plot_path = artifact_dir / f"{stem}-{suffix}.png"
            try:
                figure.savefig(plot_path, dpi=150, bbox_inches="tight")
            finally:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:
                    logger.debug("Failed to close VPC figure %s", plot_type, exc_info=True)
            artifacts.append(
                ArtifactRecord(
                    kind="plot",
                    label=label,
                    path=str(plot_path),
                    source_run_id=fit_run_id,
                    metadata={
                        **metadata,
                        "artifact_role": "plot",
                        "plot_type": plot_type,
                        "media_type": "image/png",
                    },
                )
            )

        try:
            kind = "pcVPC" if config.prediction_corrected else "VPC"
            figure = vpc_plot(
                vpc_result,
                title=f"{title} — {kind}",
            )
            _append_plot(
                figure,
                suffix="plot",
                label=f"{title} VPC plot",
                plot_type="vpc",
            )
        except Exception as exc:
            warning = f"Could not generate VPC plot: {exc}"
            warnings.append(warning)
            logger.warning("%s", warning, exc_info=True)

        try:
            figure = simulation_panel(
                vpc_result.simulated_df,
                observed_df=vpc_result.observed_df,
                title=f"{title} — simulated profiles",
            )
            _append_plot(
                figure,
                suffix="simulation-panel",
                label=f"{title} simulation panel",
                plot_type="simulation_panel",
            )
        except Exception as exc:
            warning = f"Could not generate simulation panel plot: {exc}"
            warnings.append(warning)
            logger.warning("%s", warning, exc_info=True)

        required_columns = {"bin_mid", "p50", "p5_mid", "p50_mid", "p95_mid"}
        if required_columns.issubset(summary_df.columns):
            interval_df = summary_df.loc[
                :, ["bin_mid", "p50", "p5_mid", "p50_mid", "p95_mid"]
            ].dropna()
            if not interval_df.empty:
                try:
                    figure = prediction_interval_plot(
                        interval_df["bin_mid"].to_numpy(),
                        interval_df["p50"].to_numpy(),
                        interval_df["p5_mid"].to_numpy(),
                        interval_df["p50_mid"].to_numpy(),
                        interval_df["p95_mid"].to_numpy(),
                        title=f"{title} — prediction interval",
                    )
                    _append_plot(
                        figure,
                        suffix="prediction-interval",
                        label=f"{title} prediction interval plot",
                        plot_type="prediction_interval_plot",
                    )
                except Exception as exc:
                    warning = f"Could not generate prediction interval plot: {exc}"
                    warnings.append(warning)
                    logger.warning("%s", warning, exc_info=True)

        if warnings:
            artifacts[0].metadata = {**artifacts[0].metadata, "warnings": list(warnings)}
        return artifacts, warnings
