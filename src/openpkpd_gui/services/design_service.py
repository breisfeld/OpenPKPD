"""Background optimal-design generation for the GUI Advanced workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from openpkpd.design.pfim import PFIMEngine
from openpkpd.model.parameters import ParameterSet
from openpkpd_gui.app.settings import default_workspace_root_path
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobOutcome, JobStatus
from openpkpd_gui.services.fit_service import FitService


@dataclass(slots=True)
class DesignConfig:
    """User-selected optimal-design options from the Advanced workflow."""

    n_samples: int = 6
    t_min: float = 0.0
    t_max: float = 24.0
    n_subjects: int = 10
    criterion: str = "D"
    method: str = "differential_evolution"
    n_starts: int = 10


@dataclass(slots=True)
class DesignRunResult:
    """Minimal design run result surfaced back to the GUI."""

    summary_text: str
    artifacts: list[ArtifactRecord] = field(default_factory=list)


class DesignService:
    """Create background jobs that optimize sampling schedules from a cached fit."""

    def create_job(
        self,
        workspace: Workspace,
        *,
        fit_service: FitService,
        config: DesignConfig | None = None,
        run_id: str | None = None,
    ) -> BackgroundJob:
        context = fit_service.latest_fit_context(workspace)
        if context is None:
            raise ValueError(
                "Design generation requires a successful fit from the current session for this scenario."
            )
        config = config or DesignConfig()
        title = context.problem_title or workspace.name or "Design"
        target_project_id = context.project_id
        target_scenario_id = context.scenario_id
        target_dataset_path = context.dataset_path
        target_fit_run_id = context.fit_run_id
        target_method = context.estimation_method
        target_initial_params = self._final_parameter_set(context)

        def _run(ctx) -> DesignRunResult:
            ctx.emit(f"Preparing design optimization for {title}", progress=0.1)
            engine = PFIMEngine(context.population_model, target_initial_params)
            reference_times = self._observed_sampling_times(context.population_model)
            ctx.emit(
                f"Optimizing {config.n_samples} sampling times with {config.criterion.upper()}-criterion",
                progress=0.45,
            )
            design_result = engine.optimize_design(
                n_samples=config.n_samples,
                t_min=config.t_min,
                t_max=config.t_max,
                n_subjects=config.n_subjects,
                criterion=config.criterion,
                method=config.method,
                n_starts=config.n_starts,
            )
            relative_efficiency: float | None = None
            if reference_times.size:
                try:
                    relative_efficiency = engine.efficiency(
                        design_result.sampling_times,
                        reference_times,
                        criterion=config.criterion,
                        n_subjects=config.n_subjects,
                    )
                except Exception:
                    relative_efficiency = None
            ctx.emit("Writing design artifacts", progress=0.85)
            artifacts = self._write_artifacts(
                workspace,
                title,
                design_result,
                config,
                reference_times=reference_times,
                relative_efficiency=relative_efficiency,
                run_id=run_id,
                fit_run_id=target_fit_run_id,
                estimation_method=target_method,
                project_id=target_project_id,
                scenario_id=target_scenario_id,
                dataset_path=target_dataset_path,
            )
            summary = (
                f"{title} • {config.criterion.upper()}-optimal design • {config.n_samples} samples • "
                f"{config.n_subjects} subjects"
            )
            if relative_efficiency is not None and np.isfinite(relative_efficiency):
                summary += f" • relative efficiency={relative_efficiency:.3f}"
            return DesignRunResult(summary_text=summary, artifacts=artifacts)

        return BackgroundJob(name=f"design:{title}", func=_run)

    def apply_job_outcome(self, run: RunRecord, outcome: JobOutcome) -> list[ArtifactRecord]:
        for event in outcome.events:
            run.add_log(f"[{event.kind}] {event.message}")
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, DesignRunResult):
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
        run.mark_failed(outcome.error or "Design generation failed.")
        return []

    @staticmethod
    def latest_run(workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.active_scenario.runs):
            if run.workflow == "design":
                return run
        return None

    @staticmethod
    def _slug(text: str) -> str:
        safe = "".join(
            character if character.isalnum() else "-" for character in text.strip().lower()
        )
        return safe.strip("-") or "design"

    @staticmethod
    def _final_parameter_set(context) -> ParameterSet:
        template = getattr(context.population_model, "params", None)
        result = context.estimation_result
        if template is None:
            raise ValueError(
                "Design generation requires the cached fit context to expose population-model parameters."
            )
        theta_final = getattr(result, "theta_final", None)
        omega_final = getattr(result, "omega_final", None)
        sigma_final = getattr(result, "sigma_final", None)
        if theta_final is None or omega_final is None or sigma_final is None:
            raise ValueError(
                "Design generation requires final THETA/OMEGA/SIGMA estimates from the cached fit context."
            )
        return ParameterSet(
            theta=np.asarray(theta_final, dtype=float).copy(),
            omega=np.asarray(omega_final, dtype=float).copy(),
            sigma=np.asarray(sigma_final, dtype=float).copy(),
            theta_specs=list(getattr(template, "theta_specs", [])),
            omega_specs=list(getattr(template, "omega_specs", [])),
            sigma_specs=list(getattr(template, "sigma_specs", [])),
        ).apply_bounds()

    @staticmethod
    def _observed_sampling_times(population_model: object) -> np.ndarray:
        try:
            subject_id = next(iter(population_model.subject_ids()))
            individual = population_model.individual_model(subject_id)
            times = np.asarray(getattr(individual.subject_events, "obs_times", []), dtype=float)
        except Exception:
            return np.array([], dtype=float)
        times = times[np.isfinite(times)]
        if times.size == 0:
            return np.array([], dtype=float)
        return np.unique(np.sort(times))

    @staticmethod
    def _theta_labels(count: int) -> list[str]:
        return [f"THETA({index + 1})" for index in range(count)]

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
        design_result,
        config: DesignConfig,
        *,
        reference_times: np.ndarray,
        relative_efficiency: float | None,
        run_id: str | None,
        fit_run_id: str,
        estimation_method: str,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
    ) -> list[ArtifactRecord]:
        artifact_dir = self._artifact_directory(
            workspace,
            project_id=project_id,
            scenario_id=scenario_id,
            dataset_path=dataset_path,
        )
        stem = f"{self._slug(title)}-design-{(run_id or 'run')[:8]}"
        metadata = {
            "estimation_method": estimation_method,
            "criterion": config.criterion.upper(),
            "optimization_method": config.method,
            "n_samples": config.n_samples,
            "n_subjects": config.n_subjects,
            "t_min": config.t_min,
            "t_max": config.t_max,
            "n_starts": config.n_starts,
            "fit_run_id": fit_run_id,
            "design_run_id": run_id,
        }

        summary_lines = [
            design_result.summary(),
            f"Criterion: {config.criterion.upper()}",
            f"Optimization method: {config.method}",
            f"Subjects: {config.n_subjects}",
            f"Time window: [{config.t_min}, {config.t_max}]",
        ]
        if reference_times.size:
            summary_lines.append(
                f"Observed/reference times: {np.round(reference_times, 3).tolist()}"
            )
        if relative_efficiency is not None and np.isfinite(relative_efficiency):
            summary_lines.append(f"Relative efficiency vs observed: {relative_efficiency:.6f}")
        summary_path = artifact_dir / f"{stem}-summary.txt"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        metrics_rows = [
            ("criterion", config.criterion.upper()),
            ("optimization_method", config.method),
            ("n_samples", config.n_samples),
            ("n_subjects", config.n_subjects),
            ("t_min", config.t_min),
            ("t_max", config.t_max),
            ("a_efficiency", float(design_result.a_efficiency)),
            ("condition_number", float(design_result.condition_number)),
            ("reference_sample_count", int(reference_times.size)),
            (
                "relative_efficiency",
                float(relative_efficiency)
                if relative_efficiency is not None and np.isfinite(relative_efficiency)
                else np.nan,
            ),
        ]
        metrics_df = pd.DataFrame(metrics_rows, columns=["metric", "value"])
        metrics_path = artifact_dir / f"{stem}-metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        schedule_rows = [
            {"design_kind": "optimized", "order": index + 1, "time": float(time_value)}
            for index, time_value in enumerate(
                np.asarray(design_result.sampling_times, dtype=float)
            )
        ]
        schedule_rows.extend(
            {"design_kind": "reference", "order": index + 1, "time": float(time_value)}
            for index, time_value in enumerate(reference_times)
        )
        schedule_df = pd.DataFrame(schedule_rows, columns=["design_kind", "order", "time"])
        schedule_path = artifact_dir / f"{stem}-schedule.csv"
        schedule_df.to_csv(schedule_path, index=False)

        theta_labels = self._theta_labels(len(design_result.se_theta))
        fim_df = pd.DataFrame(
            design_result.information_matrix, index=theta_labels, columns=theta_labels
        )
        fim_path = artifact_dir / f"{stem}-fim.csv"
        fim_df.to_csv(fim_path)

        se_df = pd.DataFrame(
            {
                "parameter": theta_labels,
                "expected_se": np.asarray(design_result.se_theta, dtype=float),
            }
        )
        se_path = artifact_dir / f"{stem}-theta-se.csv"
        se_df.to_csv(se_path, index=False)

        return [
            ArtifactRecord(
                kind="report",
                label=f"{title} design summary",
                path=str(summary_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "design_summary",
                    "media_type": "text/plain",
                },
            ),
            ArtifactRecord(
                kind="table",
                label=f"{title} design metrics",
                path=str(metrics_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "design_metrics",
                    "media_type": "text/csv",
                    "row_count": len(metrics_df),
                },
            ),
            ArtifactRecord(
                kind="table",
                label=f"{title} design schedule",
                path=str(schedule_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "design_schedule",
                    "media_type": "text/csv",
                    "row_count": len(schedule_df),
                },
            ),
            ArtifactRecord(
                kind="table",
                label=f"{title} design FIM",
                path=str(fim_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "design_fim",
                    "media_type": "text/csv",
                    "row_count": len(fim_df),
                },
            ),
            ArtifactRecord(
                kind="table",
                label=f"{title} design expected SE",
                path=str(se_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "design_expected_se",
                    "media_type": "text/csv",
                    "row_count": len(se_df),
                },
            ),
        ]
