"""Background bootstrap generation for the GUI Advanced workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from openpkpd.inference.bootstrap import BootstrapEngine
from openpkpd.model.parameters import ParameterSet
from openpkpd_gui.app.settings import default_workspace_root_path
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobOutcome, JobStatus
from openpkpd_gui.services.fit_service import FitService


@dataclass(slots=True)
class BootstrapConfig:
    """User-selected bootstrap options from the Advanced workflow."""

    n_boot: int = 100
    seed: int = 42
    n_jobs: int = 1
    ci_level: float = 0.95


@dataclass(slots=True)
class BootstrapRunResult:
    """Minimal bootstrap run result surfaced back to the GUI."""

    summary_text: str
    n_success: int
    artifacts: list[ArtifactRecord] = field(default_factory=list)


class BootstrapService:
    """Create background jobs that compute bootstrap artifacts from a cached fit."""

    def create_job(
        self,
        workspace: Workspace,
        *,
        fit_service: FitService,
        config: BootstrapConfig | None = None,
        run_id: str | None = None,
    ) -> BackgroundJob:
        context = fit_service.latest_fit_context(workspace)
        if context is None:
            raise ValueError(
                "Bootstrap generation requires a reusable successful fit for this scenario."
            )
        config = config or BootstrapConfig()
        title = context.problem_title or workspace.name or "Bootstrap"
        target_project_id = context.project_id
        target_scenario_id = context.scenario_id
        target_dataset_path = context.dataset_path
        target_fit_run_id = context.fit_run_id
        target_method = context.estimation_method
        target_initial_params = self._final_parameter_set(context)

        def _run(ctx) -> BootstrapRunResult:
            ctx.emit(f"Preparing bootstrap for {title}", progress=0.1)
            engine = BootstrapEngine(
                context.population_model,
                target_initial_params,
                estimation_method=target_method,
                n_boot=config.n_boot,
                n_jobs=config.n_jobs,
                seed=config.seed,
                ci_level=config.ci_level,
            )
            ctx.emit(
                f"Running {config.n_boot} bootstrap replicates with {config.n_jobs} job(s)",
                progress=0.45,
            )
            boot_result = engine.run()
            ctx.emit("Writing bootstrap artifacts", progress=0.85)
            artifacts = self._write_artifacts(
                workspace,
                title,
                boot_result,
                context.estimation_result,
                config,
                run_id=run_id,
                fit_run_id=target_fit_run_id,
                estimation_method=target_method,
                project_id=target_project_id,
                scenario_id=target_scenario_id,
                dataset_path=target_dataset_path,
            )
            summary = (
                f"{title} • bootstrap {boot_result.n_success}/{boot_result.n_boot} successful • "
                f"CI {int(config.ci_level * 100)}% • seed={config.seed}"
            )
            return BootstrapRunResult(
                summary_text=summary,
                n_success=boot_result.n_success,
                artifacts=artifacts,
            )

        return BackgroundJob(name=f"bootstrap:{title}", func=_run)

    def apply_job_outcome(self, run: RunRecord, outcome: JobOutcome) -> list[ArtifactRecord]:
        for event in outcome.events:
            run.add_log(f"[{event.kind}] {event.message}")
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, BootstrapRunResult):
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
        run.mark_failed(outcome.error or "Bootstrap generation failed.")
        return []

    @staticmethod
    def latest_run(workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.active_scenario.runs):
            if run.workflow == "bootstrap":
                return run
        return None

    @staticmethod
    def _slug(text: str) -> str:
        safe = "".join(
            character if character.isalnum() else "-" for character in text.strip().lower()
        )
        return safe.strip("-") or "bootstrap"

    @staticmethod
    def _final_parameter_set(context) -> ParameterSet:
        template = getattr(context.population_model, "params", None)
        result = context.estimation_result
        if template is None:
            raise ValueError(
                "Bootstrap generation requires the cached fit context to expose population-model parameters."
            )
        theta_final = getattr(result, "theta_final", None)
        omega_final = getattr(result, "omega_final", None)
        sigma_final = getattr(result, "sigma_final", None)
        if theta_final is None or omega_final is None or sigma_final is None:
            raise ValueError(
                "Bootstrap generation requires final THETA/OMEGA/SIGMA estimates from the cached fit context."
            )
        return ParameterSet(
            theta=np.asarray(theta_final, dtype=float).copy(),
            omega=np.asarray(omega_final, dtype=float).copy(),
            sigma=np.asarray(sigma_final, dtype=float).copy(),
            theta_specs=list(getattr(template, "theta_specs", [])),
            omega_specs=list(getattr(template, "omega_specs", [])),
            sigma_specs=list(getattr(template, "sigma_specs", [])),
        ).apply_bounds()

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
        boot_result,
        original_result,
        config: BootstrapConfig,
        *,
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
        stem = f"{self._slug(title)}-bootstrap-{(run_id or 'run')[:8]}"
        metadata = {
            "estimation_method": estimation_method,
            "n_boot": config.n_boot,
            "n_success": int(boot_result.n_success),
            "n_jobs": config.n_jobs,
            "seed": config.seed,
            "ci_level": config.ci_level,
            "fit_run_id": fit_run_id,
            "bootstrap_run_id": run_id,
        }

        summary_df = boot_result.summary()
        summary_path = artifact_dir / f"{stem}-summary.csv"
        summary_df.to_csv(summary_path, index=False)

        ci_df = boot_result.ci_table(original_result)
        ci_path = artifact_dir / f"{stem}-ci.csv"
        ci_df.to_csv(ci_path, index=False)

        samples_df = self._samples_dataframe(boot_result)
        samples_path = artifact_dir / f"{stem}-samples.csv"
        samples_df.to_csv(samples_path, index=False)

        return [
            ArtifactRecord(
                kind="table",
                label=f"{title} bootstrap summary",
                path=str(summary_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "bootstrap_summary",
                    "media_type": "text/csv",
                    "row_count": len(summary_df),
                },
            ),
            ArtifactRecord(
                kind="table",
                label=f"{title} bootstrap CI table",
                path=str(ci_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "bootstrap_ci_table",
                    "media_type": "text/csv",
                    "row_count": len(ci_df),
                },
            ),
            ArtifactRecord(
                kind="table",
                label=f"{title} bootstrap samples",
                path=str(samples_path),
                source_run_id=fit_run_id,
                metadata={
                    **metadata,
                    "artifact_role": "bootstrap_samples",
                    "media_type": "text/csv",
                    "row_count": len(samples_df),
                },
            ),
        ]

    @staticmethod
    def _samples_dataframe(boot_result) -> pd.DataFrame:
        columns: dict[str, np.ndarray] = {}
        for index in range(boot_result.theta_samples.shape[1]):
            columns[f"THETA({index + 1})"] = boot_result.theta_samples[:, index]
        for index in range(boot_result.omega_diag_samples.shape[1]):
            columns[f"OMEGA({index + 1},{index + 1})"] = boot_result.omega_diag_samples[:, index]
        for index in range(boot_result.sigma_diag_samples.shape[1]):
            columns[f"SIGMA({index + 1},{index + 1})"] = boot_result.sigma_diag_samples[:, index]
        return pd.DataFrame(columns)
