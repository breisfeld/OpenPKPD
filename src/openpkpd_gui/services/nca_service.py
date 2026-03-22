"""Prepare and run standalone NCA jobs from the selected workspace trial."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.nca import NCAEngine
from openpkpd.plots.nca import nca_boxplot, nca_distributions
from openpkpd.utils.errors import DataError
from openpkpd_gui.app.settings import default_workspace_root_path
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobOutcome, JobStatus
from openpkpd_gui.services.data_service import DatasetService
from openpkpd_gui.services.validation_service import ValidationResult


@dataclass(slots=True)
class NCAConfig:
    """User-selected NCA options from the workflow controls."""

    route: str = "oral"
    auc_method: str = "linear-log"
    min_points_lambda: int = 3
    exclude_cmax: bool = True


@dataclass(slots=True)
class NCAPreparationResult:
    """Normalized readiness state for launching NCA."""

    validation: ValidationResult = field(default_factory=ValidationResult)
    dataset_path: str | None = None
    subject_count: int = 0
    observation_count: int = 0
    row_count: int = 0

    @property
    def ready(self) -> bool:
        return self.validation.ok and bool(self.dataset_path)


@dataclass(slots=True)
class NCARunResult:
    """Minimal NCA run result surfaced back to the GUI."""

    summary_text: str
    subject_count: int
    artifacts: list[ArtifactRecord] = field(default_factory=list)


class NCAService:
    """Prepare NCA requests and create background jobs for them."""

    def prepare_run(self, workspace: Workspace) -> NCAPreparationResult:
        result = NCAPreparationResult()
        dataset_asset = workspace.active_scenario.active_dataset
        if dataset_asset is None or not dataset_asset.source_path:
            result.validation.add_error(
                "Load and save a dataset in the Data workflow before starting NCA.",
                field_name="active_dataset",
            )
            return result
        try:
            dataset = self._load_dataset(dataset_asset)
        except (DataError, OSError) as exc:
            result.validation.add_error(str(exc), field_name="active_dataset")
            return result
        result.dataset_path = dataset.source_path
        result.subject_count = dataset.n_subjects()
        result.observation_count = dataset.n_observations()
        result.row_count = len(dataset.df)
        if result.observation_count == 0:
            result.validation.add_error(
                "The active dataset has no observation rows available for NCA.",
                field_name="active_dataset",
            )
        return result

    def create_job(
        self,
        workspace: Workspace,
        config: NCAConfig | None = None,
        preparation: NCAPreparationResult | None = None,
        run_id: str | None = None,
    ) -> BackgroundJob:
        config = config or NCAConfig()
        preparation = preparation or self.prepare_run(workspace)
        if not preparation.ready:
            raise ValueError("NCA is not ready. Resolve validation issues before starting a run.")
        title = Path(preparation.dataset_path or "dataset").stem or workspace.name or "NCA"
        target_project_id = workspace.active_project.project_id
        target_scenario_id = workspace.active_scenario.scenario_id
        dataset_asset = (
            DatasetAsset.from_dict(workspace.active_scenario.active_dataset.to_dict())
            if workspace.active_scenario.active_dataset is not None
            else None
        )

        def _run(ctx) -> NCARunResult:
            ctx.emit(f"Preparing NCA for {title}", progress=0.05)
            dataset = self._load_dataset(dataset_asset)
            ctx.emit("Running NCA engine", progress=0.40)
            engine = NCAEngine(
                auc_method=config.auc_method,
                min_points_lambda=config.min_points_lambda,
                exclude_cmax=config.exclude_cmax,
            )
            results_df = engine.compute_dataset(dataset.df, route=config.route)
            ctx.emit("Writing NCA artifacts", progress=0.70)
            artifacts = self._write_artifacts(
                workspace,
                title,
                results_df,
                config,
                run_id,
                project_id=target_project_id,
                scenario_id=target_scenario_id,
                dataset_path=preparation.dataset_path,
            )
            summary_text = (
                f"{len(results_df)} subjects • route {config.route} • "
                f"AUC {config.auc_method} • min λz points {config.min_points_lambda}"
            )
            return NCARunResult(
                summary_text=summary_text,
                subject_count=len(results_df),
                artifacts=artifacts,
            )

        return BackgroundJob(name=f"nca:{title}", func=_run)

    def apply_job_outcome(self, run: RunRecord, outcome: JobOutcome) -> list[ArtifactRecord]:
        for event in outcome.events:
            run.add_log(f"[{event.kind}] {event.message}")
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, NCARunResult):
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
        run.mark_failed(outcome.error or "NCA failed.")
        return []

    @staticmethod
    def latest_run(workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.active_scenario.runs):
            if run.workflow == "nca":
                return run
        return None

    @staticmethod
    def _load_dataset(dataset_asset) -> NONMEMDataset:
        if dataset_asset is None or not dataset_asset.source_path:
            raise DataError("Active dataset path is unavailable.")
        options = DatasetService.options_from_asset(dataset_asset)
        kwargs: dict[str, object] = {}
        if getattr(dataset_asset, "input_columns", None):
            kwargs["input_columns"] = list(dataset_asset.input_columns)
        return NONMEMDataset.from_csv(
            dataset_asset.source_path,
            ignore_char=options.normalized_ignore_char,
            sep=options.effective_separator,
            **kwargs,
        )

    @staticmethod
    def _slug(text: str) -> str:
        safe = "".join(
            character if character.isalnum() else "-" for character in text.strip().lower()
        )
        return safe.strip("-") or "nca"

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
        results_df,
        config: NCAConfig,
        run_id: str | None,
        *,
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
        stem = f"{self._slug(title)}-nca-{(run_id or 'run')[:8]}"
        metadata = {
            "route": config.route,
            "auc_method": config.auc_method,
            "min_points_lambda": config.min_points_lambda,
            "exclude_cmax": config.exclude_cmax,
            "subject_count": len(results_df),
        }

        # CSV summary
        csv_path = artifact_dir / f"{stem}-summary.csv"
        results_df.to_csv(csv_path, index=False)
        artifacts: list[ArtifactRecord] = [
            ArtifactRecord(
                kind="table",
                label=f"{title} NCA summary",
                path=str(csv_path),
                source_run_id=run_id,
                metadata={**metadata, "media_type": "text/csv", "artifact_role": "nca_summary"},
            )
        ]

        # Plots — skip silently if matplotlib is unavailable or data is insufficient
        def _save_plot(fig, suffix: str, label: str, plot_type: str) -> None:
            plot_path = artifact_dir / f"{stem}-{suffix}.png"
            try:
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            finally:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:
                    pass
            artifacts.append(
                ArtifactRecord(
                    kind="plot",
                    label=label,
                    path=str(plot_path),
                    source_run_id=run_id,
                    metadata={
                        **metadata,
                        "media_type": "image/png",
                        "artifact_role": "plot",
                        "plot_type": plot_type,
                    },
                )
            )

        if len(results_df) >= 2:
            with contextlib.suppress(Exception):
                _save_plot(
                    nca_distributions(results_df, title=f"{title} — NCA parameter distributions"),
                    suffix="distributions",
                    label=f"{title} NCA distributions",
                    plot_type="nca_distributions",
                )

            with contextlib.suppress(Exception):
                _save_plot(
                    nca_boxplot(results_df, title=f"{title} — NCA parameters"),
                    suffix="boxplot",
                    label=f"{title} NCA boxplot",
                    plot_type="nca_boxplot",
                )

        return artifacts
