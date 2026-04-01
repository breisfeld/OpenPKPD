"""Prepare and run SCM (Stepwise Covariate Modeling) jobs from the workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openpkpd.data.dataset import NONMEMDataset
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.run_record import RunRecord
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobOutcome, JobStatus
from openpkpd_gui.services.model_translation_service import ModelTranslationService
from openpkpd_gui.services.validation_service import ValidationResult


@dataclass(slots=True)
class SCMCandidate:
    """One covariate–parameter relationship candidate."""

    parameter: str
    covariate: str
    effect: str  # "power", "linear", "exp", "categorical"
    reference: float = 70.0


@dataclass(slots=True)
class SCMPreparationResult:
    """Readiness state for launching an SCM run."""

    validation: ValidationResult = field(default_factory=ValidationResult)
    problem_title: str = ""
    has_builder: bool = False

    @property
    def ready(self) -> bool:
        return self.validation.ok and self.has_builder


@dataclass(slots=True)
class SCMRunResult:
    """Minimal SCM result surfaced back to the GUI."""

    summary_text: str
    step_rows: list[dict]  # list of {type, rel, delta_ofv, p_value, accepted}
    accepted_count: int
    final_ofv: float
    base_ofv: float


def generate_scm_step_plot(
    scm_result: SCMRunResult,
    *,
    run_id: str,
    output_dir: Path,
) -> ArtifactRecord | None:
    """Generate a ΔOFV significance chart for the SCM steps and return an ArtifactRecord.

    Shows −ΔOFV for each accepted forward step, sorted by effect size.
    Returns ``None`` if matplotlib is unavailable or no steps were accepted.
    """
    accepted_rows = [r for r in scm_result.step_rows if r.get("accepted")]
    if not accepted_rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        labels = [str(r.get("rel", "")) for r in accepted_rows]
        neg_delta_ofv = [-float(r.get("delta_ofv", 0.0)) for r in accepted_rows]
        # Sort by effect size (largest first, top-to-bottom)
        pairs = sorted(zip(neg_delta_ofv, labels), reverse=True)
        neg_delta_ofv_sorted = [v for v, _ in pairs]
        labels_sorted = [lbl for _, lbl in pairs]

        fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(labels_sorted) + 1.5)))
        colors = ["#2563eb" if v >= 3.84 else "#94a3b8" for v in neg_delta_ofv_sorted]
        y_pos = list(range(len(labels_sorted)))
        ax.barh(y_pos, neg_delta_ofv_sorted, color=colors, edgecolor="none", height=0.6)
        ax.axvline(3.84, color="#ef4444", linestyle="--", linewidth=1.0, label="p<0.05 (χ²=3.84)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_sorted, fontsize=9)
        ax.set_xlabel("−ΔOFV (larger = stronger covariate effect)", fontsize=9)
        ax.set_title("SCM: Accepted Covariate Steps", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"scm-step-significance-{run_id[:8]}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return ArtifactRecord(
            kind="plot",
            label="SCM: Accepted Covariate Steps",
            path=str(plot_path),
            source_run_id=run_id,
            metadata={
                "media_type": "image/png",
                "artifact_role": "scm_step_plot",
                "plot_type": "scm",
            },
        )
    except Exception:
        return None


class SCMService:
    """Prepare SCM requests and create background jobs for them."""

    def __init__(self, translation_service: ModelTranslationService | None = None) -> None:
        self._translation_service = translation_service or ModelTranslationService()

    def prepare(self, workspace: Workspace) -> SCMPreparationResult:
        result = SCMPreparationResult()
        scenario = workspace.active_scenario
        model_spec = scenario.active_model_spec
        dataset = scenario.active_dataset

        if model_spec is None:
            result.validation.add_error(
                "Configure a model in the Model workflow before running SCM.",
                field_name="active_model_spec",
            )
            return result

        if dataset is None:
            result.validation.add_error(
                "Load a dataset in the Data workflow before running SCM.",
                field_name="active_dataset",
            )
            return result

        translation = self._translation_service.translate(model_spec)
        result.problem_title = translation.problem_title or workspace.name
        result.has_builder = translation.builder is not None
        result.validation.issues.extend(translation.validation.issues)
        if translation.builder is not None:
            dataset_error = self._apply_dataset_asset_to_builder(translation.builder, dataset)
            if dataset_error is not None:
                result.validation.add_error(
                    dataset_error,
                    field_name="active_dataset",
                    target_workflow="data",
                    target_widget="data-source-path",
                )

        if not result.has_builder:
            result.validation.add_error(
                "SCM requires a builder-mode model. Switch to Model Builder mode.",
                field_name="mode",
            )
        return result

    @staticmethod
    def _dataset_load_kwargs(dataset_asset: DatasetAsset) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "sep": r"\s+" if dataset_asset.treat_as_whitespace else dataset_asset.separator,
        }
        if dataset_asset.ignore_char:
            kwargs["ignore_char"] = dataset_asset.ignore_char
        if dataset_asset.input_columns:
            kwargs["input_columns"] = list(dataset_asset.input_columns)
        return kwargs

    @classmethod
    def _apply_dataset_asset_to_builder(
        cls, builder: object, dataset_asset: DatasetAsset | None
    ) -> str | None:
        if dataset_asset is None or not dataset_asset.source_path:
            return None
        build_dataset = getattr(builder, "dataset", None)
        if not callable(build_dataset):
            return None
        try:
            dataset = NONMEMDataset.from_csv(
                dataset_asset.source_path,
                **cls._dataset_load_kwargs(dataset_asset),
            )
        except Exception as exc:
            return str(exc)
        build_dataset(dataset)
        return None

    def create_job(
        self,
        workspace: Workspace,
        candidates: list[SCMCandidate],
        forward_pvalue: float = 0.05,
        backward_pvalue: float = 0.001,
        n_jobs: int = -1,
        preparation: SCMPreparationResult | None = None,
    ) -> BackgroundJob:
        preparation = preparation or self.prepare(workspace)
        if not preparation.ready:
            raise ValueError("SCM is not ready. Resolve validation issues first.")

        scenario = workspace.active_scenario
        model_spec = scenario.active_model_spec
        title = preparation.problem_title or workspace.name or "Untitled SCM"

        def _run(ctx) -> SCMRunResult:
            from openpkpd.covariate.effects import CovariateEffect, CovariateRelationship
            from openpkpd.covariate.scm import SCMEngine

            ctx.emit("Building base model", progress=0.05)
            translation = self._translation_service.translate(model_spec)
            builder = translation.builder
            if builder is None:
                raise RuntimeError("Builder translation unavailable.")
            dataset_error = self._apply_dataset_asset_to_builder(builder, scenario.active_dataset)
            if dataset_error is not None:
                raise RuntimeError(dataset_error)

            base_pk_code = getattr(builder, "_pk_code", "") or ""

            relationships = []
            for c in candidates:
                try:
                    effect = CovariateEffect(c.effect)
                except ValueError:
                    effect = CovariateEffect.POWER
                relationships.append(
                    CovariateRelationship(
                        parameter=c.parameter,
                        covariate=c.covariate,
                        effect=effect,
                        reference=c.reference,
                    )
                )

            ctx.emit(f"Running SCM with {len(relationships)} candidates", progress=0.1)
            engine = SCMEngine(
                base_model_builder=builder,
                base_pk_code=base_pk_code,
                candidates=relationships,
                forward_pvalue=forward_pvalue,
                backward_pvalue=backward_pvalue,
                n_jobs=n_jobs,
            )
            scm_result = engine.run()

            step_rows = []
            for step in scm_result.steps:
                step_rows.append(
                    {
                        "type": step.step_type,
                        "rel": f"{step.relationship.parameter}~{step.relationship.covariate}"
                        f"({step.relationship.effect.value})",
                        "delta_ofv": step.delta_ofv,
                        "p_value": step.p_value,
                        "accepted": step.accepted,
                    }
                )

            accepted = [
                f"{r.parameter}~{r.covariate}({r.effect.value})"
                for r in scm_result.accepted_relationships
            ]
            summary_text = (
                f"{title} • {len(accepted)} accepted"
                f" • base OFV={scm_result.base_ofv:.2f}"
                f" • final OFV={scm_result.final_ofv:.2f}"
            )
            ctx.emit(summary_text, progress=1.0)
            return SCMRunResult(
                summary_text=summary_text,
                step_rows=step_rows,
                accepted_count=len(accepted),
                final_ofv=scm_result.final_ofv,
                base_ofv=scm_result.base_ofv,
            )

        return BackgroundJob(name=f"scm:{title}", func=_run)

    def apply_job_outcome(self, run: RunRecord, outcome: JobOutcome) -> SCMRunResult | None:
        for event in outcome.events:
            run.add_log(f"[{event.kind}] {event.message}")
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, SCMRunResult):
            run.mark_succeeded(outcome.value.summary_text)
            return outcome.value
        if outcome.status == JobStatus.CANCELLED:
            run.mark_cancelled(outcome.error or "Cancelled by user.")
            return None
        run.mark_failed(outcome.error or "SCM failed.")
        return None

    @staticmethod
    def latest_run(workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.active_scenario.runs):
            if run.workflow == "covariate":
                return run
        return None
