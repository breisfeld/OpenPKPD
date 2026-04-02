"""Prepare and run fit jobs from the selected workspace trial."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import platform
import subprocess
import sys
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from openpkpd import __version__ as OPENPKPD_VERSION
from openpkpd.covariance.sandwich import SandwichCovariance
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation import get_estimation_method
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import ParameterSet
from openpkpd.model.problem import Problem
from openpkpd.output.report import write_html_report
from openpkpd.parser.control_stream import ControlStream
from openpkpd.plots.diagnostics import compute_diagnostics
from openpkpd.plots.eta import eta_histograms, eta_pairs
from openpkpd.plots.gof import (
    abs_iwres_vs_ipred,
    cwres_qq,
    cwres_vs_pred,
    cwres_vs_time,
    diagnostic_panel,
    dv_vs_ipred,
    dv_vs_pred,
)
from openpkpd.plots.model_perf import ofv_history, parameter_uncertainty_plot, residual_trends_plot
from openpkpd.plots.pk import mean_profile, spaghetti_plot
from openpkpd.utils.constants import Method
from openpkpd_gui.app.settings import default_workspace_root_path
from openpkpd_gui.domain.artifact import ArtifactRecord
from openpkpd_gui.domain.dataset_asset import DatasetAsset
from openpkpd_gui.domain.model_spec import ModelSpecMode
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.jobs.base import BackgroundJob, JobOutcome, JobStatus
from openpkpd_gui.services.model_translation_service import (
    ModelTranslationResult,
    ModelTranslationService,
)
from openpkpd_gui.services.validation_service import ValidationResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FitPreparationResult:
    """Normalized readiness state for launching a fit."""

    validation: ValidationResult = field(default_factory=ValidationResult)
    translation: ModelTranslationResult | None = None
    problem_title: str = ""
    dataset_path: str | None = None
    mode: ModelSpecMode | None = None
    estimation_method: str | None = None
    covariance_enabled: bool = False
    theta_count: int = 0
    eta_count: int = 0
    eps_count: int = 0

    @property
    def ready(self) -> bool:
        return self.validation.ok and self.translation is not None and self.translation.ok


@dataclass(slots=True)
class FitRunResult:
    """Minimal run result surfaced back to the GUI."""

    problem_title: str
    estimation_method: str
    converged: bool
    ofv: float
    summary_text: str
    warning_messages: list[str] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)


@dataclass(slots=True)
class FitContext:
    """In-memory fit state needed by on-demand post-fit computations."""

    workspace_id: str
    project_id: str
    scenario_id: str
    scenario_ref: object
    fit_run_id: str
    problem_title: str
    estimation_method: str
    dataset_path: str | None
    estimation_result: EstimationResult
    population_model: object


class FitService:
    """Prepare fit requests and create background jobs for them."""

    def __init__(self, translation_service: ModelTranslationService | None = None) -> None:
        self._translation_service = translation_service or ModelTranslationService()
        self._fit_contexts: dict[tuple[str, str, str], FitContext] = {}
        self._fit_contexts_lock = threading.Lock()
        self._last_context_error: str | None = None
        self._last_restore_warnings: list[str] = []

    def prepare_run(self, workspace: Workspace) -> FitPreparationResult:
        result = FitPreparationResult()
        scenario = workspace.active_scenario
        dataset = scenario.active_dataset
        model_spec = scenario.active_model_spec

        if model_spec is None:
            if dataset is None:
                result.validation.add_error(
                    "Load a dataset in the Data workflow before starting a fit.",
                    field_name="active_dataset",
                    target_workflow="data",
                    target_widget="data-source-path",
                )
            result.validation.add_error(
                "Configure a model in the Model workflow before starting a fit.",
                field_name="active_model_spec",
                target_workflow="model",
                target_widget="model-problem-title",
            )
            return result

        translation = self._translation_service.translate(model_spec)
        if translation.mode == ModelSpecMode.BUILDER and translation.builder is not None:
            dataset_error = self._apply_dataset_asset_to_builder(translation.builder, dataset)
            if dataset_error is not None:
                result.validation.add_error(
                    dataset_error,
                    field_name="active_dataset",
                    target_workflow="data",
                    target_widget="data-source-path",
                )
        result.translation = translation
        result.problem_title = translation.problem_title or workspace.name
        # Data-screen selection overrides $DATA in control-stream mode; fall back
        # to $DATA when no dataset has been loaded on the Data screen.
        result.dataset_path = (dataset.source_path if dataset else None) or translation.dataset_path
        result.mode = model_spec.mode
        result.estimation_method = translation.estimation_method or model_spec.estimation.method
        result.covariance_enabled = translation.covariance_enabled
        result.theta_count = translation.theta_count
        result.eta_count = translation.eta_count
        result.eps_count = translation.eps_count
        result.validation.issues.extend(translation.validation.issues)
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
        # Detect tab-delimited files when the user has not opted into whitespace mode
        # and the separator is still the default comma.
        if not dataset_asset.treat_as_whitespace and dataset_asset.separator == ",":
            source = dataset_asset.source_path
            if source:
                try:
                    with open(source, encoding="utf-8", errors="replace") as _fh:
                        first_line = _fh.readline()
                    if "\t" in first_line and "," not in first_line:
                        warnings.warn(
                            f"The dataset file {source!r} appears to be tab-delimited "
                            "(first line contains tabs but no commas) but the separator "
                            "is set to comma. Consider enabling 'treat_as_whitespace' or "
                            "setting the separator to tab ('\\t').",
                            stacklevel=2,
                        )
                except OSError:
                    pass
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
        # Inject scalar LOQ as LLOQ column when user has set a GUI-level LOQ
        scalar_loq = getattr(dataset_asset, "loq", None)
        if scalar_loq is not None and float(scalar_loq) > 0.0:
            if dataset.has_lloq:
                lloq_series = dataset.df["LLOQ"]
                if lloq_series.nunique() > 1:
                    warnings.warn(
                        "The dataset already contains a non-uniform LLOQ column. "
                        "The GUI-specified scalar LOQ will not overwrite it.",
                        stacklevel=2,
                    )
                # Do not overwrite any existing LLOQ column (uniform or not).
            else:
                dataset.df["LLOQ"] = float(scalar_loq)
        build_dataset(dataset)
        return None

    def create_job(
        self,
        workspace: Workspace,
        preparation: FitPreparationResult | None = None,
        run_id: str | None = None,
        n_parallel: int = 0,
    ) -> BackgroundJob:
        preparation = preparation or self.prepare_run(workspace)
        if not preparation.ready or preparation.translation is None:
            raise ValueError("Fit is not ready. Resolve validation issues before starting a run.")

        translation = preparation.translation
        title = preparation.problem_title or workspace.name or "Untitled fit"
        target_project_id = workspace.active_project.project_id
        target_scenario_id = workspace.active_scenario.scenario_id
        target_dataset_path = preparation.dataset_path

        def _run(ctx) -> FitRunResult:
            ctx.emit(f"Preparing fit for {title}", progress=0.05)
            ctx.check_cancelled()
            if translation.mode == ModelSpecMode.BUILDER:
                return self._run_builder_fit(
                    ctx,
                    workspace,
                    translation,
                    title,
                    run_id,
                    project_id=target_project_id,
                    scenario_id=target_scenario_id,
                    dataset_path=target_dataset_path,
                    n_parallel=n_parallel,
                )
            return self._run_control_stream_fit(
                ctx,
                workspace,
                translation,
                title,
                run_id,
                project_id=target_project_id,
                scenario_id=target_scenario_id,
                dataset_path=target_dataset_path,
                n_parallel=n_parallel,
            )

        return BackgroundJob(name=f"fit:{title}", func=_run)

    def apply_job_outcome(self, run: RunRecord, outcome: JobOutcome) -> list[ArtifactRecord]:
        for event in outcome.events:
            run.add_log(f"[{event.kind}] {event.message}")
        if outcome.status == JobStatus.SUCCEEDED and isinstance(outcome.value, FitRunResult):
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
        error_text = outcome.error or "Fit failed."
        run.mark_failed(error_text)
        return []

    @staticmethod
    def latest_run(workspace: Workspace) -> RunRecord | None:
        for run in reversed(workspace.active_scenario.runs):
            if run.workflow == "fit":
                return run
        return None

    def latest_fit_context(self, workspace: Workspace) -> FitContext | None:
        with self._fit_contexts_lock:
            context = self._fit_contexts.get(self._context_key(workspace))
        if context is None or context.scenario_ref is not workspace.active_scenario:
            return None
        latest_run = self.latest_run(workspace)
        if latest_run is None or latest_run.run_id != context.fit_run_id:
            return None
        if latest_run.status != RunStatus.SUCCEEDED:
            return None
        return context

    @property
    def last_context_error(self) -> str | None:
        return self._last_context_error

    @property
    def last_restore_warnings(self) -> list[str]:
        return list(self._last_restore_warnings)

    @staticmethod
    def _context_key(workspace: Workspace) -> tuple[str, str, str]:
        return (
            workspace.workspace_id,
            workspace.active_project.project_id,
            workspace.active_scenario.scenario_id,
        )

    # ------------------------------------------------------------------
    # Snapshot serialization helpers
    # ------------------------------------------------------------------

    def all_fit_context_payloads(self, workspace: Workspace) -> dict[tuple[str, str], bytes]:
        """Return serialized fit contexts for all valid scenarios, keyed by (project_id, scenario_id)."""
        result: dict[tuple[str, str], bytes] = {}
        with self._fit_contexts_lock:
            contexts_snapshot = list(self._fit_contexts.items())
        for (ws_id, proj_id, scen_id), context in contexts_snapshot:
            if ws_id != workspace.workspace_id:
                continue
            match = workspace.find_scenario(scen_id, project_id=proj_id)
            if match is None:
                continue
            _project, scenario = match
            if context.scenario_ref is not scenario:
                continue
            latest_run = next((r for r in reversed(scenario.runs) if r.workflow == "fit"), None)
            if (
                latest_run is None
                or latest_run.run_id != context.fit_run_id
                or latest_run.status != RunStatus.SUCCEEDED
            ):
                continue
            payload = self._context_to_dict(context)
            if payload is not None:
                result[(proj_id, scen_id)] = json.dumps(payload, ensure_ascii=False).encode()
        return result

    def restore_fit_context(
        self,
        workspace: Workspace,
        payload: Mapping[str, object],
    ) -> bool:
        """Restore a fit context from a deserialized JSON payload. Returns True on success."""
        self._last_context_error = None
        try:
            er_data = payload["estimation_result"]
            estimation_result = EstimationResult(
                theta_final=np.array(er_data["theta_final"], dtype=float),
                omega_final=np.array(er_data["omega_final"], dtype=float),
                sigma_final=np.array(er_data["sigma_final"], dtype=float),
                ofv=float(er_data["ofv"]),
                converged=bool(er_data["converged"]),
                condition_number=er_data.get("condition_number"),
                eta_shrinkage=np.array(er_data.get("eta_shrinkage", []), dtype=float),
                eps_shrinkage=np.array(er_data.get("eps_shrinkage", []), dtype=float),
                post_hoc_etas={
                    int(k): np.array(v, dtype=float)
                    for k, v in er_data.get("post_hoc_etas", {}).items()
                },
                ofv_history=list(er_data.get("ofv_history", [])),
                warnings=list(er_data.get("warnings", [])),
                n_function_evals=int(er_data.get("n_function_evals", 0)),
                elapsed_time=float(er_data.get("elapsed_time", 0.0)),
                method=str(er_data.get("method", "")),
                message=str(er_data.get("message", "")),
                n_observations=int(er_data.get("n_observations", 0)),
                n_subjects=int(er_data.get("n_subjects", 0)),
            )
            proj_id = str(payload["project_id"])
            scen_id = str(payload["scenario_id"])
            match = workspace.find_scenario(scen_id, project_id=proj_id)
            if match is None:
                self._last_context_error = (
                    f"Could not restore fit context: scenario {scen_id!r} in project {proj_id!r} "
                    "was not found."
                )
                return False
            _project, scenario = match
            population_model = self._rebuild_population_model(scenario, payload.get("dataset_path"))
            if population_model is None:
                if self._last_context_error is None:
                    self._last_context_error = "Could not restore fit context: model rebuild failed."
                return False
            new_context = FitContext(
                workspace_id=workspace.workspace_id,
                project_id=proj_id,
                scenario_id=scen_id,
                scenario_ref=scenario,
                fit_run_id=str(payload["fit_run_id"]),
                problem_title=str(payload["problem_title"]),
                estimation_method=str(payload["estimation_method"]),
                dataset_path=payload.get("dataset_path"),
                estimation_result=estimation_result,
                population_model=population_model,
            )
            with self._fit_contexts_lock:
                self._fit_contexts[(workspace.workspace_id, proj_id, scen_id)] = new_context
            return True
        except Exception as exc:
            self._last_context_error = f"Could not restore fit context: {exc}"
            logger.warning("%s", self._last_context_error, exc_info=True)
            return False

    def restore_fit_context_payloads(
        self,
        workspace: Workspace,
        payloads: Mapping[tuple[str, str], bytes],
    ) -> tuple[int, list[str]]:
        """Restore multiple serialized fit contexts and return (restored_count, warnings)."""
        restored = 0
        warnings: list[str] = []
        self._last_restore_warnings = []
        for (proj_id, scen_id), payload_bytes in payloads.items():
            try:
                payload = json.loads(payload_bytes)
            except Exception as exc:
                warning = (
                    f"Could not decode saved fit state for project {proj_id!r}, "
                    f"scenario {scen_id!r}: {exc}"
                )
                warnings.append(warning)
                logger.warning("%s", warning, exc_info=True)
                continue
            if self.restore_fit_context(workspace, payload):
                restored += 1
                continue
            warning = self._last_context_error or (
                f"Could not restore fit state for project {proj_id!r}, scenario {scen_id!r}."
            )
            warnings.append(warning)
        self._last_restore_warnings = list(warnings)
        return restored, warnings

    def _rebuild_population_model(
        self,
        scenario: object,
        dataset_path: object,
    ) -> object | None:
        """Rebuild a PopulationModel from the saved model spec (no estimation)."""
        model_spec = getattr(scenario, "active_model_spec", None)
        if model_spec is None:
            self._last_context_error = "Could not rebuild fit context: scenario has no active model."
            return None
        try:
            translation = self._translation_service.translate(model_spec)
            if not translation.ok:
                messages = [issue.message for issue in translation.validation.issues]
                detail = "; ".join(messages) if messages else "model translation failed"
                self._last_context_error = f"Could not rebuild fit context: {detail}"
                return None
            if translation.mode == ModelSpecMode.BUILDER and translation.builder is not None:
                self._apply_dataset_asset_to_builder(
                    translation.builder,
                    getattr(scenario, "active_dataset", None),
                )
                built = translation.builder.build()
                return getattr(built, "population_model", None)
            if translation.control_stream is not None:
                effective_path = str(dataset_path) if dataset_path else translation.dataset_path
                problem = Problem.from_control_stream(
                    translation.control_stream, dataset_path=effective_path
                )
                return problem.population_model
        except Exception as exc:
            self._last_context_error = f"Could not rebuild fit context: {exc}"
            logger.warning("%s", self._last_context_error, exc_info=True)
        return None

    @staticmethod
    def _context_to_dict(context: FitContext) -> dict[str, object] | None:
        """Serialize a FitContext to a JSON-serializable dict."""
        try:
            result = context.estimation_result
            return {
                "format_version": 1,
                "workspace_id": context.workspace_id,
                "project_id": context.project_id,
                "scenario_id": context.scenario_id,
                "fit_run_id": context.fit_run_id,
                "problem_title": context.problem_title,
                "estimation_method": context.estimation_method,
                "dataset_path": context.dataset_path,
                "estimation_result": {
                    "theta_final": result.theta_final.tolist(),
                    "omega_final": result.omega_final.tolist(),
                    "sigma_final": result.sigma_final.tolist(),
                    "ofv": float(result.ofv),
                    "converged": bool(result.converged),
                    "condition_number": result.condition_number,
                    "eta_shrinkage": result.eta_shrinkage.tolist(),
                    "eps_shrinkage": result.eps_shrinkage.tolist(),
                    "post_hoc_etas": {str(k): v.tolist() for k, v in result.post_hoc_etas.items()},
                    "ofv_history": list(result.ofv_history),
                    "warnings": list(result.warnings),
                    "n_function_evals": int(result.n_function_evals),
                    "elapsed_time": float(result.elapsed_time),
                    "method": str(result.method),
                    "message": str(result.message),
                    "n_observations": int(result.n_observations),
                    "n_subjects": int(result.n_subjects),
                },
            }
        except Exception:
            return None

    def _cache_fit_context(
        self,
        workspace: Workspace,
        *,
        title: str,
        method: str,
        estimation_result: EstimationResult,
        population_model: object,
        fit_run_id: str,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
    ) -> None:
        new_context = FitContext(
            workspace_id=workspace.workspace_id,
            project_id=project_id,
            scenario_id=scenario_id,
            scenario_ref=workspace.active_scenario,
            fit_run_id=fit_run_id,
            problem_title=title,
            estimation_method=method,
            dataset_path=dataset_path,
            estimation_result=estimation_result,
            population_model=population_model,
        )
        with self._fit_contexts_lock:
            self._fit_contexts[(workspace.workspace_id, project_id, scenario_id)] = new_context

    def _run_builder_fit(
        self,
        ctx,
        workspace: Workspace,
        translation: ModelTranslationResult,
        title: str,
        run_id: str | None,
        *,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
        n_parallel: int = 0,
    ) -> FitRunResult:
        if translation.builder is None:
            raise ValueError("Builder translation is unavailable for this fit.")
        ctx.check_cancelled()
        ctx.emit("Building model", progress=0.18)
        built_model = translation.builder.build()
        ctx.check_cancelled()
        ctx.emit(f"Running {translation.estimation_method or 'FOCE'} estimation", progress=0.48)
        estimation_result = self._estimate_built_model(built_model, n_parallel=n_parallel, ctx=ctx)
        ctx.check_cancelled()
        ctx.emit("Collecting fitted parameters", progress=0.68)
        params = self._final_parameter_set(getattr(built_model, "params", None), estimation_result)
        return self._fit_run_result_from_estimation(
            ctx,
            workspace,
            title,
            estimation_result,
            translation.estimation_method,
            params,
            run_id,
            project_id=project_id,
            scenario_id=scenario_id,
            dataset_path=dataset_path,
            population_model=getattr(built_model, "population_model", None),
            translation=translation,
        )

    def _estimate_built_model(
        self, built_model: object, *, n_parallel: int = 0, ctx=None
    ) -> EstimationResult:
        estimation_kwargs = dict(getattr(built_model, "estimation_kwargs", {}))
        method_name = estimation_kwargs.pop("method", Method.FOCE)
        interaction = bool(estimation_kwargs.pop("interaction", False))
        maxeval = int(estimation_kwargs.pop("maxeval", 9999))
        blq_method = str(estimation_kwargs.pop("blq_method", "M1"))

        if ctx is not None:
            def _iteration_callback(iteration: int, ofv: float) -> None:
                ctx.emit(
                    f"{iteration},{ofv:.6f}",
                    kind="ofv",
                )
            estimation_kwargs["iteration_callback"] = _iteration_callback

        estimation = get_estimation_method(
            str(method_name),
            interaction=interaction,
            maxeval=maxeval,
            n_parallel=n_parallel,
            **estimation_kwargs,
        )
        population_model = built_model.population_model
        params = built_model.params
        # Apply BLQ method to population model (propagates to individual models at fit time)
        if hasattr(population_model, "blq_method"):
            population_model.blq_method = blq_method
        result = estimation.estimate(population_model, params)

        dataset = getattr(population_model, "dataset", None)
        if dataset is not None and hasattr(dataset, "n_observations"):
            result.n_observations = int(dataset.n_observations())
        result.n_subjects = population_model.n_subjects()
        result.compute_n_parameters(
            theta_specs=params.theta_specs,
            omega_specs=params.omega_specs,
            sigma_specs=params.sigma_specs,
        )

        if getattr(built_model, "do_covariance", False):
            cov_est = SandwichCovariance(**getattr(built_model, "covariance_kwargs", {}))
            final_params = ParameterSet(
                theta=result.theta_final,
                omega=result.omega_final,
                sigma=result.sigma_final,
                theta_specs=params.theta_specs,
                omega_specs=params.omega_specs,
                sigma_specs=params.sigma_specs,
            )
            cov_result = cov_est.compute(population_model, final_params, result.post_hoc_etas)
            result.warnings.extend(cov_result.warnings)

        return result

    def _run_control_stream_fit(
        self,
        ctx,
        workspace: Workspace,
        translation: ModelTranslationResult,
        title: str,
        run_id: str | None,
        *,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
        n_parallel: int = 0,
    ) -> FitRunResult:
        if translation.control_stream is None:
            raise ValueError("Control-stream translation is unavailable for this fit.")
        ctx.check_cancelled()
        ctx.emit("Assembling control-stream problem", progress=0.2)
        final_result, params, population_model = self._estimate_control_stream(
            ctx,
            translation.control_stream,
            dataset_path or translation.dataset_path,
            n_parallel=n_parallel,
        )
        method = translation.estimation_method or final_result.method or Method.FOCE
        return self._fit_run_result_from_estimation(
            ctx,
            workspace,
            title,
            final_result,
            method,
            params,
            run_id,
            project_id=project_id,
            scenario_id=scenario_id,
            dataset_path=dataset_path,
            population_model=population_model,
            translation=translation,
        )

    def _estimate_control_stream(
        self,
        ctx,
        control_stream: ControlStream,
        dataset_path: str | None,
        n_parallel: int = 0,
    ) -> tuple[EstimationResult, ParameterSet, object]:
        ctx.check_cancelled()
        problem = Problem.from_control_stream(control_stream, dataset_path=dataset_path)
        pop_model = problem.population_model
        params = pop_model.params
        estimation_records = control_stream.estimation_records or [None]
        final_result: EstimationResult | None = None

        for index, estimation_record in enumerate(estimation_records, start=1):
            ctx.check_cancelled()
            if estimation_record is None:
                method_name = Method.FOCE
                interaction = False
                maxeval = 9999
            else:
                method_name = str(estimation_record.method)
                interaction = estimation_record.interaction
                maxeval = estimation_record.maxeval
            progress = 0.35 + (0.45 * index / max(1, len(estimation_records)))
            ctx.emit(
                f"Running {method_name} estimation step {index}/{len(estimation_records)}",
                progress=progress,
            )
            estimation = get_estimation_method(
                method_name,
                **{
                    "interaction": interaction,
                    "maxeval": maxeval,
                    "n_parallel": n_parallel,
                    **(
                        {}
                        if estimation_record is None
                        else {
                            **(
                                {"n_starts": estimation_record.n_starts}
                                if getattr(estimation_record, "n_starts", 1) > 1
                                else {}
                            ),
                            **(
                                {"gtol": estimation_record.gtol}
                                if getattr(estimation_record, "gtol", 1e-5) != 1e-5
                                else {}
                            ),
                            **(
                                {"perturbation_scale": estimation_record.perturbation_scale}
                                if getattr(estimation_record, "perturbation_scale", 1.0) != 1.0
                                else {}
                            ),
                            **(
                                {"seed": estimation_record.seed}
                                if getattr(estimation_record, "seed", None) is not None
                                else {}
                            ),
                            **(
                                {"outer_optimizer": estimation_record.outer_optimizer}
                                if getattr(estimation_record, "outer_optimizer", None)
                                else {}
                            ),
                            **(
                                {
                                    "outer_fallback_optimizer": estimation_record.outer_fallback_optimizer
                                }
                                if getattr(estimation_record, "outer_fallback_optimizer", None)
                                else {}
                            ),
                            **(
                                {
                                    "outer_fallback_maxeval": estimation_record.outer_fallback_maxeval
                                }
                                if getattr(estimation_record, "outer_fallback_maxeval", None)
                                is not None
                                else {}
                            ),
                            **(
                                {"retain_best_iterate": estimation_record.retain_best_iterate}
                                if getattr(estimation_record, "retain_best_iterate", None)
                                is not None
                                else {}
                            ),
                            **(
                                {"retry_on_abnormal": estimation_record.retry_on_abnormal}
                                if getattr(estimation_record, "retry_on_abnormal", None)
                                is not None
                                else {}
                            ),
                            **(
                                {"retry_omega_scales": estimation_record.retry_omega_scales}
                                if getattr(estimation_record, "retry_omega_scales", ())
                                else {}
                            ),
                        }
                    ),
                },
            )
            final_result = estimation.estimate(pop_model, params)
            ctx.check_cancelled()
            params = ParameterSet(
                theta=final_result.theta_final,
                omega=final_result.omega_final,
                sigma=final_result.sigma_final,
                theta_specs=params.theta_specs,
                omega_specs=params.omega_specs,
                sigma_specs=params.sigma_specs,
            )

        assert final_result is not None
        if control_stream.covariance is not None:
            ctx.check_cancelled()
            ctx.emit("Running covariance step", progress=0.9)
            covariance = SandwichCovariance(matrix=control_stream.covariance.matrix)
            covariance_result = covariance.compute(pop_model, params, final_result.post_hoc_etas)
            final_result.warnings.extend(covariance_result.warnings)
            ctx.check_cancelled()
        return final_result, params, pop_model

    def _fit_run_result_from_estimation(
        self,
        ctx,
        workspace: Workspace,
        title: str,
        estimation_result: EstimationResult,
        method_name: str | None,
        params: ParameterSet | None,
        run_id: str | None,
        *,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
        population_model: object | None = None,
        translation: ModelTranslationResult | None = None,
    ) -> FitRunResult:
        method = method_name or estimation_result.method or Method.FOCE
        ctx.check_cancelled()
        ctx.emit("Finalizing fit summary", progress=0.75)
        summary_text = (
            f"{title} • {method} • converged={estimation_result.converged} "
            f"• OFV={estimation_result.ofv:.4f}"
        )
        if population_model is not None and run_id is not None:
            self._cache_fit_context(
                workspace,
                title=title,
                method=method,
                estimation_result=estimation_result,
                population_model=population_model,
                fit_run_id=run_id,
                project_id=project_id,
                scenario_id=scenario_id,
                dataset_path=dataset_path,
            )
        ctx.check_cancelled()
        ctx.emit("Generating fit outputs", progress=0.8)
        artifacts = self._generate_output_artifacts(
            ctx,
            workspace,
            title,
            method,
            estimation_result,
            params,
            run_id=run_id,
            project_id=project_id,
            scenario_id=scenario_id,
            dataset_path=dataset_path,
            population_model=population_model,
            translation=translation,
        )
        return FitRunResult(
            problem_title=title,
            estimation_method=method,
            converged=estimation_result.converged,
            ofv=estimation_result.ofv,
            summary_text=summary_text,
            warning_messages=list(estimation_result.warnings),
            artifacts=artifacts,
        )

    @staticmethod
    def _slug(text: str) -> str:
        safe = "".join(
            character if character.isalnum() else "-" for character in text.strip().lower()
        )
        return safe.strip("-") or "fit"

    @staticmethod
    def _final_parameter_set(
        params: ParameterSet | None,
        estimation_result: EstimationResult,
    ) -> ParameterSet | None:
        if params is None:
            return None
        theta_final = getattr(estimation_result, "theta_final", None)
        omega_final = getattr(estimation_result, "omega_final", None)
        sigma_final = getattr(estimation_result, "sigma_final", None)
        if theta_final is None or omega_final is None or sigma_final is None:
            return None
        return ParameterSet(
            theta=theta_final,
            omega=omega_final,
            sigma=sigma_final,
            theta_specs=params.theta_specs,
            omega_specs=params.omega_specs,
            sigma_specs=params.sigma_specs,
        )

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

    @staticmethod
    def _clean_provenance_mapping(values: Mapping[str, object]) -> dict[str, object]:
        cleaned: dict[str, object] = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, dict)) and len(value) == 0:
                continue
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _sha256_file(path: Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None

    @staticmethod
    def _dataset_provenance(dataset, dataset_path: str | None) -> dict[str, object]:
        effective_path = dataset_path or getattr(dataset, "source_path", None)
        dataset_info = FitService._clean_provenance_mapping(
            {
                "source_path": effective_path,
                "display_name": getattr(dataset, "display_name", None),
                "asset_source_path": getattr(dataset, "source_path", None),
                "separator": getattr(dataset, "separator", None),
                "columns": getattr(dataset, "columns", None),
                "row_count": getattr(dataset, "row_count", None),
                "subject_count": getattr(dataset, "subject_count", None),
                "observation_count": getattr(dataset, "observation_count", None),
            }
        )
        if not effective_path:
            return dataset_info
        dataset_file = Path(effective_path).expanduser()
        try:
            resolved_file = dataset_file.resolve(strict=False)
        except Exception:
            resolved_file = dataset_file
        dataset_info["resolved_source_path"] = str(resolved_file)
        dataset_info["exists_at_report_time"] = dataset_file.exists()
        if dataset_file.exists() and dataset_file.is_file():
            stat = dataset_file.stat()
            dataset_info["size_bytes"] = stat.st_size
            dataset_info["modified_at"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            sha256 = FitService._sha256_file(dataset_file)
            if sha256 is not None:
                dataset_info["sha256"] = sha256
        return dataset_info

    @staticmethod
    def _builder_estimation_settings(method: str, model_spec) -> dict[str, object]:
        estimation = getattr(model_spec, "estimation", None)
        options = dict(getattr(estimation, "options", {}) or {})
        effective = {
            "effective_method": method,
            "interaction": bool(
                options.get("interaction", False) or str(method).upper() == Method.FOCEI
            ),
            "maxeval": int(options.get("maxeval", 9999)),
        }
        for key, value in options.items():
            if key not in effective:
                effective[key] = value
        return FitService._clean_provenance_mapping(effective)

    @staticmethod
    def _control_stream_estimation_settings(
        translation: ModelTranslationResult, method: str
    ) -> dict[str, object]:
        control_stream = translation.control_stream
        if control_stream is None:
            return {"effective_method": method}
        steps: list[dict[str, object]] = []
        estimation_records = control_stream.estimation_records or [None]
        for index, record in enumerate(estimation_records, start=1):
            if record is None:
                steps.append(
                    {"step": index, "method": Method.FOCE, "interaction": False, "maxeval": 9999}
                )
                continue
            steps.append(
                FitService._clean_provenance_mapping(
                    {
                        "step": index,
                        "method": record.method,
                        "interaction": record.interaction,
                        "maxeval": record.maxeval,
                        "sigdig": record.sigdig,
                        "sigl": record.sigl,
                        "print_interval": record.print_interval,
                        "noabort": record.noabort,
                        "posthoc": record.posthoc,
                        "laplace": record.laplace,
                        "isample": record.isample,
                        "niter": record.niter,
                        "seed": record.seed,
                        "msfo": record.msfo,
                        "nothetaboundtest": record.nothetaboundtest,
                        "noomegaboundtest": record.noomegaboundtest,
                        "numerical": record.numerical,
                        "gradient": record.gradient,
                    }
                )
            )
        return {"effective_method": method, "estimation_steps": steps}

    @staticmethod
    def _covariance_settings(
        model_spec, translation: ModelTranslationResult | None
    ) -> dict[str, object]:
        if translation is not None and translation.control_stream is not None:
            covariance = translation.control_stream.covariance
            if covariance is None:
                return {"enabled": False}
            return FitService._clean_provenance_mapping(
                {
                    "enabled": True,
                    "matrix": covariance.matrix,
                    "unconditional": covariance.unconditional,
                    "only": covariance.only,
                    "sigl": covariance.sigl,
                    "print_e": covariance.print_e,
                }
            )
        covariance = getattr(model_spec, "covariance", None)
        if covariance is None:
            return {}
        return FitService._clean_provenance_mapping(
            {
                "enabled": covariance.enabled,
                "matrix": covariance.matrix,
                "options": dict(covariance.options or {}),
            }
        )

    @staticmethod
    def _environment_provenance() -> dict[str, object]:
        git_info = FitService._git_revision_provenance()
        return FitService._clean_provenance_mapping(
            {
                "openpkpd_version": OPENPKPD_VERSION,
                "python_version": sys.version.split()[0],
                "python_implementation": platform.python_implementation(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                **git_info,
            }
        )

    @staticmethod
    def _git_revision_provenance() -> dict[str, object]:
        repo_root = Path(__file__).resolve().parents[3]

        def _git(*args: str) -> str | None:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=repo_root,
                    capture_output=True,
                    check=True,
                    text=True,
                )
            except Exception:
                return None
            value = result.stdout.strip()
            return value or None

        commit = _git("rev-parse", "HEAD")
        if commit is None:
            return {}
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        dirty_lines = _git("status", "--porcelain")
        return FitService._clean_provenance_mapping(
            {
                "git_commit": commit,
                "git_branch": branch,
                "git_dirty_worktree": bool(dirty_lines),
            }
        )

    def _build_report_provenance(
        self,
        workspace: Workspace,
        *,
        title: str,
        method: str,
        run_id: str | None,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
        translation: ModelTranslationResult | None,
    ) -> dict[str, object]:
        project = workspace.find_project(project_id) or workspace.active_project
        scenario_lookup = workspace.find_scenario(scenario_id, project_id=project_id)
        scenario = scenario_lookup[1] if scenario_lookup is not None else project.active_scenario
        model_spec = scenario.active_model_spec
        dataset = scenario.active_dataset

        provenance: dict[str, object] = {
            "Run context": self._clean_provenance_mapping(
                {
                    "workflow": "fit",
                    "run_id": run_id,
                    "problem_title": title,
                    "workspace_id": workspace.workspace_id,
                    "workspace_name": workspace.name,
                    "workspace_root_path": workspace.root_path,
                    "project_id": project.project_id,
                    "project_name": project.name,
                    "scenario_id": scenario.scenario_id,
                    "scenario_name": scenario.name,
                    "parent_scenario_id": scenario.parent_scenario_id,
                    "scenario_updated_at": str(scenario.updated_at),
                }
            ),
            "Dataset": self._dataset_provenance(dataset, dataset_path),
            "Model": self._clean_provenance_mapping(
                {
                    "authoring_mode": getattr(
                        getattr(model_spec, "mode", None),
                        "value",
                        getattr(model_spec, "mode", None),
                    ),
                    "problem_title": getattr(model_spec, "problem_title", None) or title,
                    "model_dataset_reference": getattr(model_spec, "dataset_path", None),
                    "advan": getattr(model_spec, "advan", None),
                    "trans": getattr(model_spec, "trans", None),
                    "theta_count": translation.theta_count if translation is not None else None,
                    "eta_count": translation.eta_count if translation is not None else None,
                    "eps_count": translation.eps_count if translation is not None else None,
                    "record_count": translation.record_count if translation is not None else None,
                }
            ),
            "Model source": self._clean_provenance_mapping(
                {
                    "pk_code": getattr(model_spec, "pk_code", None),
                    "error_code": getattr(model_spec, "error_code", None),
                    "des_code": getattr(model_spec, "des_code", None),
                    "control_stream_text": getattr(model_spec, "control_stream_text", None),
                }
            ),
            "Estimation settings": self._builder_estimation_settings(method, model_spec),
            "Covariance settings": self._covariance_settings(model_spec, translation),
            "Environment": self._environment_provenance(),
        }
        if translation is not None and translation.control_stream is not None:
            provenance["Estimation settings"] = self._control_stream_estimation_settings(
                translation, method
            )
        return {
            section: payload
            for section, payload in provenance.items()
            if payload not in ({}, [], None, "")
        }

    def _generate_output_artifacts(
        self,
        ctx,
        workspace: Workspace,
        title: str,
        method: str,
        estimation_result: EstimationResult,
        params: ParameterSet | None,
        *,
        run_id: str | None,
        project_id: str,
        scenario_id: str,
        dataset_path: str | None,
        population_model: object | None = None,
        translation: ModelTranslationResult | None = None,
    ) -> list[ArtifactRecord]:
        artifact_dir = self._artifact_directory(
            workspace,
            project_id=project_id,
            scenario_id=scenario_id,
            dataset_path=dataset_path,
        )
        stem = f"{self._slug(title)}-{self._slug(method)}-{(run_id or 'run')[:8]}"
        artifacts: list[ArtifactRecord] = []

        def _append_table(table_df, suffix: str, label: str, role: str) -> None:
            table_path = artifact_dir / f"{stem}-{suffix}.csv"
            table_df.to_csv(table_path, index=False)
            artifacts.append(
                ArtifactRecord(
                    kind="table",
                    label=label,
                    path=str(table_path),
                    source_run_id=run_id,
                    metadata={
                        "media_type": "text/csv",
                        "artifact_role": role,
                        "estimation_method": method,
                    },
                )
            )

        # 1. Compute diagnostics ───────────────────────────────────────────
        diag_df = None
        if population_model is not None:
            try:
                ctx.check_cancelled()
                ctx.emit("Computing fit diagnostics", progress=0.82)
                diag_df = compute_diagnostics(population_model, estimation_result)
            except Exception:
                diag_df = None

        if diag_df is not None:
            try:
                ctx.check_cancelled()
                _append_table(
                    diag_df,
                    "diagnostics",
                    f"{title} diagnostics table",
                    "diagnostics_table",
                )
            except Exception:
                pass

        # 2. Generate plots, collect (section_title, path) for report ─────
        report_plots: list[tuple[str, str]] = []

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            def _append_plot(
                fig, suffix: str, label: str, plot_type: str, report_title: str
            ) -> None:
                plot_path = artifact_dir / f"{stem}-{suffix}.png"
                try:
                    ctx.check_cancelled()
                    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                finally:
                    plt.close(fig)
                artifacts.append(
                    ArtifactRecord(
                        kind="plot",
                        label=label,
                        path=str(plot_path),
                        source_run_id=run_id,
                        metadata={
                            "media_type": "image/png",
                            "artifact_role": "plot",
                            "plot_type": plot_type,
                            "estimation_method": method,
                        },
                    )
                )
                report_plots.append((report_title, str(plot_path)))

            ctx.check_cancelled()
            ctx.emit("Rendering diagnostic plots", progress=0.88)
            with contextlib.suppress(Exception):
                _append_plot(
                    ofv_history(estimation_result, title=f"{title} — OFV history"),
                    "ofv-history",
                    f"{title} OFV history",
                    "ofv_history",
                    "OFV History",
                )

            with contextlib.suppress(Exception):
                _append_plot(
                    parameter_uncertainty_plot(
                        estimation_result,
                        title=f"{title} — Parameter uncertainty",
                    ),
                    "parameter-uncertainty",
                    f"{title} parameter uncertainty",
                    "parameter_uncertainty",
                    "Parameter Uncertainty",
                )

            if diag_df is not None:
                with contextlib.suppress(Exception):
                    _append_plot(
                        diagnostic_panel(diag_df, title=f"{title} — GOF panel"),
                        "gof-panel",
                        f"{title} GOF panel",
                        "gof_panel",
                        "Goodness of Fit",
                    )

                for plot_function, suffix, label, plot_type, report_title in (
                    (
                        dv_vs_ipred,
                        "dv-vs-ipred",
                        f"{title} DV vs IPRED",
                        "dv_vs_ipred",
                        "DV vs IPRED",
                    ),
                    (dv_vs_pred, "dv-vs-pred", f"{title} DV vs PRED", "dv_vs_pred", "DV vs PRED"),
                    (
                        cwres_vs_time,
                        "cwres-vs-time",
                        f"{title} CWRES vs TIME",
                        "cwres_vs_time",
                        "CWRES vs TIME",
                    ),
                    (
                        cwres_vs_pred,
                        "cwres-vs-pred",
                        f"{title} CWRES vs PRED",
                        "cwres_vs_pred",
                        "CWRES vs PRED",
                    ),
                    (cwres_qq, "cwres-qq", f"{title} CWRES Q-Q", "cwres_qq", "CWRES Q-Q"),
                    (
                        abs_iwres_vs_ipred,
                        "abs-iwres-vs-ipred",
                        f"{title} |IWRES| vs IPRED",
                        "abs_iwres_vs_ipred",
                        "|IWRES| vs IPRED",
                    ),
                    (
                        residual_trends_plot,
                        "residual-trends",
                        f"{title} residual trends",
                        "residual_trends",
                        "Residual Trends",
                    ),
                    (
                        spaghetti_plot,
                        "spaghetti-plot",
                        f"{title} spaghetti plot",
                        "spaghetti_plot",
                        "Individual Profiles",
                    ),
                    (
                        mean_profile,
                        "mean-profile",
                        f"{title} mean profile",
                        "mean_profile",
                        "Mean Profile",
                    ),
                ):
                    with contextlib.suppress(Exception):
                        _append_plot(
                            plot_function(diag_df, title=f"{title} — {report_title}"),
                            suffix,
                            label,
                            plot_type,
                            report_title,
                        )

                eta_cols = [
                    str(c) for c in getattr(diag_df, "columns", []) if str(c).startswith("ETA")
                ]
                omega_final = getattr(estimation_result, "omega_final", None)
                omega_matrix = None
                if omega_final is not None:
                    try:
                        omega_matrix = np.asarray(omega_final, dtype=float)
                    except Exception:
                        omega_matrix = None

                if eta_cols and omega_matrix is not None and omega_matrix.ndim == 2:
                    with contextlib.suppress(Exception):
                        _append_plot(
                            eta_histograms(
                                diag_df, omega_matrix, title=f"{title} — ETA histograms"
                            ),
                            "eta-histograms",
                            f"{title} ETA histograms",
                            "eta_histograms",
                            "ETA Histograms",
                        )

                if eta_cols:
                    with contextlib.suppress(Exception):
                        _append_plot(
                            eta_pairs(diag_df, title=f"{title} — ETA pairs"),
                            "eta-pairs",
                            f"{title} ETA pairs",
                            "eta_pairs",
                            "ETA Pairs",
                        )
        except Exception:
            pass

        # 3. Generate HTML report with all plots embedded ──────────────────
        if params is not None:
            try:
                ctx.check_cancelled()
                ctx.emit("Generating HTML fit report", progress=0.96)
                report_path = artifact_dir / f"{stem}-report.html"
                provenance = self._build_report_provenance(
                    workspace,
                    title=title,
                    method=method,
                    run_id=run_id,
                    project_id=project_id,
                    scenario_id=scenario_id,
                    dataset_path=dataset_path,
                    translation=translation,
                )
                write_html_report(
                    str(report_path),
                    estimation_result,
                    params,
                    title=title,
                    provenance=provenance,
                    plots=report_plots or None,
                )
                artifacts.append(
                    ArtifactRecord(
                        kind="report",
                        label=f"{title} report",
                        path=str(report_path),
                        source_run_id=run_id,
                        metadata={
                            "media_type": "text/html",
                            "artifact_role": "report",
                            "estimation_method": method,
                        },
                    )
                )
            except Exception:
                pass

        ctx.check_cancelled()
        ctx.emit("Fit outputs ready", progress=0.99)
        return artifacts
