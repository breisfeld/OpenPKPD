"""Model workflow widget for editing GUI-side model specifications."""

from __future__ import annotations

import html
from pathlib import Path

from openpkpd.examples.catalog_models import ExampleEntry
from openpkpd.examples.catalog_service import ExampleCatalogService
from openpkpd.parser.control_stream import ControlStream
from openpkpd.utils.errors import ParseError
from openpkpd_gui.app.runtime import load_qt_modules
from openpkpd_gui.app.settings import (
    default_workspace_root_path,
    load_gui_preferences,
    save_gui_preferences,
    with_dismissed_hint,
    with_last_file_dialog_dir,
)
from openpkpd_gui.domain.model_spec import (
    CovarianceConfig,
    EstimationConfig,
    ModelSpec,
    ModelSpecMode,
)
from openpkpd_gui.domain.run_record import RunRecord, RunStatus
from openpkpd_gui.domain.workspace import Workspace
from openpkpd_gui.services.data_service import DatasetImportOptions, DatasetService
from openpkpd_gui.services.model_translation_service import (
    ModelTranslationResult,
    ModelTranslationService,
)
from openpkpd_gui.services.project_service import ProjectService
from openpkpd_gui.services.validation_service import ValidationIssue
from openpkpd_gui.widgets.dismissible_hint import build_dismissible_hint
from openpkpd_gui.widgets.link_formatting import (
    copy_link,
    decode_copy_target,
    external_link,
    file_link,
)
from openpkpd_gui.widgets.model_diagram import build_model_diagram_widget
from openpkpd_gui.widgets.nmtran_highlighter import NmtranHighlighter
from openpkpd_gui.widgets.responsive_layout import install_responsive_splitters
from openpkpd_gui.widgets.scrollable_page import build_scrollable_page
from openpkpd_gui.widgets.table_headers import configure_resizable_table_columns

ESTIMATION_METHODS = ("FO", "FOCE", "FOCEI", "LAPLACIAN", "SAEM", "IMP", "BAYES")

_NAMED_MODELS: tuple[tuple[str, int, int], ...] = (
    ("1-compartment IV bolus  (ADVAN1 / TRANS1)", 1, 1),
    ("1-compartment oral  (ADVAN2 / TRANS2)", 2, 2),
    ("1-compartment IV bolus, Michaelis-Menten  (ADVAN1 / TRANS2)", 1, 2),
    ("2-compartment IV bolus  (ADVAN3 / TRANS4)", 3, 4),
    ("2-compartment oral  (ADVAN4 / TRANS4)", 4, 4),
    ("N-compartment general linear  (ADVAN5 / TRANS1)", 5, 1),
    ("3-compartment IV bolus  (ADVAN11 / TRANS4)", 11, 4),
    ("3-compartment oral  (ADVAN12 / TRANS4)", 12, 4),
    ("ODE / user-defined  (ADVAN6)", 6, 1),
    ("ODE / user-defined, stiff  (ADVAN8 / LSODA)", 8, 1),
    ("Custom…", -1, -1),
)
_NAMED_MODEL_LABELS = [label for label, _, _ in _NAMED_MODELS]

_ESTIMATION_METHODS_DISPLAY: tuple[tuple[str, str], ...] = (
    ("FOCE", "FOCE — First-order conditional estimation (recommended)"),
    ("FOCEI", "FOCEI — FOCE with interaction"),
    ("FO", "FO — First-order approximation (fast, less accurate)"),
    ("SAEM", "SAEM — Stochastic approximation EM (nonlinear / complex models)"),
    ("IMP", "IMP — Importance sampling"),
    ("LAPLACIAN", "Laplacian — Laplace approximation"),
    ("BAYES", "BAYES — Full Bayesian NUTS (PyMC / built-in)"),
    ("NONPARAMETRIC", "NONPARAMETRIC — Nonparametric support point estimation"),
)

_NP_METHODS = frozenset({"NONPARAMETRIC"})
_NP_BASE_METHODS: tuple[tuple[str, str], ...] = (
    ("FOCE", "FOCE"),
    ("FOCEI", "FOCEI"),
    ("FO", "FO"),
)

_ESTIMATION_HELP_TEXT = (
    "FOCE (recommended): First-order conditional estimation with eta-linearization. "
    "Accurate for most nonlinear mixed-effects models.\n\n"
    "FOCEI: FOCE with interaction term — use when residual error depends on ETAs.\n\n"
    "FO: First-order approximation. Fastest but least accurate; suitable for screening.\n\n"
    "SAEM: Stochastic Approximation EM. Best for complex/multimodal models.\n\n"
    "IMP: Importance sampling — alternative to SAEM for complex posteriors.\n\n"
    "Laplacian: Laplace approximation, similar accuracy to FOCE.\n\n"
    "BAYES: Full Bayesian posterior sampling via NUTS (No-U-Turn Sampler). "
    "Returns the full posterior distribution, credible intervals, R-hat convergence "
    "diagnostics, and effective sample size. Supports PyMC or the built-in native "
    "backend. Slower than FOCE/SAEM but provides uncertainty quantification without "
    "linearisation assumptions.\n\n"
    "NONPARAMETRIC: Two-step nonparametric estimation (NPML / support point approximation). "
    "Step 1 runs a base parametric method (FOCE/FOCEI/FO) to obtain population parameters "
    "and empirical Bayes estimates. Step 2 treats EBEs as support points and optimises their "
    "weights by EM to maximise the marginal likelihood. Distribution-free IIV representation."
)
_ADVAN_HELP_TEXT = (
    "ADVAN selects the PK structural model (compartment structure).\n"
    "TRANS selects the parameterization (e.g., CL/V vs micro-rate constants).\n\n"
    "Common choices:\n"
    "  ADVAN2/TRANS2 — 1-compartment, oral, CL/V/KA\n"
    "  ADVAN4/TRANS4 — 2-compartment, oral, CL/V1/Q/V2/KA\n"
    "  ADVAN5/TRANS1 — N-compartment linear, micro rate constants Kij/Ki0\n"
    "  ADVAN6        — ODE-based user-defined model (non-stiff RK45)\n"
    "  ADVAN8        — ODE-based user-defined model (stiff, LSODA)\n\n"
    "Use ADVAN8 when your $DES model has widely separated time scales\n"
    "(e.g. fast distribution K12 >> slow elimination K10), PBPK systems,\n"
    "or receptor-binding models. ADVAN6 handles most standard PK ODEs and\n"
    "automatically falls back to the stiff solver if the step limit is hit.\n\n"
    "Tip: install openpkpd[jit] to enable Numba acceleration for $DES models.\n"
    "When active, the FOCE/FOCEI inner loop and Hessian computation use a\n"
    "compiled ODE probe instead of repeated Python evaluations — no model\n"
    "changes needed."
)
_OMEGA_HELP_TEXT = (
    "OMEGA is the covariance matrix of inter-individual variability (IIV).\n"
    "Each diagonal element represents the variance of one ETA (random effect).\n"
    "Off-diagonal elements represent correlations between ETAs.\n\n"
    "Typical diagonal value: 0.09 → CV ≈ 30% for a lognormal parameter."
)
_SIGMA_HELP_TEXT = (
    "SIGMA is the covariance matrix of residual (within-subject) error.\n"
    "For a proportional error model: SIGMA(1,1) = 0.04 → 20% CV residual."
)
_COV_HELP_TEXT = (
    "The covariance step estimates standard errors (SE) and the correlation matrix "
    "of all population parameters.\n\n"
    "SR (sandwich): robust estimator, recommended.\n"
    "R: Fisher information matrix.\n"
    "S: Cross-product approximation (fast but can be unstable)."
)
_EST_OPTIONS_HELP_TEXT = (
    "Max eval sets the outer optimizer budget for the selected estimation method.\n\n"
    "Multi-start runs the optimizer N times from randomly perturbed initial values and "
    "keeps the best result. Useful for models with local minima (e.g. 2-compartment IV).\n\n"
    "Tight gradient (gtol=1e-6) forces more optimizer iterations in flat likelihood "
    "regions. Recommended for covariate-rich models (power-law WT/AGE exponents).\n\n"
    "Advanced FOCE/FOCEI outer-optimizer controls such as OUTEROPT/FALLBACKOPT, "
    "best-iterate retention, and retry-on-abnormal are available through control "
    "streams and the Python API, and are available here for gradient-based methods."
)
_GRADIENT_METHODS = {"FOCE", "FOCEI", "LAPLACIAN"}
_INTERACTION_METHODS = {"FOCEI", "LAPLACIAN"}
_OUTER_OPTIMIZERS: tuple[str, ...] = ("L-BFGS-B", "Powell")
_FALLBACK_OPTIMIZER_CHOICES: tuple[tuple[str, str | None], ...] = (
    ("Engine default", None),
    ("Powell", "Powell"),
    ("L-BFGS-B", "L-BFGS-B"),
)


def _default_maxeval(method: str) -> int:
    return 9999


def _default_outer_optimizer(method: str) -> str:
    return "L-BFGS-B"


def _default_fallback_optimizer(method: str) -> str | None:
    if method in _INTERACTION_METHODS:
        return None
    return None


def _default_fallback_maxeval(method: str) -> int:
    return 40


def _default_retain_best_iterate(method: str) -> bool:
    return True


def _default_retry_on_abnormal(method: str) -> bool:
    return method in _INTERACTION_METHODS


def _default_retry_omega_scales(method: str) -> str:
    if method in _INTERACTION_METHODS:
        return "0.5, 0.25, 0.1"
    return ""


def _parse_retry_omega_scales(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        values.append(float(piece))
    return tuple(values)


def _advan_trans_to_named_model_index(advan: int, trans: int) -> int:
    for i, (_, a, t) in enumerate(_NAMED_MODELS):
        if a == advan and t == trans:
            return i
    return len(_NAMED_MODELS) - 1  # "Custom…"


CONTROL_STREAM_FILE_FILTER = "NONMEM control streams (*.ctl *.mod *.txt);;All files (*)"


def _format_catalog_control_stream_label(entry: ExampleEntry) -> str:
    """Return a compact control-stream example label for the selector."""
    category = entry.manifest.category.upper() if entry.manifest.category else "EXAMPLE"
    return f"{category}: {entry.manifest.title}"


def _format_catalog_control_stream_details(entry: ExampleEntry | None) -> str:
    """Return helper text for the selected curated control-stream example."""
    if entry is None:
        return ""
    parts = [entry.manifest.description.strip() or entry.manifest.title]
    meta: list[str] = []
    if entry.manifest.route:
        meta.append(f"route: {entry.manifest.route}")
    meta.append(f"difficulty: {entry.manifest.difficulty}")
    if entry.dataset_path is not None:
        meta.append(f"dataset: {entry.dataset_path.name}")
    else:
        meta.append("dataset: not bundled")
    if meta:
        parts.append(" • ".join(meta))
    provenance_parts = [f"source: {entry.manifest.source.kind}"]
    if entry.manifest.source.license:
        provenance_parts.append(f"license: {entry.manifest.source.license}")
    if entry.manifest.source.url:
        provenance_parts.append(f"url: {entry.manifest.source.url}")
    parts.append("Provenance: " + " • ".join(provenance_parts))
    if entry.readme_path is not None:
        parts.append(f"Bundle notes: {entry.readme_path}")
    return "\n".join(parts)


def _format_catalog_control_stream_details_html(entry: ExampleEntry | None) -> str:
    """Return rich-text helper text with clickable links for a curated control-stream example."""
    if entry is None:
        return ""
    parts = [f"<b>{html.escape(entry.manifest.title)}</b>"]
    description = entry.manifest.description.strip() or entry.manifest.title
    parts.append(html.escape(description))
    meta: list[str] = []
    if entry.manifest.route:
        meta.append(f"route: {html.escape(entry.manifest.route)}")
    meta.append(f"difficulty: {html.escape(entry.manifest.difficulty)}")
    if entry.dataset_path is not None:
        meta.append(
            "dataset: "
            + file_link(entry.dataset_path, label=entry.dataset_path.name)
            + " • "
            + copy_link(entry.dataset_path, label="Copy path")
        )
    else:
        meta.append("dataset: not bundled")
    parts.append(" • ".join(meta))
    provenance_parts = [f"source: {html.escape(entry.manifest.source.kind)}"]
    if entry.manifest.source.license:
        provenance_parts.append(f"license: {html.escape(entry.manifest.source.license)}")
    if entry.manifest.source.url:
        provenance_parts.append(
            "url: "
            + external_link(entry.manifest.source.url)
            + " • "
            + copy_link(entry.manifest.source.url, label="Copy URL")
        )
    parts.append("Provenance: " + " • ".join(provenance_parts))
    if entry.readme_path is not None:
        parts.append(
            "Bundle notes: "
            + file_link(entry.readme_path)
            + " • "
            + copy_link(entry.readme_path, label="Copy path")
        )
    actions = [
        file_link(entry.control_stream_path, label="Open control stream"),
        copy_link(entry.control_stream_path, label="Copy control-stream path"),
        file_link(entry.bundle_dir, label="Open bundle folder"),
        copy_link(entry.bundle_dir, label="Copy bundle path"),
    ]
    if entry.readme_path is not None:
        actions.append(file_link(entry.readme_path, label="Open bundle notes"))
    if entry.manifest.source.url:
        actions.append(external_link(entry.manifest.source.url, label="Open upstream source"))
    parts.append("Actions: " + " • ".join(actions))
    return "<br/>".join(parts)


DEFAULT_PK_CODE = "\n".join(
    [
        "KA = THETA(1) * EXP(ETA(1))",
        "CL = THETA(2) * EXP(ETA(2))",
        "V  = THETA(3) * EXP(ETA(3))",
    ]
)
DEFAULT_ERROR_CODE = "Y = F * (1 + EPS(1))"
DEFAULT_THETA_ROWS = (
    {"label": "KA", "lower": 0.01, "init": 1.5, "upper": 20.0, "fixed": False},
    {"label": "CL", "lower": 0.001, "init": 0.08, "upper": 5.0, "fixed": False},
    {"label": "V", "lower": 0.1, "init": 30.0, "upper": 500.0, "fixed": False},
)
DEFAULT_OMEGA_VALUES = (
    (0.3, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (0.0, 0.0, 0.2),
)
DEFAULT_SIGMA_VALUES = ((0.1,),)
THETA_TABLE_HEADERS = ("Label", "Lower", "Init", "Upper", "Fixed")
MODEL_RESPONSIVE_LAYOUT_BREAKPOINT = 1180

# ---------------------------------------------------------------------------
# P3-C: Per-ADVAN/TRANS typical starting parameter values
# ---------------------------------------------------------------------------
# Keys are (advan, trans) pairs; values are (theta_rows, omega_diag, sigma_diag)
_ADVAN_PRESETS: dict[
    tuple[int, int],
    tuple[list[dict[str, object]], list[float], list[float]],
] = {
    # 1-cmt IV bolus, CL/V parameterisation
    (1, 2): (
        [
            {"label": "CL", "lower": 0.001, "init": 5.0, "upper": 100.0, "fixed": False},
            {"label": "V", "lower": 0.1, "init": 30.0, "upper": 500.0, "fixed": False},
        ],
        [0.2, 0.2],
        [0.1],
    ),
    # 1-cmt oral, CL/V/KA parameterisation
    (2, 2): (
        [
            {"label": "KA", "lower": 0.01, "init": 1.5, "upper": 20.0, "fixed": False},
            {"label": "CL", "lower": 0.001, "init": 0.08, "upper": 5.0, "fixed": False},
            {"label": "V", "lower": 0.1, "init": 30.0, "upper": 500.0, "fixed": False},
        ],
        [0.3, 0.2, 0.2],
        [0.1],
    ),
    # 2-cmt IV bolus, CL/V1/Q/V2 parameterisation
    (3, 4): (
        [
            {"label": "CL", "lower": 0.001, "init": 5.0, "upper": 100.0, "fixed": False},
            {"label": "V1", "lower": 0.1, "init": 30.0, "upper": 500.0, "fixed": False},
            {"label": "Q", "lower": 0.001, "init": 2.0, "upper": 50.0, "fixed": False},
            {"label": "V2", "lower": 0.1, "init": 50.0, "upper": 500.0, "fixed": False},
        ],
        [0.2, 0.2, 0.2, 0.2],
        [0.1],
    ),
    # 2-cmt oral, CL/V1/Q/V2/KA parameterisation
    (4, 4): (
        [
            {"label": "KA", "lower": 0.01, "init": 1.5, "upper": 20.0, "fixed": False},
            {"label": "CL", "lower": 0.001, "init": 5.0, "upper": 100.0, "fixed": False},
            {"label": "V1", "lower": 0.1, "init": 30.0, "upper": 500.0, "fixed": False},
            {"label": "Q", "lower": 0.001, "init": 2.0, "upper": 50.0, "fixed": False},
            {"label": "V2", "lower": 0.1, "init": 50.0, "upper": 500.0, "fixed": False},
        ],
        [0.3, 0.2, 0.2, 0.2, 0.2],
        [0.1],
    ),
}


def suggest_theta_rows_for_advan(
    advan: int, trans: int
) -> list[dict[str, object]] | None:
    """Return typical THETA starting values for *advan*/*trans*, or ``None`` if unknown."""
    preset = _ADVAN_PRESETS.get((advan, trans))
    if preset is None:
        return None
    theta_rows, _, _ = preset
    return [dict(row) for row in theta_rows]


def suggest_omega_values_for_advan(
    advan: int, trans: int
) -> list[list[float]] | None:
    """Return typical diagonal OMEGA values for *advan*/*trans*, or ``None`` if unknown."""
    preset = _ADVAN_PRESETS.get((advan, trans))
    if preset is None:
        return None
    _, omega_diag, _ = preset
    n = len(omega_diag)
    return [[omega_diag[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


def _default_theta_rows() -> list[dict[str, object]]:
    return [dict(row) for row in DEFAULT_THETA_ROWS]


def _default_omega_values() -> list[list[float]]:
    return [list(row) for row in DEFAULT_OMEGA_VALUES]


def _default_sigma_values() -> list[list[float]]:
    return [list(row) for row in DEFAULT_SIGMA_VALUES]


def default_theta_row(index: int) -> dict[str, object]:
    """Return a default THETA row for table-driven editing."""
    return {
        "label": f"THETA{index}",
        "lower": 0.0,
        "init": 1.0,
        "upper": 10.0,
        "fixed": False,
    }


def normalize_theta_rows(theta_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return THETA rows with all expected fields populated."""
    rows = theta_rows or [default_theta_row(1)]
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        default = default_theta_row(index)
        normalized.append(
            {
                "label": row.get("label", default["label"]),
                "lower": row.get("lower", default["lower"]),
                "init": row.get("init", default["init"]),
                "upper": row.get("upper", default["upper"]),
                "fixed": row.get("fixed", default["fixed"]),
            }
        )
    return normalized


def resize_square_matrix(
    values: list[list[object]],
    size: int,
    *,
    diagonal_fill: float,
) -> list[list[object]]:
    """Resize a square matrix while preserving any existing entered values."""
    size = max(1, size)
    resized: list[list[object]] = []
    for row_index in range(size):
        row: list[object] = []
        for column_index in range(size):
            existing = None
            if row_index < len(values) and column_index < len(values[row_index]):
                existing = values[row_index][column_index]
            if existing not in (None, ""):
                row.append(existing)
            elif row_index == column_index:
                row.append(diagonal_fill)
            else:
                row.append(0.0)
        resized.append(row)
    return resized


def format_parameter_summary(model_spec: ModelSpec) -> str:
    """Return a compact description of the current parameter-table state."""
    theta_count = len(model_spec.theta_rows)
    omega_size = len(model_spec.omega_values)
    sigma_size = len(model_spec.sigma_values)
    return (
        f"{theta_count} THETA rows • OMEGA {omega_size}×{omega_size} • "
        f"SIGMA {sigma_size}×{sigma_size}"
    )


def format_model_draft_status(has_unsaved_changes: bool) -> str:
    """Return helper text describing whether the current model edits are saved."""
    return "Unsaved model changes" if has_unsaved_changes else ""


def latest_successful_fit_run(project: Workspace) -> RunRecord | None:
    """Return the newest successful fit run for the active scenario."""
    for run in reversed(project.runs):
        if run.workflow == "fit" and run.status == RunStatus.SUCCEEDED:
            return run
    return None


def recommend_model_next_action(
    project: Workspace,
    *,
    current_translation_result: ModelTranslationResult | None,
) -> tuple[str, str, str] | None:
    """Return the primary handoff CTA for the Model workflow."""
    if project.active_dataset is None:
        return (
            "Open Data",
            "data",
            "Load a dataset in the Data workflow before handing this model off to fitting.",
        )

    if current_translation_result is None or not current_translation_result.ok:
        return None

    # In control-stream mode the active project dataset is used for fitting regardless
    # of the $DATA path inside the file, so skip the path-match check.
    if current_translation_result.mode != ModelSpecMode.CONTROL_STREAM:
        active_dataset_path = (project.active_dataset.source_path or "").strip()
        translation_dataset_path = (current_translation_result.dataset_path or "").strip()
        if active_dataset_path and translation_dataset_path != active_dataset_path:
            return None

    if latest_successful_fit_run(project) is not None:
        return (
            "Open Results",
            "results",
            "A successful fit is already available. Review the latest outputs in Results.",
        )

    return ("Save model and open Fit", "fit", "Model is valid and dataset is ready — go to Fit.")


def default_model_spec(project: Workspace) -> ModelSpec:
    """Return the current model spec or a sensible default for the project."""
    if project.active_model_spec is not None:
        model_spec = ModelSpec.from_dict(project.active_model_spec.to_dict())
        if (
            not model_spec.dataset_path
            and project.active_dataset is not None
            and project.active_dataset.source_path
        ):
            model_spec.dataset_path = project.active_dataset.source_path
    else:
        model_spec = ModelSpec(
            problem_title=project.name,
            dataset_path=project.active_dataset.source_path if project.active_dataset else None,
            pk_code=DEFAULT_PK_CODE,
            error_code=DEFAULT_ERROR_CODE,
            theta_rows=_default_theta_rows(),
            omega_values=_default_omega_values(),
            sigma_values=_default_sigma_values(),
        )
    model_spec.theta_rows = normalize_theta_rows(model_spec.theta_rows)
    model_spec.omega_values = resize_square_matrix(
        [list(row) for row in model_spec.omega_values],
        len(model_spec.omega_values) or 1,
        diagonal_fill=0.1,
    )
    model_spec.sigma_values = resize_square_matrix(
        [list(row) for row in model_spec.sigma_values],
        len(model_spec.sigma_values) or 1,
        diagonal_fill=0.1,
    )
    return model_spec


def format_model_summary(model_spec: ModelSpec) -> str:
    """Return a compact description of the current model workspace state."""
    mode_label = "Builder" if model_spec.mode == ModelSpecMode.BUILDER else "Control stream"
    dataset_text = model_spec.dataset_path or "no dataset selected"
    title = model_spec.problem_title or "Untitled model"
    return (
        f"{title} — {mode_label} mode • ADVAN{model_spec.advan}/TRANS{model_spec.trans} "
        f"• method {model_spec.estimation.method} • dataset {dataset_text}"
    )


def format_translation_summary(result: ModelTranslationResult) -> str:
    """Return a compact translation preview for the current model state."""
    status = "Ready for engine translation" if result.ok else "Translation needs attention"
    method_text = result.estimation_method or "unspecified"
    if result.mode == ModelSpecMode.BUILDER:
        return (
            f"{status} — Builder mode • {result.theta_count} THETA • "
            f"{result.eta_count} ETA • {result.eps_count} EPS • method {method_text}"
        )
    title = result.problem_title or "Control stream"
    dataset_text = result.dataset_path or "dataset unresolved"
    return (
        f"{status} — {title} • {result.record_count} records • "
        f"method {method_text} • dataset {dataset_text}"
    )


def format_validation_issue(issue: ValidationIssue) -> str:
    """Render a validation issue for display in the model workflow."""
    field_suffix = f" [{issue.field_name}]" if issue.field_name else ""
    return f"{issue.severity.value.title()}{field_suffix}: {issue.message}"


def resolve_control_stream_dataset_path(
    control_stream: ControlStream, source_path: str
) -> str | None:
    """Resolve the control-stream $DATA path relative to the control-stream file."""
    data_filename = control_stream.data.filename if control_stream.data is not None else ""
    cleaned_filename = data_filename.strip()
    if not cleaned_filename:
        return None
    dataset_path = Path(cleaned_filename)
    if not dataset_path.is_absolute():
        dataset_path = Path(source_path).resolve().parent / dataset_path
    return str(dataset_path.resolve())


def load_control_stream_model_spec(path: str, *, base_spec: ModelSpec | None = None) -> ModelSpec:
    """Build a control-stream model spec from a NONMEM file on disk."""
    control_stream = ControlStream.from_file(path)
    model_spec = (
        ModelSpec.from_dict(base_spec.to_dict())
        if base_spec is not None
        else ModelSpec(
            pk_code=DEFAULT_PK_CODE,
            error_code=DEFAULT_ERROR_CODE,
            theta_rows=_default_theta_rows(),
            omega_values=_default_omega_values(),
            sigma_values=_default_sigma_values(),
        )
    )
    model_spec.mode = ModelSpecMode.CONTROL_STREAM
    model_spec.control_stream_text = control_stream.source_text
    if control_stream.problem is not None and control_stream.problem.title.strip():
        model_spec.problem_title = control_stream.problem.title.strip()
    resolved_dataset_path = resolve_control_stream_dataset_path(control_stream, path)
    if resolved_dataset_path is not None:
        model_spec.dataset_path = resolved_dataset_path
    if control_stream.subroutines is not None:
        model_spec.advan = control_stream.subroutines.advan
        model_spec.trans = control_stream.subroutines.trans
    if control_stream.estimation_records:
        model_spec.estimation.method = str(control_stream.estimation_records[0].method)
    model_spec.covariance.enabled = control_stream.covariance is not None

    # Populate parameter tables from parsed records
    theta_rows: list[dict[str, object]] = []
    for record in control_stream.theta_records:
        for i, spec in enumerate(record.specs, start=len(theta_rows) + 1):
            theta_rows.append(
                {
                    "label": spec.label or f"THETA{i}",
                    "lower": float(spec.lower) if spec.lower is not None else 0.0,
                    "init": float(spec.init),
                    "upper": float(spec.upper) if spec.upper is not None else float(spec.init) * 10,
                    "fixed": bool(spec.fixed),
                }
            )
    if theta_rows:
        model_spec.theta_rows = theta_rows

    omega_values: list[list[float]] = []
    for record in control_stream.omega_records:
        for spec in record.specs:
            if not spec.same:
                omega_values.append([float(v) for v in spec.values])
    if omega_values:
        model_spec.omega_values = omega_values

    sigma_values: list[list[float]] = []
    for record in control_stream.sigma_records:
        for spec in record.specs:
            sigma_values.append([float(v) for v in spec.values])
    if sigma_values:
        model_spec.sigma_values = sigma_values

    return model_spec


def load_control_stream_dataset_asset(
    data_service: DatasetService,
    model_spec: ModelSpec,
    *,
    display_name: str | None = None,
):
    options = DatasetImportOptions()
    input_columns: list[str] | None = None
    control_stream_text = model_spec.control_stream_text.strip()
    if control_stream_text:
        try:
            control_stream = ControlStream.from_string(control_stream_text)
        except ParseError:
            control_stream = None
        if control_stream is not None:
            if control_stream.data is not None:
                options.ignore_char = control_stream.data.ignore_char
            if control_stream.input is not None:
                input_columns = list(control_stream.input.columns)
    load_result = data_service.load_csv(
        model_spec.dataset_path,
        options=options,
        input_columns=input_columns,
    )
    if display_name and load_result.dataset_asset is not None:
        load_result.dataset_asset.display_name = display_name
    return load_result


def write_control_stream_text(path: str, text: str) -> None:
    """Persist authored control-stream text without rewriting its contents."""
    Path(path).write_text(text, encoding="utf-8")


def build_model_workflow(
    project: Workspace,
    project_service: ProjectService | None = None,
    data_service: DatasetService | None = None,
    example_catalog_service: ExampleCatalogService | None = None,
):
    """Build the first real Model workflow page."""
    qt_core, qt_gui, qt_widgets = load_qt_modules()
    project_service = project_service or ProjectService()
    repo_root = Path(__file__).resolve().parents[3]
    data_service = data_service or DatasetService()
    example_catalog_service = example_catalog_service or ExampleCatalogService(
        catalog_root=repo_root / "examples" / "catalog",
        shared_data_root=repo_root / "examples" / "shared_data",
    )
    translation_service = ModelTranslationService()
    model_spec = default_model_spec(project)
    translation_result = translation_service.translate(model_spec)

    root, _, layout, scroll_area = build_scrollable_page(
        qt_widgets, root_object_name="model-workflow"
    )

    title_label = qt_widgets.QLabel("Model workflow")
    title_font = title_label.font()
    title_font.setPointSize(title_font.pointSize() + 4)
    title_font.setBold(True)
    title_label.setFont(title_font)

    _hint_prefs = load_gui_preferences()

    def _save_dismissed_hint_model():
        save_gui_preferences(with_dismissed_hint(load_gui_preferences(), "hint_model"))

    hint_widget, _ = build_dismissible_hint(
        "Define GUI-side model state for builder and control-stream modes. "
        "Use Validate translation to preview engine-facing state before fitting.",
        dismissed="hint_model" in _hint_prefs.dismissed_hints,
        on_dismiss=_save_dismissed_hint_model,
    )

    form_layout = qt_widgets.QFormLayout()
    problem_title_input = qt_widgets.QLineEdit(model_spec.problem_title)
    problem_title_input.setObjectName("model-problem-title")

    mode_radio_builder = qt_widgets.QRadioButton("Builder")
    mode_radio_builder.setObjectName("model-mode-radio-builder")
    mode_radio_ctl = qt_widgets.QRadioButton("Control stream (.ctl)")
    mode_radio_ctl.setObjectName("model-mode-radio-ctl")
    mode_radio_group = qt_widgets.QButtonGroup()
    mode_radio_group.addButton(mode_radio_builder, 0)
    mode_radio_group.addButton(mode_radio_ctl, 1)
    mode_radio_row = qt_widgets.QHBoxLayout()
    mode_radio_row.setContentsMargins(0, 0, 0, 0)
    mode_radio_row.addWidget(mode_radio_builder)
    mode_radio_row.addWidget(mode_radio_ctl)
    mode_radio_row.addStretch(1)
    mode_radio_widget = qt_widgets.QWidget()
    mode_radio_widget.setLayout(mode_radio_row)
    if model_spec.mode == ModelSpecMode.BUILDER:
        mode_radio_builder.setChecked(True)
    else:
        mode_radio_ctl.setChecked(True)

    dataset_row = qt_widgets.QHBoxLayout()
    _active_source = project.active_dataset.source_path if project.active_dataset else None
    _initial_dataset_path = model_spec.dataset_path or _active_source or ""
    _was_autofilled = not model_spec.dataset_path and bool(_active_source)
    dataset_path_input = qt_widgets.QLineEdit(_initial_dataset_path)
    dataset_path_input.setObjectName("model-dataset-path")
    use_active_dataset_button = qt_widgets.QPushButton("Use active dataset")
    use_active_dataset_button.setObjectName("model-use-active-dataset")
    dataset_autofill_hint = qt_widgets.QLabel("Auto-filled from active dataset")
    dataset_autofill_hint.setObjectName("model-dataset-autofill-hint")
    _hint_style = "font-style: italic; color: #888; font-size: 11px;"
    dataset_autofill_hint.setStyleSheet(_hint_style)
    dataset_autofill_hint.setVisible(_was_autofilled)
    dataset_row.addWidget(dataset_path_input, 1)
    dataset_row.addWidget(use_active_dataset_button)

    model_combo = qt_widgets.QComboBox()
    model_combo.setObjectName("model-named-model-combo")
    model_combo.addItems(_NAMED_MODEL_LABELS)
    model_combo.setCurrentIndex(
        _advan_trans_to_named_model_index(model_spec.advan, model_spec.trans)
    )

    custom_subroutine_row = qt_widgets.QWidget()
    custom_subroutine_layout = qt_widgets.QHBoxLayout(custom_subroutine_row)
    custom_subroutine_layout.setContentsMargins(0, 0, 0, 0)
    advan_spin = qt_widgets.QSpinBox()
    advan_spin.setObjectName("model-advan-spin")
    advan_spin.setRange(1, 99)
    advan_spin.setValue(model_spec.advan)
    trans_spin = qt_widgets.QSpinBox()
    trans_spin.setObjectName("model-trans-spin")
    trans_spin.setRange(1, 99)
    trans_spin.setValue(model_spec.trans)
    custom_subroutine_layout.addWidget(qt_widgets.QLabel("ADVAN"))
    custom_subroutine_layout.addWidget(advan_spin)
    custom_subroutine_layout.addSpacing(12)
    custom_subroutine_layout.addWidget(qt_widgets.QLabel("TRANS"))
    custom_subroutine_layout.addWidget(trans_spin)
    custom_subroutine_layout.addStretch(1)
    custom_subroutine_row.setVisible(model_combo.currentText().startswith("Custom"))

    diagram_widget = build_model_diagram_widget(model_spec.advan, model_spec.trans)

    subroutine_row = qt_widgets.QVBoxLayout()
    subroutine_row.setContentsMargins(0, 0, 0, 0)
    subroutine_row.setSpacing(4)
    subroutine_row.addWidget(model_combo)
    subroutine_row.addWidget(custom_subroutine_row)
    subroutine_row.addWidget(diagram_widget)

    def _make_help_button(title: str, text: str) -> qt_widgets.QToolButton:
        btn = qt_widgets.QToolButton()
        btn.setText("?")
        btn.setFixedSize(20, 20)
        btn.setStyleSheet("QToolButton { border-radius: 4px; padding: 1px; font-size: 10px; }")
        btn.setToolTip(f"<b>{title}</b><hr>{text.replace(chr(10), '<br>')}")
        btn.clicked.connect(lambda: qt_widgets.QMessageBox.information(None, title, text))
        return btn

    estimation_row = qt_widgets.QHBoxLayout()
    estimation_combo = qt_widgets.QComboBox()
    estimation_combo.setObjectName("model-estimation-combo")
    for _code, _label in _ESTIMATION_METHODS_DISPLAY:
        estimation_combo.addItem(_label, userData=_code)
    _current_method_index = next(
        (
            i
            for i, (c, _) in enumerate(_ESTIMATION_METHODS_DISPLAY)
            if c == model_spec.estimation.method
        ),
        0,
    )
    estimation_combo.setCurrentIndex(_current_method_index)
    covariance_checkbox = qt_widgets.QCheckBox("Run covariance step")
    covariance_checkbox.setObjectName("model-covariance-checkbox")
    covariance_checkbox.setChecked(model_spec.covariance.enabled)
    _COV_MATRIX_TOOLTIPS = {
        "SR": "Sandwich (robust) estimator — recommended for most models",
        "R": "Fisher information matrix",
        "S": "Cross-product approximation",
    }
    covariance_matrix_combo = qt_widgets.QComboBox()
    covariance_matrix_combo.setObjectName("model-covariance-matrix-combo")
    for _cov_key in ("SR", "R", "S"):
        covariance_matrix_combo.addItem(_cov_key)
        covariance_matrix_combo.setItemData(
            covariance_matrix_combo.count() - 1,
            _COV_MATRIX_TOOLTIPS[_cov_key],
            qt_core.Qt.ItemDataRole.ToolTipRole,
        )
    covariance_matrix_combo.setCurrentText(model_spec.covariance.matrix)
    covariance_matrix_combo.setToolTip(_COV_MATRIX_TOOLTIPS.get(model_spec.covariance.matrix, ""))
    covariance_matrix_combo.currentTextChanged.connect(
        lambda text: covariance_matrix_combo.setToolTip(_COV_MATRIX_TOOLTIPS.get(text, ""))
    )
    # Nonparametric base method sub-selector (P3-D)
    nonparam_base_label = qt_widgets.QLabel("Base method:")
    nonparam_base_label.setObjectName("model-nonparam-base-method-label")
    nonparam_base_combo = qt_widgets.QComboBox()
    nonparam_base_combo.setObjectName("model-nonparam-base-method-combo")
    for _code, _label in _NP_BASE_METHODS:
        nonparam_base_combo.addItem(_label, userData=_code)
    _np_base_default = str(model_spec.estimation.options.get("base_method", "FOCE"))
    _np_base_idx = max(0, nonparam_base_combo.findData(_np_base_default))
    nonparam_base_combo.setCurrentIndex(_np_base_idx)

    # BLQ method selector (P2-C)
    blq_method_label = qt_widgets.QLabel("BLQ:")
    blq_method_label.setObjectName("model-blq-method-label")
    blq_method_combo = qt_widgets.QComboBox()
    blq_method_combo.setObjectName("model-blq-method-combo")
    blq_method_combo.addItem("M1 — Ignore BLQ", userData="M1")
    blq_method_combo.addItem("M3 — Censored likelihood", userData="M3")
    blq_method_combo.setToolTip(
        "BLQ handling method.\n"
        "M1: exclude below-LOQ observations (default).\n"
        "M3: replace BLQ likelihood with Φ((LOQ−IPRED)/σ). "
        "Requires an LLOQ column in the dataset or a scalar LOQ set in the Data workflow."
    )
    _blq_init = str(model_spec.estimation.options.get("blq_method", "M1"))
    _blq_idx = max(0, blq_method_combo.findData(_blq_init))
    blq_method_combo.setCurrentIndex(_blq_idx)

    estimation_row.addWidget(estimation_combo)
    estimation_row.addWidget(_make_help_button("Estimation Methods", _ESTIMATION_HELP_TEXT))
    estimation_row.addWidget(nonparam_base_label)
    estimation_row.addWidget(nonparam_base_combo)
    estimation_row.addWidget(blq_method_label)
    estimation_row.addWidget(blq_method_combo)
    estimation_row.addWidget(covariance_checkbox)
    estimation_row.addWidget(covariance_matrix_combo)
    estimation_row.addWidget(_make_help_button("Covariance Step", _COV_HELP_TEXT))
    estimation_row.addStretch(1)

    # --- Estimation options grid (advanced controls for gradient methods) ---
    est_options_row = qt_widgets.QGridLayout()
    est_options_row.setContentsMargins(0, 0, 0, 0)
    est_options_row.setHorizontalSpacing(12)
    est_options_row.setVerticalSpacing(8)

    maxeval_label = qt_widgets.QLabel("Max eval:")
    maxeval_spin = qt_widgets.QSpinBox()
    maxeval_spin.setObjectName("model-maxeval-spin")
    maxeval_spin.setRange(1, 100000)
    maxeval_spin.setValue(
        int(model_spec.estimation.options.get("maxeval", _default_maxeval(model_spec.estimation.method)))
    )
    maxeval_spin.setToolTip("Outer optimizer evaluation budget for the selected method.")

    nstarts_label = qt_widgets.QLabel("Multi-start runs:")
    nstarts_spin = qt_widgets.QSpinBox()
    nstarts_spin.setObjectName("model-nstarts-spin")
    nstarts_spin.setRange(1, 20)
    nstarts_spin.setValue(int(model_spec.estimation.options.get("n_starts", 1)))
    nstarts_spin.setToolTip(
        "Number of independent optimizer restarts (best OFV kept). 1 = no multi-start."
    )
    tight_gtol_checkbox = qt_widgets.QCheckBox("Tight gradient (gtol=1e-6)")
    tight_gtol_checkbox.setObjectName("model-tight-gtol-checkbox")
    tight_gtol_checkbox.setChecked(float(model_spec.estimation.options.get("gtol", 1e-5)) < 1e-5)
    tight_gtol_checkbox.setToolTip(
        "Tighten outer convergence criterion — recommended for covariate power-law models."
    )

    outer_optimizer_label = qt_widgets.QLabel("Outer optimizer:")
    outer_optimizer_combo = qt_widgets.QComboBox()
    outer_optimizer_combo.setObjectName("model-outer-optimizer-combo")
    for optimizer in _OUTER_OPTIMIZERS:
        outer_optimizer_combo.addItem(optimizer, userData=optimizer)

    fallback_optimizer_label = qt_widgets.QLabel("Fallback optimizer:")
    fallback_optimizer_combo = qt_widgets.QComboBox()
    fallback_optimizer_combo.setObjectName("model-fallback-optimizer-combo")
    for label, value in _FALLBACK_OPTIMIZER_CHOICES:
        fallback_optimizer_combo.addItem(label, userData=value)

    fallback_maxeval_label = qt_widgets.QLabel("Fallback max eval:")
    fallback_maxeval_spin = qt_widgets.QSpinBox()
    fallback_maxeval_spin.setObjectName("model-fallback-maxeval-spin")
    fallback_maxeval_spin.setRange(1, 10000)

    retain_best_checkbox = qt_widgets.QCheckBox("Retain best iterate")
    retain_best_checkbox.setObjectName("model-retain-best-checkbox")
    retain_best_checkbox.setToolTip(
        "Keep the best OFV point visited even if the terminal optimizer iterate is worse."
    )

    retry_on_abnormal_checkbox = qt_widgets.QCheckBox("Retry on abnormal termination")
    retry_on_abnormal_checkbox.setObjectName("model-retry-on-abnormal-checkbox")
    retry_on_abnormal_checkbox.setToolTip(
        "Rerun FOCEI/Laplacian from structured alternate starts if the main run terminates abnormally."
    )

    retry_omega_scales_label = qt_widgets.QLabel("Retry OMEGA scales:")
    retry_omega_scales_input = qt_widgets.QLineEdit()
    retry_omega_scales_input.setObjectName("model-retry-omega-scales-input")
    retry_omega_scales_input.setPlaceholderText("0.5, 0.25, 0.1")
    retry_omega_scales_input.setToolTip(
        "Comma-separated OMEGA scaling factors used for structured retries."
    )

    help_button = _make_help_button("Estimation Options", _EST_OPTIONS_HELP_TEXT)
    est_options_row.addWidget(maxeval_label, 0, 0)
    est_options_row.addWidget(maxeval_spin, 0, 1)
    est_options_row.addWidget(nstarts_label, 0, 2)
    est_options_row.addWidget(nstarts_spin, 0, 3)
    est_options_row.addWidget(tight_gtol_checkbox, 0, 4)
    est_options_row.addWidget(help_button, 0, 5)
    est_options_row.addWidget(outer_optimizer_label, 1, 0)
    est_options_row.addWidget(outer_optimizer_combo, 1, 1)
    est_options_row.addWidget(fallback_optimizer_label, 1, 2)
    est_options_row.addWidget(fallback_optimizer_combo, 1, 3)
    est_options_row.addWidget(fallback_maxeval_label, 1, 4)
    est_options_row.addWidget(fallback_maxeval_spin, 1, 5)
    est_options_row.addWidget(retain_best_checkbox, 2, 0, 1, 2)
    est_options_row.addWidget(retry_on_abnormal_checkbox, 2, 2, 1, 2)
    est_options_row.addWidget(retry_omega_scales_label, 2, 4)
    est_options_row.addWidget(retry_omega_scales_input, 2, 5)
    est_options_widget = qt_widgets.QWidget()
    est_options_widget.setObjectName("model-est-options-widget")
    est_options_widget.setLayout(est_options_row)
    est_options_widget.setVisible(estimation_combo.currentData() in _GRADIENT_METHODS)

    def _apply_estimation_option_defaults_for_method(method: str) -> None:
        outer_index = max(0, outer_optimizer_combo.findData(_default_outer_optimizer(method)))
        outer_optimizer_combo.setCurrentIndex(outer_index)
        fallback_index = max(
            0,
            fallback_optimizer_combo.findData(_default_fallback_optimizer(method)),
        )
        fallback_optimizer_combo.setCurrentIndex(fallback_index)
        fallback_maxeval_spin.setValue(_default_fallback_maxeval(method))
        retain_best_checkbox.setChecked(_default_retain_best_iterate(method))
        retry_on_abnormal_checkbox.setChecked(_default_retry_on_abnormal(method))
        retry_omega_scales_input.setText(_default_retry_omega_scales(method))
        maxeval_spin.setValue(_default_maxeval(method))

    def _load_estimation_option_values_from_spec() -> None:
        method = model_spec.estimation.method
        options = dict(model_spec.estimation.options)
        maxeval_spin.setValue(int(options.get("maxeval", _default_maxeval(method))))
        outer_index = max(
            0,
            outer_optimizer_combo.findData(options.get("outer_optimizer", _default_outer_optimizer(method))),
        )
        outer_optimizer_combo.setCurrentIndex(outer_index)
        fallback_index = max(
            0,
            fallback_optimizer_combo.findData(
                options.get("outer_fallback_optimizer", _default_fallback_optimizer(method))
            ),
        )
        fallback_optimizer_combo.setCurrentIndex(fallback_index)
        fallback_maxeval_spin.setValue(
            int(options.get("outer_fallback_maxeval", _default_fallback_maxeval(method)))
        )
        retain_best_checkbox.setChecked(
            bool(options.get("retain_best_iterate", _default_retain_best_iterate(method)))
        )
        retry_on_abnormal_checkbox.setChecked(
            bool(options.get("retry_on_abnormal", _default_retry_on_abnormal(method)))
        )
        retry_omega_scales = options.get("retry_omega_scales", _default_retry_omega_scales(method))
        if isinstance(retry_omega_scales, (list, tuple)):
            retry_omega_scales_text = ", ".join(str(value) for value in retry_omega_scales)
        else:
            retry_omega_scales_text = str(retry_omega_scales)
        retry_omega_scales_input.setText(retry_omega_scales_text)
        np_base = str(options.get("base_method", "FOCE"))
        np_base_idx = max(0, nonparam_base_combo.findData(np_base))
        nonparam_base_combo.setCurrentIndex(np_base_idx)
        blq = str(options.get("blq_method", "M1"))
        blq_idx = max(0, blq_method_combo.findData(blq))
        blq_method_combo.setCurrentIndex(blq_idx)

    def _refresh_estimation_option_affordances() -> None:
        method = str(estimation_combo.currentData())
        advanced_enabled = method in _GRADIENT_METHODS
        interaction_enabled = method in _INTERACTION_METHODS
        np_enabled = method in _NP_METHODS
        est_options_widget.setVisible(advanced_enabled and not np_enabled)
        nonparam_base_label.setVisible(np_enabled)
        nonparam_base_combo.setVisible(np_enabled)
        for widget in (
            maxeval_label,
            nstarts_label,
            outer_optimizer_label,
            fallback_optimizer_label,
            fallback_maxeval_label,
        ):
            widget.setEnabled(advanced_enabled)
        for widget in (
            maxeval_spin,
            nstarts_spin,
            tight_gtol_checkbox,
            outer_optimizer_combo,
            fallback_optimizer_combo,
            fallback_maxeval_spin,
            retain_best_checkbox,
        ):
            widget.setEnabled(advanced_enabled)
        for widget in (
            retry_on_abnormal_checkbox,
            retry_omega_scales_input,
        ):
            widget.setEnabled(interaction_enabled)
        retry_omega_scales_label.setEnabled(interaction_enabled)

    _load_estimation_option_values_from_spec()
    _refresh_estimation_option_affordances()

    subroutine_container = qt_widgets.QWidget()
    subroutine_container.setLayout(subroutine_row)
    subroutine_header_row = qt_widgets.QHBoxLayout()
    subroutine_header_row.setContentsMargins(0, 0, 0, 0)
    subroutine_header_row.addWidget(subroutine_container, 1)
    subroutine_header_row.addWidget(_make_help_button("PK Model Selection", _ADVAN_HELP_TEXT))

    form_layout.addRow("Problem title", problem_title_input)
    form_layout.addRow("Mode", mode_radio_widget)
    form_layout.addRow("Dataset", dataset_row)
    form_layout.addRow("", dataset_autofill_hint)
    form_layout.addRow("Model", subroutine_header_row)
    form_layout.addRow("Estimation", estimation_row)
    form_layout.addRow("", est_options_widget)

    editor_stack = qt_widgets.QStackedWidget()
    editor_stack.setObjectName("model-editor-stack")

    builder_page = qt_widgets.QWidget()
    builder_layout = qt_widgets.QVBoxLayout(builder_page)
    pk_edit = qt_widgets.QPlainTextEdit(model_spec.pk_code)
    pk_edit.setObjectName("model-pk-code")
    pk_edit.setPlaceholderText("$PK code, e.g. CL = THETA(1)")
    error_edit = qt_widgets.QPlainTextEdit(model_spec.error_code)
    error_edit.setObjectName("model-error-code")
    error_edit.setPlaceholderText("$ERROR code")
    des_edit = qt_widgets.QPlainTextEdit(model_spec.des_code)
    des_edit.setObjectName("model-des-code")
    des_edit.setPlaceholderText("$DES code for ODE models")
    NmtranHighlighter.attach(pk_edit)
    NmtranHighlighter.attach(error_edit)
    NmtranHighlighter.attach(des_edit)
    builder_layout.addWidget(qt_widgets.QLabel("$PK"))
    builder_layout.addWidget(pk_edit, 2)
    builder_layout.addWidget(qt_widgets.QLabel("$ERROR"))
    builder_layout.addWidget(error_edit, 2)
    builder_layout.addWidget(qt_widgets.QLabel("$DES"))
    builder_layout.addWidget(des_edit, 2)

    control_stream_page = qt_widgets.QWidget()
    control_stream_layout = qt_widgets.QVBoxLayout(control_stream_page)
    control_stream_button_row = qt_widgets.QHBoxLayout()
    open_control_stream_button = qt_widgets.QPushButton("Open NONMEM file…")
    open_control_stream_button.setObjectName("model-control-stream-open-button")
    open_control_stream_button.setToolTip(
        "Open a NONMEM control stream (.ctl/.mod/.txt).\n"
        "If the file's $DATA path resolves to an existing CSV, that dataset is\n"
        "automatically loaded on the Data screen, replacing any current selection."
    )
    save_control_stream_button = qt_widgets.QPushButton("Save NONMEM file…")
    save_control_stream_button.setObjectName("model-control-stream-save-button")
    control_stream_button_row.addWidget(open_control_stream_button)
    control_stream_button_row.addWidget(save_control_stream_button)
    control_stream_button_row.addStretch(1)
    control_stream_edit = qt_widgets.QPlainTextEdit(model_spec.control_stream_text)
    control_stream_edit.setObjectName("model-control-stream-text")
    control_stream_edit.setPlaceholderText("Paste or author a NONMEM control stream here.")
    NmtranHighlighter.attach(control_stream_edit)
    control_stream_layout.addLayout(control_stream_button_row)
    ctl_dataset_note = qt_widgets.QLabel(
        "Dataset: opening or loading a control stream automatically loads its $DATA file "
        "on the Data screen, replacing any current selection. "
        "You can then re-select a different dataset there to override $DATA for fitting."
    )
    ctl_dataset_note.setObjectName("model-ctl-dataset-note")
    ctl_dataset_note.setWordWrap(True)
    ctl_dataset_note.setStyleSheet("font-size: 11px; color: #64748b; font-style: italic;")
    control_stream_layout.addWidget(ctl_dataset_note)
    control_stream_layout.addWidget(qt_widgets.QLabel("Control stream"))
    control_stream_layout.addWidget(control_stream_edit, 1)

    # Curated example control streams — kept outside the editor_stack so they
    # are always reachable without being squeezed by the text editor's stretch.
    ctl_examples = example_catalog_service.list_control_stream_examples()
    ctl_examples_by_id = {entry.manifest.id: entry for entry in ctl_examples}
    if ctl_examples:
        example_group = qt_widgets.QGroupBox("Or load a curated example control stream")
        example_group_layout = qt_widgets.QVBoxLayout(example_group)
        example_row = qt_widgets.QHBoxLayout()
        example_selector = qt_widgets.QComboBox()
        example_selector.setObjectName("model-ctl-example-selector")
        example_selector.addItem("Choose a curated example control stream…", None)
        for entry in ctl_examples:
            example_selector.addItem(_format_catalog_control_stream_label(entry), entry.manifest.id)
        load_example_button = qt_widgets.QPushButton("Load example")
        load_example_button.setObjectName("model-ctl-example-load-button")
        load_example_button.setEnabled(False)
        load_example_button.setToolTip(
            "Load the selected curated example control stream.\n"
            "If the manifest declares a dataset, it is automatically loaded on the\n"
            "Data screen, replacing any current dataset selection."
        )
        example_row.addWidget(example_selector, 1)
        example_row.addWidget(load_example_button)
        example_details_label = qt_widgets.QLabel("")
        example_details_label.setObjectName("model-ctl-example-details")
        example_details_label.setOpenExternalLinks(False)
        example_details_label.setTextInteractionFlags(
            qt_core.Qt.TextInteractionFlag.TextBrowserInteraction
        )
        example_details_label.setWordWrap(True)
        example_group_layout.addLayout(example_row)
        example_group_layout.addWidget(example_details_label)
    else:
        example_group = None  # type: ignore[assignment]
        example_selector = None  # type: ignore[assignment]
        load_example_button = None  # type: ignore[assignment]
        example_details_label = None  # type: ignore[assignment]

    editor_stack.addWidget(builder_page)
    editor_stack.addWidget(control_stream_page)

    summary_label = qt_widgets.QLabel(format_model_summary(model_spec))
    summary_label.setObjectName("model-summary-label")
    summary_label.setWordWrap(True)
    parameter_summary_label = qt_widgets.QLabel(format_parameter_summary(model_spec))
    parameter_summary_label.setObjectName("model-parameter-summary")
    parameter_summary_label.setWordWrap(True)
    translation_label = qt_widgets.QLabel(format_translation_summary(translation_result))
    translation_label.setObjectName("model-translation-summary")
    translation_label.setWordWrap(True)
    validation_list = qt_widgets.QListWidget()
    validation_list.setObjectName("model-validation-list")

    theta_table = qt_widgets.QTableWidget()
    theta_table.setObjectName("model-theta-table")
    theta_table.setColumnCount(len(THETA_TABLE_HEADERS))
    theta_table.setHorizontalHeaderLabels(list(THETA_TABLE_HEADERS))
    configure_resizable_table_columns(theta_table, qt_widgets)
    theta_heading = qt_widgets.QLabel("THETA parameter table")

    theta_button_row = qt_widgets.QHBoxLayout()
    add_theta_button = qt_widgets.QPushButton("Add THETA row")
    add_theta_button.setObjectName("model-add-theta-row")
    remove_theta_button = qt_widgets.QPushButton("Remove THETA row")
    remove_theta_button.setObjectName("model-remove-theta-row")
    suggest_theta_button = qt_widgets.QPushButton("Suggest typical values")
    suggest_theta_button.setObjectName("model-suggest-theta-button")
    suggest_theta_button.setToolTip(
        "Pre-fill THETA and OMEGA with typical starting values for the selected model structure"
    )
    theta_button_row.addWidget(add_theta_button)
    theta_button_row.addWidget(remove_theta_button)
    theta_button_row.addWidget(suggest_theta_button)
    theta_button_row.addStretch(1)

    omega_table = qt_widgets.QTableWidget()
    omega_table.setObjectName("model-omega-table")
    configure_resizable_table_columns(omega_table, qt_widgets)
    add_omega_button = qt_widgets.QPushButton("Add ETA")
    add_omega_button.setObjectName("model-add-omega-button")
    remove_omega_button = qt_widgets.QPushButton("Remove last ETA")
    remove_omega_button.setObjectName("model-remove-omega-button")

    sigma_table = qt_widgets.QTableWidget()
    sigma_table.setObjectName("model-sigma-table")
    configure_resizable_table_columns(sigma_table, qt_widgets)
    add_sigma_button = qt_widgets.QPushButton("Add EPS")
    add_sigma_button.setObjectName("model-add-sigma-button")
    remove_sigma_button = qt_widgets.QPushButton("Remove last EPS")
    remove_sigma_button.setObjectName("model-remove-sigma-button")

    content_row_widget = qt_widgets.QSplitter(root)
    content_row_widget.setObjectName("model-content-row")
    content_row_widget.setChildrenCollapsible(False)
    content_row_widget.setHandleWidth(8)

    configuration_panel = qt_widgets.QWidget(content_row_widget)
    configuration_panel.setObjectName("model-configuration-panel")
    configuration_layout = qt_widgets.QVBoxLayout(configuration_panel)
    configuration_layout.setContentsMargins(12, 12, 12, 12)
    configuration_layout.setSpacing(8)
    configuration_layout.addLayout(form_layout)
    configuration_layout.addWidget(summary_label)
    configuration_layout.addWidget(parameter_summary_label)
    configuration_layout.addWidget(theta_heading)
    configuration_layout.addWidget(theta_table)
    configuration_layout.addLayout(theta_button_row)

    omega_btn_row = qt_widgets.QHBoxLayout()
    omega_btn_row.addWidget(qt_widgets.QLabel("OMEGA  (inter-individual variability)"))
    omega_btn_row.addWidget(
        _make_help_button("OMEGA — Inter-individual Variability", _OMEGA_HELP_TEXT)
    )
    omega_btn_row.addStretch(1)
    omega_btn_row.addWidget(add_omega_button)
    omega_btn_row.addWidget(remove_omega_button)
    configuration_layout.addLayout(omega_btn_row)
    configuration_layout.addWidget(omega_table)

    sigma_btn_row = qt_widgets.QHBoxLayout()
    sigma_btn_row.addWidget(qt_widgets.QLabel("SIGMA  (residual error)"))
    sigma_btn_row.addWidget(_make_help_button("SIGMA — Residual Error", _SIGMA_HELP_TEXT))
    sigma_btn_row.addStretch(1)
    sigma_btn_row.addWidget(add_sigma_button)
    sigma_btn_row.addWidget(remove_sigma_button)
    configuration_layout.addLayout(sigma_btn_row)
    configuration_layout.addWidget(sigma_table)

    translation_panel = qt_widgets.QWidget(content_row_widget)
    translation_panel.setObjectName("model-translation-panel")
    translation_layout = qt_widgets.QVBoxLayout(translation_panel)
    translation_layout.setContentsMargins(12, 12, 12, 12)
    translation_layout.setSpacing(8)
    translation_layout.addWidget(translation_label)
    translation_layout.addWidget(validation_list)
    translation_layout.addWidget(editor_stack, 1)
    if example_group is not None:
        translation_layout.addWidget(example_group)

    content_row_widget.addWidget(configuration_panel)
    content_row_widget.addWidget(translation_panel)
    content_row_widget.setStretchFactor(0, 3)
    content_row_widget.setStretchFactor(1, 2)
    content_row_widget.setSizes([600, 400])

    next_action_label = qt_widgets.QLabel("")
    next_action_label.setObjectName("model-next-action-label")
    next_action_label.setWordWrap(True)
    next_action_label.setVisible(False)
    next_action_button = qt_widgets.QPushButton("")
    next_action_button.setObjectName("model-next-action-button")
    next_action_button.setProperty("primaryAction", True)
    next_action_button.setMinimumHeight(36)
    next_action_button.setVisible(False)

    layout.addWidget(title_label)
    layout.addWidget(hint_widget)
    layout.addWidget(content_row_widget, 1)
    layout.addWidget(next_action_label)
    layout.addWidget(next_action_button)

    _apply_responsive_layout = install_responsive_splitters(
        root,
        breakpoint=MODEL_RESPONSIVE_LAYOUT_BREAKPOINT,
        width_provider=lambda: (
            (root.parentWidget().width() if root.parentWidget() is not None else 0) or root.width()
        ),
        splitters=(content_row_widget,),
    )

    last_synced_project_payload: dict[str, object] | None = None
    last_synced_editor_state: dict[str, object] | None = None
    last_reported_dirty_state = False
    next_action_target = [""]
    # H-11: track the last confirmed mode so we can revert to it if the user
    # cancels a mode-switch when unsaved changes exist.
    last_confirmed_mode: list[ModelSpecMode] = [ModelSpecMode.BUILDER]

    def _current_mode() -> ModelSpecMode:
        return ModelSpecMode.CONTROL_STREAM if mode_radio_ctl.isChecked() else ModelSpecMode.BUILDER

    def _sync_mode() -> None:
        is_cs = _current_mode() == ModelSpecMode.CONTROL_STREAM
        editor_stack.setCurrentIndex(1 if is_cs else 0)
        if example_group is not None:
            example_group.setVisible(is_cs)

    def _set_table_item(table, row_index: int, column_index: int, value: object) -> None:
        item = table.item(row_index, column_index)
        if item is None:
            item = qt_widgets.QTableWidgetItem()
            table.setItem(row_index, column_index, item)
        item.setText("" if value is None else str(value))

    def _cell_text(table, row_index: int, column_index: int) -> str:
        item = table.item(row_index, column_index)
        return item.text().strip() if item is not None else ""

    def _populate_theta_table(rows: list[dict[str, object]]) -> None:
        normalized_rows = normalize_theta_rows(rows)
        theta_table.setRowCount(len(normalized_rows))
        for row_index, row in enumerate(normalized_rows):
            _set_table_item(theta_table, row_index, 0, row.get("label") or "")
            _set_table_item(theta_table, row_index, 1, row.get("lower"))
            _set_table_item(theta_table, row_index, 2, row.get("init"))
            _set_table_item(theta_table, row_index, 3, row.get("upper"))
            _set_table_item(theta_table, row_index, 4, "true" if row.get("fixed") else "false")

    def _populate_matrix_table(table, values: list[list[object]], *, diagonal_fill: float) -> None:
        normalized_values = resize_square_matrix(
            values, len(values) or 1, diagonal_fill=diagonal_fill
        )
        table.setRowCount(len(normalized_values))
        table.setColumnCount(len(normalized_values))
        for row_index, row in enumerate(normalized_values):
            for column_index, value in enumerate(row):
                _set_table_item(table, row_index, column_index, value)

    def _read_theta_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row_index in range(theta_table.rowCount()):
            label = _cell_text(theta_table, row_index, 0) or None
            lower = _cell_text(theta_table, row_index, 1) or None
            init = _cell_text(theta_table, row_index, 2)
            upper = _cell_text(theta_table, row_index, 3) or None
            fixed_text = _cell_text(theta_table, row_index, 4).lower()
            rows.append(
                {
                    "label": label,
                    "lower": lower,
                    "init": init,
                    "upper": upper,
                    "fixed": fixed_text if fixed_text else False,
                }
            )
        return rows or [default_theta_row(1)]

    def _read_matrix_table(table) -> list[list[object]]:
        rows: list[list[object]] = []
        for row_index in range(table.rowCount()):
            rows.append(
                [
                    _cell_text(table, row_index, column_index)
                    for column_index in range(table.columnCount())
                ]
            )
        return rows

    def _current_editor_state() -> dict[str, object]:
        return {
            "problem_title": problem_title_input.text(),
            "mode": _current_mode().value,
            "dataset_path": dataset_path_input.text(),
            "advan": advan_spin.value(),
            "trans": trans_spin.value(),
            "estimation_method": estimation_combo.currentData(),
            "covariance_enabled": covariance_checkbox.isChecked(),
            "covariance_matrix": covariance_matrix_combo.currentText(),
            "pk_code": pk_edit.toPlainText(),
            "error_code": error_edit.toPlainText(),
            "des_code": des_edit.toPlainText(),
            "control_stream_text": control_stream_edit.toPlainText(),
            "theta_rows": _read_theta_rows(),
            "omega_values": _read_matrix_table(omega_table),
            "sigma_values": _read_matrix_table(sigma_table),
            "maxeval": maxeval_spin.value(),
            "n_starts": nstarts_spin.value(),
            "gtol_tight": tight_gtol_checkbox.isChecked(),
            "outer_optimizer": outer_optimizer_combo.currentData(),
            "outer_fallback_optimizer": fallback_optimizer_combo.currentData(),
            "outer_fallback_maxeval": fallback_maxeval_spin.value(),
            "retain_best_iterate": retain_best_checkbox.isChecked(),
            "retry_on_abnormal": retry_on_abnormal_checkbox.isChecked(),
            "retry_omega_scales": retry_omega_scales_input.text(),
        }

    def _project_model_payload() -> dict[str, object]:
        return default_model_spec(project).to_dict()

    def _collect_model_spec() -> ModelSpec:
        retry_omega_scales = _parse_retry_omega_scales(retry_omega_scales_input.text())
        return ModelSpec(
            mode=_current_mode(),
            problem_title=problem_title_input.text().strip(),
            dataset_path=dataset_path_input.text().strip() or None,
            control_stream_text=control_stream_edit.toPlainText(),
            advan=advan_spin.value(),
            trans=trans_spin.value(),
            pk_code=pk_edit.toPlainText(),
            error_code=error_edit.toPlainText(),
            des_code=des_edit.toPlainText(),
            theta_rows=_read_theta_rows(),
            omega_values=_read_matrix_table(omega_table),
            sigma_values=_read_matrix_table(sigma_table),
            estimation=EstimationConfig(
                method=estimation_combo.currentData(),
                options={
                    "maxeval": maxeval_spin.value(),
                    "n_starts": nstarts_spin.value(),
                    "gtol": 1e-6 if tight_gtol_checkbox.isChecked() else 1e-5,
                    "outer_optimizer": outer_optimizer_combo.currentData(),
                    "outer_fallback_optimizer": fallback_optimizer_combo.currentData(),
                    "outer_fallback_maxeval": fallback_maxeval_spin.value(),
                    "retain_best_iterate": retain_best_checkbox.isChecked(),
                    "retry_on_abnormal": retry_on_abnormal_checkbox.isChecked(),
                    "retry_omega_scales": retry_omega_scales,
                    "base_method": nonparam_base_combo.currentData(),
                    "blq_method": blq_method_combo.currentData(),
                },
            ),
            covariance=CovarianceConfig(
                enabled=covariance_checkbox.isChecked(),
                matrix=covariance_matrix_combo.currentText(),
                options=dict(model_spec.covariance.options),
            ),
        )

    def _apply_model_spec(new_model_spec: ModelSpec, *, mark_synced: bool = True) -> None:
        nonlocal model_spec, last_synced_project_payload, last_synced_editor_state
        # H-11: keep last_confirmed_mode in sync when the spec is loaded
        # programmatically so the mode-switch guard uses the correct baseline.
        last_confirmed_mode[0] = new_model_spec.mode
        model_spec = ModelSpec.from_dict(new_model_spec.to_dict())

        mode_radio_group.blockSignals(True)
        if model_spec.mode == ModelSpecMode.BUILDER:
            mode_radio_builder.setChecked(True)
        else:
            mode_radio_ctl.setChecked(True)
        mode_radio_group.blockSignals(False)

        problem_title_input.setText(model_spec.problem_title)
        dataset_path_input.setText(model_spec.dataset_path or "")
        advan_spin.setValue(model_spec.advan)
        trans_spin.setValue(model_spec.trans)
        model_combo.blockSignals(True)
        model_combo.setCurrentIndex(
            _advan_trans_to_named_model_index(model_spec.advan, model_spec.trans)
        )
        custom_subroutine_row.setVisible(model_combo.currentText().startswith("Custom"))
        model_combo.blockSignals(False)
        _est_idx = next(
            (
                i
                for i, (c, _) in enumerate(_ESTIMATION_METHODS_DISPLAY)
                if c == model_spec.estimation.method
            ),
            0,
        )
        estimation_combo.setCurrentIndex(_est_idx)
        _load_estimation_option_values_from_spec()
        _refresh_estimation_option_affordances()
        covariance_checkbox.setChecked(model_spec.covariance.enabled)
        covariance_matrix_combo.setCurrentText(model_spec.covariance.matrix)
        pk_edit.setPlainText(model_spec.pk_code)
        error_edit.setPlainText(model_spec.error_code)
        des_edit.setPlainText(model_spec.des_code)
        control_stream_edit.setPlainText(model_spec.control_stream_text)

        _populate_theta_table(model_spec.theta_rows)
        _populate_matrix_table(omega_table, model_spec.omega_values, diagonal_fill=0.1)
        _update_header_labels(omega_table, "ETA")
        _shade_upper_triangle(omega_table)
        _set_diagonal_tooltips(omega_table)
        _populate_matrix_table(sigma_table, model_spec.sigma_values, diagonal_fill=0.1)
        _update_header_labels(sigma_table, "EPS")
        _shade_upper_triangle(sigma_table)
        _set_diagonal_tooltips(sigma_table)
        _sync_mode()

        summary_label.setText(format_model_summary(model_spec))
        parameter_summary_label.setText(format_parameter_summary(model_spec))
        _render_translation(translation_service.translate(model_spec))

        if mark_synced:
            last_synced_project_payload = _project_model_payload()
            last_synced_editor_state = _current_editor_state()
        _update_unsaved_indicator(notify_project=False)

    def _refresh_from_project_if_pristine() -> None:
        project_payload = _project_model_payload()
        if (
            last_synced_editor_state is not None
            and _current_editor_state() != last_synced_editor_state
        ):
            return
        if project_payload != last_synced_project_payload:
            _apply_model_spec(ModelSpec.from_dict(project_payload))
            return
        _refresh_next_action()

    def _reload_from_project() -> None:
        _apply_model_spec(default_model_spec(project))

    def _has_unsaved_changes() -> bool:
        return (
            last_synced_editor_state is not None
            and _current_editor_state() != last_synced_editor_state
        )

    def _update_unsaved_indicator(*, notify_project: bool = True) -> None:
        nonlocal last_reported_dirty_state
        dirty = _has_unsaved_changes()
        _refresh_next_action()
        if notify_project and dirty != last_reported_dirty_state:
            _notify_project_state_changed()
        last_reported_dirty_state = dirty

    def _notify_project_state_changed() -> None:
        callback = getattr(root, "_project_state_changed", None)
        if callable(callback):
            callback()

    def _refresh_next_action() -> None:
        result = translation_service.translate(_collect_model_spec())
        _render_translation(result)
        action = recommend_model_next_action(project, current_translation_result=result)
        if action is None:
            next_action_target[0] = ""
            next_action_label.clear()
            next_action_label.setVisible(False)
            next_action_button.setText("")
            next_action_button.setToolTip("")
            next_action_button.setVisible(False)
            return
        button_text, workflow_id, summary = action
        next_action_target[0] = workflow_id
        next_action_label.setText(summary)
        next_action_label.setVisible(True)
        next_action_button.setText(button_text)
        next_action_button.setToolTip(summary)
        next_action_button.setVisible(True)

    def _navigate_to_next_action() -> None:
        target = next_action_target[0]
        if not target:
            return
        _save_model_spec()
        callback = getattr(root, "_navigate_to_workflow", None)
        if callable(callback):
            callback(target)

    def _save_model_spec() -> None:
        nonlocal model_spec, last_synced_project_payload, last_synced_editor_state
        model_spec = _collect_model_spec()
        if (
            not model_spec.dataset_path
            and project.active_dataset is not None
            and project.active_dataset.source_path
        ):
            dataset_path_input.setText(project.active_dataset.source_path)
            model_spec = _collect_model_spec()
        project_service.set_model_spec(project, model_spec)
        summary_label.setText(format_model_summary(model_spec))
        parameter_summary_label.setText(format_parameter_summary(model_spec))
        last_synced_project_payload = _project_model_payload()
        last_synced_editor_state = _current_editor_state()
        _update_unsaved_indicator(notify_project=False)
        _notify_project_state_changed()

    def _render_translation(result: ModelTranslationResult) -> None:
        translation_label.setText(format_translation_summary(result))
        validation_list.clear()
        if not result.validation.issues:
            validation_list.addItem("No validation issues.")
            return
        for issue in result.validation.issues:
            validation_list.addItem(format_validation_issue(issue))

    def _use_active_dataset() -> None:
        if project.active_dataset is not None and project.active_dataset.source_path:
            dataset_path_input.setText(project.active_dataset.source_path)
            _refresh_next_action()

    def _default_control_stream_path() -> str:
        dataset_path = dataset_path_input.text().strip()
        if dataset_path:
            return str(Path(dataset_path).resolve().parent)
        if project.root_path:
            return str(Path(project.root_path).resolve())
        return str(default_workspace_root_path())

    def _default_control_stream_export_name() -> str:
        stem = "".join(
            character.lower() if character.isalnum() else "-"
            for character in (problem_title_input.text().strip() or project.name or "model")
        ).strip("-")
        return f"{stem or 'model'}.ctl"

    def _on_ctl_example_selected(index: int) -> None:
        if load_example_button is None:
            return
        example_id = example_selector.currentData()
        load_example_button.setEnabled(bool(example_id))
        if example_id and example_details_label is not None:
            example_details_label.setText(
                _format_catalog_control_stream_details_html(ctl_examples_by_id.get(str(example_id)))
            )
        elif example_details_label is not None:
            example_details_label.setText("")

    def _handle_ctl_example_details_link(href: str) -> None:
        copy_target = decode_copy_target(href)
        if copy_target is not None:
            clipboard = qt_widgets.QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(copy_target)
            return
        qt_gui.QDesktopServices.openUrl(qt_core.QUrl(href))

    def _auto_load_ctl_dataset(imported_spec: ModelSpec) -> None:
        """If the control stream's $DATA path exists, load it as the active dataset."""
        dataset_path = imported_spec.dataset_path
        if not dataset_path or not Path(dataset_path).exists():
            return
        load_result = load_control_stream_dataset_asset(data_service, imported_spec)
        if load_result.dataset_asset is not None:
            project_service.attach_dataset(project, load_result.dataset_asset)

    def _load_ctl_example() -> None:
        if example_selector is None:
            return
        example_id = example_selector.currentData()
        if not example_id:
            return
        entry = ctl_examples_by_id.get(str(example_id))
        if entry is None or entry.control_stream_path is None:
            return
        try:
            imported_model_spec = load_control_stream_model_spec(
                str(entry.control_stream_path),
                base_spec=_collect_model_spec(),
            )
        except (OSError, ParseError) as exc:
            qt_widgets.QMessageBox.warning(root, "Load example control stream", str(exc))
            return
        if entry.dataset_path is not None:
            imported_model_spec.dataset_path = str(entry.dataset_path)
            load_result = load_control_stream_dataset_asset(
                data_service,
                imported_model_spec,
                display_name=entry.manifest.title,
            )
            if load_result.dataset_asset is not None:
                project_service.attach_dataset(project, load_result.dataset_asset)
        else:
            _auto_load_ctl_dataset(imported_model_spec)
        _apply_model_spec(imported_model_spec, mark_synced=False)
        _notify_project_state_changed()

    def _open_control_stream_file() -> None:
        preferences = load_gui_preferences()
        source_path, _selected_filter = qt_widgets.QFileDialog.getOpenFileName(
            root,
            "Open NONMEM control stream",
            preferences.last_file_dialog_dir or _default_control_stream_path(),
            CONTROL_STREAM_FILE_FILTER,
        )
        if not source_path:
            return
        save_gui_preferences(with_last_file_dialog_dir(preferences, source_path))
        try:
            imported_model_spec = load_control_stream_model_spec(
                source_path,
                base_spec=_collect_model_spec(),
            )
        except (OSError, ParseError) as exc:
            qt_widgets.QMessageBox.warning(root, "Open NONMEM control stream", str(exc))
            return
        _auto_load_ctl_dataset(imported_model_spec)
        _apply_model_spec(imported_model_spec, mark_synced=False)
        _notify_project_state_changed()

    def _save_control_stream_file() -> None:
        preferences = load_gui_preferences()
        start_dir = preferences.last_file_dialog_dir or _default_control_stream_path()
        destination_path, _selected_filter = qt_widgets.QFileDialog.getSaveFileName(
            root,
            "Save NONMEM control stream",
            str(Path(start_dir) / _default_control_stream_export_name()),
            CONTROL_STREAM_FILE_FILTER,
        )
        if not destination_path:
            return
        try:
            write_control_stream_text(destination_path, control_stream_edit.toPlainText())
            save_gui_preferences(with_last_file_dialog_dir(preferences, destination_path))
        except OSError as exc:
            qt_widgets.QMessageBox.warning(root, "Save NONMEM control stream", str(exc))

    def _add_theta_row() -> None:
        rows = _read_theta_rows()
        rows.append(default_theta_row(len(rows) + 1))
        _populate_theta_table(rows)
        parameter_summary_label.setText(format_parameter_summary(_collect_model_spec()))

    def _remove_theta_row() -> None:
        rows = _read_theta_rows()
        if len(rows) <= 1:
            _populate_theta_table([default_theta_row(1)])
        else:
            remove_index = (
                theta_table.currentRow() if theta_table.currentRow() >= 0 else len(rows) - 1
            )
            rows.pop(remove_index)
            _populate_theta_table(rows)
        parameter_summary_label.setText(format_parameter_summary(_collect_model_spec()))

    def _suggest_theta_values() -> None:
        """Pre-fill THETA and OMEGA with typical values for the current ADVAN/TRANS."""
        advan = advan_spin.value()
        trans = trans_spin.value()
        theta_rows = suggest_theta_rows_for_advan(advan, trans)
        omega_values = suggest_omega_values_for_advan(advan, trans)
        if theta_rows is None:
            qt_widgets.QMessageBox.information(
                root,
                "Suggest typical values",
                f"No preset available for ADVAN{advan}/TRANS{trans}.\n"
                "Typical values are provided for ADVAN1/TRANS2, ADVAN2/TRANS2, "
                "ADVAN3/TRANS4, and ADVAN4/TRANS4.",
            )
            return
        _populate_theta_table(theta_rows)
        if omega_values is not None:
            _populate_matrix_table(omega_table, omega_values, diagonal_fill=0.1)
            _update_header_labels(omega_table, "ETA")
            _shade_upper_triangle(omega_table)
            _set_diagonal_tooltips(omega_table)
        parameter_summary_label.setText(format_parameter_summary(_collect_model_spec()))
        _update_unsaved_indicator()

    def _update_header_labels(table: qt_widgets.QTableWidget, prefix: str) -> None:
        n = table.columnCount()
        labels = [f"{prefix}{i + 1}" for i in range(n)]
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)

    def _set_diagonal_tooltips(table: qt_widgets.QTableWidget) -> None:
        import math

        for i in range(table.rowCount()):
            item = table.item(i, i)
            if item is None:
                continue
            try:
                v = float(item.text())
                cv = math.sqrt(math.exp(v) - 1) * 100
                item.setToolTip(f"CV ≈ {cv:.0f}% (lognormal IIV)")
            except (ValueError, OverflowError):
                item.setToolTip("")

    def _shade_upper_triangle(table: qt_widgets.QTableWidget) -> None:
        grey = qt_gui.QColor(230, 230, 230)
        flags_no_edit = qt_core.Qt.ItemFlag.ItemIsEnabled
        flags_editable = (
            qt_core.Qt.ItemFlag.ItemIsEnabled
            | qt_core.Qt.ItemFlag.ItemIsEditable
            | qt_core.Qt.ItemFlag.ItemIsSelectable
        )
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                item = table.item(r, c)
                if item is None:
                    continue
                if c > r:
                    item.setBackground(grey)
                    item.setFlags(flags_no_edit)
                else:
                    item.setFlags(flags_editable)

    def _add_omega() -> None:
        current = _read_matrix_table(omega_table)
        size = len(current) + 1
        _populate_matrix_table(
            omega_table, resize_square_matrix(current, size, diagonal_fill=0.1), diagonal_fill=0.1
        )
        _update_header_labels(omega_table, "ETA")
        _shade_upper_triangle(omega_table)
        _set_diagonal_tooltips(omega_table)
        parameter_summary_label.setText(format_parameter_summary(_collect_model_spec()))
        _update_unsaved_indicator()

    def _remove_omega() -> None:
        current = _read_matrix_table(omega_table)
        if len(current) <= 1:
            return
        _populate_matrix_table(
            omega_table,
            resize_square_matrix(current, len(current) - 1, diagonal_fill=0.1),
            diagonal_fill=0.1,
        )
        _update_header_labels(omega_table, "ETA")
        _shade_upper_triangle(omega_table)
        _set_diagonal_tooltips(omega_table)
        parameter_summary_label.setText(format_parameter_summary(_collect_model_spec()))
        _update_unsaved_indicator()

    def _add_sigma() -> None:
        current = _read_matrix_table(sigma_table)
        size = len(current) + 1
        _populate_matrix_table(
            sigma_table, resize_square_matrix(current, size, diagonal_fill=0.1), diagonal_fill=0.1
        )
        _update_header_labels(sigma_table, "EPS")
        _shade_upper_triangle(sigma_table)
        _set_diagonal_tooltips(sigma_table)
        parameter_summary_label.setText(format_parameter_summary(_collect_model_spec()))
        _update_unsaved_indicator()

    def _remove_sigma() -> None:
        current = _read_matrix_table(sigma_table)
        if len(current) <= 1:
            return
        _populate_matrix_table(
            sigma_table,
            resize_square_matrix(current, len(current) - 1, diagonal_fill=0.1),
            diagonal_fill=0.1,
        )
        _update_header_labels(sigma_table, "EPS")
        _shade_upper_triangle(sigma_table)
        _set_diagonal_tooltips(sigma_table)
        parameter_summary_label.setText(format_parameter_summary(_collect_model_spec()))
        _update_unsaved_indicator()

    def _handle_mode_changed() -> None:
        # H-11: warn the user before discarding unsaved edits on mode switch.
        new_mode = _current_mode()
        if _has_unsaved_changes() and new_mode != last_confirmed_mode[0]:
            answer = qt_widgets.QMessageBox.question(
                root,
                "Switch mode?",
                "You have unsaved changes. Switch mode and discard them?",
                qt_widgets.QMessageBox.StandardButton.Yes
                | qt_widgets.QMessageBox.StandardButton.No,
                qt_widgets.QMessageBox.StandardButton.No,
            )
            if answer != qt_widgets.QMessageBox.StandardButton.Yes:
                # Revert the radio button without re-triggering this handler.
                mode_radio_group.blockSignals(True)
                if last_confirmed_mode[0] == ModelSpecMode.CONTROL_STREAM:
                    mode_radio_ctl.setChecked(True)
                else:
                    mode_radio_builder.setChecked(True)
                mode_radio_group.blockSignals(False)
                return
        last_confirmed_mode[0] = new_mode
        _sync_mode()
        _refresh_next_action()

    def _on_model_combo_changed(index: int) -> None:
        _, advan, trans = _NAMED_MODELS[index]
        is_custom = advan == -1
        custom_subroutine_row.setVisible(is_custom)
        if not is_custom:
            advan_spin.setValue(advan)
            trans_spin.setValue(trans)
            diagram_widget.update_diagram(advan, trans)
        _update_unsaved_indicator()

    def _sync_model_combo_from_spins() -> None:
        idx = _advan_trans_to_named_model_index(advan_spin.value(), trans_spin.value())
        model_combo.blockSignals(True)
        model_combo.setCurrentIndex(idx)
        model_combo.blockSignals(False)
        diagram_widget.update_diagram(advan_spin.value(), trans_spin.value())
        _update_unsaved_indicator()

    mode_radio_group.buttonToggled.connect(lambda _btn, _checked: _handle_mode_changed())
    model_combo.currentIndexChanged.connect(_on_model_combo_changed)
    use_active_dataset_button.clicked.connect(_use_active_dataset)
    open_control_stream_button.clicked.connect(_open_control_stream_file)
    save_control_stream_button.clicked.connect(_save_control_stream_file)
    if example_selector is not None:
        example_selector.currentIndexChanged.connect(_on_ctl_example_selected)
        load_example_button.clicked.connect(_load_ctl_example)
        example_details_label.linkActivated.connect(_handle_ctl_example_details_link)
    add_theta_button.clicked.connect(_add_theta_row)
    remove_theta_button.clicked.connect(_remove_theta_row)
    suggest_theta_button.clicked.connect(_suggest_theta_values)
    add_omega_button.clicked.connect(_add_omega)
    remove_omega_button.clicked.connect(_remove_omega)
    add_sigma_button.clicked.connect(_add_sigma)
    remove_sigma_button.clicked.connect(_remove_sigma)
    problem_title_input.textChanged.connect(lambda _: _update_unsaved_indicator())
    dataset_path_input.textChanged.connect(lambda _: _update_unsaved_indicator())
    dataset_path_input.textChanged.connect(
        lambda text: dataset_autofill_hint.setVisible(
            bool(_active_source) and text == _active_source
        )
    )
    advan_spin.valueChanged.connect(lambda _: _sync_model_combo_from_spins())
    trans_spin.valueChanged.connect(lambda _: _sync_model_combo_from_spins())
    def _on_estimation_method_changed() -> None:
        _apply_estimation_option_defaults_for_method(str(estimation_combo.currentData()))
        _refresh_estimation_option_affordances()
        _update_unsaved_indicator()

    estimation_combo.currentIndexChanged.connect(lambda _: _on_estimation_method_changed())
    covariance_checkbox.toggled.connect(lambda _: _update_unsaved_indicator())
    covariance_matrix_combo.currentIndexChanged.connect(lambda _: _update_unsaved_indicator())
    pk_edit.textChanged.connect(_update_unsaved_indicator)
    error_edit.textChanged.connect(_update_unsaved_indicator)
    des_edit.textChanged.connect(_update_unsaved_indicator)
    control_stream_edit.textChanged.connect(_update_unsaved_indicator)
    maxeval_spin.valueChanged.connect(lambda _: _update_unsaved_indicator())
    nstarts_spin.valueChanged.connect(lambda _: _update_unsaved_indicator())
    tight_gtol_checkbox.toggled.connect(lambda _: _update_unsaved_indicator())
    outer_optimizer_combo.currentIndexChanged.connect(lambda _: _update_unsaved_indicator())
    fallback_optimizer_combo.currentIndexChanged.connect(lambda _: _update_unsaved_indicator())
    fallback_maxeval_spin.valueChanged.connect(lambda _: _update_unsaved_indicator())
    retain_best_checkbox.toggled.connect(lambda _: _update_unsaved_indicator())
    retry_on_abnormal_checkbox.toggled.connect(lambda _: _update_unsaved_indicator())
    retry_omega_scales_input.textChanged.connect(lambda _: _update_unsaved_indicator())
    theta_table.itemChanged.connect(lambda _: _update_unsaved_indicator())
    omega_table.itemChanged.connect(
        lambda item: (_set_diagonal_tooltips(omega_table), _update_unsaved_indicator())
    )
    sigma_table.itemChanged.connect(
        lambda item: (_set_diagonal_tooltips(sigma_table), _update_unsaved_indicator())
    )
    next_action_button.clicked.connect(_navigate_to_next_action)
    save_shortcut = qt_gui.QShortcut(qt_gui.QKeySequence("Ctrl+S"), root)
    save_shortcut.activated.connect(_save_model_spec)
    root._apply_responsive_layout = _apply_responsive_layout  # type: ignore[attr-defined]
    root._has_unsaved_changes = _has_unsaved_changes  # type: ignore[attr-defined]
    root._refresh_workflow = _refresh_from_project_if_pristine  # type: ignore[attr-defined]
    root._load_project = _reload_from_project  # type: ignore[attr-defined]
    root._on_leave = lambda: _save_model_spec() if _has_unsaved_changes() else None  # type: ignore[attr-defined]

    _apply_model_spec(model_spec)
    _apply_responsive_layout()
    return root
