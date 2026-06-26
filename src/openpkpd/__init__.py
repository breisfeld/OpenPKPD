"""
OpenPKPD — Open-source Python reimplementation of NONMEM.

Quick start (pure Python API):
    from openpkpd import ModelBuilder

    result = (
        ModelBuilder()
        .problem("My model")
        .data("data.csv")
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([1.5, 0.1, 30])
        .omega([0.5, 0.3, 0.3])
        .sigma(0.1)
        .estimation(method="FOCE", interaction=True)
        .build()
        .fit()
    )

Quick start (control stream):
    from openpkpd.parser.control_stream import ControlStream
    from openpkpd.cli.runner import run_model
    result = run_model("model.ctl")
"""

from __future__ import annotations

import contextlib

from openpkpd.api.model_builder import BuiltModel, ModelBuilder
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.output.report import (
    ReportExportError,
    estimation_result_to_html,
    export_html_report_to_pdf,
    write_html_report,
    write_pdf_report,
)
from openpkpd.parser.control_stream import ControlStream

__version__ = "0.3.1"

# ── Optional / progressive imports ──────────────────────────────────────────
# These use try/except so the package can be imported even if sub-modules
# created by other agents are not yet present.

with contextlib.suppress(ImportError):
    from openpkpd.library import get_model, list_models

with contextlib.suppress(ImportError):
    from openpkpd.nca.nca import NCAEngine

with contextlib.suppress(ImportError):
    from openpkpd.inference.model_comparison import compare_models, lrt

__all__ = [
    "ModelBuilder",
    "BuiltModel",
    "ControlStream",
    "EstimationResult",
    "ParameterSet",
    "ThetaSpec",
    "OmegaSpec",
    "SigmaSpec",
    "ReportExportError",
    "estimation_result_to_html",
    "export_html_report_to_pdf",
    "write_html_report",
    "write_pdf_report",
    # Model library
    "list_models",
    "get_model",
    # NCA (optional — created by another agent)
    "NCAEngine",
    # Inference (optional — created by another agent)
    "lrt",
    "compare_models",
    # plots subpackage is importable as openpkpd.plots (matplotlib optional)
    "plots",
]
