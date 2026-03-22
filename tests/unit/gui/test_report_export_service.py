from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openpkpd import write_pdf_report
from openpkpd.estimation.base import EstimationResult
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd_gui.app.runtime import load_qt_modules, qt_widgets_available


def _make_result() -> EstimationResult:
    return EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=np.diag([0.2]),
        sigma_final=np.diag([0.1]),
        ofv=1.23,
        converged=True,
        n_observations=5,
        method="FOCE",
    )


def _make_params() -> ParameterSet:
    return ParameterSet.from_specs(
        [ThetaSpec(init=1.0)],
        [OmegaSpec(block_size=1, values=[0.2])],
        [SigmaSpec(block_size=1, values=[0.1])],
    )


@pytest.mark.unit
def test_write_pdf_report_creates_pdf_with_expected_report_content(tmp_path: Path) -> None:
    if not qt_widgets_available():
        pytest.skip("Qt GUI modules are unavailable in this environment")

    qt_core, _, qt_widgets = load_qt_modules()
    try:
        from PySide6 import QtPdf
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Qt PDF modules are unavailable in this environment: {exc}")

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication(
        ["test", "-platform", "offscreen"]
    )
    parent = qt_core.QObject()
    pdf_path = tmp_path / "report.pdf"

    write_pdf_report(
        pdf_path,
        _make_result(),
        _make_params(),
        title="PDF export regression",
        provenance={
            "Run context": {"run_id": "run-123", "scenario_name": "Baseline"},
            "Dataset": {"source_path": "/tmp/theo.csv", "sha256": "abc123"},
            "Environment": {"openpkpd_version": "0.1.0", "platform": "Linux"},
        },
        timeout_ms=60_000,
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0

    document = QtPdf.QPdfDocument(parent)
    try:
        status = document.load(str(pdf_path))
        assert status == QtPdf.QPdfDocument.Error.None_
        assert document.pageCount() >= 1
        extracted_text = "\n".join(
            document.getAllText(page).text() for page in range(document.pageCount())
        )
    finally:
        document.close()

    assert "PDF export regression" in extracted_text
    assert "Provenance & Reproducibility" in extracted_text
    assert "run id run-123" in extracted_text.lower()
    assert "/tmp/theo.csv" in extracted_text
    assert "abc123" in extracted_text
    assert "openpkpd version 0.1.0" in extracted_text.lower()
    assert "THETA(1)" in extracted_text

    parent.deleteLater()
    qt_core.QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
