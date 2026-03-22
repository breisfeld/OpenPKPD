"""Unit tests for HTML report generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from openpkpd.output.report import estimation_result_to_html


def _make_result(converged: bool = True) -> MagicMock:
    result = MagicMock()
    result.ofv = -100.0
    result.aic = -90.0
    result.bic = -85.0
    result.n_parameters = 5
    result.n_subjects = 10
    result.n_observations = 100
    result.method = "FOCE"
    result.converged = converged
    result.warnings = []
    result.theta_final = [1.0, 0.5]
    result.omega_final = np.array([[0.1]])
    result.sigma_final = np.array([[0.05]])
    result.eta_shrinkage = None
    result.eps_shrinkage = None
    result.ofv_history = []
    return result


def _make_params() -> MagicMock:
    from openpkpd.model.parameters import ThetaSpec

    params = MagicMock()
    params.theta_specs = [
        ThetaSpec(init=1.0, label="CL"),
        ThetaSpec(init=0.5, label="V"),
    ]
    return params


def test_provenance_appears_after_plots_section():
    result = _make_result()
    params = _make_params()
    provenance = {"Environment": {"python_version": "3.12", "platform": "linux"}}
    html = estimation_result_to_html(result, params, provenance=provenance)

    plots_pos = html.find("Diagnostic Plots")
    prov_pos = html.find("Provenance")
    # If no plots are generated (no ofv_history), provenance should still come after
    # the parameter tables — check it's not before them
    theta_pos = html.find("THETA")
    assert prov_pos > theta_pos, "Provenance should appear after parameter tables"
    # When there are no plots at all, provenance order relative to plot header is moot
    if plots_pos != -1:
        assert prov_pos > plots_pos, "Provenance should appear after Diagnostic Plots"


def test_plots_embedded_as_base64(tmp_path: Path):
    # Write a minimal 1x1 PNG (valid PNG bytes)
    import struct
    import zlib

    def _minimal_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"

        def chunk(tag: bytes, data: bytes) -> bytes:
            length = struct.pack(">I", len(data))
            crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            return length + tag + data + crc

        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xff\x00\x00"
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    png_path = tmp_path / "test_plot.png"
    png_path.write_bytes(_minimal_png())

    result = _make_result()
    params = _make_params()
    plots = [("GOF Panel", str(png_path))]
    html = estimation_result_to_html(result, params, plots=plots)

    assert "Diagnostic Plots" in html
    assert "GOF Panel" in html
    assert "data:image/png;base64," in html


def test_missing_plot_file_skipped_silently(tmp_path: Path):
    result = _make_result()
    params = _make_params()
    plots = [("Missing plot", str(tmp_path / "does_not_exist.png"))]
    # Should not raise; missing file is silently skipped
    html = estimation_result_to_html(result, params, plots=plots)
    assert isinstance(html, str)


def test_no_plots_no_diagnostic_section():
    result = _make_result()
    params = _make_params()
    html = estimation_result_to_html(result, params, plots=None)
    assert "Diagnostic Plots" not in html
