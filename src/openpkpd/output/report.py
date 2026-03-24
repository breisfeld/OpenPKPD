"""
HTML report generation for estimation results.

Produces a self-contained HTML report embedding parameter tables, OFV/AIC/BIC,
convergence status, covariance results, and diagnostic plots as base64 PNG.

Usage:
    from openpkpd.output.report import (
        estimation_result_to_html,
        export_html_report_to_pdf,
        write_html_report,
        write_pdf_report,
    )

    write_html_report("results.html", result, params, title="Theophylline FOCE")
    write_pdf_report("results.pdf", result, params, title="Theophylline FOCE")

    # Or get HTML string directly:
    html = estimation_result_to_html(result, params)
"""

from __future__ import annotations

import base64
import datetime
import html
import io
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openpkpd.covariance.sandwich import CovarianceResult
    from openpkpd.estimation.base import EstimationResult
    from openpkpd.model.parameters import ParameterSet


_CSS = """
@page { size: A4 portrait; margin: 12mm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 2em; color: #222; background: #fafafa; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }
h2 { color: #34495e; margin-top: 1.8em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
th { background: #3498db; color: white; padding: 8px 12px; text-align: left; }
td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) { background: #f2f2f2; }
.provenance-table td:first-child { width: 24%; font-weight: 600; white-space: nowrap; }
.converged { color: #27ae60; font-weight: bold; }
.not-converged { color: #e74c3c; font-weight: bold; }
.warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 0.8em 1em; margin: 0.5em 0; }
.stat-grid { display: flex; gap: 1.5em; flex-wrap: wrap; margin: 1em 0; }
.stat-card { background: white; border: 1px solid #ddd; border-radius: 6px;
             padding: 1em 1.5em; min-width: 140px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.stat-card .label { font-size: 0.75em; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-card .value { font-size: 1.4em; font-weight: bold; color: #2c3e50; margin-top: 0.2em; }
img { max-width: 100%; margin: 1em 0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
details { margin: 0.5em 0; }
details summary { cursor: pointer; color: #2980b9; font-size: 0.9em; padding: 0.3em 0; }
details summary:hover { text-decoration: underline; }
.footer { margin-top: 3em; font-size: 0.8em; color: #999; border-top: 1px solid #eee; padding-top: 1em; }
@media print {
  body { margin: 0; background: white; }
  .stat-card { break-inside: avoid; box-shadow: none; }
  table, pre, img, .warning { break-inside: avoid; }
  h1, h2, h3 { break-after: avoid; }
}
"""

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
<h1>{title}</h1>
<p>Generated: {timestamp}</p>

<div class="stat-grid">
  <div class="stat-card"><div class="label">OFV</div><div class="value">{ofv}</div></div>
  <div class="stat-card"><div class="label">AIC</div><div class="value">{aic}</div></div>
  <div class="stat-card"><div class="label">BIC</div><div class="value">{bic}</div></div>
  <div class="stat-card"><div class="label">Parameters</div><div class="value">{n_params}</div></div>
  <div class="stat-card"><div class="label">Subjects</div><div class="value">{n_subjects}</div></div>
  <div class="stat-card"><div class="label">Observations</div><div class="value">{n_obs}</div></div>
  <div class="stat-card"><div class="label">Method</div><div class="value">{method}</div></div>
  <div class="stat-card"><div class="label">Converged</div>
    <div class="value {conv_class}">{conv_text}</div></div>
</div>

{warnings_html}

<h2>Parameter Estimates</h2>
{theta_table}
{omega_table}
{sigma_table}

{shrinkage_html}

{covariance_html}

{plots_html}

{provenance_html}

<div class="footer">
  OpenPKPD — <a href="https://github.com">github.com/openpkpd/openpkpd</a>
</div>
</body>
</html>
"""


class ReportExportError(RuntimeError):
    """Raised when PDF report export cannot be completed."""


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0.0):
        return f"{v:.{digits}E}"
    return f"{v:.{digits}f}"


def _rse(se: float | None, est: float) -> str:
    if se is None or est == 0.0 or not np.isfinite(se):
        return "—"
    return f"{abs(se / est) * 100:.1f}%"


def _theta_table_html(
    result: EstimationResult,
    params: ParameterSet,
    se_vec: list[float] | None,
) -> str:
    rows = []
    for i, (val, spec) in enumerate(zip(result.theta_final, params.theta_specs, strict=False)):
        label = spec.label or f"TH{i + 1}"
        se = se_vec[i] if se_vec and i < len(se_vec) else None
        rows.append(
            f"<tr><td>THETA({i + 1})</td><td>{label}</td>"
            f"<td>{_fmt(val)}</td><td>{_fmt(se)}</td>"
            f"<td>{_rse(se, val)}</td>"
            f"<td>{_fmt(spec.lower)}</td><td>{_fmt(spec.upper)}</td></tr>"
        )
    body = "\n".join(rows)
    return (
        "<h3>THETA (Fixed Effects)</h3>"
        "<table><thead><tr><th>Name</th><th>Label</th><th>Estimate</th>"
        "<th>SE</th><th>RSE%</th><th>Lower</th><th>Upper</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _matrix_table_html(name: str, mat: np.ndarray) -> str:
    n = mat.shape[0]
    if n == 0:
        return ""
    rows = []
    for r in range(n):
        for c in range(r + 1):
            rows.append(f"<tr><td>{name}({r + 1},{c + 1})</td><td>{_fmt(mat[r, c])}</td></tr>")
    body = "\n".join(rows)
    return (
        f"<h3>{name} (Random Effects Covariance)</h3>"
        f"<table><thead><tr><th>Element</th><th>Estimate</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _shrinkage_html(result: EstimationResult) -> str:
    parts = []
    high_shrinkage_indices: list[int] = []

    if result.eta_shrinkage is not None and len(result.eta_shrinkage) > 0:
        n_eta = result.omega_final.shape[0]
        rows = []
        for k, sh in enumerate(result.eta_shrinkage):
            omega_kk = float(result.omega_final[k, k]) if k < n_eta else 0.0
            target_sd = float(np.sqrt(max(omega_kk, 0.0)))
            corr_str = f"{1.0 / (1.0 - sh):.3f}" if sh < 1.0 - 1e-8 else "&#8734;"
            high = sh > 0.3
            if high:
                high_shrinkage_indices.append(k)
            flag = (
                '<td style="color:#e74c3c;font-weight:bold">HIGH</td>'
                if high
                else "<td></td>"
            )
            rows.append(
                f"<tr><td>ETA({k + 1})</td><td>{sh * 100:.1f}%</td>"
                f"<td>{target_sd:.4f}</td><td>{corr_str}</td>{flag}</tr>"
            )
        body = "".join(rows)
        parts.append(
            "<h3>ETA Shrinkage</h3>"
            "<table><thead><tr><th>ETA</th><th>Shrinkage</th>"
            "<th>Target SD (&radic;&Omega;<sub>kk</sub>)</th>"
            "<th>Correction&nbsp;(1&nbsp;/&nbsp;1&minus;sh)</th>"
            "<th>Note</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

        if high_shrinkage_indices and result.post_hoc_etas:
            deshrunk = result.compute_deshrinkage_etas()
            if deshrunk:
                # Explanation callout
                parts.append(
                    "<div style='background:#e8f4fd;border-left:4px solid #3498db;"
                    "padding:0.8em 1em;margin:0.8em 0'>"
                    "<b>De-shrunken EBEs (Combes 2013 correction)</b><br>"
                    "EBEs from FOCE/FOCEI are biased toward zero: shrinkage &gt; 30% means "
                    "the raw EBEs underestimate between-subject variability and should not be "
                    "used directly for covariate plots or ETA histograms. "
                    "The correction below rescales each subject&rsquo;s ETA so that "
                    "SD(adj&nbsp;&eta;<sub>k</sub>)&nbsp;=&nbsp;&radic;&Omega;<sub>kk</sub> exactly: "
                    "<i>&eta;&#770;<sub>adj,ik</sub> = &eta;&#770;<sub>ik</sub> / "
                    "(1 &minus; shrinkage<sub>k</sub>)</i>. "
                    "Relative subject ordering is preserved."
                    "</div>"
                )

                # SD summary: raw vs de-shrunken vs target
                subjects = list(result.post_hoc_etas.keys())
                raw_matrix = np.array(
                    [result.post_hoc_etas[sid] for sid in subjects], dtype=float
                )
                adj_matrix = np.array(
                    [deshrunk[sid] for sid in subjects], dtype=float
                )
                n_cols = raw_matrix.shape[1] if raw_matrix.ndim > 1 else 1
                stat_rows = []
                for k in range(min(n_eta, n_cols)):
                    raw_sd = float(np.std(raw_matrix[:, k], ddof=1)) if len(subjects) > 1 else 0.0
                    adj_sd = float(np.std(adj_matrix[:, k], ddof=1)) if len(subjects) > 1 else 0.0
                    omega_kk = float(result.omega_final[k, k])
                    tgt_sd = float(np.sqrt(max(omega_kk, 0.0)))
                    stat_rows.append(
                        f"<tr><td>ETA({k + 1})</td><td>{raw_sd:.4f}</td>"
                        f"<td>{adj_sd:.4f}</td><td>{tgt_sd:.4f}</td></tr>"
                    )
                parts.append(
                    "<h4>SD summary: raw vs de-shrunken EBEs</h4>"
                    "<table><thead><tr><th>ETA</th><th>SD (raw EBEs)</th>"
                    "<th>SD (de-shrunken)</th>"
                    "<th>Target &radic;&Omega;<sub>kk</sub></th></tr></thead>"
                    f"<tbody>{''.join(stat_rows)}</tbody></table>"
                )

                # Collapsible per-subject table
                hdr = "".join(f"<th>ETA({k + 1}) adj</th>" for k in range(n_cols))
                detail_rows = []
                for i, sid in enumerate(subjects):
                    vals = adj_matrix[i] if adj_matrix.ndim > 1 else [float(adj_matrix[i])]
                    cells = "".join(f"<td>{float(v):.4f}</td>" for v in vals)
                    detail_rows.append(f"<tr><td>{sid}</td>{cells}</tr>")
                parts.append(
                    "<details><summary><b>Per-subject de-shrunken ETAs</b>"
                    " &#9660; click to expand</summary>"
                    f"<table><thead><tr><th>Subject</th>{hdr}</tr></thead>"
                    f"<tbody>{''.join(detail_rows)}</tbody></table>"
                    "</details>"
                )

    if result.eps_shrinkage is not None and len(result.eps_shrinkage) > 0:
        rows = "".join(
            f"<tr><td>EPS({k + 1})</td><td>{sh * 100:.1f}%</td></tr>"
            for k, sh in enumerate(result.eps_shrinkage)
        )
        parts.append(
            "<h3>EPS Shrinkage</h3>"
            "<table><thead><tr><th>EPS</th><th>Shrinkage</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return "\n".join(parts) if parts else ""


def _covariance_html(cov_result: CovarianceResult | None) -> str:
    if cov_result is None:
        return ""
    parts = ["<h2>Covariance Step</h2>"]
    if cov_result.condition_number is not None:
        parts.append(f"<p>Condition number: <b>{cov_result.condition_number:.2E}</b></p>")
    eigenvalues = getattr(cov_result, "eigenvalues", None)
    if eigenvalues is not None and len(eigenvalues) > 0:
        eigs = ", ".join(f"{e:.3E}" for e in eigenvalues)
        parts.append(f"<p>Eigenvalues: {eigs}</p>")
    if hasattr(cov_result, "se") and cov_result.se is not None and len(cov_result.se) > 0:
        rows = "".join(
            f"<tr><td>THETA({i + 1})</td><td>{_fmt(se)}</td></tr>"
            for i, se in enumerate(cov_result.se)
        )
        parts.append(
            "<table><thead><tr><th>Parameter</th><th>SE</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return "\n".join(parts)


def _provenance_value_html(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        if not value.strip():
            return "—"
        escaped = html.escape(value)
        return f"<pre>{escaped}</pre>" if "\n" in value else escaped
    if isinstance(value, (Mapping, list, tuple)):
        try:
            serialized = json.dumps(value, indent=2, sort_keys=True, default=str)
        except TypeError:
            serialized = str(value)
        return f"<pre>{html.escape(serialized)}</pre>"
    return html.escape(str(value))


def _provenance_html(provenance: Mapping[str, object] | None) -> str:
    if not provenance:
        return ""
    parts = ["<h2>Provenance &amp; Reproducibility</h2>"]
    for section_name, section_payload in provenance.items():
        if section_payload in (None, "", [], {}):
            continue
        parts.append(f"<h3>{html.escape(str(section_name))}</h3>")
        if isinstance(section_payload, Mapping):
            rows = []
            for key, value in section_payload.items():
                if value in (None, "", [], {}):
                    continue
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(key).replace('_', ' '))}</td>"
                    f"<td>{_provenance_value_html(value)}</td>"
                    "</tr>"
                )
            if rows:
                parts.append(
                    "<table class='provenance-table'><tbody>" + "".join(rows) + "</tbody></table>"
                )
            continue
        parts.append(_provenance_value_html(section_payload))
    return "\n".join(parts) if len(parts) > 1 else ""


def _plot_to_b64(fig) -> str:
    """Convert a matplotlib Figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _make_ofv_history_b64(result: EstimationResult) -> str | None:
    """Return base64 PNG of OFV history, or None if unavailable."""
    if not result.ofv_history or len(result.ofv_history) < 2:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result.ofv_history, color="#3498db", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("OFV")
        ax.set_title("OFV History")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        b64 = _plot_to_b64(fig)
        plt.close(fig)
        return b64
    except Exception:
        return None


def _embedded_plots_html(
    result: EstimationResult,
    plots: Sequence[tuple[str, str | Path]] | None,
) -> str:
    """Build the Diagnostic Plots section, embedding PNGs as base64.

    ``plots`` is an ordered sequence of ``(section_title, png_path)`` pairs
    that have already been saved to disk.  OFV history is prepended if it can
    be generated inline from ``result.ofv_history``.
    """
    items: list[tuple[str, str]] = []  # (title, b64)

    ofv_b64 = _make_ofv_history_b64(result)
    if ofv_b64 is not None:
        items.append(("OFV History", ofv_b64))

    if plots:
        for section_title, png_path in plots:
            try:
                data = Path(png_path).read_bytes()
                items.append((section_title, base64.b64encode(data).decode("ascii")))
            except Exception:
                pass

    if not items:
        return ""

    parts = ["<h2>Diagnostic Plots</h2>"]
    for section_title, b64 in items:
        escaped = html.escape(section_title)
        parts.append(f'<h3>{escaped}</h3><img src="data:image/png;base64,{b64}" alt="{escaped}">')
    return "\n".join(parts)


def estimation_result_to_html(
    result: EstimationResult,
    params: ParameterSet,
    title: str = "OpenPKPD Estimation Report",
    cov_result: CovarianceResult | None = None,
    provenance: Mapping[str, object] | None = None,
    plots: Sequence[tuple[str, str | Path]] | None = None,
) -> str:
    """
    Render an estimation result as a self-contained HTML string.

    Args:
        result:     EstimationResult from any estimation method.
        params:     ParameterSet with specs for labelling.
        title:      Report title shown in the browser tab and heading.
        cov_result: Optional CovarianceResult for SE / condition number.
        provenance: Optional provenance mapping (shown near the end).
        plots:      Optional list of ``(section_title, png_path)`` pairs to
                    embed as inline base64 images in the Diagnostic Plots
                    section.  OFV history is always prepended when available.

    Returns:
        HTML string (UTF-8, self-contained, no external dependencies).
    """
    # SE vector from covariance result
    se_vec: list[float] | None = None
    if cov_result is not None and hasattr(cov_result, "se") and cov_result.se is not None:
        se_vec = list(cov_result.se)

    # Warnings
    all_warnings = list(result.warnings)
    if hasattr(result, "shrinkage_warnings"):
        all_warnings.extend(result.shrinkage_warnings)
    warnings_html = ""
    if all_warnings:
        items = "\n".join(f"<div class='warning'>⚠ {w}</div>" for w in all_warnings)
        warnings_html = f"<h2>Warnings</h2>{items}"
    provenance_html = _provenance_html(provenance)

    plots_html = _embedded_plots_html(result, plots)

    html = _HTML_TEMPLATE.format(
        title=title,
        css=_CSS,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ofv=_fmt(result.ofv, 4),
        aic=_fmt(result.aic, 2),
        bic=_fmt(result.bic, 2),
        n_params=result.n_parameters,
        n_subjects=result.n_subjects,
        n_obs=result.n_observations,
        method=result.method or "—",
        conv_class="converged" if result.converged else "not-converged",
        conv_text="YES" if result.converged else "NO",
        warnings_html=warnings_html,
        provenance_html=provenance_html,
        theta_table=_theta_table_html(result, params, se_vec),
        omega_table=_matrix_table_html("OMEGA", result.omega_final),
        sigma_table=_matrix_table_html("SIGMA", result.sigma_final),
        shrinkage_html=_shrinkage_html(result),
        covariance_html=_covariance_html(cov_result),
        plots_html=plots_html,
    )
    return html


def write_html_report(
    path: str,
    result: EstimationResult,
    params: ParameterSet,
    title: str = "OpenPKPD Estimation Report",
    cov_result: CovarianceResult | None = None,
    provenance: Mapping[str, object] | None = None,
    plots: Sequence[tuple[str, str | Path]] | None = None,
) -> None:
    """
    Write a self-contained HTML estimation report to disk.

    Args:
        path:       Output file path (e.g. "results.html").
        result:     EstimationResult.
        params:     ParameterSet for parameter labels.
        title:      Report heading.
        cov_result: Optional CovarianceResult for SE / condition number.
        provenance: Optional provenance mapping (shown near the end).
        plots:      Optional list of ``(section_title, png_path)`` pairs to
                    embed as inline images.
    """
    html = estimation_result_to_html(
        result, params, title=title, cov_result=cov_result, provenance=provenance, plots=plots
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def export_html_report_to_pdf(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    timeout_ms: int = 30_000,
    qapplication_args: list[str] | None = None,
) -> None:
    """Convert an existing HTML report into a PDF.

    This convenience wrapper uses the optional Qt GUI stack under the hood. It
    requires the ``openpkpd[gui]`` extras (PySide6 / Qt WebEngine) to be
    installed and available at runtime.
    """

    try:
        from openpkpd_gui.app.runtime import load_qt_modules
        from openpkpd_gui.services.report_export_service import ReportExportService
    except Exception as exc:  # pragma: no cover - environment specific
        raise ReportExportError(
            "PDF report export requires the optional GUI dependencies. "
            "Install the package with the 'gui' extra, e.g. `pip install openpkpd[gui]`."
        ) from exc

    try:
        qt_core, _, qt_widgets = load_qt_modules()
        app = qt_widgets.QApplication.instance()
        if app is None:
            app = qt_widgets.QApplication(qapplication_args or ["openpkpd-report-export"])
        success, message = ReportExportService().export_html_report_to_pdf(
            parent=None,
            source_path=source_path,
            destination_path=destination_path,
            timeout_ms=timeout_ms,
        )
    except Exception as exc:  # pragma: no cover - environment specific
        raise ReportExportError(f"Failed to initialize the Qt PDF export runtime: {exc}") from exc
    qt_core.QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
    if not success:
        raise ReportExportError(message or f"Failed to export PDF report from {source_path}")


def write_pdf_report(
    path: str | Path,
    result: EstimationResult,
    params: ParameterSet,
    title: str = "OpenPKPD Estimation Report",
    cov_result: CovarianceResult | None = None,
    provenance: Mapping[str, object] | None = None,
    plots: Sequence[tuple[str, str | Path]] | None = None,
    *,
    timeout_ms: int = 30_000,
    intermediate_html_path: str | Path | None = None,
    qapplication_args: list[str] | None = None,
) -> None:
    """Write a PDF estimation report to disk.

    The report HTML is generated using the same renderer as
    :func:`write_html_report`, then converted to PDF via the optional Qt GUI
    stack.
    """

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    cleanup_intermediate = intermediate_html_path is None
    if intermediate_html_path is None:
        with tempfile.NamedTemporaryFile(
            prefix="openpkpd-report-", suffix=".html", delete=False
        ) as handle:
            html_path = Path(handle.name)
    else:
        html_path = Path(intermediate_html_path).expanduser()
        html_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write_html_report(
            str(html_path),
            result,
            params,
            title=title,
            cov_result=cov_result,
            provenance=provenance,
            plots=plots,
        )
        export_html_report_to_pdf(
            html_path,
            destination,
            timeout_ms=timeout_ms,
            qapplication_args=qapplication_args,
        )
    finally:
        if cleanup_intermediate:
            html_path.unlink(missing_ok=True)
