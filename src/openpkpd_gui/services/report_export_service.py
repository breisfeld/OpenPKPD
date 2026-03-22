"""Helpers for exporting HTML report artifacts to PDF."""

from __future__ import annotations

from pathlib import Path

from openpkpd_gui.app.runtime import load_qt_modules


class ReportExportService:
    """Export self-contained HTML reports to PDF via Qt WebEngine."""

    def export_html_report_to_pdf(
        self,
        *,
        parent,
        source_path: str | Path,
        destination_path: str | Path,
        timeout_ms: int = 30_000,
    ) -> tuple[bool, str | None]:
        qt_core, qt_gui, _ = load_qt_modules()
        from shiboken6 import delete

        try:
            from PySide6 import QtWebEngineCore
        except Exception as exc:  # pragma: no cover - environment specific
            return False, f"Qt WebEngine is unavailable for PDF export: {exc}"

        source = Path(source_path).expanduser()
        if not source.exists() or not source.is_file():
            return False, f"Report file is not available on disk: {source}"
        if source.suffix.lower() not in {".html", ".htm"}:
            return False, f"PDF export currently supports HTML reports only: {source.name}"

        destination = Path(destination_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        page = QtWebEngineCore.QWebEnginePage()
        timer = qt_core.QTimer()
        timer.setSingleShot(True)
        loop = qt_core.QEventLoop()
        layout = qt_gui.QPageLayout(
            qt_gui.QPageSize(qt_gui.QPageSize.PageSizeId.A4),
            qt_gui.QPageLayout.Orientation.Portrait,
            qt_core.QMarginsF(12.7, 12.7, 12.7, 12.7),
        )
        state = {"done": False, "success": False, "message": None}

        def _finish(success: bool, message: str | None = None) -> None:
            if state["done"]:
                return
            state["done"] = True
            state["success"] = success
            state["message"] = message
            if timer.isActive():
                timer.stop()
            loop.quit()

        def _on_load_finished(ok: bool) -> None:
            if not ok:
                _finish(False, f"Qt could not load the HTML report for PDF export: {source}")
                return
            try:
                page.printToPdf(str(destination), layout)
            except Exception as exc:  # pragma: no cover - defensive
                _finish(False, f"Qt could not start PDF export: {exc}")

        def _on_pdf_printed(_path: str, ok: bool) -> None:
            if ok and destination.exists():
                _finish(True)
                return
            _finish(False, f"Qt did not finish writing the PDF report: {destination}")

        page.loadFinished.connect(_on_load_finished)
        page.pdfPrintingFinished.connect(_on_pdf_printed)
        timer.timeout.connect(
            lambda: _finish(
                False, f"Timed out while exporting the report PDF after {timeout_ms} ms."
            )
        )
        timer.start(timeout_ms)
        page.load(qt_core.QUrl.fromLocalFile(str(source.resolve())))
        loop.exec()
        page.loadFinished.disconnect(_on_load_finished)
        page.pdfPrintingFinished.disconnect(_on_pdf_printed)
        timer.timeout.disconnect()
        delete(page)
        delete(timer)
        delete(loop)
        qt_core.QCoreApplication.sendPostedEvents(None, 0)
        qt_core.QCoreApplication.processEvents()

        if state["success"]:
            return True, None
        return False, str(state["message"] or f"PDF export failed for {source.name}")
