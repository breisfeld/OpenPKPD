"""
Integration tests: marimo notebook smoke tests.

Each test runs a notebook in headless mode via ``marimo run`` and verifies
it exits with code 0 (no unhandled exceptions in any cell).

Notebooks are skipped automatically when:
  - marimo is not installed (ImportError)
  - matplotlib is not installed (many notebooks require it)
  - The test is collected without the ``notebooks`` extra

Tests are marked ``slow`` because each notebook runs a full estimation
pipeline (FO/FOCE fit on embedded data), typically 5–30 s each.

Run with:
    just run-notebook notebooks/01_quickstart.py   # single notebook
    pytest tests/integration/test_notebooks.py -v  # full suite
    pytest tests/integration/test_notebooks.py -v -m "not slow"  # skip slow
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parents[2] / "notebooks"

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


def _marimo_available() -> bool:
    try:
        import marimo  # noqa: F401

        return True
    except ImportError:
        return False


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


skip_no_marimo = pytest.mark.skipif(
    not _marimo_available(),
    reason="marimo not installed (install with: uv sync --extra notebooks)",
)

skip_no_matplotlib = pytest.mark.skipif(
    not _matplotlib_available(),
    reason="matplotlib not installed (install with: uv sync --extra notebooks)",
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_notebook(path: Path, timeout: int = 120) -> tuple[subprocess.CompletedProcess[str], str]:
    """Execute a marimo notebook by exporting it to HTML and return the HTML text."""
    with tempfile.TemporaryDirectory(prefix="openpkpd-notebook-") as tmpdir:
        output_path = Path(tmpdir) / f"{path.stem}.html"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html",
                str(path),
                "-o",
                str(output_path),
                "-f",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        html = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        return result, html


def _assert_notebook_output_contains(
    result: subprocess.CompletedProcess[str],
    html: str,
    notebook: str,
    expected_text: str,
) -> None:
    output = f"{result.stdout}\n{result.stderr}\n{html}"
    assert expected_text in output, (
        f"Notebook {notebook} did not emit expected text: {expected_text!r}\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}\n"
        f"HTML:\n{html[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Parametrised smoke tests
# ---------------------------------------------------------------------------

# Notebooks that only require openpkpd core (no matplotlib)
_CORE_ONLY_NOTEBOOKS = [
    "00_index.py",
    "02_data_handling.py",
]

# Notebooks that require matplotlib in addition to openpkpd
_PLOT_NOTEBOOKS = [
    "01_quickstart.py",
    "03_estimation_methods.py",
    "04_simulation_vpc_npde.py",
    "05_nca.py",
    "06_pk_subroutines.py",
    "07_pkpd_models.py",
    "08_diagnostics_plots.py",
    "09_covariate_modeling.py",
    "10_inference_bootstrap.py",
    "11_advanced.py",
]


@skip_no_marimo
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("notebook", _CORE_ONLY_NOTEBOOKS)
def test_notebook_core(notebook: str) -> None:
    """Notebook runs without errors (core dependencies only)."""
    path = NOTEBOOKS_DIR / notebook
    if not path.exists():
        pytest.skip(f"Notebook not found: {path}")

    result, _html = _run_notebook(path)
    assert result.returncode == 0, (
        f"Notebook {notebook} failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )


@skip_no_marimo
@skip_no_matplotlib
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("notebook", _PLOT_NOTEBOOKS)
def test_notebook_plots(notebook: str) -> None:
    """Notebook runs without errors (requires matplotlib)."""
    path = NOTEBOOKS_DIR / notebook
    if not path.exists():
        pytest.skip(f"Notebook not found: {path}")

    result, _html = _run_notebook(path, timeout=180)
    assert result.returncode == 0, (
        f"Notebook {notebook} failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )


@skip_no_marimo
@skip_no_matplotlib
@pytest.mark.slow
@pytest.mark.integration
def test_estimation_notebook_reports_focei_advanced_options() -> None:
    notebook = "03_estimation_methods.py"
    result, html = _run_notebook(NOTEBOOKS_DIR / notebook, timeout=180)
    assert result.returncode == 0
    _assert_notebook_output_contains(result, html, notebook, "FOCEI advanced options")
    _assert_notebook_output_contains(result, html, notebook, "outer=L-BFGS-B")
    _assert_notebook_output_contains(result, html, notebook, "fallback=Powell")


@skip_no_marimo
@skip_no_matplotlib
@pytest.mark.slow
@pytest.mark.integration
def test_pk_subroutines_notebook_reports_solver_options() -> None:
    notebook = "06_pk_subroutines.py"
    result, html = _run_notebook(NOTEBOOKS_DIR / notebook, timeout=180)
    assert result.returncode == 0
    _assert_notebook_output_contains(result, html, notebook, "ODE solver options")
    _assert_notebook_output_contains(result, html, notebook, "ADVAN6 method=RK45")
    _assert_notebook_output_contains(result, html, notebook, "ADVAN8 method=Radau")


@skip_no_marimo
@skip_no_matplotlib
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    ("notebook", "expected_texts"),
    [
        (
            "00_index.py",
            [
                "OpenPKPD — Notebook Library",
                "OpenPKPD version:",
                "Quick Start",
                "Advanced Topics",
            ],
        ),
        (
            "02_data_handling.py",
            [
                "Subjects: 3",
                "Observations: 18",
                "Covariates: ['WT', 'AGE', 'SEX']",
                "BLQ",
                "Occasion Variability",
            ],
        ),
        (
            "01_quickstart.py",
            [
                "Method: FO",
                "Converged: True",
            ],
        ),
        (
            "04_simulation_vpc_npde.py",
            [
                "Method: FOCEI",
                "OFV: 117.2815",
                "Converged: True",
                "VPC summary",
                "Observed bins: 8",
                "Simulated replicates: 200",
                "NPDE Summary",
                "Mean NPDE : +0.1388",
                "Observed within PI:   97.5%",
                "Numerical Predictive Check",
            ],
        ),
        (
            "05_nca.py",
            [
                "NCA Parameters",
                "Average Bioequivalence — AUC",
                "Required N per sequence",
            ],
        ),
        (
            "07_pkpd_models.py",
            [
                "Direct Emax Model",
                "Indirect Response Model",
                "Effect Compartment (Ce) Model",
            ],
        ),
        (
            "09_covariate_modeling.py",
            [
                "Base model OFV:",
                "Candidate relationships: 4",
                "Stepwise Covariate Modeling (SCM) Summary",
                "Retained relationship count:",
                "Top SCM signal:",
            ],
        ),
        (
            "10_inference_bootstrap.py",
            [
                "1-cmt FO   OFV =",
                "1-cmt FO+WT OFV =",
                "1-cmt FOCE OFV =",
            ],
        ),
        (
            "11_advanced.py",
            [
                "Prior specification:",
                "FIM trace (sum of diagonal):",
                "FIM trace (sum of diagonal): 0.143",
                "FIM det (D-criterion): 6.6788e-07",
                "Report HTML length:",
            ],
        ),
        (
            "03_estimation_methods.py",
            [
                # Existing method sections
                "FOCE",
                "SAEM",
                # Bayesian section added in P1.8+P1.10
                # Note: marimo URL-encodes source in HTML (= → %3D, " → %22),
                # so only alphanumeric/underscore strings can be checked literally.
                "Bayesian",
                "nsamples",
                "posterior_samples_by_chain",
                "mcmc_trace_by_chain_plot",
                "rhat_plot",
                "ess_plot",
                # Convergence interpretation table
                "R-hat",
                "result.converged",
            ],
        ),
        (
            "08_diagnostics_plots.py",
            [
                # Section 7: MCMC Diagnostic Plots
                "MCMC Diagnostic Plots",
                "mcmc_trace_by_chain_plot",
                "rhat_plot",
                "ess_plot",
                "compute_rhat",
                "compute_ess",
                "compute_autocorr",
                # Interpretation guidance
                "Excellent convergence",
                "posterior_density_plot",
                "posterior_forest_plot",
            ],
        ),
    ],
)
def test_notebooks_emit_expected_results(notebook: str, expected_texts: list[str]) -> None:
    result, html = _run_notebook(NOTEBOOKS_DIR / notebook, timeout=180)
    assert result.returncode == 0
    for expected_text in expected_texts:
        _assert_notebook_output_contains(result, html, notebook, expected_text)
