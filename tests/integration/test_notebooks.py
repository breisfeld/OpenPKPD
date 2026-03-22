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


def _run_notebook(path: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a marimo notebook headlessly and return the CompletedProcess."""
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "run", "--headless", str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


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

    result = _run_notebook(path)
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

    result = _run_notebook(path, timeout=180)
    assert result.returncode == 0, (
        f"Notebook {notebook} failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
