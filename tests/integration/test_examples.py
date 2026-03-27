"""
Integration tests for shipped numbered example scripts.

The suite is intentionally tiered:
  - A smoke layer ensures every shipped example script launches successfully.
  - A contract layer asserts concrete outputs for a curated flagship subset.

This keeps CI coverage broad without reducing all example validation to
"process exited with code 0".
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLES_DIR = ROOT / "examples"
ALL_EXAMPLES = tuple(range(1, 25))
PLOT_EXAMPLES = {1, 2, 3, 4, 5, 7, 9, 11, 12, 14, 16, 17, 20, 21, 22, 23, 24}

SMOKE_TIMEOUTS = {
    14: 180,
    18: 180,
    20: 180,
}

CONTRACT_TIMEOUTS = {
    1: 120,
    6: 120,
    12: 120,
    14: 180,
    20: 180,
}


def _example_path(num: int) -> Path:
    matches = sorted(EXAMPLES_DIR.glob(f"{num:02d}_*.py"))
    if len(matches) != 1:
        raise AssertionError(f"Could not uniquely resolve example {num:02d}")
    return matches[0]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_example(
    num: int,
    tmp_path: Path,
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    output_dir = tmp_path / f"example_{num:02d}_output"
    mpl_config_dir = tmp_path / f"example_{num:02d}_mplconfig"
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(mpl_config_dir)
    env["OPENPKPD_EXAMPLE_OUTPUT"] = str(output_dir)

    return subprocess.run(
        [sys.executable, str(_example_path(num))],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_example_passed(num: int, result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"Example {num:02d} failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout[-4000:]}\n"
        f"STDERR:\n{result.stderr[-4000:]}"
    )


def _skip_if_optional_deps_missing(num: int) -> None:
    if num in PLOT_EXAMPLES and not _module_available("matplotlib"):
        pytest.skip("matplotlib not installed (install with: uv sync --extra plots)")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("num", ALL_EXAMPLES)
def test_example_scripts_smoke(num: int, tmp_path: Path) -> None:
    """Each shipped numbered example script runs successfully."""
    _skip_if_optional_deps_missing(num)

    result = _run_example(num, tmp_path, timeout=SMOKE_TIMEOUTS.get(num, 120))
    _assert_example_passed(num, result)


@pytest.mark.integration
@pytest.mark.slow
def test_example_01_contract(tmp_path: Path) -> None:
    result = _run_example(1, tmp_path, timeout=CONTRACT_TIMEOUTS[1])
    _assert_example_passed(1, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_01_output"
    assert "Running FO estimation..." in stdout
    assert "Converged: True" in stdout
    assert "KA =" in stdout
    assert "CL =" in stdout
    assert "V  =" in stdout
    assert "Figures saved to" in stdout
    assert (output_dir / "01_spaghetti.png").exists()
    assert (output_dir / "01_conc_time.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_06_contract(tmp_path: Path) -> None:
    result = _run_example(6, tmp_path, timeout=CONTRACT_TIMEOUTS[6])
    _assert_example_passed(6, result)

    stdout = result.stdout
    assert "Parsed control stream:" in stdout
    assert "Problem: Theophylline via control stream" in stdout
    assert "ADVAN: 2" in stdout
    assert "n_theta: 4" in stdout
    assert "Running estimation" in stdout
    assert "Converged: True" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_12_contract(tmp_path: Path) -> None:
    result = _run_example(12, tmp_path, timeout=CONTRACT_TIMEOUTS[12])
    _assert_example_passed(12, result)

    stdout = result.stdout
    assert "Example 12: Non-Compartmental Analysis (NCA)" in stdout
    assert "NCA Summary Table" in stdout
    assert "Geometric mean AUC_inf: 19.18" in stdout
    assert "Geometric mean CL/F:    5.214" in stdout
    assert "Median t½:              6.56 hr" in stdout
    assert "Average Bioequivalence — AUC0-inf" in stdout
    assert "Average Bioequivalence — Cmax" in stdout
    assert "Done." in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_14_contract(tmp_path: Path) -> None:
    result = _run_example(14, tmp_path, timeout=CONTRACT_TIMEOUTS[14])
    _assert_example_passed(14, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_14_output"
    assert "VPCEngine computation complete." in stdout
    assert "VPC percentiles computed for 30 time points." in stdout
    assert "Created 5 figures." in stdout
    assert "All figures saved to" in stdout
    assert (output_dir / "14_vpc_prediction_interval.png").exists()
    assert (output_dir / "14_vpc_engine.png").exists()
    assert (output_dir / "14_vpc_model_perf.png").exists()
    assert (output_dir / "14_simulation_panel.png").exists()
    assert (output_dir / "14_spaghetti.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_20_contract(tmp_path: Path) -> None:
    result = _run_example(20, tmp_path, timeout=CONTRACT_TIMEOUTS[20])
    _assert_example_passed(20, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_20_output"
    assert "Example 20: SAEM Estimation" in stdout
    assert "Method: SAEM" in stdout
    assert "Converged: True" in stdout
    assert "Running FOCE (reference comparison)..." in stdout
    assert "Parameter           True       FOCE       SAEM" in stdout
    assert "OFV  FOCE =" in stdout
    assert "OFV  SAEM =" in stdout
    assert (output_dir / "20_saem_convergence.png").exists()
