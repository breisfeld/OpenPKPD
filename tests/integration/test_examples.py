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
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLES_DIR = ROOT / "examples"
ALL_EXAMPLES = tuple(sorted(int(path.name[:2]) for path in EXAMPLES_DIR.glob("[0-9][0-9]_*.py")))
PLOT_EXAMPLES = {1, 2, 3, 4, 5, 7, 9, 11, 12, 14, 16, 17, 20, 21, 22, 23, 24, 33, 34}

SMOKE_TIMEOUTS = {
    10: 360,
    14: 180,
    18: 180,
    20: 180,
}

CONTRACT_TIMEOUTS = {
    1: 120,
    2: 120,
    3: 120,
    4: 120,
    5: 120,
    7: 120,
    8: 120,
    9: 180,
    10: 360,
    11: 120,
    6: 120,
    12: 120,
    13: 120,
    14: 180,
    15: 120,
    16: 120,
    17: 120,
    18: 180,
    19: 120,
    20: 180,
    21: 180,
    22: 120,
    23: 120,
    24: 120,
    25: 120,
    26: 120,
    27: 120,
    28: 120,
    29: 120,
    30: 180,
    31: 120,
    32: 120,
    33: 120,
    34: 120,
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


def _extract_float(stdout: str, pattern: str) -> float:
    match = re.search(pattern, stdout)
    assert match is not None, f"Pattern not found: {pattern!r}"
    return float(match.group(1))


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
def test_example_02_contract(tmp_path: Path) -> None:
    result = _run_example(2, tmp_path, timeout=CONTRACT_TIMEOUTS[2])
    _assert_example_passed(2, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_02_output"
    ofv = _extract_float(stdout, r"OFV:\s+(-?[0-9.]+)")
    assert "Running FOCE estimation..." in stdout
    assert "Method: FOCEI" in stdout
    assert "Converged: True" in stdout
    assert ofv < 0.0
    assert "Near-singular Omega" in stdout
    assert "Figures saved to" in stdout
    assert (output_dir / "02_gof_panel.png").exists()
    assert (output_dir / "02_eta_hist.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_03_contract(tmp_path: Path) -> None:
    result = _run_example(3, tmp_path, timeout=CONTRACT_TIMEOUTS[3])
    _assert_example_passed(3, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_03_output"
    ofv = _extract_float(stdout, r"OFV:\s+(-?[0-9.]+)")
    assert "Running FO on 2-cmt IV model..." in stdout
    assert "Method: FO" in stdout
    assert "Converged: True" in stdout
    assert ofv < 0.0
    assert "THETA:" in stdout
    assert (output_dir / "03_log_conc_time.png").exists()
    assert (output_dir / "03_spaghetti_log.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_04_contract(tmp_path: Path) -> None:
    result = _run_example(4, tmp_path, timeout=CONTRACT_TIMEOUTS[4])
    _assert_example_passed(4, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_04_output"
    k = _extract_float(stdout, r"K=([0-9.]+),")
    v = _extract_float(stdout, r"V=([0-9.]+),")
    e0 = _extract_float(stdout, r"E0=([0-9.]+),")
    emax = _extract_float(stdout, r"Emax=([0-9.]+),")
    ec50 = _extract_float(stdout, r"EC50=([0-9.]+),")
    assert "Running FO on Emax PD model..." in stdout
    assert "Method: FO" in stdout
    assert "Converged: True" in stdout
    assert 0.05 < k < 1.0
    assert 5.0 < v < 100.0
    assert 0.0 < e0 < 10.0
    assert 0.0 < emax < 30.0
    assert 1.0 < ec50 < 100.0
    assert (output_dir / "04_emax_curve.png").exists()
    assert (output_dir / "04_effect_time.png").exists()
    assert (output_dir / "04_gof_panel.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_05_contract(tmp_path: Path) -> None:
    result = _run_example(5, tmp_path, timeout=CONTRACT_TIMEOUTS[5])
    _assert_example_passed(5, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_05_output"
    ofv = _extract_float(stdout, r"OFV:\s+(-?[0-9.]+)")
    assert "Simulating indirect response data..." in stdout
    assert "Fitting..." in stdout
    assert "Method: FO" in stdout
    assert "Converged: True" in stdout
    assert 0.0 < ofv < 100.0
    assert (output_dir / "05_effect_time.png").exists()
    assert (output_dir / "05_hysteresis.png").exists()


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
def test_example_07_contract(tmp_path: Path) -> None:
    result = _run_example(7, tmp_path, timeout=CONTRACT_TIMEOUTS[7])
    _assert_example_passed(7, result)

    stdout = result.stdout
    stderr = result.stderr
    output_dir = tmp_path / "example_07_output"
    assert "Running FOCE..." in stdout
    assert "Method: FOCEI" in stdout
    match = re.search(r"OFV:\s+([0-9]+\.[0-9]+)", stdout)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(75.98, abs=0.02)
    assert "Converged: True" in stdout
    assert "Created 14 figures." in stdout
    assert "Figures saved to" in stdout
    assert "Near-singular Omega" in stdout
    assert "ETA3 shrinkage is 99.8% (>30%)" in stdout
    assert "divide by zero encountered in divide" not in stderr
    assert (output_dir / "07_gof_panel.png").exists()
    assert (output_dir / "07_cwres_qq.png").exists()
    assert (output_dir / "07_eta_hist.png").exists()
    assert (output_dir / "07_ofv_history.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_08_contract(tmp_path: Path) -> None:
    result = _run_example(8, tmp_path, timeout=CONTRACT_TIMEOUTS[8])
    _assert_example_passed(8, result)

    stdout = result.stdout
    assert "Example 08: Transit Compartment Absorption (ADVAN6)" in stdout
    assert "Fitting transit compartment model (FO, maxeval=80)..." in stdout
    assert "Estimation complete:" in stdout
    assert "Usable optimum: False" in stdout
    assert "Warning: the objective remained on the penalty surface." in stdout
    assert "demonstrates ADVAN6 transit-model setup" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_09_contract(tmp_path: Path) -> None:
    result = _run_example(9, tmp_path, timeout=CONTRACT_TIMEOUTS[9])
    _assert_example_passed(9, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_09_output"
    assert "Example 09: 3-Compartment IV Model (ADVAN11)" in stdout
    assert "Running FOCE estimation..." in stdout
    assert "Converged = True" in stdout
    cl = _extract_float(stdout, r"THETA\(1\) \[CL \(L/h\)\]: est=([0-9.]+)")
    v1 = _extract_float(stdout, r"THETA\(2\) \[V1 \(L\)\]: est=([0-9.]+)")
    q2 = _extract_float(stdout, r"THETA\(3\) \[Q2 \(L/h\)\]: est=([0-9.]+)")
    assert cl == pytest.approx(2.0, abs=0.4)
    assert v1 == pytest.approx(10.0, abs=2.5)
    assert q2 == pytest.approx(1.5, abs=0.4)
    assert "Simulating 3 replicates from fitted model..." in stdout
    assert (output_dir / "09_three_cmt_profile.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_10_contract(tmp_path: Path) -> None:
    result = _run_example(10, tmp_path, timeout=CONTRACT_TIMEOUTS[10])
    _assert_example_passed(10, result)

    stdout = result.stdout
    blq = _extract_float(stdout, r"BLQ observations\s+:\s+([0-9]+)")
    blq_pct = _extract_float(stdout, r"BLQ observations\s+:\s+[0-9]+\s+\(([0-9.]+)%\)")
    m1_ofv_match = re.search(r"Fitting BLQ method: M1.*?OFV\s+=\s+([0-9.]+)", stdout, re.S)
    m3_ofv_match = re.search(r"Fitting BLQ method: M3.*?OFV\s+=\s+([0-9.]+)", stdout, re.S)
    m5_ofv_match = re.search(r"Fitting BLQ method: M5.*?OFV\s+=\s+([0-9.]+)", stdout, re.S)
    m1_match = re.search(r"Fitting BLQ method: M1.*?Converged = (True|False).*?n_obs\s+=\s+([0-9]+)", stdout, re.S)
    m3_match = re.search(r"Fitting BLQ method: M3.*?Converged = (True|False).*?n_obs\s+=\s+([0-9]+)", stdout, re.S)
    m5_match = re.search(r"Fitting BLQ method: M5.*?Converged = (True|False).*?n_obs\s+=\s+([0-9]+)", stdout, re.S)
    assert m1_ofv_match is not None
    assert m3_ofv_match is not None
    assert m5_ofv_match is not None
    assert m1_match is not None
    assert m3_match is not None
    assert m5_match is not None
    assert "Example 10: BLQ Handling — M1 vs M3 vs M5" in stdout
    assert blq >= 10
    assert 10.0 <= blq_pct <= 30.0
    assert m1_match.group(1) == "True"
    assert m3_match.group(1) == "True"
    assert m5_match.group(1) == "True"
    assert int(m1_match.group(2)) < int(m3_match.group(2))
    assert int(m3_match.group(2)) == int(m5_match.group(2))
    assert "Best model by AIC: M3" in stdout
    assert float(m3_ofv_match.group(1)) < float(m1_ofv_match.group(1))
    assert float(m3_ofv_match.group(1)) < float(m5_ofv_match.group(1))


@pytest.mark.integration
@pytest.mark.slow
def test_example_11_contract(tmp_path: Path) -> None:
    result = _run_example(11, tmp_path, timeout=CONTRACT_TIMEOUTS[11])
    _assert_example_passed(11, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_11_output"
    const_aic = _extract_float(stdout, r"ConstantHazard AIC\s+:\s+([0-9.]+)")
    weib_aic = _extract_float(stdout, r"Weibull AIC\s+:\s+([0-9.]+)")
    scale = _extract_float(stdout, r"Fitted scale\s+:\s+([0-9.]+)")
    shape = _extract_float(stdout, r"Fitted shape\s+:\s+([0-9.]+)")
    assert "Example 11: Time-to-Event Survival Analysis" in stdout
    assert "Weibull model preferred" in stdout
    assert weib_aic < const_aic
    assert scale == pytest.approx(15.0, abs=2.0)
    assert shape == pytest.approx(1.8, abs=0.2)
    assert (output_dir / "11_tte_survival.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_12_contract(tmp_path: Path) -> None:
    result = _run_example(12, tmp_path, timeout=CONTRACT_TIMEOUTS[12])
    _assert_example_passed(12, result)

    stdout = result.stdout
    assert "Example 12: Non-Compartmental Analysis (NCA)" in stdout
    assert "NCA Summary Table" in stdout
    auc_inf = _extract_float(stdout, r"Geometric mean AUC_inf:\s+([0-9.]+)")
    cl_f = _extract_float(stdout, r"Geometric mean CL/F:\s+([0-9.]+)")
    t_half = _extract_float(stdout, r"Median t½:\s+([0-9.]+)\s+hr")
    assert auc_inf == pytest.approx(19.18, abs=0.2)
    assert cl_f == pytest.approx(5.214, abs=0.05)
    assert t_half == pytest.approx(6.56, abs=0.1)
    assert "Average Bioequivalence — AUC0-inf" in stdout
    assert "Average Bioequivalence — Cmax" in stdout
    assert "Done." in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_13_contract(tmp_path: Path) -> None:
    result = _run_example(13, tmp_path, timeout=CONTRACT_TIMEOUTS[13])
    _assert_example_passed(13, result)

    stdout = result.stdout
    assert "Example 13: Covariate Search — Theophylline PK" in stdout
    assert "Dataset: NONMEMDataset(n_subjects=6, n_rows=64" in stdout
    assert "Fitting base model (FOCE, maxeval=80)..." in stdout
    base_ofv = _extract_float(stdout, r"OFV:\s+(-?[0-9.]+)")
    assert "Converged: True" in stdout
    assert "Manual LRT: WT (power) on CL" in stdout
    delta_ofv = _extract_float(stdout, r"ΔOFV\s+:\s+(-?[0-9.]+)")
    p_value = _extract_float(stdout, r"p-value \(LRT, 1df\):\s+([0-9.]+)")
    match = re.search(r"THETA\(4\) \[WT→CL\]\s+:\s+(-?[0-9]+\.[0-9]+)", stdout)
    assert match is not None
    assert base_ofv == pytest.approx(-109.70, abs=0.05)
    assert delta_ofv == pytest.approx(0.0038, abs=0.002)
    assert p_value == pytest.approx(0.9506, abs=0.02)
    assert float(match.group(1)) == pytest.approx(0.0213, abs=0.001)
    assert ">>> WT (power) on CL is NOT significant at 5% level." in stdout
    assert "SCMEngine: Automatic Stepwise Covariate Search" in stdout
    final_ofv = _extract_float(stdout, r"Final OFV\s+:\s+(-?[0-9.]+)")
    assert final_ofv == pytest.approx(base_ofv, abs=0.02)
    assert "No covariate relationships were accepted." in stdout
    assert "Example 13 complete." in stdout


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
def test_example_15_contract(tmp_path: Path) -> None:
    result = _run_example(15, tmp_path, timeout=CONTRACT_TIMEOUTS[15])
    _assert_example_passed(15, result)

    stdout = result.stdout
    assert "Example 15: Bayesian Estimation via MAP and Laplace Posterior Approximation" in stdout
    assert "Dataset: 12 subjects, 132 rows" in stdout
    assert "Backend used: laplace" in stdout
    assert "THETA(1) [KA (hr⁻¹)]" in stdout
    assert "THETA(2) [CL (L/hr)]" in stdout
    assert "THETA(3) [V (L)]" in stdout
    theta1_match = re.search(r"THETA\(1\)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+1\.0000\s+2000", stdout)
    theta2_match = re.search(r"THETA\(2\)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+1\.0000\s+2000", stdout)
    theta3_match = re.search(r"THETA\(3\)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+1\.0000\s+2000", stdout)
    assert theta1_match is not None
    assert theta2_match is not None
    assert theta3_match is not None
    assert float(theta1_match.group(1)) == pytest.approx(1.7168, abs=0.01)
    assert float(theta2_match.group(1)) == pytest.approx(2.8269, abs=0.01)
    assert float(theta3_match.group(1)) == pytest.approx(33.2203, abs=0.05)
    assert "Laplace Approximation Diagnostics" not in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_16_contract(tmp_path: Path) -> None:
    result = _run_example(16, tmp_path, timeout=CONTRACT_TIMEOUTS[16])
    _assert_example_passed(16, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_16_output"
    assert "Example 16: Delay Differential Equation (DDE) PK model" in stdout
    assert "ODE (no delay)" in stdout
    assert "Analytical" in stdout
    ode_err = _extract_float(stdout, r"Max ODE vs analytical error:\s+([0-9.eE+-]+)")
    dde_gap = _extract_float(stdout, r"Max DDE vs ODE difference:\s+([0-9.eE+-]+)")
    assert ode_err < 1e-4
    assert dde_gap > 0.0
    assert "DDE model with tau=0.5 h produces delayed elimination — as expected." in stdout
    assert (output_dir / "16_dde_model.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_17_contract(tmp_path: Path) -> None:
    result = _run_example(17, tmp_path, timeout=CONTRACT_TIMEOUTS[17])
    _assert_example_passed(17, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_17_output"
    assert "Example 17: SBML model import and simulation" in stdout
    assert "Two-compartment IV model:" in stdout
    assert "Mass balance check: PASSED" in stdout
    assert "ThetaSpec round-trip: PASSED" in stdout
    assert "SBML import example complete." in stdout
    assert (output_dir / "17_sbml_import.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_18_contract(tmp_path: Path) -> None:
    result = _run_example(18, tmp_path, timeout=CONTRACT_TIMEOUTS[18])
    _assert_example_passed(18, result)

    stdout = result.stdout
    conv_match = re.search(r"Converged:\s+([0-9]+)/([0-9]+)", stdout)
    assert conv_match is not None
    assert "Example 18: Parallel bootstrap with get_backend()" in stdout
    assert "Backend: _MultiprocessingBackend" in stdout
    assert conv_match.group(1) == conv_match.group(2)
    assert "Bootstrap 95% confidence intervals:" in stdout
    assert "KA    :" in stdout
    assert "CL    :" in stdout
    assert "V     :" in stdout
    assert "map([1..5], x*2) = [2, 4, 6, 8, 10]" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_19_contract(tmp_path: Path) -> None:
    result = _run_example(19, tmp_path, timeout=CONTRACT_TIMEOUTS[19])
    _assert_example_passed(19, result)

    stdout = result.stdout
    mean_count = _extract_float(stdout, r"Mean count:\s+([0-9.]+)")
    poisson_match = re.search(r"--- Poisson model ---.*?AIC:\s+([0-9.]+)", stdout, re.S)
    zip_match = re.search(r"--- Zero-Inflated Poisson model \(excess zeros\) ---.*?AIC:\s+([0-9.]+)", stdout, re.S)
    assert poisson_match is not None
    assert zip_match is not None
    assert "PART 1: Count PD models" in stdout
    assert "PART 2: Categorical PD models" in stdout
    assert mean_count > 0.0
    assert "P(cat) at C=0.0:" in stdout
    assert "Fitted transition matrix:" in stdout
    assert "Fitted rate matrix Q:" in stdout
    assert float(zip_match.group(1)) < float(poisson_match.group(1))
    assert "Example 19 complete." in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_20_contract(tmp_path: Path) -> None:
    result = _run_example(20, tmp_path, timeout=CONTRACT_TIMEOUTS[20])
    _assert_example_passed(20, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_20_output"
    assert "Example 20: SAEM Estimation" in stdout
    assert "Method: SAEM" in stdout
    assert "Converged:" in stdout
    assert "Estimation warnings:" in stdout
    assert "Running FOCE (reference comparison)..." in stdout
    assert "Parameter           True       FOCE       SAEM" in stdout
    assert "OFV  FOCE =" in stdout
    assert "OFV  SAEM =" in stdout
    ka_match = re.search(r"KA \(hr⁻¹\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", stdout)
    cl_match = re.search(r"CL \(L/hr\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", stdout)
    v_match = re.search(r"V \(L\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", stdout)
    assert ka_match is not None
    assert cl_match is not None
    assert v_match is not None
    assert float(ka_match.group(3)) == pytest.approx(1.5, abs=0.15)
    assert float(cl_match.group(3)) == pytest.approx(2.8, abs=0.2)
    assert float(v_match.group(3)) == pytest.approx(32.9, abs=1.0)
    assert (output_dir / "20_saem_convergence.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_21_contract(tmp_path: Path) -> None:
    result = _run_example(21, tmp_path, timeout=CONTRACT_TIMEOUTS[21])
    _assert_example_passed(21, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_21_output"
    assert "Example 21: Laplacian Estimation with Prior Augmentation" in stdout
    assert "Running FOCE (baseline)..." in stdout
    assert "Running Laplacian (no prior)..." in stdout
    assert "Running Laplacian + prior" in stdout
    foce_match = re.search(r"FOCE OFV =\s+(-?[0-9.]+)", stdout)
    lap_match = re.search(r"Laplacian OFV =\s+(-?[0-9.]+)", stdout)
    prior_match = re.search(r"Laplacian\+Prior OFV =\s+(-?[0-9.]+)", stdout)
    assert foce_match is not None
    assert lap_match is not None
    assert prior_match is not None
    foce_ofv = float(foce_match.group(1))
    lap_ofv = float(lap_match.group(1))
    prior_ofv = float(prior_match.group(1))
    assert foce_ofv < 0.0
    assert lap_ofv > 0.0
    assert prior_ofv > 0.0

    ka_match = re.search(
        r"KA \(hr⁻¹\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        stdout,
    )
    cl_match = re.search(
        r"CL \(L/hr\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        stdout,
    )
    v_match = re.search(
        r"V \(L\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        stdout,
    )
    assert ka_match is not None
    assert cl_match is not None
    assert v_match is not None
    ka_prior, ka_foce, ka_lap, ka_lap_prior = [float(ka_match.group(i)) for i in range(1, 5)]
    cl_prior, cl_foce, cl_lap, cl_lap_prior = [float(cl_match.group(i)) for i in range(1, 5)]
    v_prior, v_foce, v_lap, v_lap_prior = [float(v_match.group(i)) for i in range(1, 5)]

    assert ka_prior == pytest.approx(0.9000, abs=1e-4)
    assert cl_prior == pytest.approx(0.1300, abs=1e-4)
    assert v_prior == pytest.approx(8.7000, abs=1e-4)

    # The informative prior should shrink sparse-data Laplacian estimates toward the prior mean.
    assert abs(ka_lap_prior - ka_prior) < abs(ka_lap - ka_prior)
    assert abs(cl_lap_prior - cl_prior) < abs(cl_lap - cl_prior)
    assert abs(v_lap_prior - v_prior) < abs(v_lap - v_prior)

    # The estimates should remain finite and physiologically plausible.
    assert 0.01 < ka_foce < 20.0
    assert 0.01 < ka_lap < 20.0
    assert 0.01 < ka_lap_prior < 20.0
    assert 0.01 < cl_foce < 5.0
    assert 0.01 < cl_lap < 5.0
    assert 0.01 < cl_lap_prior < 5.0
    assert 1.0 < v_foce < 200.0
    assert 1.0 < v_lap < 200.0
    assert 1.0 < v_lap_prior < 200.0
    assert "These OFVs are not directly comparable across methods." in stdout
    assert "ADVAN2 requires KA > 0" not in stdout
    assert (output_dir / "21_prior_shrinkage.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_22_contract(tmp_path: Path) -> None:
    result = _run_example(22, tmp_path, timeout=CONTRACT_TIMEOUTS[22])
    _assert_example_passed(22, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_22_output"
    assert "Example 22: PBPK Model — 5-Organ Human Template" in stdout
    assert "Compartments: ['lung', 'liver', 'kidney', 'gut', 'central']" in stdout
    assert "Output compartment: 'central' (index 5)" in stdout
    assert "Plasma (central) concentration profile" in stdout
    assert "Tissue concentration comparison at key times" in stdout
    assert (output_dir / "22_pbpk.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_23_contract(tmp_path: Path) -> None:
    result = _run_example(23, tmp_path, timeout=CONTRACT_TIMEOUTS[23])
    _assert_example_passed(23, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_23_output"
    assert "Example 23: Inter-Occasion Variability (IOV) Modelling" in stdout
    assert "Dataset: 8 subjects, 2 occasions, 128 rows" in stdout
    assert "OCC column present: True" in stdout
    assert "Fitting BSV-only model (1 ETA on CL, FO method)..." in stdout
    assert "Fitting BSV+IOV model (BSV on CL + per-occasion IOV, FO method)..." in stdout
    assert "ΔOFV =" in stdout
    assert "(df=2)" in stdout
    assert "-> BSV-only model not significantly improved by adding IOV" in stdout
    assert "ω²_IOV_occ1 =" in stdout
    assert "ω²_IOV_occ2 =" in stdout
    assert "True ω²_BSV_CL ≈ 0.0400  True ω²_IOV_CL ≈ 0.0225" in stdout
    assert (output_dir / "23_iov_etas.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_24_contract(tmp_path: Path) -> None:
    result = _run_example(24, tmp_path, timeout=CONTRACT_TIMEOUTS[24])
    _assert_example_passed(24, result)

    stdout = result.stdout
    assert "Example 24: Advanced PD Models" in stdout
    assert "1. Effect Compartment Model (biophase, Hill equation)" in stdout
    assert "2. Turnover Model (production stimulation, IDR type 1)" in stdout
    assert "3. Tumor Growth Inhibition (Simeoni 2004)" in stdout
    assert "4. Placebo Response Model (disease progression)" in stdout

    effect_match = re.search(
        r"Ke0\s+=\s+([0-9.]+)\s+\(true: 0\.8\).*?"
        r"Emax =\s+([0-9.]+)\s+\(true: 90\.0\).*?"
        r"EC50 =\s+([0-9.]+)\s+\(true: 2\.0\).*?"
        r"n\s+=\s+([0-9.]+)\s+\(true: 2\.0\).*?"
        r"OFV\s+=\s+(-?[0-9.]+)\s+AIC =\s+(-?[0-9.]+)\s+converged = (True|False)",
        stdout,
        re.S,
    )
    turnover_match = re.search(
        r"Kin\s+=\s+([0-9.]+)\s+\(true: 2\.0\).*?"
        r"Kout\s+=\s+([0-9.]+)\s+\(true: 0\.5\).*?"
        r"EC50_in =\s+([0-9.]+)\s+\(true: 1\.5\).*?"
        r"Emax_in =\s+([0-9.]+)\s+\(true: 1\.0\).*?"
        r"OFV =\s+(-?[0-9.]+)\s+AIC =\s+(-?[0-9.]+)\s+converged = (True|False)",
        stdout,
        re.S,
    )
    tgi_match = re.search(
        r"lambda0 =\s+([0-9.]+)\s+\(true: 0\.25\).*?"
        r"lambda1 =\s+([0-9.]+)\s+\(true: 2\.0\).*?"
        r"K1\s+=\s+([0-9.]+)\s+\(true: 0\.2\).*?"
        r"K2\s+=\s+([0-9.]+)\s+\(true: 0\.025\).*?"
        r"X0\s+=\s+([0-9.]+)\s+\(true: 150\.0\).*?"
        r"OFV =\s+(-?[0-9.]+)\s+AIC =\s+(-?[0-9.]+)\s+converged = (True|False)",
        stdout,
        re.S,
    )
    placebo_match = re.search(
        r"E0\s+=\s+([0-9.]+)\s+\(true: 60\.0\).*?"
        r"kdeg\s+=\s+([0-9.]+)\s+\(true: 0\.02\).*?"
        r"Eplacebo =\s+([0-9.]+)\s+\(true: 20\.0\).*?"
        r"kpl\s+=\s+([0-9.]+)\s+\(true: 0\.05\).*?"
        r"OFV =\s+(-?[0-9.]+)\s+AIC =\s+(-?[0-9.]+)\s+converged = (True|False)",
        stdout,
        re.S,
    )
    assert effect_match is not None
    assert turnover_match is not None
    assert tgi_match is not None
    assert placebo_match is not None

    ke0, emax, ec50, hill_n, effect_ofv, effect_aic, effect_conv = effect_match.groups()
    kin, kout, ec50_in, emax_in, turnover_ofv, turnover_aic, turnover_conv = turnover_match.groups()
    lambda0, lambda1, k1, k2, x0, tgi_ofv, tgi_aic, tgi_conv = tgi_match.groups()
    e0, kdeg, eplacebo, kpl, placebo_ofv, placebo_aic, placebo_conv = placebo_match.groups()

    # Plausible recovery for each example fit.
    assert float(ke0) == pytest.approx(0.8, abs=0.35)
    assert float(emax) == pytest.approx(90.0, abs=15.0)
    assert float(ec50) == pytest.approx(2.0, abs=0.25)
    assert float(hill_n) == pytest.approx(2.0, abs=1.5)

    assert float(kin) == pytest.approx(2.0, abs=0.15)
    assert float(kout) == pytest.approx(0.5, abs=0.08)
    assert float(ec50_in) == pytest.approx(1.5, abs=0.25)
    assert float(emax_in) == pytest.approx(1.0, abs=0.1)

    assert float(lambda0) == pytest.approx(0.25, abs=0.08)
    assert float(lambda1) == pytest.approx(2.0, abs=0.1)
    assert float(k1) == pytest.approx(0.2, abs=0.03)
    assert float(k2) == pytest.approx(0.025, abs=0.01)
    assert float(x0) == pytest.approx(150.0, abs=5.0)

    assert float(e0) == pytest.approx(60.0, abs=2.0)
    assert float(kdeg) == pytest.approx(0.02, abs=0.01)
    assert float(eplacebo) == pytest.approx(20.0, abs=7.0)
    assert float(kpl) == pytest.approx(0.05, abs=0.02)

    # AIC accounting must remain consistent with the number of fitted parameters.
    assert float(effect_aic) == pytest.approx(float(effect_ofv) + 8.0, abs=0.02)
    assert float(turnover_aic) == pytest.approx(float(turnover_ofv) + 8.0, abs=0.02)
    assert float(tgi_aic) == pytest.approx(float(tgi_ofv) + 12.0, abs=0.02)
    assert float(placebo_aic) == pytest.approx(float(placebo_ofv) + 8.0, abs=0.02)

    # Turnover should remain the best-fitting model in this synthetic set.
    aics = {
        "effect": float(effect_aic),
        "turnover": float(turnover_aic),
        "tgi": float(tgi_aic),
        "placebo": float(placebo_aic),
    }
    assert min(aics, key=aics.get) == "turnover"
    assert turnover_conv == "True"
    assert placebo_conv == "True"

    assert "--- Model AIC summary ---" in stdout
    assert "Turnover" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_25_contract(tmp_path: Path) -> None:
    result = _run_example(25, tmp_path, timeout=CONTRACT_TIMEOUTS[25])
    _assert_example_passed(25, result)

    stdout = result.stdout
    assert "Method: FOCEI" in stdout
    assert "Converged: True" in stdout
    assert "OFV:" in stdout
    assert "OMEGA (diagonal):" in stdout
    assert "SIGMA (diagonal):" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_26_contract(tmp_path: Path) -> None:
    result = _run_example(26, tmp_path, timeout=CONTRACT_TIMEOUTS[26])
    _assert_example_passed(26, result)

    stdout = result.stdout
    assert "Problem: Warfarin PK — FOCEI optimizer controls demo" in stdout
    assert "Method: FOCE, interaction=True" in stdout
    assert "Outer optimizer: L-BFGS-B" in stdout
    assert "Fallback optimizer: POWELL" in stdout
    assert "Retry OMEGA scales: (0.5, 0.25, 0.1)" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_27_contract(tmp_path: Path) -> None:
    result = _run_example(27, tmp_path, timeout=CONTRACT_TIMEOUTS[27])
    _assert_example_passed(27, result)

    stdout = result.stdout
    assert "Running phenobarbital FO estimation..." in stdout
    assert "Method: FO" in stdout
    assert "Converged: True" in stdout
    assert "CL/kg =" in stdout
    assert "V/kg  =" in stdout
    assert "t1/2  =" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_28_contract(tmp_path: Path) -> None:
    result = _run_example(28, tmp_path, timeout=CONTRACT_TIMEOUTS[28])
    _assert_example_passed(28, result)

    stdout = result.stdout
    assert "Example 28: Indometh NCA" in stdout
    assert "Subject" in stdout
    assert "AUCinf" in stdout
    assert "Mean AUCinf:" in stdout
    assert "Mean t_half:" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_29_contract(tmp_path: Path) -> None:
    result = _run_example(29, tmp_path, timeout=CONTRACT_TIMEOUTS[29])
    _assert_example_passed(29, result)

    stdout = result.stdout
    assert "Example 29: Optimal design with PFIM" in stdout
    assert "Reference times:" in stdout
    assert "Optimized times:" in stdout
    assert "D-efficiency vs reference:" in stdout
    assert "Expected SE:" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_30_contract(tmp_path: Path) -> None:
    result = _run_example(30, tmp_path, timeout=CONTRACT_TIMEOUTS[30])
    _assert_example_passed(30, result)

    stdout = result.stdout
    assert "Example 30: 4-Compartment General Linear Model (ADVAN5)" in stdout
    assert "Fitting ADVAN5 (N=4) via FOCE" in stdout
    assert "Max |ΔIPRED|" in stdout
    assert "Done." in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_31_contract(tmp_path: Path) -> None:
    result = _run_example(31, tmp_path, timeout=CONTRACT_TIMEOUTS[31])
    _assert_example_passed(31, result)

    stdout = result.stdout
    assert "Example 31: IMPMAP warm-start diagnostics on warfarin PK" in stdout
    assert "Method: IMPMAP" in stdout
    assert "Short-run converged: False" in stdout
    assert "Warm start used: True" in stdout
    assert "Warm start method: FOCEI" in stdout
    assert "Warm start converged: True" in stdout
    assert "Recorded OFV evaluations:" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_32_contract(tmp_path: Path) -> None:
    result = _run_example(32, tmp_path, timeout=CONTRACT_TIMEOUTS[32])
    _assert_example_passed(32, result)

    stdout = result.stdout
    assert "Example 32: Nonparametric support-point estimation" in stdout
    assert "Synthetic dataset: 12 subjects, 1 ETA on CL, seed=42." in stdout
    assert "Method: NONPARAMETRIC" in stdout
    assert "n_support_points: 12" in stdout
    top_match = re.search(r"rank=1 weight=([0-9.]+) eta=\[(-?[0-9.]+)\]", stdout)
    mean_match = re.search(r"mean ETA:\s+\[(-?[0-9.]+)\]", stdout)
    var_match = re.search(r"variance ETA:\s+\[([0-9.]+)\]", stdout)
    assert top_match is not None
    assert mean_match is not None
    assert var_match is not None
    assert float(top_match.group(1)) == pytest.approx(0.4992, abs=1e-4)
    assert float(top_match.group(2)) == pytest.approx(-0.0780737, abs=1e-6)
    assert float(mean_match.group(1)) == pytest.approx(-0.0046, abs=5e-4)
    assert float(var_match.group(1)) == pytest.approx(0.0159, abs=5e-4)


@pytest.mark.integration
@pytest.mark.slow
def test_example_33_contract(tmp_path: Path) -> None:
    result = _run_example(33, tmp_path, timeout=CONTRACT_TIMEOUTS[33])
    _assert_example_passed(33, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_33_output"
    full_peak = _extract_float(stdout, r"Full TMDD\s+peak C:\s+([0-9.]+)\s+nmol/L")
    qssa_peak = _extract_float(stdout, r"QSSA\s+peak C:\s+([0-9.]+)\s+nmol/L")
    mm_peak = _extract_float(stdout, r"MM\s+peak C:\s+([0-9.]+)\s+nmol/L")
    full_auc = _extract_float(stdout, r"Full TMDD\s+AUC\(0-168h\)\s+≈\s+([0-9.]+)\s+nmol·h/L")
    qssa_auc = _extract_float(stdout, r"QSSA\s+AUC\(0-168h\)\s+≈\s+([0-9.]+)\s+nmol·h/L")
    mm_auc = _extract_float(stdout, r"MM\s+AUC\(0-168h\)\s+≈\s+([0-9.]+)\s+nmol·h/L")
    assert full_peak == pytest.approx(0.99, abs=0.05)
    assert qssa_peak == pytest.approx(0.99, abs=0.05)
    assert mm_peak == pytest.approx(1.00, abs=0.05)
    assert full_auc < mm_auc
    assert qssa_auc < mm_auc
    assert (output_dir / "33_tmdd_model.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_34_contract(tmp_path: Path) -> None:
    result = _run_example(34, tmp_path, timeout=CONTRACT_TIMEOUTS[34])
    _assert_example_passed(34, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_34_output"
    assert "Single-dose NCA" in stdout
    cmax = _extract_float(stdout, r"Cmax\s+=\s+([0-9.]+)\s+mg/L")
    auc_inf = _extract_float(stdout, r"AUC\(0-∞\)\s+=\s+([0-9.]+)\s+mg·h/L")
    assert "Steady-state NCA (SS=1, tau=12 h)" in stdout
    ctrough = _extract_float(stdout, r"Ctrough\s+=\s+([0-9.]+)\s+mg/L")
    cpeak_ss = _extract_float(stdout, r"Cpeak_ss\s+=\s+([0-9.]+)\s+mg/L")
    cavg_ss = _extract_float(stdout, r"Cavg_ss\s+=\s+([0-9.]+)\s+mg/L")
    auctau = _extract_float(stdout, r"AUCtau\s+=\s+([0-9.]+)\s+mg·h/L")
    rac = _extract_float(stdout, r"R_ac\s+=\s+([0-9.]+)")
    fluct = _extract_float(stdout, r"%Fluct\s+=\s+([0-9.]+)%")
    assert cmax == pytest.approx(5.57, abs=0.1)
    assert auc_inf == pytest.approx(53.18, abs=0.5)
    assert ctrough < cpeak_ss
    assert ctrough < cavg_ss < cpeak_ss
    assert auctau == pytest.approx(49.69, abs=0.5)
    assert 0.7 < rac < 1.2
    assert fluct > 50.0
    assert (output_dir / "34_multidose_ss_nca.png").exists()
