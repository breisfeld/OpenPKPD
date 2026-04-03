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
ALL_EXAMPLES = tuple(sorted(int(path.name[:2]) for path in EXAMPLES_DIR.glob("[0-9][0-9]_*.py")))
PLOT_EXAMPLES = {1, 2, 3, 4, 5, 7, 9, 11, 12, 14, 16, 17, 20, 21, 22, 23, 24, 33, 34}

SMOKE_TIMEOUTS = {
    14: 180,
    18: 180,
    20: 180,
}

CONTRACT_TIMEOUTS = {
    1: 120,
    6: 120,
    12: 120,
    13: 120,
    14: 180,
    15: 120,
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
def test_example_13_contract(tmp_path: Path) -> None:
    result = _run_example(13, tmp_path, timeout=CONTRACT_TIMEOUTS[13])
    _assert_example_passed(13, result)

    stdout = result.stdout
    assert "Example 13: Covariate Search — Theophylline PK" in stdout
    assert "Dataset: NONMEMDataset(n_subjects=6, n_rows=64" in stdout
    assert "Fitting base model (FOCE, maxeval=80)..." in stdout
    assert "OFV: -109.7020" in stdout
    assert "Converged: True" in stdout
    assert "Manual LRT: WT (power) on CL" in stdout
    assert "ΔOFV              : 0.0038" in stdout
    assert "p-value (LRT, 1df): 0.9506" in stdout
    assert "THETA(4) [WT→CL]  : 0.0212" in stdout
    assert ">>> WT (power) on CL is NOT significant at 5% level." in stdout
    assert "SCMEngine: Automatic Stepwise Covariate Search" in stdout
    assert "Final OFV : -109.7020" in stdout
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
    assert "THETA(1) [KA (hr⁻¹)] = 1.4792" in stdout
    assert "THETA(2) [CL (L/hr)] = 2.7181" in stdout
    assert "THETA(3) [V (L)] = 32.4609" in stdout
    assert "THETA(1)       1.7134" in stdout
    assert "THETA(2)       2.8306" in stdout
    assert "THETA(3)      33.2099" in stdout
    assert "Laplace Approximation Diagnostics" not in stdout


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


@pytest.mark.integration
@pytest.mark.slow
def test_example_21_contract(tmp_path: Path) -> None:
    result = _run_example(21, tmp_path, timeout=CONTRACT_TIMEOUTS[21])
    _assert_example_passed(21, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_21_output"
    assert "Example 21: Laplacian Estimation with Prior Augmentation" in stdout
    assert "Running FOCE (baseline)..." in stdout
    assert "FOCE OFV = -58.7740" in stdout
    assert "Running Laplacian (no prior)..." in stdout
    assert "Laplacian OFV = -116.8088" in stdout
    assert "Running Laplacian + prior" in stdout
    assert "Laplacian+Prior OFV = -101.3085" in stdout
    assert "KA (hr⁻¹)         0.9000     0.5363       0.5528       0.5510" in stdout
    assert "CL (L/hr)         0.1300     0.2794       0.2676       0.2667" in stdout
    assert "V (L)             8.7000    11.4485      11.9031      11.8525" in stdout
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
    assert "0.25             1.5935" in stdout
    assert "0.25       1.5935       8.0686       4.6843" in stdout
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
    assert "Ke0  = 0.499  (true: 0.8)" in stdout
    assert "Emax = 80.07  (true: 90.0)" in stdout
    assert "OFV  = 181.29  AIC = 189.29  converged = True" in stdout
    assert "2. Turnover Model (production stimulation, IDR type 1)" in stdout
    assert "Kin     = 2.051  (true: 2.0)" in stdout
    assert "OFV = -110.84  AIC = -102.84  converged = True" in stdout
    assert "3. Tumor Growth Inhibition (Simeoni 2004)" in stdout
    assert "lambda0 = 0.3007  (true: 0.25)" in stdout
    assert "OFV = 42.27  AIC = 54.27  converged = True" in stdout
    assert "4. Placebo Response Model (disease progression)" in stdout
    assert "E0       = 60.51  (true: 60.0)" in stdout
    assert "OFV = 31.16  AIC = 39.16  converged = True" in stdout
    assert "--- Model AIC summary ---" in stdout
    assert "Turnover             AIC = -102.84  (converged=True)" in stdout


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
    assert "rank=1 weight=0.4992 eta=[-0.07807372]" in stdout
    assert "mean ETA:     [-0.0046]" in stdout
    assert "variance ETA: [0.0159]" in stdout


@pytest.mark.integration
@pytest.mark.slow
def test_example_33_contract(tmp_path: Path) -> None:
    result = _run_example(33, tmp_path, timeout=CONTRACT_TIMEOUTS[33])
    _assert_example_passed(33, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_33_output"
    assert "Full TMDD  peak C: 0.989 nmol/L" in stdout
    assert "QSSA       peak C: 0.994 nmol/L" in stdout
    assert "MM         peak C: 1.000 nmol/L" in stdout
    assert "Full TMDD  AUC(0-168h) ≈ 1.6 nmol·h/L" in stdout
    assert "QSSA       AUC(0-168h) ≈ 1.4 nmol·h/L" in stdout
    assert "MM         AUC(0-168h) ≈ 15.0 nmol·h/L" in stdout
    assert (output_dir / "33_tmdd_model.png").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_example_34_contract(tmp_path: Path) -> None:
    result = _run_example(34, tmp_path, timeout=CONTRACT_TIMEOUTS[34])
    _assert_example_passed(34, result)

    stdout = result.stdout
    output_dir = tmp_path / "example_34_output"
    assert "Single-dose NCA" in stdout
    assert "Cmax     = 5.572 mg/L" in stdout
    assert "AUC(0-∞) = 53.18 mg·h/L" in stdout
    assert "Steady-state NCA (SS=1, tau=12 h)" in stdout
    assert "Ctrough  = 1.697 mg/L" in stdout
    assert "Cpeak_ss = 6.830 mg/L" in stdout
    assert "Cavg_ss  = 4.141 mg/L" in stdout
    assert "AUCtau   = 49.69 mg·h/L" in stdout
    assert "R_ac     = 0.934" in stdout
    assert "%%Fluct  = 123.9%%" in stdout
    assert (output_dir / "34_multidose_ss_nca.png").exists()
