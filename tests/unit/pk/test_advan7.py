"""
Unit tests for ADVAN7 (general N-compartment linear model, expm backend).

Tests verify:
  - 4-cmt bolus matches scipy.linalg.expm reference
  - 4-cmt infusion matches scipy.linalg.expm reference
  - Near-degenerate systems remain numerically aligned with expm reference
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan5 import _build_rate_matrix, _infer_n_and_parse_rates
from openpkpd.pk.analytical.advan7 import ADVAN7

_PARAMS_4CMT = {
    "K": 0.1,
    "K12": 0.05,
    "K21": 0.025,
    "K13": 0.02,
    "K31": 0.0067,
    "K14": 0.01,
    "K41": 0.005,
    "V1": 10.0,
}

_DEGENERATE_PARAMS = {
    "K": 0.1,
    "K12": 0.04,
    "K21": 0.04,
    "K13": 0.04,
    "K31": 0.04,
    "V1": 10.0,
}


def _scipy_bolus_amounts(params: dict[str, float], dose: float, dose_cmt: int, times: np.ndarray) -> np.ndarray:
    n, kij, ki0 = _infer_n_and_parse_rates(params)
    M = _build_rate_matrix(n, kij, ki0)
    a0 = np.zeros(n)
    a0[dose_cmt - 1] = dose
    out = np.zeros((len(times), n))
    for i, t in enumerate(times):
        if t > 0:
            out[i] = expm(M * t) @ a0
    return out


def _scipy_infusion_amounts(
    params: dict[str, float],
    rate: float,
    duration: float,
    dose_cmt: int,
    times: np.ndarray,
) -> np.ndarray:
    n, kij, ki0 = _infer_n_and_parse_rates(params)
    M = _build_rate_matrix(n, kij, ki0)
    b = np.zeros(n)
    b[dose_cmt - 1] = rate
    eye = np.eye(n)
    out = np.zeros((len(times), n))
    a_end = np.linalg.solve(M, (expm(M * duration) - eye) @ b)
    for i, t in enumerate(times):
        if t <= 0:
            continue
        if t <= duration:
            out[i] = np.linalg.solve(M, (expm(M * t) - eye) @ b)
        else:
            out[i] = expm(M * (t - duration)) @ a_end
    return out


@pytest.mark.unit
def test_advan7_4cmt_bolus_matches_scipy_expm():
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    dose = [DoseEvent(time=0.0, amount=100.0)]

    sol = ADVAN7().solve(_PARAMS_4CMT, dose, times)
    expected = _scipy_bolus_amounts(_PARAMS_4CMT, 100.0, 1, times)

    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, expected[:, 0] / _PARAMS_4CMT["V1"], rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan7_4cmt_infusion_matches_scipy_expm():
    rate, duration = 50.0, 2.0
    dose = [DoseEvent(time=0.0, amount=rate * duration, rate=rate)]
    times = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])

    sol = ADVAN7().solve(_PARAMS_4CMT, dose, times)
    expected = _scipy_infusion_amounts(_PARAMS_4CMT, rate, duration, 1, times)

    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan7_near_degenerate_system_matches_scipy_expm():
    times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    dose = [DoseEvent(time=0.0, amount=100.0)]

    sol = ADVAN7().solve(_DEGENERATE_PARAMS, dose, times)
    expected = _scipy_bolus_amounts(_DEGENERATE_PARAMS, 100.0, 1, times)

    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, expected[:, 0] / _DEGENERATE_PARAMS["V1"], rtol=1e-10, atol=1e-10)
