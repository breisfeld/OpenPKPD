"""Exact and boundary tests for ADVAN3 (2-compartment IV)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.utils.errors import PKError


def _one_cmt_iv_infusion_amounts(rate: float, duration: float, k: float, t) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    amount = np.zeros_like(t_arr)
    during = t_arr <= duration
    after = ~during
    amount[during] = rate / k * (1.0 - np.exp(-k * t_arr[during]))
    a_end = rate / k * (1.0 - np.exp(-k * duration))
    amount[after] = a_end * np.exp(-k * (t_arr[after] - duration))
    return amount


def _two_cmt_bolus_amounts(dose: float, k: float, k12: float, k21: float, t) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    out = np.zeros((len(t_arr), 2), dtype=float)
    m = np.array([[-(k + k12), k21], [k12, -k21]], dtype=float)
    y0 = np.array([dose, 0.0], dtype=float)
    positive = t_arr > 0.0
    for i, dt in enumerate(t_arr[positive]):
        out[np.where(positive)[0][i]] = expm(m * dt) @ y0
    return out


def _two_cmt_infusion_amounts(
    rate: float, duration: float, k: float, k12: float, k21: float, t
) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    out = np.zeros((len(t_arr), 2), dtype=float)
    m = np.array([[-(k + k12), k21], [k12, -k21]], dtype=float)
    u = np.array([rate, 0.0], dtype=float)
    eye = np.eye(2)
    a_end = np.linalg.solve(m, (expm(m * duration) - eye) @ u)

    for i, ti in enumerate(t_arr):
        if ti <= duration:
            out[i] = np.linalg.solve(m, (expm(m * ti) - eye) @ u)
        else:
            out[i] = expm(m * (ti - duration)) @ a_end
    return out


@pytest.mark.unit
def test_advan3_matches_one_compartment_iv_bolus_when_distribution_is_off():
    advan3 = ADVAN3()
    params = {"K": 0.2, "K12": 0.0, "K21": 0.0, "V1": 10.0}
    dose = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

    sol = advan3.solve(params, dose, times)
    expected_a1 = 100.0 * np.exp(-params["K"] * times)

    np.testing.assert_allclose(sol.amounts[:, 0], expected_a1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.amounts[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol.ipred, expected_a1 / params["V1"], rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan3_matches_one_compartment_iv_infusion_when_distribution_is_off():
    advan3 = ADVAN3()
    params = {"K": 0.2, "K12": 0.0, "K21": 0.0, "V1": 10.0}
    dose = [DoseEvent(time=0.0, amount=90.0, rate=30.0, compartment=1)]
    times = np.array([0.5, 1.0, 2.0, 3.0, 6.0, 12.0])

    sol = advan3.solve(params, dose, times)
    expected_a1 = _one_cmt_iv_infusion_amounts(dose[0].rate, 3.0, params["K"], times)

    np.testing.assert_allclose(sol.amounts[:, 0], expected_a1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.amounts[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol.ipred, expected_a1 / params["V1"], rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan3_general_infusion_matches_exact_matrix_exponential_solution():
    advan3 = ADVAN3()
    params = {"K": 0.2, "K12": 0.15, "K21": 0.05, "V1": 10.0}
    rate = 40.0
    duration = 2.0
    dose = [DoseEvent(time=0.0, amount=rate * duration, rate=rate, compartment=1)]
    times = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0])

    sol = advan3.solve(params, dose, times)
    expected = _two_cmt_infusion_amounts(
        rate, duration, params["K"], params["K12"], params["K21"], times
    )

    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, expected[:, 0] / params["V1"], rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan3_multidose_bolus_matches_exact_matrix_exponential_superposition():
    advan3 = ADVAN3()
    params = {"K": 0.12, "K12": 0.08, "K21": 0.03, "V1": 9.0}
    doses = [
        DoseEvent(time=0.0, amount=80.0, compartment=1),
        DoseEvent(time=6.0, amount=50.0, compartment=1),
    ]
    times = np.array([0.5, 2.0, 6.0, 6.5, 8.0, 12.0, 24.0])

    sol = advan3.solve(params, doses, times)
    expected = sum(
        _two_cmt_bolus_amounts(
            dose.amount, params["K"], params["K12"], params["K21"], times - dose.time
        )
        for dose in doses
    )

    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, expected[:, 0] / params["V1"], rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan3_uses_v_alias_when_v1_is_absent():
    advan3 = ADVAN3()
    times = np.array([0.5, 1.0, 2.0, 4.0])
    dose = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
    params_v1 = {"K": 0.2, "K12": 0.07, "K21": 0.04, "V1": 11.0}
    params_v = {"K": 0.2, "K12": 0.07, "K21": 0.04, "V": 11.0}

    sol_v1 = advan3.solve(params_v1, dose, times)
    sol_v = advan3.solve(params_v, dose, times)

    np.testing.assert_allclose(sol_v.ipred, sol_v1.ipred, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol_v.amounts, sol_v1.amounts, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_advan3_rejects_explicit_zero_v1_instead_of_falling_back_to_v():
    advan3 = ADVAN3()
    params = {"K": 0.2, "K12": 0.07, "K21": 0.04, "V1": 0.0, "V": 11.0}

    with pytest.raises(PKError, match="V1"):
        advan3.solve(params, [DoseEvent(time=0.0, amount=100.0, compartment=1)], np.array([1.0]))
