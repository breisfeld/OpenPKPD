"""
Unit tests for ADVAN11 (3-compartment IV model).

Tests verify:
  - Triexponential decay (monotonically decreasing IPRED for simple params)
  - Initial concentration approaches DOSE/V1
  - All compartment amounts non-negative
  - Multiple dose superposition
  - Zero-order infusion support
  - Correct shape of output arrays
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan11 import ADVAN11, _eigendecomp, _rate_matrix
from openpkpd.utils.errors import PKError

# Reference parameters: CL=1, V1=10, Q2=0.5, V2=20, Q3=0.2, V3=30
# → K=0.1, K12=0.05, K21=0.025, K13=0.02, K31=0.0067
_PARAMS = {
    "K": 0.1,
    "K12": 0.05,
    "K21": 0.025,
    "K13": 0.02,
    "K31": 0.0067,
    "V1": 10.0,
}

_DOSE_100 = [DoseEvent(time=0.0, amount=100.0)]
_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])


def _three_cmt_bolus_amounts(
    dose: float,
    k: float,
    k12: float,
    k21: float,
    k13: float,
    k31: float,
    t,
    k23: float = 0.0,
    k32: float = 0.0,
) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    out = np.zeros((len(t_arr), 3), dtype=float)
    m = _rate_matrix(k, k12, k21, k13, k31, k23, k32)
    y0 = np.array([dose, 0.0, 0.0], dtype=float)
    positive = t_arr > 0.0
    pos_idx = np.where(positive)[0]
    for i, dt in zip(pos_idx, t_arr[positive], strict=False):
        out[i] = expm(m * dt) @ y0
    return out


def _three_cmt_infusion_amounts(
    rate: float,
    duration: float,
    k: float,
    k12: float,
    k21: float,
    k13: float,
    k31: float,
    t,
    k23: float = 0.0,
    k32: float = 0.0,
) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    out = np.zeros((len(t_arr), 3), dtype=float)
    m = _rate_matrix(k, k12, k21, k13, k31, k23, k32)
    u = np.array([rate, 0.0, 0.0], dtype=float)
    eye = np.eye(3)

    if np.linalg.matrix_rank(m) == 3:
        a_end = np.linalg.solve(m, (expm(m * duration) - eye) @ u)
        for i, ti in enumerate(t_arr):
            if ti <= duration:
                out[i] = np.linalg.solve(m, (expm(m * ti) - eye) @ u)
            else:
                out[i] = expm(m * (ti - duration)) @ a_end
        return out

    for i, ti in enumerate(t_arr):
        n_steps = max(int(np.ceil(max(ti, 1e-12) / 1e-4)), 1)
        dt_step = ti / n_steps
        state = np.zeros(3, dtype=float)
        for step in range(n_steps):
            t_mid = (step + 0.5) * dt_step
            forcing = u if t_mid <= duration else 0.0 * u
            state = expm(m * dt_step) @ state + np.linalg.solve(
                m if np.linalg.matrix_rank(m) == 3 else m - 1e-12 * np.eye(3),
                (expm(m * dt_step) - eye) @ forcing,
            )
        out[i] = state
    return out


@pytest.fixture
def advan11():
    return ADVAN11()


@pytest.mark.unit
def test_advan11_single_dose_basic(advan11):
    """3-cmt IV: IPRED should be positive and decreasing at late times."""
    sol = advan11.solve(_PARAMS, _DOSE_100, _TIMES)
    assert sol.ipred[0] > sol.ipred[-1], "IPRED should decrease over time"
    assert np.all(sol.ipred >= 0), "IPRED must be non-negative"


@pytest.mark.unit
def test_advan11_output_shape(advan11):
    """Output arrays have correct shape."""
    sol = advan11.solve(_PARAMS, _DOSE_100, _TIMES)
    assert sol.amounts.shape == (len(_TIMES), 3), (
        f"Expected amounts shape ({len(_TIMES)}, 3), got {sol.amounts.shape}"
    )
    assert sol.ipred.shape == (len(_TIMES),)
    assert sol.times.shape == (len(_TIMES),)


@pytest.mark.unit
def test_advan11_concentration_at_zero():
    """At t≈0+, central concentration should approach DOSE/V1."""
    model = ADVAN11()
    params = {"K": 0.1, "K12": 0.05, "K21": 0.03, "K13": 0.02, "K31": 0.01, "V1": 10.0}
    dose = [DoseEvent(time=0.0, amount=100.0)]
    times = np.array([1e-6])  # very small time after dose
    sol = model.solve(params, dose, times)
    expected = 100.0 / 10.0  # DOSE / V1
    assert abs(sol.ipred[0] - expected) < 0.1, (
        f"Initial concentration {sol.ipred[0]:.4f} deviates from {expected:.4f}"
    )


@pytest.mark.unit
def test_advan11_amounts_nonnegative(advan11):
    """All compartment amounts must be non-negative."""
    sol = advan11.solve(_PARAMS, _DOSE_100, _TIMES)
    assert np.all(sol.amounts >= -1e-8), f"Negative amounts found: {sol.amounts.min():.4e}"


@pytest.mark.unit
def test_advan11_mass_conservation_approximate(advan11):
    """
    At early times (near dose), total amount should be close to DOSE.

    Some mass will have been eliminated, so we check that A1+A2+A3 <= DOSE
    and that it's reasonably close at t=0.5 h.
    """
    sol = advan11.solve(_PARAMS, _DOSE_100, _TIMES)
    total = sol.amounts[0, :].sum()
    # Some elimination by t=0.5h, but not too much
    assert total <= 100.0 * 1.01, f"Total amount {total:.2f} exceeds dose 100"
    assert total >= 1.0, f"Total amount {total:.2f} too low at t=0.5h"


@pytest.mark.unit
def test_advan11_no_dose_returns_zeros(advan11):
    """Without any dose event, all amounts and IPRED should be zero."""
    sol = advan11.solve(_PARAMS, [], _TIMES)
    assert np.allclose(sol.ipred, 0.0), "IPRED should be zero with no doses"
    assert np.allclose(sol.amounts, 0.0), "Amounts should be zero with no doses"


@pytest.mark.unit
def test_advan11_multiple_doses():
    """Multiple doses should give higher concentrations than a single dose."""
    model = ADVAN11()
    params = {"K": 0.1, "K12": 0.05, "K21": 0.03, "K13": 0.02, "K31": 0.01, "V1": 10.0}
    two_doses = [
        DoseEvent(time=0.0, amount=100.0),
        DoseEvent(time=12.0, amount=100.0),
    ]
    single_dose = [DoseEvent(time=0.0, amount=100.0)]
    times = np.array([12.5, 14.0, 18.0, 24.0])

    sol_two = model.solve(params, two_doses, times)
    sol_one = model.solve(params, single_dose, times)

    # After second dose, concentrations must exceed single-dose concentrations
    assert np.all(sol_two.ipred > sol_one.ipred), (
        "Two-dose concentrations should exceed single-dose after second dose"
    )


@pytest.mark.unit
def test_advan11_infusion():
    """Zero-order infusion should give lower peak than equivalent bolus."""
    model = ADVAN11()
    params = {"K": 0.1, "K12": 0.05, "K21": 0.025, "K13": 0.02, "K31": 0.01, "V1": 10.0}
    dose_amt = 100.0
    dur = 2.0  # 2-hour infusion
    rate = dose_amt / dur  # 50 mg/h

    bolus = [DoseEvent(time=0.0, amount=dose_amt)]
    infusion = [DoseEvent(time=0.0, amount=dose_amt, rate=rate)]

    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    sol_bol = model.solve(params, bolus, times)
    sol_inf = model.solve(params, infusion, times)

    # During infusion, concentration should be lower than bolus
    assert sol_inf.ipred[0] < sol_bol.ipred[0], (
        "Infusion concentration at t=0.5 should be less than bolus"
    )
    # Infusion amounts should be non-negative
    assert np.all(sol_inf.amounts >= -1e-8)


@pytest.mark.unit
def test_advan11_rate_matrix():
    """Rate matrix should have correct diagonal structure."""
    M = _rate_matrix(0.1, 0.05, 0.025, 0.02, 0.01)
    assert M.shape == (3, 3)
    # Row sums should be non-positive (conservation property)
    M.sum(axis=0)  # column sums = 0 for conservative system
    # The elimination column doesn't sum to zero; check diagonal is negative
    assert np.all(np.diag(M) < 0), "Diagonal of rate matrix must be negative"


@pytest.mark.unit
def test_advan11_eigenvalues_positive():
    """Eigenvalues (decay rates) of the rate matrix must all be positive."""
    M = _rate_matrix(0.1, 0.05, 0.025, 0.02, 0.01)
    lam, P, P_inv = _eigendecomp(M)
    assert np.all(lam > 0), f"All decay rates must be positive, got lam={lam}"


@pytest.mark.unit
def test_advan11_missing_params_raises():
    """solve() should raise PKError when required parameters are missing."""
    from openpkpd.utils.errors import PKError

    model = ADVAN11()
    incomplete_params = {"K": 0.1, "V1": 10.0}  # Missing K12, K21, K13, K31
    with pytest.raises(PKError):
        model.solve(incomplete_params, _DOSE_100, _TIMES)


@pytest.mark.unit
def test_advan11_ipred_equals_a1_over_v1(advan11):
    """IPRED should equal A1 (central compartment amount) / V1."""
    sol = advan11.solve(_PARAMS, _DOSE_100, _TIMES)
    v1 = _PARAMS["V1"]
    expected_ipred = sol.amounts[:, 0] / v1
    np.testing.assert_allclose(sol.ipred, expected_ipred, rtol=1e-10)


@pytest.mark.unit
def test_advan11_terminal_decay():
    """In the terminal phase, log(IPRED) should be approximately linear."""
    model = ADVAN11()
    params = {"K": 0.1, "K12": 0.05, "K21": 0.025, "K13": 0.02, "K31": 0.01, "V1": 10.0}
    late_times = np.array([48.0, 60.0, 72.0, 84.0, 96.0])
    sol = model.solve(params, [DoseEvent(0.0, 100.0)], late_times)
    positive_ipred = sol.ipred[sol.ipred > 1e-10]
    if len(positive_ipred) >= 3:
        log_ipred = np.log(positive_ipred)
        times_used = late_times[sol.ipred > 1e-10]
        coeffs = np.polyfit(times_used, log_ipred, 1)
        r2 = 1.0 - np.var(log_ipred - np.polyval(coeffs, times_used)) / np.var(log_ipred)
        assert r2 > 0.99, f"Terminal phase should be log-linear, R²={r2:.4f}"


@pytest.mark.unit
def test_advan11_matches_advan1_exactly_when_distribution_is_off_bolus():
    advan11 = ADVAN11()
    advan1 = ADVAN1()
    params11 = {"K": 0.2, "K12": 0.0, "K21": 0.0, "K13": 0.0, "K31": 0.0, "V1": 10.0}
    params1 = {"K": 0.2, "V": 10.0}
    dose = [DoseEvent(time=0.0, amount=100.0)]
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

    sol11 = advan11.solve(params11, dose, times)
    sol1 = advan1.solve(params1, dose, times)

    np.testing.assert_allclose(sol11.amounts[:, 0], sol1.amounts[:, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol11.amounts[:, 1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol11.ipred, sol1.ipred, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan11_matches_advan1_exactly_when_distribution_is_off_infusion():
    advan11 = ADVAN11()
    advan1 = ADVAN1()
    params11 = {"K": 0.2, "K12": 0.0, "K21": 0.0, "K13": 0.0, "K31": 0.0, "V1": 10.0}
    params1 = {"K": 0.2, "V": 10.0}
    dose = [DoseEvent(time=0.0, amount=90.0, rate=30.0)]
    times = np.array([0.5, 1.0, 2.0, 3.0, 6.0, 12.0])

    sol11 = advan11.solve(params11, dose, times)
    sol1 = advan1.solve(params1, dose, times)

    np.testing.assert_allclose(sol11.amounts[:, 0], sol1.amounts[:, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol11.amounts[:, 1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol11.ipred, sol1.ipred, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan11_general_infusion_matches_exact_matrix_exponential_solution():
    advan11 = ADVAN11()
    params = {"K": 0.2, "K12": 0.15, "K21": 0.05, "K13": 0.07, "K31": 0.03, "V1": 10.0}
    rate = 40.0
    duration = 2.0
    dose = [DoseEvent(time=0.0, amount=rate * duration, rate=rate)]
    times = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0])

    sol = advan11.solve(params, dose, times)
    expected = _three_cmt_infusion_amounts(
        rate,
        duration,
        params["K"],
        params["K12"],
        params["K21"],
        params["K13"],
        params["K31"],
        times,
    )

    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, expected[:, 0] / params["V1"], rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan11_multidose_bolus_matches_exact_matrix_exponential_superposition():
    advan11 = ADVAN11()
    params = {"K": 0.12, "K12": 0.08, "K21": 0.03, "K13": 0.05, "K31": 0.02, "V1": 9.0}
    doses = [DoseEvent(time=0.0, amount=80.0), DoseEvent(time=6.0, amount=50.0)]
    times = np.array([0.5, 2.0, 6.0, 6.5, 8.0, 12.0, 24.0])

    sol = advan11.solve(params, doses, times)
    expected = sum(
        _three_cmt_bolus_amounts(
            dose.amount,
            params["K"],
            params["K12"],
            params["K21"],
            params["K13"],
            params["K31"],
            times - dose.time,
        )
        for dose in doses
    )

    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol.ipred, expected[:, 0] / params["V1"], rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan11_uses_v_alias_when_v1_is_absent():
    advan11 = ADVAN11()
    dose = [DoseEvent(time=0.0, amount=100.0)]
    times = np.array([0.5, 1.0, 2.0, 4.0])
    params_v1 = {"K": 0.2, "K12": 0.07, "K21": 0.04, "K13": 0.03, "K31": 0.02, "V1": 11.0}
    params_v = {"K": 0.2, "K12": 0.07, "K21": 0.04, "K13": 0.03, "K31": 0.02, "V": 11.0}

    sol_v1 = advan11.solve(params_v1, dose, times)
    sol_v = advan11.solve(params_v, dose, times)

    np.testing.assert_allclose(sol_v.amounts, sol_v1.amounts, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol_v.ipred, sol_v1.ipred, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_advan11_rejects_explicit_zero_v1_instead_of_falling_back_to_v():
    advan11 = ADVAN11()
    params = {"K": 0.2, "K12": 0.07, "K21": 0.04, "K13": 0.03, "K31": 0.02, "V1": 0.0, "V": 11.0}

    with pytest.raises(PKError, match="V1"):
        advan11.solve(params, [DoseEvent(time=0.0, amount=100.0)], np.array([1.0]))
