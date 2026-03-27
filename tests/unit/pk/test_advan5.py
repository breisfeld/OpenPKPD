"""
Unit tests for ADVAN5 (General N-Compartment Linear Model).

Tests verify:
  - 2-cmt equivalence with ADVAN3 (N=2 special case)
  - 3-cmt equivalence with ADVAN11 (N=3 special case)
  - 4-cmt bolus and infusion vs scipy.linalg.expm reference
  - Multiple dose superposition
  - Dosing into non-central compartments
  - Non-central output compartment
  - Volume fallback (V vs V{n})
  - N-inference from Kij/Ki0 parameter keys
  - Parameter key aliases ("K" → K10)
  - Non-negativity across random parameter space (Hypothesis)
  - Error cases: missing volume, invalid output/dose compartment
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.linalg import expm

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.analytical.advan5 import (
    ADVAN5,
    _build_rate_matrix,
    _eigendecomp_n,
    _infer_n_and_parse_rates,
)
from openpkpd.pk.analytical.advan11 import ADVAN11
from openpkpd.utils.errors import PKError

# ---------------------------------------------------------------------------
# Reference parameters
# ---------------------------------------------------------------------------

# 3-cmt params (shared with ADVAN11 tests for cross-validation)
_PARAMS_3CMT = {
    "K": 0.1,
    "K12": 0.05,
    "K21": 0.025,
    "K13": 0.02,
    "K31": 0.0067,
    "V1": 10.0,
}

# 4-cmt params: central (1) ↔ three peripherals (2, 3, 4)
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

_DOSE_100 = [DoseEvent(time=0.0, amount=100.0)]
_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])


# ---------------------------------------------------------------------------
# scipy.linalg.expm reference helpers
# ---------------------------------------------------------------------------

def _scipy_bolus_amounts(params: dict, dose: float, dose_cmt: int, times: np.ndarray) -> np.ndarray:
    """Compute N-cmt bolus amounts via scipy matrix exponential."""
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
    params: dict, rate: float, duration: float, dose_cmt: int, times: np.ndarray
) -> np.ndarray:
    """Compute N-cmt infusion amounts via scipy matrix exponential."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def advan5():
    return ADVAN5()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_advan5_2cmt_equivalence_bolus():
    """ADVAN5 with K,K12,K21,V1 must produce identical results to ADVAN3."""
    params = {"K": 0.15, "K12": 0.06, "K21": 0.03, "V1": 12.0}
    dose = [DoseEvent(time=0.0, amount=100.0)]
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

    sol3 = ADVAN3().solve(params, dose, times)
    sol5 = ADVAN5().solve(params, dose, times)

    np.testing.assert_allclose(sol5.ipred, sol3.ipred, rtol=1e-10, atol=1e-10)
    # ADVAN5 amounts has 2 columns; ADVAN3 also has 2 columns
    np.testing.assert_allclose(sol5.amounts, sol3.amounts, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan5_3cmt_equivalence_bolus():
    """ADVAN5 with K,K12,K21,K13,K31,V1 must match ADVAN11 exactly."""
    sol11 = ADVAN11().solve(_PARAMS_3CMT, _DOSE_100, _TIMES)
    sol5 = ADVAN5().solve(_PARAMS_3CMT, _DOSE_100, _TIMES)

    np.testing.assert_allclose(sol5.ipred, sol11.ipred, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol5.amounts, sol11.amounts, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_advan5_3cmt_equivalence_infusion():
    """ADVAN5 infusion must match ADVAN11 infusion exactly."""
    dose = [DoseEvent(time=0.0, amount=200.0, rate=100.0)]  # 2-hour infusion
    times = np.array([0.5, 1.0, 2.0, 3.0, 6.0, 12.0])

    sol11 = ADVAN11().solve(_PARAMS_3CMT, dose, times)
    sol5 = ADVAN5().solve(_PARAMS_3CMT, dose, times)

    np.testing.assert_allclose(sol5.ipred, sol11.ipred, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sol5.amounts, sol11.amounts, rtol=1e-10, atol=1e-10)



@pytest.mark.unit
def test_advan5_4cmt_bolus_matches_scipy_expm():
    """4-cmt bolus must match scipy.linalg.expm reference, rtol=1e-8."""
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    sol = ADVAN5().solve(_PARAMS_4CMT, _DOSE_100, times)
    expected = _scipy_bolus_amounts(_PARAMS_4CMT, 100.0, 1, times)
    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        sol.ipred, expected[:, 0] / _PARAMS_4CMT["V1"], rtol=1e-8, atol=1e-8
    )


@pytest.mark.unit
def test_advan5_4cmt_infusion_matches_scipy_expm():
    """4-cmt zero-order infusion must match scipy.linalg.expm reference."""
    rate, duration = 50.0, 2.0
    dose = [DoseEvent(time=0.0, amount=rate * duration, rate=rate)]
    times = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])

    sol = ADVAN5().solve(_PARAMS_4CMT, dose, times)
    expected = _scipy_infusion_amounts(_PARAMS_4CMT, rate, duration, 1, times)
    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.unit
def test_advan5_multiple_dosing_superposition():
    """Two-dose solution must equal sum of individual single-dose solutions."""
    params = _PARAMS_4CMT
    dose1 = DoseEvent(time=0.0, amount=100.0)
    dose2 = DoseEvent(time=12.0, amount=100.0)
    times = np.array([0.5, 6.0, 12.0, 12.5, 18.0, 24.0])

    sol_both = ADVAN5().solve(params, [dose1, dose2], times)
    sol_d1 = ADVAN5().solve(params, [dose1], times)
    sol_d2 = ADVAN5().solve(params, [dose2], times)

    np.testing.assert_allclose(
        sol_both.amounts, sol_d1.amounts + sol_d2.amounts, rtol=1e-10, atol=1e-10
    )


@pytest.mark.unit
def test_advan5_dose_into_compartment_2():
    """Bolus into compartment 2: at t→0+ A[1]≈dose, A[0]≈0; matches scipy."""
    params = _PARAMS_4CMT
    dose = [DoseEvent(time=0.0, amount=100.0, compartment=2)]
    times = np.array([1e-6, 0.5, 1.0, 4.0, 12.0])

    sol = ADVAN5().solve(params, dose, times)
    assert sol.amounts[0, 1] > 90.0, "Most mass should remain in cmt 2 at t≈0"
    assert sol.amounts[0, 0] < 1.0, "Central should receive almost nothing at t≈0"

    expected = _scipy_bolus_amounts(params, 100.0, 2, times)
    np.testing.assert_allclose(sol.amounts, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_advan5_output_compartment_2():
    """ADVAN5(output_compartment=2) gives IPRED = A[:,1] / V2."""
    params = {**_PARAMS_4CMT, "V2": 30.0}
    sol = ADVAN5(output_compartment=2).solve(params, _DOSE_100, _TIMES)
    expected_ipred = sol.amounts[:, 1] / 30.0
    np.testing.assert_allclose(sol.ipred, expected_ipred, rtol=1e-12)


@pytest.mark.unit
def test_advan5_volume_fallback():
    """Plain 'V' is accepted as volume when 'V1' is absent."""
    params_v1 = {**_PARAMS_3CMT}
    params_v = {k: v for k, v in _PARAMS_3CMT.items() if k != "V1"}
    params_v["V"] = _PARAMS_3CMT["V1"]

    sol_v1 = ADVAN5().solve(params_v1, _DOSE_100, _TIMES)
    sol_v = ADVAN5().solve(params_v, _DOSE_100, _TIMES)

    np.testing.assert_allclose(sol_v.ipred, sol_v1.ipred, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol_v.amounts, sol_v1.amounts, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_advan5_no_doses_returns_zeros(advan5):
    """Empty dose list → all zeros."""
    sol = advan5.solve(_PARAMS_3CMT, [], _TIMES)
    assert np.allclose(sol.ipred, 0.0)
    assert np.allclose(sol.amounts, 0.0)


@pytest.mark.unit
def test_advan5_output_shape(advan5):
    """amounts.shape == (n_times, N) and ipred.shape == (n_times,)."""
    sol = advan5.solve(_PARAMS_4CMT, _DOSE_100, _TIMES)
    assert sol.amounts.shape == (len(_TIMES), 4)
    assert sol.ipred.shape == (len(_TIMES),)
    assert sol.times.shape == (len(_TIMES),)


@pytest.mark.unit
def test_advan5_k_alias():
    """Plain 'K' is treated as K10 (elimination from cmt 1)."""
    params_with_k = {"K": 0.1, "K12": 0.05, "K21": 0.025, "V1": 10.0}
    params_with_k10 = {"K10": 0.1, "K12": 0.05, "K21": 0.025, "V1": 10.0}

    n_k, kij_k, ki0_k = _infer_n_and_parse_rates(params_with_k)
    n_10, kij_10, ki0_10 = _infer_n_and_parse_rates(params_with_k10)

    assert ki0_k == ki0_10
    assert n_k == n_10
    assert kij_k == kij_10

    sol_k = ADVAN5().solve(params_with_k, _DOSE_100, _TIMES)
    sol_k10 = ADVAN5().solve(params_with_k10, _DOSE_100, _TIMES)
    np.testing.assert_allclose(sol_k.ipred, sol_k10.ipred, rtol=1e-12)


@pytest.mark.unit
def test_advan5_n_inferred_from_k23():
    """K23/K32/K30 keys → N inferred as 3."""
    params = {
        "K": 0.1, "K12": 0.05, "K21": 0.02,
        "K23": 0.03, "K32": 0.01, "K30": 0.005, "V1": 10.0,
    }
    n, kij, ki0 = _infer_n_and_parse_rates(params)
    assert n == 3
    assert (2, 3) in kij
    assert (3, 2) in kij
    assert 3 in ki0


@pytest.mark.unit
def test_advan5_eigenvalues_positive():
    """_eigendecomp_n on a valid 4×4 rate matrix → all lam > 0."""
    _, kij, ki0 = _infer_n_and_parse_rates(_PARAMS_4CMT)
    M = _build_rate_matrix(4, kij, ki0)
    lam, _, _ = _eigendecomp_n(M)
    assert np.all(lam > 0), f"All decay rates must be positive, got lam={lam}"


@pytest.mark.unit
def test_advan5_amounts_nonnegative(advan5):
    """All compartment amounts must be non-negative."""
    sol = advan5.solve(_PARAMS_4CMT, _DOSE_100, _TIMES)
    assert np.all(sol.amounts >= -1e-8), f"Negative amounts: {sol.amounts.min():.2e}"


@pytest.mark.unit
def test_advan5_missing_volume_raises(advan5):
    """PKError when neither V1 nor V is in pk_params."""
    params_no_v = {k: v for k, v in _PARAMS_3CMT.items() if k not in ("V1", "V")}
    with pytest.raises(PKError, match="volume"):
        advan5.solve(params_no_v, _DOSE_100, _TIMES)


@pytest.mark.unit
def test_advan5_invalid_output_compartment_raises():
    """PKError when output_compartment > N."""
    model = ADVAN5(output_compartment=5)
    with pytest.raises(PKError, match="output_compartment"):
        model.solve(_PARAMS_3CMT, _DOSE_100, _TIMES)


@pytest.mark.unit
def test_advan5_invalid_dose_compartment_raises(advan5):
    """PKError when DoseEvent.compartment > N."""
    dose = [DoseEvent(time=0.0, amount=100.0, compartment=5)]
    with pytest.raises(PKError, match="compartment"):
        advan5.solve(_PARAMS_3CMT, dose, _TIMES)


@pytest.mark.unit
def test_advan5_no_rate_constants_raises():
    """PKError when pk_params contains no Kij or Ki0 keys."""
    with pytest.raises(PKError, match="rate-constant"):
        ADVAN5().solve({"V1": 10.0, "CL": 2.0}, _DOSE_100, _TIMES)


@pytest.mark.unit
@given(
    k=st.floats(min_value=0.01, max_value=2.0),
    k12=st.floats(min_value=0.001, max_value=1.0),
    k21=st.floats(min_value=0.001, max_value=1.0),
    k13=st.floats(min_value=0.001, max_value=1.0),
    k31=st.floats(min_value=0.001, max_value=1.0),
)
@settings(max_examples=50)
def test_advan5_hypothesis_amounts_nonnegative(k, k12, k21, k13, k31):
    """Compartment amounts are non-negative for random 3-cmt parameters."""
    params = {"K": k, "K12": k12, "K21": k21, "K13": k13, "K31": k31, "V1": 10.0}
    sol = ADVAN5().solve(params, _DOSE_100, _TIMES)
    assert np.all(sol.amounts >= -1e-8), (
        f"Negative amounts for params={params}: min={sol.amounts.min():.2e}"
    )
    assert np.all(np.isfinite(sol.amounts)), "Non-finite amounts encountered"
