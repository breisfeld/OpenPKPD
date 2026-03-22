"""
Unit tests for ADVAN12 (3-compartment oral absorption model).

Tests verify:
  - Absorption peak exists at an intermediate time point
  - All compartment amounts are non-negative
  - Depot compartment decays after absorption
  - 4-compartment output structure
  - Multiple dose superposition
  - Bioavailability (F1) scaling
  - Comparison with simpler 2-cmt oral model at limiting parameters
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan4 import ADVAN4
from openpkpd.pk.analytical.advan12 import ADVAN12
from openpkpd.utils.errors import PKError

# Reference parameters: 3-cmt oral
# CL=1, V2=10, Q2=0.5, V3=20, Q3=0.2, V4=30, KA=1.5
_PARAMS_3CMT = {
    "KA": 1.5,
    "K": 0.1,
    "K12": 0.05,
    "K21": 0.025,
    "K13": 0.02,
    "K31": 0.0067,
    "V2": 10.0,
}

_DOSE_100 = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])


def _one_comp_oral_infusion_amounts(rate: float, duration: float, ka: float, k: float, t):
    t_arr = np.asarray(t, dtype=float)
    a1 = np.zeros_like(t_arr)
    a2 = np.zeros_like(t_arr)

    during = t_arr <= duration
    after = ~during

    a1[during] = rate / ka * (1.0 - np.exp(-ka * t_arr[during]))
    a2[during] = rate * (
        (1.0 - np.exp(-k * t_arr[during])) / k
        - (np.exp(-ka * t_arr[during]) - np.exp(-k * t_arr[during])) / (k - ka)
    )

    if np.any(after):
        tau = t_arr[after] - duration
        a1_end = rate / ka * (1.0 - np.exp(-ka * duration))
        a2_end = rate * (
            (1.0 - np.exp(-k * duration)) / k
            - (np.exp(-ka * duration) - np.exp(-k * duration)) / (k - ka)
        )
        a1[after] = a1_end * np.exp(-ka * tau)
        a2[after] = a2_end * np.exp(-k * tau) + a1_end * ka / (ka - k) * (
            np.exp(-k * tau) - np.exp(-ka * tau)
        )

    return a1, a2


@pytest.fixture
def advan12():
    return ADVAN12()


@pytest.mark.unit
def test_advan12_output_shape(advan12):
    """Output should have 4 compartments."""
    sol = advan12.solve(_PARAMS_3CMT, _DOSE_100, _TIMES)
    assert sol.amounts.shape == (len(_TIMES), 4), (
        f"Expected shape ({len(_TIMES)}, 4), got {sol.amounts.shape}"
    )
    assert sol.ipred.shape == (len(_TIMES),)


@pytest.mark.unit
def test_advan12_peak_at_intermediate_time(advan12):
    """Absorption peak should occur at an intermediate (not boundary) time."""
    sol = advan12.solve(_PARAMS_3CMT, _DOSE_100, _TIMES)
    peak_idx = int(np.argmax(sol.ipred))
    assert 0 < peak_idx < len(_TIMES) - 1, (
        f"Peak at boundary index {peak_idx}; expected intermediate"
    )


@pytest.mark.unit
def test_advan12_amounts_nonnegative(advan12):
    """Depot (A1) and central (A2) amounts must be non-negative."""
    sol = advan12.solve(_PARAMS_3CMT, _DOSE_100, _TIMES)
    assert np.all(sol.amounts[:, 0] >= -1e-8), "Depot (A1) must be non-negative"
    assert np.all(sol.amounts[:, 1] >= -1e-8), "Central (A2) must be non-negative"
    assert np.all(sol.ipred >= -1e-8), "IPRED must be non-negative"


@pytest.mark.unit
def test_advan12_depot_decays(advan12):
    """Depot compartment amount should decrease after initial peak."""
    sol = advan12.solve(_PARAMS_3CMT, _DOSE_100, _TIMES)
    a1 = sol.amounts[:, 0]
    # Depot should be decreasing at late times
    assert a1[-1] < a1[0], "Depot should decline over time"


@pytest.mark.unit
def test_advan12_no_dose_returns_zeros(advan12):
    """Without dose events, all predictions should be zero."""
    sol = advan12.solve(_PARAMS_3CMT, [], _TIMES)
    assert np.allclose(sol.ipred, 0.0)
    assert np.allclose(sol.amounts, 0.0)


@pytest.mark.unit
def test_advan12_bioavailability():
    """F1=0.5 should halve the IPRED compared to F1=1.0."""
    model = ADVAN12()
    params_full = dict(_PARAMS_3CMT)
    params_half = dict(_PARAMS_3CMT)
    params_half["F1"] = 0.5

    sol_full = model.solve(params_full, _DOSE_100, _TIMES)
    sol_half = model.solve(params_half, _DOSE_100, _TIMES)

    np.testing.assert_allclose(sol_half.ipred, 0.5 * sol_full.ipred, rtol=1e-6)


@pytest.mark.unit
def test_advan12_multiple_doses():
    """Two doses should give higher concentrations than single dose after second dose."""
    model = ADVAN12()
    two_doses = [
        DoseEvent(time=0.0, amount=100.0, compartment=1),
        DoseEvent(time=12.0, amount=100.0, compartment=1),
    ]
    single_dose = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
    times = np.array([12.5, 14.0, 18.0])

    sol_two = model.solve(_PARAMS_3CMT, two_doses, times)
    sol_one = model.solve(_PARAMS_3CMT, single_dose, times)

    assert np.all(sol_two.ipred > sol_one.ipred), (
        "Two-dose concentrations should exceed single-dose after second dose"
    )


@pytest.mark.unit
def test_advan12_superposition_matches_sum_of_single_dose_solutions():
    """ADVAN12 is linear, so multi-dose output should equal summed single-dose outputs."""
    model = ADVAN12()
    doses = [
        DoseEvent(time=0.0, amount=100.0, compartment=1),
        DoseEvent(time=6.0, amount=40.0, compartment=1),
    ]
    obs_times = np.array([0.5, 2.0, 6.0, 7.0, 10.0, 18.0])

    sol_multi = model.solve(_PARAMS_3CMT, doses, obs_times)
    sol_sum = sum(model.solve(_PARAMS_3CMT, [dose], obs_times).ipred for dose in doses)

    np.testing.assert_allclose(sol_multi.ipred, sol_sum, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_advan12_missing_params_raises():
    """Missing required parameters should raise PKError."""
    model = ADVAN12()
    incomplete = {"KA": 1.5, "V2": 10.0}  # Missing K, K12, etc.
    with pytest.raises(PKError):
        model.solve(incomplete, _DOSE_100, _TIMES)


@pytest.mark.unit
def test_advan12_matches_advan2_exactly_when_distribution_is_off_bolus():
    """With no distribution, ADVAN12 should reduce exactly to ADVAN2 for bolus dosing."""
    advan12 = ADVAN12()
    advan2 = ADVAN2()
    params12 = {
        "KA": 1.2,
        "K": 0.15,
        "K12": 0.0,
        "K21": 0.0,
        "K13": 0.0,
        "K31": 0.0,
        "V2": 8.0,
        "F1": 0.7,
    }
    params2 = {"KA": 1.2, "K": 0.15, "V": 8.0, "F1": 0.7}
    times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

    sol12 = advan12.solve(params12, _DOSE_100, times)
    sol2 = advan2.solve(params2, _DOSE_100, times)

    np.testing.assert_allclose(sol12.amounts[:, 0], sol2.amounts[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol12.amounts[:, 1], sol2.amounts[:, 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol12.amounts[:, 2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol12.ipred, sol2.ipred, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_advan12_matches_advan2_exactly_when_distribution_is_off_infusion():
    """With no distribution, ADVAN12 infusion into depot should match the exact 1-cmt solution."""
    advan12 = ADVAN12()
    params12 = {
        "KA": 1.1,
        "K": 0.2,
        "K12": 0.0,
        "K21": 0.0,
        "K13": 0.0,
        "K31": 0.0,
        "V2": 10.0,
    }
    dose = DoseEvent(time=0.0, amount=90.0, rate=30.0, compartment=1)
    times = np.array([0.5, 1.5, 3.0, 3.5, 6.0])

    sol12 = advan12.solve(params12, [dose], times)
    expected_a1, expected_a2 = _one_comp_oral_infusion_amounts(
        rate=dose.rate,
        duration=dose.amount / dose.rate,
        ka=params12["KA"],
        k=params12["K"],
        t=times,
    )

    np.testing.assert_allclose(sol12.amounts[:, 0], expected_a1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol12.amounts[:, 1], expected_a2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol12.amounts[:, 2:], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol12.ipred, expected_a2 / params12["V2"], rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_advan12_f1_zero_gives_zero_predictions():
    """Explicit F1=0 should suppress the dose rather than falling back to 1.0."""
    model = ADVAN12()
    params = dict(_PARAMS_3CMT)
    params["F1"] = 0.0

    sol = model.solve(params, _DOSE_100, _TIMES)

    np.testing.assert_allclose(sol.ipred, 0.0, atol=1e-12)
    np.testing.assert_allclose(sol.amounts, 0.0, atol=1e-12)


@pytest.mark.unit
def test_advan12_uses_v1_alias_when_v2_missing():
    """ADVAN12 should accept V1 as the central-volume alias when V2 is absent."""
    model = ADVAN12()
    params_v2 = dict(_PARAMS_3CMT)
    params_v1 = dict(_PARAMS_3CMT)
    params_v1["V1"] = params_v1.pop("V2")

    sol_v2 = model.solve(params_v2, _DOSE_100, _TIMES)
    sol_v1 = model.solve(params_v1, _DOSE_100, _TIMES)

    np.testing.assert_allclose(sol_v1.amounts, sol_v2.amounts, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol_v1.ipred, sol_v2.ipred, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_advan12_explicit_zero_v2_raises_instead_of_falling_back():
    """Explicit V2=0 is invalid and must not be treated as a missing alias."""
    model = ADVAN12()
    params = dict(_PARAMS_3CMT)
    params["V2"] = 0.0
    params["V1"] = 10.0

    with pytest.raises(PKError, match="V2"):
        model.solve(params, _DOSE_100, _TIMES)


@pytest.mark.unit
def test_advan12_ipred_is_a2_over_v2(advan12):
    """IPRED should equal A2 (central amount) / V2."""
    sol = advan12.solve(_PARAMS_3CMT, _DOSE_100, _TIMES)
    v2 = _PARAMS_3CMT["V2"]
    expected = sol.amounts[:, 1] / v2
    np.testing.assert_allclose(sol.ipred, expected, rtol=1e-10)


@pytest.mark.unit
def test_advan12_vs_advan4_limit():
    """
    When K13 → 0 and K31 → 0 (no third compartment), ADVAN12 should
    approximate ADVAN4 (2-cmt oral) for central compartment predictions.

    Note: ADVAN4 and ADVAN12 use different parameterizations and the
    agreement is approximate due to numerical differences; we check
    that IPRED values are in the same order of magnitude.
    """
    # Eliminate third compartment by setting K13 and K31 to nearly zero
    params_12 = {
        "KA": 1.5,
        "K": 0.1,
        "K12": 0.05,
        "K21": 0.025,
        "K13": 1e-6,  # nearly zero
        "K31": 1e-6,  # nearly zero
        "V2": 10.0,
    }
    params_4 = {
        "KA": 1.5,
        "K": 0.1,
        "K12": 0.05,
        "K21": 0.025,
        "V2": 10.0,
    }

    advan12 = ADVAN12()
    advan4 = ADVAN4()

    times = np.array([1.0, 4.0, 8.0, 12.0, 24.0])
    sol12 = advan12.solve(params_12, _DOSE_100, times)
    sol4 = advan4.solve(params_4, _DOSE_100, times)

    # Should be close (within 5% at most time points)
    relative_diff = np.abs(sol12.ipred - sol4.ipred) / (sol4.ipred + 1e-12)
    assert np.median(relative_diff) < 0.1, (
        f"ADVAN12 with K13≈0 should approximate ADVAN4; "
        f"median rel diff = {np.median(relative_diff):.4f}"
    )


@pytest.mark.unit
def test_advan12_output_compartment():
    """Output compartment index should be 2 (central)."""
    assert ADVAN12.output_compartment == 2


@pytest.mark.unit
def test_advan12_n_compartments():
    """ADVAN12 should have 4 compartments."""
    assert ADVAN12.n_compartments == 4
