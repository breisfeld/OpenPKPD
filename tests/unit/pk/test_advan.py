"""Unit tests for ADVAN PK subroutines."""

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan3 import ADVAN3

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_bolus(t=0.0, amt=100.0, cmt=1):
    return [DoseEvent(time=t, amount=amt, compartment=cmt)]


def make_obs_times():
    return np.array([0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])


# ── ADVAN1 Tests ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestADVAN1:
    def test_mono_decay(self):
        """Concentration should be monotonically decreasing after bolus dose."""
        advan = ADVAN1()
        params = {"K": 0.1, "V": 20.0}
        sol = advan.solve(params, make_bolus(), make_obs_times())
        assert np.all(np.diff(sol.ipred) < 0), "Concentrations should decrease monotonically"

    def test_initial_concentration(self):
        """At t≈0+, concentration should be ~DOSE/V."""
        advan = ADVAN1()
        params = {"K": 0.1, "V": 20.0}
        dose = 100.0
        times = np.array([0.001])
        sol = advan.solve(params, make_bolus(amt=dose), times)
        expected = dose / 20.0  # DOSE/V
        assert sol.ipred[0] == pytest.approx(expected, rel=0.01)

    def test_analytical_formula(self):
        """Compare to manual computation C(t) = DOSE/V * exp(-K*t)."""
        advan = ADVAN1()
        K, V, D = 0.15, 25.0, 50.0
        params = {"K": K, "V": V}
        times = make_obs_times()
        sol = advan.solve(params, make_bolus(amt=D), times)
        expected = D / V * np.exp(-K * times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-6)

    def test_amounts_nonnegative(self):
        """Compartment amounts should always be ≥ 0."""
        advan = ADVAN1()
        params = {"K": 0.2, "V": 10.0}
        sol = advan.solve(params, make_bolus(), make_obs_times())
        assert np.all(sol.amounts >= 0)

    def test_multiple_doses(self):
        """Two doses should sum via superposition."""
        advan = ADVAN1()
        params = {"K": 0.1, "V": 20.0}
        doses = [DoseEvent(time=0.0, amount=50.0), DoseEvent(time=12.0, amount=50.0)]
        times = np.array([6.0, 12.0, 18.0, 24.0])
        sol = advan.solve(params, doses, times)
        # At t=12, concentration from first dose only (second dose just administered)
        c_first_at_12 = 50.0 / 20.0 * math.exp(-0.1 * 12.0)
        assert sol.ipred[1] == pytest.approx(c_first_at_12, rel=0.01)
        # After second dose, concentration should rise
        assert sol.ipred[2] > sol.ipred[1]

    def test_infusion(self):
        """Infusion should give different profile from bolus."""
        advan = ADVAN1()
        params = {"K": 0.1, "V": 20.0}
        # Infusion: 100 mg over 1 hour = 100 mg/hr rate
        infusion = [DoseEvent(time=0.0, amount=100.0, rate=100.0)]
        bolus = make_bolus(amt=100.0)
        times = np.array([0.5, 1.0, 2.0, 4.0])
        sol_inf = advan.solve(params, infusion, times)
        sol_bol = advan.solve(params, bolus, times)
        # During infusion, concentration from infusion should be lower than bolus (gradual input)
        assert sol_inf.ipred[0] < sol_bol.ipred[0]


# ── ADVAN2 Tests ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestADVAN2:
    def test_peak_time(self):
        """Peak concentration should occur at Tmax = log(KA/K)/(KA-K)."""
        advan = ADVAN2()
        KA, K, V = 1.5, 0.08 / 30.0, 30.0
        params = {"KA": KA, "K": K, "V": V}
        tmax = math.log(KA / K) / (KA - K)
        times = np.linspace(0.1, 10, 100)
        sol = advan.solve(params, make_bolus(amt=100.0), times)
        peak_idx = np.argmax(sol.ipred)
        # Allow ±10% tolerance on peak time
        assert times[peak_idx] == pytest.approx(tmax, rel=0.1)

    def test_amounts_nonnegative(self):
        """Both compartments should have non-negative amounts."""
        advan = ADVAN2()
        params = {"KA": 1.5, "K": 0.003, "V": 30.0}
        sol = advan.solve(params, make_bolus(), make_obs_times())
        assert np.all(sol.amounts >= -1e-10)  # Allow tiny numerical errors

    def test_depot_decays(self):
        """Depot compartment should decay to zero over time."""
        advan = ADVAN2()
        params = {"KA": 1.5, "K": 0.003, "V": 30.0}
        times = np.array([0.1, 1.0, 5.0, 12.0, 24.0, 48.0])
        sol = advan.solve(params, make_bolus(amt=100.0), times)
        # Last depot amount should be much less than initial
        assert sol.amounts[-1, 0] < sol.amounts[0, 0]

    def test_can_skip_amount_construction_for_prediction_only(self):
        advan = ADVAN2()
        params = {"KA": 1.5, "K": 0.003, "V": 30.0}
        times = np.array([0.1, 1.0, 5.0, 12.0])

        full = advan.solve(params, make_bolus(amt=100.0), times)
        pred_only = advan.solve(params, make_bolus(amt=100.0), times, return_amounts=False)

        np.testing.assert_allclose(pred_only.ipred, full.ipred)
        assert pred_only.amounts.shape == (len(times), 0)

    def test_repeated_solve_cache_keeps_parameter_dependent_infusion_behavior(self):
        advan = ADVAN2()
        doses = [DoseEvent(time=0.0, amount=60.0, rate=20.0, compartment=1)]
        times = np.array([0.5, 1.5, 3.0, 4.0, 6.0])
        params_a = {"KA": 1.5, "K": 0.1, "V": 10.0, "F1": 0.6}
        params_b = {"KA": 0.9, "K": 0.08, "V": 12.0, "F1": 1.0}

        sol_a = advan.solve(params_a, doses, times)
        sol_b = advan.solve(params_b, doses, times)

        fresh_a = ADVAN2().solve(params_a, doses, times)
        fresh_b = ADVAN2().solve(params_b, doses, times)

        np.testing.assert_allclose(sol_a.ipred, fresh_a.ipred)
        np.testing.assert_allclose(sol_a.amounts, fresh_a.amounts)
        np.testing.assert_allclose(sol_b.ipred, fresh_b.ipred)
        np.testing.assert_allclose(sol_b.amounts, fresh_b.amounts)


# ── ADVAN3 Tests ──────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestADVAN3:
    def test_biexponential_decay(self):
        """2-cmt IV should show biexponential decline."""
        advan = ADVAN3()
        params = {"K": 0.1, "K12": 0.05, "K21": 0.02, "V1": 10.0}
        times = np.linspace(0.1, 24, 50)
        sol = advan.solve(params, make_bolus(amt=100.0), times)
        assert np.all(sol.amounts >= -1e-8)
        # Eventually should approach zero
        assert sol.ipred[-1] < sol.ipred[0]

    def test_amounts_nonnegative(self):
        advan = ADVAN3()
        params = {"K": 0.1, "K12": 0.05, "K21": 0.02, "V1": 10.0}
        sol = advan.solve(params, make_bolus(), make_obs_times())
        assert np.all(sol.amounts >= -1e-8)


# ── Hypothesis property tests ─────────────────────────────────────────────────


@pytest.mark.unit
@given(
    ka=st.floats(min_value=0.1, max_value=5.0),
    k=st.floats(min_value=0.001, max_value=1.0),
    v=st.floats(min_value=1.0, max_value=100.0),
    dose=st.floats(min_value=0.1, max_value=500.0),
)
@settings(max_examples=50, deadline=5000)
def test_advan2_amounts_nonneg(ka, k, v, dose):
    """ADVAN2 compartment amounts are always non-negative for valid parameters."""
    advan = ADVAN2()
    params = {"KA": ka, "K": k, "V": v}
    times = np.array([1.0, 6.0, 12.0, 24.0])
    sol = advan.solve(params, make_bolus(amt=dose), times)
    assert np.all(np.isfinite(sol.amounts))
    assert np.all(sol.amounts >= -1e-8)


@pytest.mark.unit
@given(
    k=st.floats(min_value=0.001, max_value=2.0),
    v=st.floats(min_value=1.0, max_value=100.0),
    dose=st.floats(min_value=0.1, max_value=500.0),
)
@settings(max_examples=50, deadline=5000)
def test_advan1_amounts_nonneg(k, v, dose):
    """ADVAN1 compartment amounts are always non-negative for valid parameters."""
    advan = ADVAN1()
    params = {"K": k, "V": v}
    times = np.array([1.0, 6.0, 12.0, 24.0])
    sol = advan.solve(params, make_bolus(amt=dose), times)
    assert np.all(sol.amounts >= -1e-8)
