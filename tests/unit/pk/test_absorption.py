"""
Tests for transit, parallel, and EHC absorption models.
"""

from __future__ import annotations

import numpy as np


class _MockDoseEvent:
    """Minimal dose event for testing."""

    def __init__(self, time: float, amount: float):
        self.time = time
        self.amount = amount


def _bateman_concentration(dose: float, ka: float, k: float, v: float, t) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    conc = np.zeros_like(t_arr, dtype=float)
    mask = t_arr >= 0.0
    dt = t_arr[mask]
    if abs(ka - k) > 1e-12 * max(ka, k, 1.0):
        conc[mask] = (dose * ka) / (v * (ka - k)) * (np.exp(-k * dt) - np.exp(-ka * dt))
    else:
        conc[mask] = (dose * ka / v) * dt * np.exp(-k * dt)
    return conc


def _zero_order_concentration(dose: float, duration: float, k: float, v: float, t) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    conc = np.zeros_like(t_arr, dtype=float)
    mask = t_arr >= 0.0
    dt = t_arr[mask]
    rate = dose / duration
    during = dt <= duration
    after = ~during
    conc_masked = np.zeros_like(dt)
    conc_masked[during] = (rate / (v * k)) * (1.0 - np.exp(-k * dt[during]))
    conc_masked[after] = (
        (rate / (v * k)) * (1.0 - np.exp(-k * duration)) * np.exp(-k * (dt[after] - duration))
    )
    conc[mask] = conc_masked
    return conc


class TestTransitAbsorption:
    def test_import(self):
        from openpkpd.pk.absorption.transit import TransitAbsorption

        assert TransitAbsorption is not None

    def test_basic_solve(self):
        from openpkpd.pk.absorption.transit import TransitAbsorption

        model = TransitAbsorption()
        pk_params = {"KTR": 1.0, "N_TRANSIT": 3, "CL": 2.0, "V": 20.0}
        dose_events = [_MockDoseEvent(0.0, 100.0)]
        obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0], dtype=float)
        sol = model.solve(pk_params, dose_events, obs_times)

        assert sol.ipred.shape == obs_times.shape
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0)

    def test_concentration_peak(self):
        """Transit absorption: concentration rises then falls."""
        from openpkpd.pk.absorption.transit import TransitAbsorption

        model = TransitAbsorption()
        pk_params = {"KTR": 2.0, "N_TRANSIT": 2, "CL": 1.0, "V": 10.0}
        dose_events = [_MockDoseEvent(0.0, 100.0)]
        obs_times = np.linspace(0.1, 12.0, 50)
        sol = model.solve(pk_params, dose_events, obs_times)

        # Should have a peak somewhere
        assert np.max(sol.ipred) > 0

    def test_zero_dose(self):
        """Zero dose -> zero concentrations."""
        from openpkpd.pk.absorption.transit import TransitAbsorption

        model = TransitAbsorption()
        pk_params = {"KTR": 1.0, "N_TRANSIT": 3, "CL": 2.0, "V": 20.0}
        dose_events = [_MockDoseEvent(0.0, 0.0)]
        obs_times = np.array([1.0, 2.0, 4.0], dtype=float)
        sol = model.solve(pk_params, dose_events, obs_times)
        assert np.allclose(sol.ipred, 0.0, atol=1e-6)

    def test_solution_structure(self):
        """PKSolution has correct shapes."""
        from openpkpd.pk.absorption.transit import TransitAbsorption

        model = TransitAbsorption()
        pk_params = {"KTR": 1.5, "N_TRANSIT": 2, "CL": 1.5, "V": 15.0}
        dose_events = [_MockDoseEvent(0.0, 50.0)]
        obs_times = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
        sol = model.solve(pk_params, dose_events, obs_times)
        assert sol.amounts.shape[0] == len(obs_times)
        assert sol.times.shape == obs_times.shape

    def test_single_transit_matches_exact_bateman_reduction(self):
        """N_TRANSIT=1 reduces exactly to a one-compartment oral Bateman model."""
        from openpkpd.pk.absorption.transit import TransitAbsorption

        model = TransitAbsorption()
        dose = 100.0
        ktr = 1.3
        cl = 1.5
        v = 12.0
        f1 = 0.7
        obs_times = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)

        sol = model.solve(
            {"KTR": ktr, "N_TRANSIT": 1, "CL": cl, "V": v, "F1": f1},
            [_MockDoseEvent(0.0, dose)],
            obs_times,
        )

        expected = _bateman_concentration(dose * f1, ktr, cl / v, v, obs_times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=5e-4, atol=1e-8)

    def test_single_transit_equal_rate_limit_matches_closed_form(self):
        """The N_TRANSIT=1 reduction also matches the KA=K limit form."""
        from openpkpd.pk.absorption.transit import TransitAbsorption

        model = TransitAbsorption()
        dose = 80.0
        ktr = 0.4
        v = 20.0
        cl = ktr * v
        obs_times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0], dtype=float)

        sol = model.solve(
            {"KTR": ktr, "N_TRANSIT": 1, "CL": cl, "V": v},
            [_MockDoseEvent(0.0, dose)],
            obs_times,
        )

        expected = _bateman_concentration(dose, ktr, cl / v, v, obs_times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=5e-4, atol=1e-8)

    def test_single_transit_multiple_doses_match_exact_superposition(self):
        """For N_TRANSIT=1, linear superposition should match shifted Bateman sums."""
        from openpkpd.pk.absorption.transit import TransitAbsorption

        model = TransitAbsorption()
        ktr = 0.9
        cl = 1.8
        v = 15.0
        doses = [_MockDoseEvent(0.0, 100.0), _MockDoseEvent(3.0, 40.0)]
        obs_times = np.array([0.5, 2.0, 3.0, 3.5, 5.0, 8.0], dtype=float)

        sol = model.solve(
            {"KTR": ktr, "N_TRANSIT": 1, "CL": cl, "V": v},
            doses,
            obs_times,
        )

        expected = sum(
            _bateman_concentration(dose.amount, ktr, cl / v, v, obs_times - dose.time)
            for dose in doses
        )
        np.testing.assert_allclose(sol.ipred, expected, rtol=8e-4, atol=1e-8)


class TestParallelAbsorption:
    def test_import(self):
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        assert ParallelAbsorption is not None

    def test_basic_solve(self):
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        model = ParallelAbsorption()
        pk_params = {"KA": 1.0, "D1": 2.0, "F1_FO": 0.5, "F1_ZO": 0.5, "CL": 2.0, "V": 20.0}
        dose_events = [_MockDoseEvent(0.0, 100.0)]
        obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
        sol = model.solve(pk_params, dose_events, obs_times)

        assert sol.ipred.shape == obs_times.shape
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0)

    def test_all_fo_equals_bateman(self):
        """F1_ZO=0 gives the exact one-compartment Bateman solution."""
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        model = ParallelAbsorption()
        pk_params = {"KA": 1.0, "D1": 1.0, "F1_FO": 1.0, "F1_ZO": 0.0, "CL": 1.0, "V": 10.0}
        dose_events = [_MockDoseEvent(0.0, 100.0)]
        obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
        sol = model.solve(pk_params, dose_events, obs_times)
        expected = _bateman_concentration(100.0, 1.0, 0.1, 10.0, obs_times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-12, atol=1e-12)

    def test_all_zo_matches_exact_zero_order_solution(self):
        """F1_FO=0 gives the exact central zero-order-input solution."""
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        model = ParallelAbsorption()
        pk_params = {"KA": 1.3, "D1": 2.5, "F1_FO": 0.0, "F1_ZO": 1.0, "CL": 1.2, "V": 8.0}
        obs_times = np.array([0.5, 1.0, 2.5, 3.0, 6.0], dtype=float)

        sol = model.solve(pk_params, [_MockDoseEvent(0.0, 60.0)], obs_times)
        expected = _zero_order_concentration(60.0, 2.5, 1.2 / 8.0, 8.0, obs_times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-12, atol=1e-12)

    def test_equal_rate_limit_matches_lhopital_form(self):
        """KA = CL/V should use the exact L'Hôpital limit for the FO branch."""
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        model = ParallelAbsorption()
        ka = 0.35
        v = 10.0
        cl = ka * v
        obs_times = np.array([0.25, 0.75, 1.5, 3.0, 5.0], dtype=float)

        sol = model.solve(
            {"KA": ka, "D1": 2.0, "F1_FO": 1.0, "F1_ZO": 0.0, "CL": cl, "V": v},
            [_MockDoseEvent(0.0, 90.0)],
            obs_times,
        )

        expected = _bateman_concentration(90.0, ka, cl / v, v, obs_times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-12, atol=1e-12)

    def test_parallel_superposition_matches_shifted_exact_sum(self):
        """Parallel absorption is linear across doses and branch contributions."""
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        model = ParallelAbsorption()
        pk_params = {
            "KA": 0.8,
            "D1": 1.5,
            "F1_FO": 0.4,
            "F1_ZO": 0.6,
            "CL": 1.4,
            "V": 10.0,
            "F1": 0.75,
        }
        doses = [_MockDoseEvent(0.0, 100.0), _MockDoseEvent(2.0, 50.0)]
        obs_times = np.array([0.5, 1.5, 2.0, 2.5, 4.0, 6.0], dtype=float)

        sol = model.solve(pk_params, doses, obs_times)

        expected = np.zeros_like(obs_times)
        for dose in doses:
            scaled_amt = dose.amount * pk_params["F1"]
            dt = obs_times - dose.time
            expected += _bateman_concentration(
                scaled_amt * pk_params["F1_FO"],
                pk_params["KA"],
                pk_params["CL"] / pk_params["V"],
                pk_params["V"],
                dt,
            )
            expected += _zero_order_concentration(
                scaled_amt * pk_params["F1_ZO"],
                pk_params["D1"],
                pk_params["CL"] / pk_params["V"],
                pk_params["V"],
                dt,
            )

        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-12, atol=1e-12)

    def test_zero_dose_gives_zero(self):
        """Zero dose -> zero predictions."""
        from openpkpd.pk.absorption.parallel import ParallelAbsorption

        model = ParallelAbsorption()
        pk_params = {"KA": 1.0, "D1": 1.0, "F1_FO": 0.5, "F1_ZO": 0.5, "CL": 1.0, "V": 10.0}
        dose_events = [_MockDoseEvent(0.0, 0.0)]
        obs_times = np.array([1.0, 2.0, 4.0], dtype=float)
        sol = model.solve(pk_params, dose_events, obs_times)
        assert np.allclose(sol.ipred, 0.0, atol=1e-10)


class TestEHCModel:
    def test_import(self):
        from openpkpd.pk.absorption.ehc import EnterohepatiCRecirculation

        assert EnterohepatiCRecirculation is not None

    def test_basic_solve(self):
        from openpkpd.pk.absorption.ehc import EnterohepatiCRecirculation

        model = EnterohepatiCRecirculation()
        pk_params = {
            "CL": 1.0,
            "V": 10.0,
            "KGB": 0.5,
            "FGBMAX": 0.1,
            "EHC_INTERVAL": 6.0,
            "KA": 0.0,
        }
        dose_events = [_MockDoseEvent(0.0, 100.0)]
        obs_times = np.linspace(0.5, 24.0, 20)
        sol = model.solve(pk_params, dose_events, obs_times)

        assert sol.ipred.shape == obs_times.shape
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0)

    def test_ehc_secondary_peak(self):
        """EHC model should produce a secondary bump after initial decline."""
        from openpkpd.pk.absorption.ehc import EnterohepatiCRecirculation

        model = EnterohepatiCRecirculation()
        pk_params = {
            "CL": 0.5,
            "V": 10.0,
            "KGB": 2.0,
            "FGBMAX": 0.3,
            "EHC_INTERVAL": 6.0,
            "EHC_DURATION": 1.0,
            "KA": 0.0,
        }
        dose_events = [_MockDoseEvent(0.0, 100.0)]
        obs_times = np.linspace(0.5, 24.0, 48)
        sol = model.solve(pk_params, dose_events, obs_times)

        # Concentrations should be positive
        assert np.max(sol.ipred) > 0

    def test_solution_non_negative(self):
        """EHC predictions should never be negative."""
        from openpkpd.pk.absorption.ehc import EnterohepatiCRecirculation

        model = EnterohepatiCRecirculation()
        pk_params = {
            "CL": 1.0,
            "V": 10.0,
            "KGB": 0.5,
            "FGBMAX": 0.15,
            "EHC_INTERVAL": 4.0,
            "KA": 1.0,
        }
        dose_events = [_MockDoseEvent(0.0, 100.0)]
        obs_times = np.linspace(0.1, 24.0, 100)
        sol = model.solve(pk_params, dose_events, obs_times)
        assert np.all(sol.ipred >= 0)

    def test_zero_dose_gives_zero_predictions(self):
        from openpkpd.pk.absorption.ehc import EnterohepatiCRecirculation

        model = EnterohepatiCRecirculation()
        pk_params = {
            "CL": 1.0,
            "V": 10.0,
            "KGB": 0.5,
            "FGBMAX": 0.2,
            "EHC_INTERVAL": 6.0,
            "KA": 0.0,
        }
        obs_times = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)

        sol = model.solve(pk_params, [_MockDoseEvent(0.0, 0.0)], obs_times)

        np.testing.assert_allclose(sol.ipred, 0.0, atol=1e-10)
        np.testing.assert_allclose(sol.amounts, 0.0, atol=1e-10)

    def test_fgbmax_zero_reduces_to_one_compartment_iv_bolus(self):
        from openpkpd.pk.absorption.ehc import EnterohepatiCRecirculation

        model = EnterohepatiCRecirculation()
        dose = 100.0
        cl = 1.0
        v = 10.0
        obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
        pk_params = {
            "CL": cl,
            "V": v,
            "KGB": 0.5,
            "FGBMAX": 0.0,
            "EHC_INTERVAL": 6.0,
            "KA": 0.0,
        }

        sol = model.solve(pk_params, [_MockDoseEvent(0.0, dose)], obs_times)

        expected = (dose / v) * np.exp(-(cl / v) * obs_times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(sol.amounts[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(sol.amounts[:, 2], 0.0, atol=1e-8)

    def test_kgb_zero_prevents_return_to_central_while_gallbladder_accumulates(self):
        from openpkpd.pk.absorption.ehc import EnterohepatiCRecirculation

        model = EnterohepatiCRecirculation()
        dose = 100.0
        cl = 1.0
        v = 10.0
        obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
        pk_params = {
            "CL": cl,
            "V": v,
            "KGB": 0.0,
            "FGBMAX": 0.2,
            "EHC_INTERVAL": 6.0,
            "KA": 0.0,
        }

        sol = model.solve(pk_params, [_MockDoseEvent(0.0, dose)], obs_times)

        expected = (dose / v) * np.exp(-(cl / v) * obs_times)
        np.testing.assert_allclose(sol.ipred, expected, rtol=1e-6, atol=1e-8)
        assert np.all(sol.amounts[:, 2] >= 0.0)
        assert np.all(np.diff(sol.amounts[:, 2]) >= -1e-10)
