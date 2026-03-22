"""
Tests for ADVAN13 forward-mode sensitivity equations.

Verifies that solve_with_sensitivity() produces:
  1. A PKSolution with a non-None `sensitivity` field.
  2. Sensitivity shape (n_times, n_params).
  3. Finite sensitivity values for a simple 1-compartment ODE.
  4. Numerical accuracy: ∂IPRED/∂V matches a finite-difference approximation.
  5. Graceful fallback when des_callable is None.
  6. PKSolution.sensitivity field is None by default.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from scipy.linalg import expm

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.base import PKSolution
from openpkpd.pk.ode.advan8 import ADVAN8
from openpkpd.pk.ode.advan13 import ADVAN13

# ── Shared fixtures ────────────────────────────────────────────────────────────


def _make_dose_event(amount: float = 100.0, time: float = 0.0, compartment: int = 1):
    return DoseEvent(time=time, amount=amount, compartment=compartment, rate=0.0)


def _one_cmt_des_callable(t, a, pk_params, theta, eta):
    """Simple 1-compartment: dA/dt = -k * A.  Signature: (t, a, pk_params, theta, eta)."""
    k = float(pk_params.get("K", 0.1))
    dadt = [-k * a[0]]
    return dadt


def _two_cmt_des_callable(t, a, pk_params, theta, eta):
    """2-compartment: dA1/dt = -k10*A1 - k12*A1 + k21*A2; dA2/dt = k12*A1 - k21*A2."""
    k10 = float(pk_params.get("K10", 0.1))
    k12 = float(pk_params.get("K12", 0.05))
    k21 = float(pk_params.get("K21", 0.03))
    dadt = [-(k10 + k12) * a[0] + k21 * a[1], k12 * a[0] - k21 * a[1]]
    return dadt


def _two_cmt_exact_amounts(times, doses, k10, k12, k21):
    """Exact 2-cmt linear bolus superposition via the matrix exponential."""
    system = np.array(
        [
            [-(k10 + k12), k21],
            [k12, -k21],
        ]
    )
    amounts = np.zeros((len(times), 2))

    for i, t in enumerate(times):
        state = np.zeros(2)
        for dose in doses:
            if dose.rate != 0.0:
                raise ValueError("Helper only supports bolus doses")
            if t >= dose.time - 1e-12:
                state += expm(system * (t - dose.time)) @ np.array([dose.amount, 0.0])
        amounts[i] = state

    return amounts


# ── PKSolution.sensitivity default ────────────────────────────────────────────


class TestPKSolutionSensitivity:
    def test_sensitivity_none_by_default(self):
        sol = PKSolution(
            times=np.array([1.0, 2.0]),
            amounts=np.zeros((2, 1)),
            ipred=np.array([1.0, 0.5]),
        )
        assert sol.sensitivity is None

    def test_sensitivity_stored_when_provided(self):
        sens = np.array([[0.1, 0.2], [0.05, 0.1]])
        sol = PKSolution(
            times=np.array([1.0, 2.0]),
            amounts=np.zeros((2, 1)),
            ipred=np.array([1.0, 0.5]),
            sensitivity=sens,
        )
        assert sol.sensitivity is not None
        assert sol.sensitivity.shape == (2, 2)


# ── solve_with_sensitivity: basic interface ────────────────────────────────────


class TestSolveWithSensitivityInterface:
    @pytest.fixture()
    def advan13(self):
        return ADVAN13(n_compartments=1, rtol=1e-8, atol=1e-10, force_scipy=True)

    def test_returns_pk_solution(self, advan13):
        obs_times = np.array([1.0, 2.0, 4.0])
        pk_params = {"K": 0.1, "V": 10.0}
        sol = advan13.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=[_make_dose_event(100.0)],
            obs_times=obs_times,
            des_callable=_one_cmt_des_callable,
        )
        assert isinstance(sol, PKSolution)

    def test_sensitivity_not_none(self, advan13):
        obs_times = np.array([1.0, 2.0, 4.0])
        pk_params = {"K": 0.1, "V": 10.0}
        sol = advan13.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=[_make_dose_event(100.0)],
            obs_times=obs_times,
            des_callable=_one_cmt_des_callable,
        )
        assert sol.sensitivity is not None

    def test_sensitivity_shape(self, advan13):
        obs_times = np.array([1.0, 2.0, 4.0, 8.0])
        pk_params = {"K": 0.1, "V": 10.0}
        sol = advan13.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=[_make_dose_event(100.0)],
            obs_times=obs_times,
            des_callable=_one_cmt_des_callable,
            param_names=["K", "V"],
        )
        n_times = len(obs_times)
        n_params = 2  # K and V
        assert sol.sensitivity.shape == (n_times, n_params)

    def test_sensitivity_finite(self, advan13):
        obs_times = np.array([0.5, 1.0, 2.0])
        pk_params = {"K": 0.2, "V": 5.0}
        sol = advan13.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=[_make_dose_event(50.0)],
            obs_times=obs_times,
            des_callable=_one_cmt_des_callable,
            param_names=["K"],
        )
        assert np.all(np.isfinite(sol.sensitivity))

    def test_no_des_callable_falls_back(self, advan13):
        """Without des_callable, falls back to base solve (sensitivity=None)."""
        obs_times = np.array([1.0, 2.0])
        pk_params = {"K": 0.1, "V": 10.0}
        sol = advan13.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=[_make_dose_event(100.0)],
            obs_times=obs_times,
            des_callable=None,
        )
        assert isinstance(sol, PKSolution)
        # Falls back to base solve which has no sensitivity
        assert sol.sensitivity is None

    def test_ipred_matches_base_solve(self, advan13):
        """IPRED from solve_with_sensitivity should match base solve closely."""
        obs_times = np.array([1.0, 2.0, 4.0])
        pk_params = {"K": 0.15, "V": 8.0}
        dose = [_make_dose_event(80.0)]

        sol_sens = advan13.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=dose,
            obs_times=obs_times,
            des_callable=_one_cmt_des_callable,
            param_names=["K"],
        )
        sol_base = advan13.solve(pk_params, dose, obs_times, None, _one_cmt_des_callable)

        np.testing.assert_allclose(
            sol_sens.ipred,
            sol_base.ipred,
            rtol=1e-4,
            err_msg="IPRED from sensitivity solve should match base solve",
        )


class TestADVAN13Solve:
    def test_force_scipy_matches_stiff_linear_multi_dose_oracle(self):
        """ADVAN13 scipy fallback should solve a stiff linear multi-dose system accurately."""
        advan13 = ADVAN13(n_compartments=2, rtol=1e-9, atol=1e-11, force_scipy=True)
        pk_params = {"K10": 0.1, "K12": 40.0, "K21": 0.02, "V": 10.0}
        dose_events = [
            _make_dose_event(100.0, time=0.0),
            _make_dose_event(40.0, time=1.5),
        ]
        obs_times = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 1.6, 2.0, 4.0])

        sol = advan13.solve(pk_params, dose_events, obs_times, None, _two_cmt_des_callable)
        expected_amounts = _two_cmt_exact_amounts(
            obs_times,
            dose_events,
            pk_params["K10"],
            pk_params["K12"],
            pk_params["K21"],
        )

        np.testing.assert_allclose(sol.amounts, expected_amounts, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(
            sol.ipred, expected_amounts[:, 0] / pk_params["V"], rtol=2e-5, atol=1e-8
        )

    def test_fake_jax_path_uses_compiled_des_signature_without_scipy_fallback(self, monkeypatch):
        """The JAX path should call DES as des(t, a, pk_params, theta, eta)."""
        fake_jax = types.ModuleType("jax")
        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.zeros = lambda n: np.zeros(n, dtype=float)
        fake_jnp.array = lambda values, dtype=float: np.array(values, dtype=dtype)
        fake_jax.numpy = fake_jnp

        fake_diffrax = types.ModuleType("diffrax")

        class _FakeODETerm:
            def __init__(self, vector_field):
                self.vector_field = vector_field

        class _FakeSaveAt:
            def __init__(self, ts):
                self.ts = np.array(ts, dtype=float)

        class _FakePIDController:
            def __init__(self, rtol, atol):
                self.rtol = rtol
                self.atol = atol

        class _FakeKvaerno5:
            pass

        class _FakeAdjoint:
            pass

        def _fake_diffeqsolve(
            term, solver, t0, t1, dt0, y0, saveat, stepsize_controller, adjoint, max_steps
        ):
            ys = []
            y = np.array(y0, dtype=float)
            for t in saveat.ts:
                term.vector_field(float(t), y, None)
                ys.append(y.copy())
            return types.SimpleNamespace(ys=np.array(ys))

        fake_diffrax.ODETerm = _FakeODETerm
        fake_diffrax.SaveAt = _FakeSaveAt
        fake_diffrax.PIDController = _FakePIDController
        fake_diffrax.Kvaerno5 = _FakeKvaerno5
        fake_diffrax.Adjoint = _FakeAdjoint
        fake_diffrax.diffeqsolve = _fake_diffeqsolve

        monkeypatch.setitem(sys.modules, "jax", fake_jax)
        monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
        monkeypatch.setitem(sys.modules, "diffrax", fake_diffrax)

        def _unexpected_fallback(*args, **kwargs):
            raise AssertionError("unexpected scipy fallback")

        monkeypatch.setattr(ADVAN8, "solve", _unexpected_fallback)

        advan13 = ADVAN13(n_compartments=1, rtol=1e-8, atol=1e-10, force_scipy=False)
        obs_times = np.array([0.25, 1.0])

        sol = advan13.solve(
            {"K": 0.1, "V": 10.0},
            [],
            obs_times,
            None,
            _one_cmt_des_callable,
        )

        np.testing.assert_array_equal(sol.ipred, np.zeros_like(obs_times))
        assert sol.amounts.shape == (len(obs_times), 1)


# ── Numerical accuracy: compare to FD ─────────────────────────────────────────


class TestSensitivityAccuracy:
    """
    Verify ∂IPRED/∂K against a finite-difference approximation for a
    simple 1-compartment ODE with known analytical solution.

    Analytical: IPRED(t) = Dose / V * exp(-K*t)
    ∂IPRED/∂K = -t * Dose / V * exp(-K*t) = -t * IPRED(t)
    """

    @pytest.fixture()
    def advan13(self):
        return ADVAN13(n_compartments=1, rtol=1e-9, atol=1e-11, force_scipy=True)

    def _analytical_sensitivity_K(self, t, dose, k, v):
        """∂IPRED/∂K = -t * dose/v * exp(-k*t)."""
        return -t * (dose / v) * np.exp(-k * t)

    def test_dk_sensitivity_vs_analytical(self, advan13):
        dose = 100.0
        k = 0.2
        v = 10.0
        obs_times = np.array([0.5, 1.0, 2.0, 4.0])

        sol = advan13.solve_with_sensitivity(
            pk_params={"K": k, "V": v},
            dose_events=[_make_dose_event(dose)],
            obs_times=obs_times,
            des_callable=_one_cmt_des_callable,
            param_names=["K"],
        )

        expected = self._analytical_sensitivity_K(obs_times, dose, k, v)
        # Allow 1% relative tolerance (FD error in sensitivity computation)
        np.testing.assert_allclose(
            sol.sensitivity[:, 0],
            expected,
            rtol=0.01,
            err_msg="∂IPRED/∂K does not match analytical value",
        )

    def test_dk_sensitivity_vs_fd(self, advan13):
        """Cross-check against brute-force FD on IPRED."""
        dose = 80.0
        k = 0.15
        v = 5.0
        obs_times = np.array([1.0, 3.0, 6.0])
        eps = 1e-4

        sol_ref = advan13.solve_with_sensitivity(
            pk_params={"K": k, "V": v},
            dose_events=[_make_dose_event(dose)],
            obs_times=obs_times,
            des_callable=_one_cmt_des_callable,
            param_names=["K"],
        )

        sol_p = advan13.solve(
            {"K": k + eps, "V": v},
            [_make_dose_event(dose)],
            obs_times,
            None,
            _one_cmt_des_callable,
        )

        fd_sens = (sol_p.ipred - sol_ref.ipred) / eps
        np.testing.assert_allclose(
            sol_ref.sensitivity[:, 0],
            fd_sens,
            rtol=0.01,
            err_msg="Forward sensitivity does not match FD approximation",
        )


# ── 2-compartment: shape and finite ───────────────────────────────────────────


class TestSensitivity2Cmt:
    @pytest.fixture()
    def advan13_2cmt(self):
        return ADVAN13(n_compartments=2, rtol=1e-8, atol=1e-10, force_scipy=True)

    def test_two_cmt_sensitivity_shape(self, advan13_2cmt):
        obs_times = np.array([0.5, 1.0, 2.0, 4.0])
        pk_params = {"K10": 0.1, "K12": 0.05, "K21": 0.03, "V": 10.0}
        sol = advan13_2cmt.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=[_make_dose_event(100.0)],
            obs_times=obs_times,
            des_callable=_two_cmt_des_callable,
            param_names=["K10", "K12", "K21"],
        )
        assert sol.sensitivity is not None
        assert sol.sensitivity.shape == (len(obs_times), 3)

    def test_two_cmt_sensitivity_finite(self, advan13_2cmt):
        obs_times = np.array([1.0, 2.0])
        pk_params = {"K10": 0.1, "K12": 0.05, "K21": 0.03, "V": 10.0}
        sol = advan13_2cmt.solve_with_sensitivity(
            pk_params=pk_params,
            dose_events=[_make_dose_event(100.0)],
            obs_times=obs_times,
            des_callable=_two_cmt_des_callable,
        )
        assert np.all(np.isfinite(sol.sensitivity))
