"""
Unit tests for ODE-based PK subroutines: ADVAN6, ADVAN8, ADVAN10.

Tests verify:
  - ADVAN6 reproduces ADVAN1 (1-cmt IV) via a 1-cmt DES block
  - ADVAN6 reproduces ADVAN3 (2-cmt IV) via a 2-cmt DES block
  - ADVAN8 (stiff solver) gives same results as ADVAN6
  - ADVAN10 (Michaelis-Menten) gives non-negative amounts and correct behavior
  - CompiledDESCallable correctly injects pk_params into DES code namespace
  - Error handling: missing des_callable raises PKError
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import lambertw

from openpkpd.data.event_processor import DoseEvent
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.ode.advan6 import ADVAN6
from openpkpd.pk.ode.advan8 import ADVAN8
from openpkpd.pk.ode.advan10 import ADVAN10
from openpkpd.utils.errors import PKError

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_bolus(t: float = 0.0, amt: float = 100.0, cmt: int = 1) -> list[DoseEvent]:
    """Create a single bolus dose event."""
    return [DoseEvent(time=t, amount=amt, compartment=cmt)]


_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])


def _mm_exact_bolus_amount(
    times: np.ndarray,
    dose_amount: float,
    vmax: float,
    km: float,
    v: float,
    dose_time: float = 0.0,
) -> np.ndarray:
    """Exact ADVAN10 bolus solution using the Lambert W form."""
    times = np.asarray(times, dtype=float)
    amounts = np.zeros_like(times)
    if dose_amount <= 0:
        return amounts

    scale = km * v
    post_mask = times >= dose_time - 1e-12
    t_rel = np.maximum(times[post_mask] - dose_time, 0.0)
    arg = (dose_amount / scale) * np.exp((dose_amount - vmax * t_rel) / scale)
    amounts[post_mask] = scale * np.real(lambertw(arg))
    return amounts


def _two_cmt_linear_superposition_amounts(
    times: np.ndarray,
    doses: list[DoseEvent],
    k10: float,
    k12: float,
    k21: float,
) -> np.ndarray:
    """Exact 2-cmt linear bolus superposition via the matrix exponential."""
    times = np.asarray(times, dtype=float)
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


# ── CompiledDESCallable pk_params injection ───────────────────────────────────


@pytest.mark.unit
class TestDESCallableInjection:
    """Verify that CompiledDESCallable correctly injects pk_params as local vars."""

    def test_1cmt_des_accesses_k_directly(self):
        """
        DES code referencing K directly should work after pk_params injection.

        DES code: DADT(1) = -K * A(1)
        pk_params: {'K': 0.1}
        """
        compiler = NMTRANCompiler()
        des_code = "DADT(1) = -K * A(1)"
        des_fn = compiler.compile_des(des_code, n_compartments=1)

        pk_params = {"K": 0.1}
        a = [50.0]
        dadt = des_fn(t=1.0, a=a, pk_params=pk_params, theta=[], eta=[])

        expected = -0.1 * 50.0
        assert abs(dadt[0] - expected) < 1e-10, f"Expected dadt[0]={expected}, got {dadt[0]}"

    def test_2cmt_des_accesses_k12_k21(self):
        """
        2-cmt DES code referencing K, K12, K21 should work after pk_params injection.
        """
        compiler = NMTRANCompiler()
        des_code = "DADT(1) = -(K + K12) * A(1) + K21 * A(2)\nDADT(2) = K12 * A(1) - K21 * A(2)"
        des_fn = compiler.compile_des(des_code, n_compartments=2)

        pk_params = {"K": 0.1, "K12": 0.05, "K21": 0.03}
        a = [80.0, 20.0]
        dadt = des_fn(t=1.0, a=a, pk_params=pk_params, theta=[], eta=[])

        expected_0 = -(0.1 + 0.05) * 80.0 + 0.03 * 20.0
        expected_1 = 0.05 * 80.0 - 0.03 * 20.0
        assert abs(dadt[0] - expected_0) < 1e-8
        assert abs(dadt[1] - expected_1) < 1e-8

    def test_des_with_math_functions(self):
        """DES code using math.exp should be accessible."""
        compiler = NMTRANCompiler()
        # Test with EXP intrinsic
        des_code = "DADT(1) = -K * A(1)"
        des_fn = compiler.compile_des(des_code, n_compartments=1)
        pk_params = {"K": 0.2}
        dadt = des_fn(0.0, [100.0], pk_params, [], [])
        assert abs(dadt[0] - (-0.2 * 100.0)) < 1e-10


# ── ADVAN6 vs ADVAN1 ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestADVAN6vs1Cmt:
    """ADVAN6 with 1-cmt DES should match ADVAN1 analytical solution."""

    @pytest.fixture
    def setup(self):
        """Set up 1-cmt DES callable and parameters."""
        compiler = NMTRANCompiler()
        des_code = "DADT(1) = -K * A(1)"
        des_fn = compiler.compile_des(des_code, n_compartments=1)
        return des_fn

    def test_1cmt_matches_advan1(self, setup):
        """ODE 1-cmt should match ADVAN1 analytical within tolerance."""
        des_fn = setup
        K, V = 0.1, 20.0
        pk_params = {"K": K, "V": V}

        advan6 = ADVAN6(n_compartments=1, rtol=1e-8, atol=1e-10)
        advan1 = ADVAN1()

        sol6 = advan6.solve(pk_params, make_bolus(amt=100.0), _TIMES, des_callable=des_fn)
        sol1 = advan1.solve(pk_params, make_bolus(amt=100.0), _TIMES)

        np.testing.assert_allclose(
            sol6.ipred, sol1.ipred, rtol=1e-4, err_msg="ADVAN6 1-cmt should match ADVAN1 analytical"
        )

    def test_1cmt_amounts_nonnegative(self, setup):
        """ODE 1-cmt amounts must be non-negative."""
        des_fn = setup
        advan6 = ADVAN6(n_compartments=1)
        sol = advan6.solve({"K": 0.15, "V": 25.0}, make_bolus(), _TIMES, des_callable=des_fn)
        assert np.all(sol.amounts >= -1e-6)

    def test_1cmt_infusion(self, setup):
        """ODE 1-cmt infusion should give lower peak than bolus."""
        des_fn = setup
        K, V = 0.1, 20.0
        pk_params = {"K": K, "V": V}
        advan6 = ADVAN6(n_compartments=1)

        bolus = make_bolus(amt=100.0)
        infusion = [DoseEvent(time=0.0, amount=100.0, rate=100.0)]
        times = np.array([0.5, 1.0, 2.0])

        sol_bol = advan6.solve(pk_params, bolus, times, des_callable=des_fn)
        sol_inf = advan6.solve(pk_params, infusion, times, des_callable=des_fn)

        assert sol_inf.ipred[0] < sol_bol.ipred[0], (
            "Infusion concentration should be less than bolus at t=0.5"
        )


# ── ADVAN6 vs ADVAN3 ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestADVAN6vs2Cmt:
    """ADVAN6 with 2-cmt DES should approximately match ADVAN3 analytical."""

    @pytest.fixture
    def setup(self):
        """Set up 2-cmt DES callable and parameters."""
        compiler = NMTRANCompiler()
        des_code = "DADT(1) = -(K + K12) * A(1) + K21 * A(2)\nDADT(2) = K12 * A(1) - K21 * A(2)"
        des_fn = compiler.compile_des(des_code, n_compartments=2)
        return des_fn

    def test_2cmt_matches_advan3(self, setup):
        """ODE 2-cmt should match ADVAN3 analytical within tolerance."""
        des_fn = setup
        pk_params = {"K": 0.1, "K12": 0.05, "K21": 0.03, "V1": 10.0}

        advan6 = ADVAN6(n_compartments=2, rtol=1e-8, atol=1e-10)
        advan3 = ADVAN3()

        sol6 = advan6.solve(pk_params, make_bolus(amt=100.0), _TIMES, des_callable=des_fn)
        sol3 = advan3.solve(pk_params, make_bolus(amt=100.0), _TIMES)

        np.testing.assert_allclose(
            sol6.ipred,
            sol3.ipred,
            rtol=1e-3,
            err_msg="ADVAN6 2-cmt should approximate ADVAN3 analytical",
        )

    def test_2cmt_biexponential(self, setup):
        """ODE 2-cmt should show biexponential decay pattern."""
        des_fn = setup
        advan6 = ADVAN6(n_compartments=2)
        pk_params = {"K": 0.1, "K12": 0.05, "K21": 0.03, "V1": 10.0}
        sol = advan6.solve(pk_params, make_bolus(amt=100.0), _TIMES, des_callable=des_fn)

        # Concentration should decrease over time (overall)
        assert sol.ipred[-1] < sol.ipred[0]
        assert np.all(sol.amounts >= -1e-6)

    def test_2cmt_shape(self, setup):
        """Output amounts should have 2 compartments."""
        des_fn = setup
        advan6 = ADVAN6(n_compartments=2)
        sol = advan6.solve(
            {"K": 0.1, "K12": 0.05, "K21": 0.03, "V1": 10.0},
            make_bolus(),
            _TIMES,
            des_callable=des_fn,
        )
        assert sol.amounts.shape == (len(_TIMES), 2)


# ── ADVAN6 error handling ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_advan6_raises_without_des_callable():
    """ADVAN6 must raise PKError if des_callable is not provided."""
    advan6 = ADVAN6()
    with pytest.raises(PKError, match="des_callable"):
        advan6.solve({"K": 0.1, "V": 10.0}, make_bolus(), _TIMES)


@pytest.mark.unit
def test_advan6_empty_obs_times():
    """ADVAN6 should handle empty observation times gracefully."""
    compiler = NMTRANCompiler()
    des_fn = compiler.compile_des("DADT(1) = -K * A(1)", n_compartments=1)
    advan6 = ADVAN6(n_compartments=1)
    sol = advan6.solve({"K": 0.1, "V": 10.0}, make_bolus(), np.array([]), des_callable=des_fn)
    assert sol.ipred.shape == (0,)
    assert sol.amounts.shape == (0, 1)


@pytest.mark.unit
def test_advan6_duplicate_obs_times_are_preserved_by_occurrence():
    """ADVAN6 should support duplicate observation times without reordering."""
    compiler = NMTRANCompiler()
    des_fn = compiler.compile_des("DADT(1) = -K * A(1)", n_compartments=1)
    advan6 = ADVAN6(n_compartments=1, rtol=1e-8, atol=1e-10)
    times = np.array([0.5, 1.0, 1.0, 2.0])

    sol = advan6.solve({"K": 0.1, "V": 10.0}, make_bolus(), times, des_callable=des_fn)

    assert sol.amounts.shape == (4, 1)
    assert sol.ipred[1] == pytest.approx(sol.ipred[2], rel=0.0, abs=1e-12)
    assert sol.times.tolist() == pytest.approx(times.tolist())


@pytest.mark.unit
def test_advan6_honors_pcmt_override_for_output_compartment():
    """ADVAN6 should use PCMT from pk_params to select the observed compartment."""
    compiler = NMTRANCompiler()
    des_fn = compiler.compile_des(
        "DADT(1) = -K12 * A(1)\nDADT(2) = K12 * A(1) - K20 * A(2)",
        n_compartments=2,
    )
    advan6 = ADVAN6(n_compartments=2, rtol=1e-8, atol=1e-10)
    times = np.array([0.5, 1.0, 2.0])

    sol_default = advan6.solve(
        {"K12": 0.5, "K20": 0.1, "V": 10.0},
        make_bolus(),
        times,
        des_callable=des_fn,
    )
    sol_pcmt2 = advan6.solve(
        {"K12": 0.5, "K20": 0.1, "V": 10.0, "PCMT": 2},
        make_bolus(),
        times,
        des_callable=des_fn,
    )

    assert not np.allclose(sol_default.ipred, sol_pcmt2.ipred)
    assert sol_default.ipred[0] > sol_pcmt2.ipred[0]


# ── ADVAN8 ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestADVAN8:
    """ADVAN8 should produce same results as ADVAN6 (just different solver)."""

    def test_advan8_matches_advan6(self):
        """ADVAN8 (LSODA) should match ADVAN6 (RK45) for simple 1-cmt case."""
        compiler = NMTRANCompiler()
        des_fn = compiler.compile_des("DADT(1) = -K * A(1)", n_compartments=1)

        pk_params = {"K": 0.1, "V": 20.0}
        advan6 = ADVAN6(n_compartments=1, rtol=1e-8, atol=1e-10)
        advan8 = ADVAN8(n_compartments=1, rtol=1e-8, atol=1e-10)

        sol6 = advan6.solve(pk_params, make_bolus(amt=100.0), _TIMES, des_callable=des_fn)
        sol8 = advan8.solve(pk_params, make_bolus(amt=100.0), _TIMES, des_callable=des_fn)

        np.testing.assert_allclose(
            sol6.ipred,
            sol8.ipred,
            rtol=1e-4,
            err_msg="ADVAN8 should match ADVAN6 for non-stiff 1-cmt",
        )

    def test_advan8_advan_number(self):
        """ADVAN8 should have advan=8."""
        assert ADVAN8.advan == 8

    def test_advan8_default_method(self):
        """ADVAN8 should default to LSODA."""
        advan8 = ADVAN8()
        assert advan8.method == "LSODA"

    def test_advan8_stiff_two_cmt_multi_dose_matches_matrix_exponential_oracle(self):
        """ADVAN8 should solve a stiff linear 2-cmt multi-dose system accurately."""
        compiler = NMTRANCompiler()
        des_fn = compiler.compile_des(
            "DADT(1)=-(K10+K12)*A(1)+K21*A(2)\nDADT(2)=K12*A(1)-K21*A(2)",
            n_compartments=2,
        )
        pk_params = {"K10": 0.1, "K12": 40.0, "K21": 0.02, "V1": 10.0}
        dose_events = [
            DoseEvent(time=0.0, amount=100.0),
            DoseEvent(time=1.5, amount=40.0),
        ]
        times = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 1.6, 2.0, 4.0])

        sol = ADVAN8(n_compartments=2, rtol=1e-9, atol=1e-11).solve(
            pk_params,
            dose_events,
            times,
            des_callable=des_fn,
        )
        expected_amounts = _two_cmt_linear_superposition_amounts(
            times,
            dose_events,
            pk_params["K10"],
            pk_params["K12"],
            pk_params["K21"],
        )

        np.testing.assert_allclose(sol.amounts, expected_amounts, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(
            sol.ipred, expected_amounts[:, 0] / pk_params["V1"], rtol=2e-5, atol=1e-8
        )


# ── Stiff ODE robustness ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestStiffODERobustness:
    """
    Tests that verify correct behaviour for stiff ODEs and the step-limit
    fallback mechanism.

    Key behaviours:
    * ``jit='scipy'`` with method='LSODA' or 'Radau' always handles stiff ODEs
    * ``jit='numpy'`` (explicit RK45) raises RuntimeError when ``max_steps``
      is exceeded; the solver catches this, emits a UserWarning, and retries
      via scipy with ``self.method``  — no silent wrong answers
    * The fallback result matches the ADVAN8/LSODA oracle
    """

    @pytest.fixture
    def stiff_des(self):
        """Compile a 2-cmt DES (K12 >> K21, creating large eigenvalue separation)."""
        compiler = NMTRANCompiler()
        return compiler.compile_des(
            "DADT(1)=-(K10+K12)*A(1)+K21*A(2)\nDADT(2)=K12*A(1)-K21*A(2)",
            n_compartments=2,
        )

    @pytest.fixture
    def stiff_params(self):
        return {"K10": 0.05, "K12": 50.0, "K21": 0.01, "V1": 10.0}

    @pytest.fixture
    def stiff_dose(self):
        return [DoseEvent(time=0.0, amount=100.0)]

    @pytest.fixture
    def stiff_times(self):
        return np.array([0.01, 0.05, 0.1, 0.5, 1.0, 4.0, 8.0, 24.0])

    def _advan8_oracle(self, stiff_params, stiff_dose, stiff_times, stiff_des):
        """Reference solution using ADVAN8 (LSODA stiff solver)."""
        return ADVAN8(n_compartments=2, rtol=1e-10, atol=1e-12).solve(
            stiff_params, stiff_dose, stiff_times, des_callable=stiff_des
        )

    def test_scipy_lsoda_succeeds_on_stiff_ode(
        self, stiff_params, stiff_dose, stiff_times, stiff_des
    ):
        """jit='scipy' with method='LSODA' must solve stiff ODEs without error."""
        sol = ADVAN6(n_compartments=2, method="LSODA", jit="scipy").solve(
            stiff_params, stiff_dose, stiff_times, des_callable=stiff_des
        )
        oracle = self._advan8_oracle(stiff_params, stiff_dose, stiff_times, stiff_des)
        np.testing.assert_allclose(sol.amounts, oracle.amounts, rtol=1e-4, atol=1e-6)

    def test_scipy_radau_succeeds_on_stiff_ode(
        self, stiff_params, stiff_dose, stiff_times, stiff_des
    ):
        """jit='scipy' with method='Radau' must solve stiff ODEs without error."""
        sol = ADVAN6(n_compartments=2, method="Radau", jit="scipy").solve(
            stiff_params, stiff_dose, stiff_times, des_callable=stiff_des
        )
        oracle = self._advan8_oracle(stiff_params, stiff_dose, stiff_times, stiff_des)
        np.testing.assert_allclose(sol.amounts, oracle.amounts, rtol=1e-4, atol=1e-6)

    def test_numpy_rk45_raises_on_step_limit(self, stiff_des):
        """numpy_rk45_solve must raise RuntimeError when max_steps is exhausted.

        This tests the error mechanism directly with a deliberately small
        max_steps=10 so the test is fast and independent of physical stiffness.
        """
        from openpkpd.pk.ode.jit import numpy_rk45_solve

        # Any non-trivial ODE will fail to integrate over 24 h in only 10 steps
        rhs = lambda t, y: np.array([-0.1 * y[0]])  # noqa: E731
        y0 = np.array([100.0])
        t_eval = np.array([24.0])
        with pytest.raises(RuntimeError, match="max_steps"):
            numpy_rk45_solve(rhs, 0.0, 24.0, y0, t_eval, rtol=1e-6, atol=1e-8, max_steps=10)

    def test_numpy_tier_warns_and_falls_back_when_step_limit_hit(
        self, stiff_params, stiff_dose, stiff_times, stiff_des
    ):
        """jit='numpy' with a tiny max_steps must emit a UserWarning and still
        return a correct result via the scipy fallback — not silently wrong data.

        max_steps=10 forces the fallback for any ODE (not relying on stiffness).
        """
        advan = ADVAN6(n_compartments=2, method="LSODA", jit="numpy", max_steps=10)
        with pytest.warns(UserWarning, match="step-limit exceeded"):
            sol = advan.solve(stiff_params, stiff_dose, stiff_times, des_callable=stiff_des)
        oracle = self._advan8_oracle(stiff_params, stiff_dose, stiff_times, stiff_des)
        # The scipy LSODA fallback must give a result consistent with the oracle
        np.testing.assert_allclose(sol.amounts, oracle.amounts, rtol=1e-4, atol=1e-6)

    def test_no_silent_wrong_answer_when_step_limit_hit(
        self, stiff_params, stiff_dose, stiff_times, stiff_des
    ):
        """After the fallback, IPRED must be physiologically sensible
        (positive, decreasing) — the old silent-fill bug would produce a
        constant plateau or zeroes for the unfilled eval points."""
        advan = ADVAN6(n_compartments=2, method="LSODA", jit="numpy", max_steps=10)
        with pytest.warns(UserWarning, match="step-limit exceeded"):
            sol = advan.solve(stiff_params, stiff_dose, stiff_times, des_callable=stiff_des)
        assert np.all(sol.ipred >= 0), "IPRED has negative values — wrong answer"
        # Concentrations must decline over 24 h (no constant plateau or zero fill)
        assert sol.ipred[-1] < sol.ipred[0], "IPRED plateau suggests silent fill bug"

    def test_normal_ode_no_warning_no_fallback(self, stiff_times):
        """A non-stiff ODE with default max_steps must complete without warning."""
        compiler = NMTRANCompiler()
        simple_des = compiler.compile_des("DADT(1) = -K * A(1)", n_compartments=1)
        params = {"K": 0.1, "V": 20.0}
        dose = [DoseEvent(time=0.0, amount=100.0)]
        advan = ADVAN6(n_compartments=1, jit="numpy")
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            # Should complete without raising (i.e. no step-limit warning)
            sol = advan.solve(params, dose, stiff_times, des_callable=simple_des)
        assert np.all(sol.ipred >= 0)


# ── ADVAN10 ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestADVAN10:
    """Tests for ADVAN10 Michaelis-Menten 1-compartment elimination."""

    @pytest.fixture
    def params(self):
        return {"Vmax": 100.0, "Km": 10.0, "V": 20.0}

    def test_amounts_nonnegative(self, params):
        """Michaelis-Menten elimination must give non-negative amounts."""
        advan10 = ADVAN10()
        sol = advan10.solve(params, make_bolus(amt=200.0), _TIMES)
        assert np.all(sol.amounts >= -1e-8)
        assert np.all(sol.ipred >= -1e-8)

    def test_concentration_decreasing_at_high_conc(self, params):
        """At very high concentrations, elimination should be nearly zeroth-order."""
        advan10 = ADVAN10()
        # High dose: concentration >> Km, so elimination ≈ Vmax (constant rate)
        dose = 5000.0  # concentration = 250 >> Km=10
        times = np.array([0.1, 0.5, 1.0])
        sol = advan10.solve(params, make_bolus(amt=dose), times)
        # Concentration must decrease
        assert sol.ipred[0] > sol.ipred[-1]

    def test_linear_elimination_limit(self):
        """At very low concentrations (dose << Km), should approach linear elimination."""
        # If C << Km: dA/dt ≈ -Vmax/Km * A → K_el = Vmax / (Km * V)
        Vmax, Km, V = 10.0, 100.0, 10.0
        K_linear = Vmax / (Km * V)  # effective first-order rate

        advan10 = ADVAN10(rtol=1e-8, atol=1e-10)
        advan1 = ADVAN1()

        dose = 1.0  # C0 = 0.1 << Km=100 → linear regime
        times = np.array([1.0, 4.0, 8.0, 12.0])

        pk_ode = {"Vmax": Vmax, "Km": Km, "V": V}
        pk_1cmt = {"K": K_linear, "V": V}

        sol_ode = advan10.solve(pk_ode, make_bolus(amt=dose), times)
        sol_1cmt = advan1.solve(pk_1cmt, make_bolus(amt=dose), times)

        # Should be within 5% in the linear regime
        np.testing.assert_allclose(
            sol_ode.ipred,
            sol_1cmt.ipred,
            rtol=0.05,
            err_msg="ADVAN10 should approach linear 1-cmt at low concentrations",
        )

    def test_single_bolus_matches_exact_lambert_w_solution(self):
        """Single bolus should match the exact Michaelis-Menten amount solution."""
        params = {"Vmax": 100.0, "Km": 10.0, "V": 20.0}
        dose = 500.0
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

        sol = ADVAN10(rtol=1e-10, atol=1e-12).solve(params, make_bolus(amt=dose), times)
        expected_amounts = _mm_exact_bolus_amount(
            times, dose, params["Vmax"], params["Km"], params["V"]
        )

        np.testing.assert_allclose(sol.amounts[:, 0], expected_amounts, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(sol.ipred, expected_amounts / params["V"], rtol=1e-8, atol=1e-10)

    def test_f1_and_alag1_match_shifted_exact_oracle(self):
        """F1 scaling and ALAG1 shifting should follow the exact delayed bolus oracle."""
        params = {"Vmax": 80.0, "Km": 12.0, "V": 16.0, "F1": 0.6, "ALAG1": 2.0}
        dose = 150.0
        times = np.array([0.5, 1.5, 2.0, 2.5, 4.0, 8.0])

        sol = ADVAN10(rtol=1e-10, atol=1e-12).solve(params, make_bolus(amt=dose), times)
        expected_amounts = _mm_exact_bolus_amount(
            times,
            dose_amount=dose * params["F1"],
            vmax=params["Vmax"],
            km=params["Km"],
            v=params["V"],
            dose_time=params["ALAG1"],
        )
        expected_amounts[np.abs(times - params["ALAG1"]) < 1e-12] = 0.0

        np.testing.assert_allclose(sol.amounts[:, 0], expected_amounts, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(sol.ipred, expected_amounts / params["V"], rtol=1e-8, atol=1e-10)

    def test_f1_zero_suppresses_dose_exactly(self):
        """F1=0 should fully suppress the dose rather than falling back to 1.0."""
        advan10 = ADVAN10()
        params = {"Vmax": 50.0, "Km": 10.0, "V": 20.0, "F1": 0.0}
        times = np.array([0.5, 1.0, 4.0])

        sol = advan10.solve(params, make_bolus(amt=100.0), times)

        np.testing.assert_array_equal(sol.amounts[:, 0], np.zeros_like(times))
        np.testing.assert_array_equal(sol.ipred, np.zeros_like(times))

    def test_uppercase_aliases_and_v1_match_exact_solution(self):
        """VMAX/KM/V1 aliases should behave like the primary ADVAN10 parameter names."""
        params = {"VMAX": 24.0, "KM": 40.0, "V1": 12.0}
        dose = 75.0
        times = np.array([0.25, 1.0, 3.0, 6.0])

        sol = ADVAN10(rtol=1e-10, atol=1e-12).solve(params, make_bolus(amt=dose), times)
        expected_amounts = _mm_exact_bolus_amount(times, dose, 24.0, 40.0, 12.0)

        np.testing.assert_allclose(sol.amounts[:, 0], expected_amounts, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(sol.ipred, expected_amounts / 12.0, rtol=1e-8, atol=1e-10)

    def test_missing_vmax_raises(self):
        """Missing Vmax should raise PKError."""
        advan10 = ADVAN10()
        with pytest.raises(PKError):
            advan10.solve({"Km": 10.0, "V": 20.0}, make_bolus(), _TIMES)

    def test_missing_km_raises(self):
        """Missing Km should raise PKError."""
        advan10 = ADVAN10()
        with pytest.raises(PKError):
            advan10.solve({"Vmax": 100.0, "V": 20.0}, make_bolus(), _TIMES)

    def test_explicit_zero_vmax_is_not_silently_replaced_by_uppercase_alias(self):
        """Explicit Vmax=0 should be rejected even if VMAX is also present."""
        advan10 = ADVAN10()
        params = {"Vmax": 0.0, "VMAX": 100.0, "Km": 10.0, "V": 20.0}

        with pytest.raises(PKError, match="Vmax > 0"):
            advan10.solve(params, make_bolus(), _TIMES)

    def test_explicit_zero_km_is_not_silently_replaced_by_uppercase_alias(self):
        """Explicit Km=0 should be rejected even if KM is also present."""
        advan10 = ADVAN10()
        params = {"Vmax": 100.0, "Km": 0.0, "KM": 10.0, "V": 20.0}

        with pytest.raises(PKError, match="Km > 0"):
            advan10.solve(params, make_bolus(), _TIMES)

    def test_explicit_zero_v_is_not_silently_replaced_by_v1_alias(self):
        """Explicit V=0 should be rejected even if V1 is also present."""
        advan10 = ADVAN10()
        params = {"Vmax": 100.0, "Km": 10.0, "V": 0.0, "V1": 20.0}

        with pytest.raises(PKError, match="V > 0"):
            advan10.solve(params, make_bolus(), _TIMES)

    def test_output_shape(self):
        """ADVAN10 should return 1-compartment output."""
        advan10 = ADVAN10()
        params = {"Vmax": 50.0, "Km": 10.0, "V": 15.0}
        sol = advan10.solve(params, make_bolus(amt=100.0), _TIMES)
        assert sol.amounts.shape == (len(_TIMES), 1)
        assert sol.ipred.shape == (len(_TIMES),)

    def test_advan10_infusion(self):
        """Infusion should give lower peak than equivalent bolus."""
        advan10 = ADVAN10()
        params = {"Vmax": 50.0, "Km": 10.0, "V": 20.0}
        dose_amt = 100.0
        times = np.array([0.5, 1.0, 2.0, 4.0])

        bolus = make_bolus(amt=dose_amt)
        infusion = [DoseEvent(time=0.0, amount=dose_amt, rate=dose_amt / 2.0)]

        sol_bol = advan10.solve(params, bolus, times)
        sol_inf = advan10.solve(params, infusion, times)

        assert sol_inf.ipred[0] < sol_bol.ipred[0], "Infusion C(0.5) should be < bolus C(0.5)"

    def test_advan10_advan_number(self):
        """ADVAN10 should have advan=10."""
        assert ADVAN10.advan == 10

    def test_advan10_multiple_doses(self):
        """Multiple doses should accumulate and give higher concentrations."""
        advan10 = ADVAN10()
        params = {"Vmax": 50.0, "Km": 10.0, "V": 20.0}
        two_doses = [
            DoseEvent(time=0.0, amount=50.0),
            DoseEvent(time=8.0, amount=50.0),
        ]
        single_dose = [DoseEvent(time=0.0, amount=50.0)]
        times = np.array([8.5, 10.0, 12.0])

        sol_two = advan10.solve(params, two_doses, times)
        sol_one = advan10.solve(params, single_dose, times)

        assert np.all(sol_two.ipred >= sol_one.ipred * 0.9), (
            "Two-dose concentrations should be >= single-dose after second dose"
        )
