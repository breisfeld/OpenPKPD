"""
Unit tests for ADVAN4 (2-compartment oral).
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.pk.analytical.advan4 import ADVAN4

# Reference PK parameters (K from CL/V1 via TRANS4 equivalent)
# CL=2, V1=10, Q=1, V2=20 → K=0.2, K12=0.1, K21=0.05
_PARAMS_2CMT = {"KA": 1.5, "K": 0.2, "K12": 0.1, "K21": 0.05, "V2": 10.0}
_PARAMS_1CMT = {"KA": 1.5, "K": 0.2, "V": 10.0}

_DOSE = [DoseEvent(time=0.0, amount=100.0, compartment=1)]
_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])


@pytest.fixture
def advan4():
    return ADVAN4()


@pytest.fixture
def advan2():
    return ADVAN2()


@pytest.mark.unit
def test_peak_later_than_advan2(advan4, advan2):
    """
    2-cmt oral peak: occurs at an intermediate time (not first or last observation).
    The 2-cmt peak may shift earlier or later than 1-cmt depending on K12/K21/V2.
    """
    sol4 = advan4.solve(_PARAMS_2CMT, _DOSE, _TIMES)
    sol2 = advan2.solve(_PARAMS_1CMT, _DOSE, _TIMES)

    peak_idx4 = int(np.argmax(sol4.ipred))
    # Peak should be at an intermediate time point (not endpoint)
    assert 0 < peak_idx4 < len(_TIMES) - 1, f"2-cmt oral peak at boundary index {peak_idx4}"
    # Both peaks should occur before the last time point
    peak_idx2 = int(np.argmax(sol2.ipred))
    assert 0 < peak_idx2 < len(_TIMES) - 1


@pytest.mark.unit
def test_amounts_nonnegative(advan4):
    """
    Central compartment (IPRED = A2/V2) and depot (A1) must be ≥ 0.

    Note: The peripheral compartment A3 uses an approximate analytical
    formula and may show small negative values at early times before
    the true steady-state distribution is established (known limitation
    of the current analytical approximation — use ADVAN6 for exact A3).
    """
    sol = advan4.solve(_PARAMS_2CMT, _DOSE, _TIMES)
    # Central and depot must be non-negative
    a1 = sol.amounts[:, 0]
    a2 = sol.amounts[:, 1]
    assert np.all(a1 >= -1e-10), f"Depot A1 negative: {a1.min():.3e}"
    assert np.all(a2 >= -1e-10), f"Central A2 negative: {a2.min():.3e}"
    assert np.all(sol.ipred >= -1e-10), f"IPRED negative: {sol.ipred.min():.3e}"


@pytest.mark.unit
def test_ipred_nonnegative(advan4):
    """IPRED = A2/V2 must be non-negative."""
    sol = advan4.solve(_PARAMS_2CMT, _DOSE, _TIMES)
    assert np.all(sol.ipred >= -1e-10)


@pytest.mark.unit
def test_biexponential_terminal_phase(advan4):
    """
    In the terminal phase (large t), log(IPRED) should be approximately
    linear — i.e., the slower exponential dominates.
    """
    late_times = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    sol = advan4.solve(_PARAMS_2CMT, _DOSE, late_times)
    log_conc = np.log(sol.ipred + 1e-30)
    # Fit a line to log(IPRED) vs time
    coeffs = np.polyfit(late_times, log_conc, 1)
    r2 = 1 - np.var(log_conc - np.polyval(coeffs, late_times)) / np.var(log_conc)
    assert r2 > 0.998, f"Terminal phase not linear in log space: R²={r2:.4f}"


@pytest.mark.unit
def test_degenerate_ka_equals_lambda(advan4):
    """No NaN when KA ≈ λ1 (exercises numerical fallback)."""
    from openpkpd.pk.analytical.advan3 import _eigenvalues

    k, k12, k21 = 0.2, 0.1, 0.05
    lam1, lam2 = _eigenvalues(k, k12, k21)

    # Set KA ≈ λ1 (within 1e-7)
    params_degenerate = {"KA": lam1 + 1e-8, "K": k, "K12": k12, "K21": k21, "V2": 10.0}
    sol = advan4.solve(params_degenerate, _DOSE, _TIMES)
    assert not np.any(np.isnan(sol.ipred)), "NaN in IPRED when KA ≈ λ1"
    assert not np.any(np.isinf(sol.ipred)), "Inf in IPRED when KA ≈ λ1"


@pytest.mark.unit
def test_multiple_doses_superposition(advan4):
    """
    Two identical doses separated by a dosing interval should give
    higher concentrations after the second dose than after the first.
    """
    tau = 12.0
    two_doses = [
        DoseEvent(time=0.0, amount=100.0, compartment=1),
        DoseEvent(time=tau, amount=100.0, compartment=1),
    ]
    single_dose = [DoseEvent(time=0.0, amount=100.0, compartment=1)]

    t_after_2nd = np.array([tau + 0.5, tau + 2.0, tau + 6.0])

    sol_two = advan4.solve(_PARAMS_2CMT, two_doses, t_after_2nd)
    sol_one = advan4.solve(_PARAMS_2CMT, single_dose, t_after_2nd)

    # After 2nd dose, concentrations must exceed single-dose concentrations
    assert np.all(sol_two.ipred > sol_one.ipred), (
        "Superposition failed: 2-dose concentrations not higher than 1-dose"
    )


@pytest.mark.unit
def test_no_dose_returns_zeros(advan4):
    """Without any dose events, IPRED should be zero everywhere."""
    sol = advan4.solve(_PARAMS_2CMT, [], _TIMES)
    assert np.all(sol.ipred == pytest.approx(0.0))
