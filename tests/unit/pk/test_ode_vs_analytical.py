"""
C3: ODE vs analytical comparison test.

For a 2-compartment IV bolus, compare ADVAN6+DES predictions against
ADVAN3 (analytical) predictions.  Results should agree to ODE tolerance 1e-4.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent
from openpkpd.parser.code_compiler import NMTRANCompiler
from openpkpd.pk.analytical.advan3 import ADVAN3
from openpkpd.pk.ode.advan6 import ADVAN6

# ── Helpers ───────────────────────────────────────────────────────────────────

OBS_TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])

# 2-compartment IV bolus parameters
# K10 = elimination from central, K12/K21 = inter-compartmental exchange
# ADVAN3 uses "K" for K10; ADVAN6 DES uses K10 directly
PK_PARAMS_MICRO = {
    "K": 0.1,  # elimination rate (=K10=CL/V1) for ADVAN3
    "K10": 0.1,  # same value for the DES block variable K10
    "K12": 0.3,  # central → peripheral
    "K21": 0.15,  # peripheral → central
    "V": 10.0,  # V1 = central volume
    "V1": 10.0,
}


def make_bolus_dose(amt: float = 100.0) -> list[DoseEvent]:
    return [DoseEvent(time=0.0, amount=amt, compartment=1)]


# ── 2-cmt DES code ───────────────────────────────────────────────────────────

DES_2CMT = """
DADT(1) = -(K10 + K12)*A(1) + K21*A(2)
DADT(2) = K12*A(1) - K21*A(2)
"""


@pytest.mark.unit
class TestODEVsAnalytical:
    """Compare ADVAN6 ODE predictions against ADVAN3 analytical solutions."""

    def _compile_des(self) -> object:
        compiler = NMTRANCompiler(use_jax=False)
        return compiler.compile_des(DES_2CMT, n_compartments=2)

    def test_2cmt_iv_ipred_agrees(self):
        """
        ADVAN6 + 2-cmt DES should match ADVAN3 IPRED to within 1e-4 rel. tol.

        ADVAN3 requires micro-parameters: K10, K12, K21, V (=V1).
        ADVAN6 integrates the same ODEs numerically.
        """
        des_callable = self._compile_des()
        dose = make_bolus_dose(100.0)

        # --- ADVAN3 analytical ---
        advan3 = ADVAN3()
        sol3 = advan3.solve(PK_PARAMS_MICRO, dose, OBS_TIMES)
        ipred_analytical = sol3.ipred

        # --- ADVAN6 ODE ---
        advan6 = ADVAN6(n_compartments=2)
        sol6 = advan6.solve(PK_PARAMS_MICRO, dose, OBS_TIMES, des_callable=des_callable)
        ipred_ode = sol6.ipred

        np.testing.assert_allclose(
            ipred_ode,
            ipred_analytical,
            rtol=1e-4,
            atol=1e-6,
            err_msg="ADVAN6 ODE predictions diverge from ADVAN3 analytical",
        )

    def test_2cmt_iv_amounts_non_negative(self):
        """Compartment amounts must be non-negative at all observation times."""
        des_callable = self._compile_des()
        advan6 = ADVAN6(n_compartments=2)
        sol = advan6.solve(PK_PARAMS_MICRO, make_bolus_dose(), OBS_TIMES, des_callable=des_callable)
        assert np.all(sol.amounts >= -1e-9), "Negative amounts found in ODE solution"

    def test_monotone_elimination(self):
        """
        For a simple 1-cmt IV bolus, ODE concentration should be
        monotonically decreasing.
        """
        des_1cmt = """
DADT(1) = -K10 * A(1)
"""
        compiler = NMTRANCompiler(use_jax=False)
        des_callable = compiler.compile_des(des_1cmt, n_compartments=1)
        pk_params = {"K10": 0.1, "V": 10.0, "V1": 10.0}
        advan6 = ADVAN6(n_compartments=1)
        sol = advan6.solve(pk_params, make_bolus_dose(100.0), OBS_TIMES, des_callable=des_callable)
        ipred = sol.ipred
        diffs = np.diff(ipred)
        assert np.all(diffs <= 1e-9), "IPRED not monotonically decreasing for 1-cmt IV"

    def test_mass_conservation(self):
        """
        Total amount across all compartments should equal dose minus eliminated
        (i.e., A1+A2 ≤ dose at all times for a 2-cmt system with elimination).
        """
        des_callable = self._compile_des()
        dose_amt = 100.0
        advan6 = ADVAN6(n_compartments=2)
        sol = advan6.solve(
            PK_PARAMS_MICRO, make_bolus_dose(dose_amt), OBS_TIMES, des_callable=des_callable
        )
        total = sol.amounts.sum(axis=1)
        assert np.all(total <= dose_amt + 1e-6), "Total drug exceeds dose (mass violation)"
        assert np.all(total >= 0.0), "Negative total amount (mass violation)"
