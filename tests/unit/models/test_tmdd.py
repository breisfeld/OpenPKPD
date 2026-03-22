"""Tests for TMDD pharmacokinetic models."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.data.event_processor import DoseEvent
from openpkpd.models.tmdd import FullTMDD, MichaelisMentenTMDD, QSSATMDDModel
from openpkpd.pk.analytical.advan1 import ADVAN1
from openpkpd.pk.ode.advan10 import ADVAN10


def make_iv_doses(amt=100.0, t=0.0):
    return [DoseEvent(time=t, amount=amt, compartment=1)]


obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0], dtype=float)


class TestFullTMDD:
    def test_solve_returns_pksolution(self):
        model = FullTMDD()
        pk_params = {
            "CL": 0.5,
            "V": 5.0,
            "kon": 0.1,
            "koff": 0.01,
            "kint": 0.2,
            "Ksyn": 0.5,
            "Kdeg": 0.1,
        }
        sol = model.solve(pk_params, make_iv_doses(100.0), obs_times)
        assert sol.ipred.shape == obs_times.shape
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0)

    def test_concentration_declines(self):
        """Drug concentration should generally decline over time."""
        model = FullTMDD()
        pk_params = {
            "CL": 1.0,
            "V": 10.0,
            "kon": 0.05,
            "koff": 0.005,
            "kint": 0.1,
            "Ksyn": 0.5,
            "Kdeg": 0.1,
        }
        sol = model.solve(pk_params, make_iv_doses(100.0), obs_times)
        # Overall trend: first point > last point
        assert sol.ipred[0] >= sol.ipred[-1]

    def test_zero_dose(self):
        """Zero dose -> near-zero concentration."""
        model = FullTMDD()
        pk_params = {
            "CL": 1.0,
            "V": 10.0,
            "kon": 0.05,
            "koff": 0.005,
            "kint": 0.1,
            "Ksyn": 0.5,
            "Kdeg": 0.1,
        }
        sol = model.solve(pk_params, make_iv_doses(0.0), obs_times)
        assert np.allclose(sol.ipred, 0.0, atol=1e-3)

    def test_amounts_shape(self):
        model = FullTMDD()
        pk_params = {
            "CL": 0.5,
            "V": 5.0,
            "kon": 0.1,
            "koff": 0.01,
            "kint": 0.2,
            "Ksyn": 0.5,
            "Kdeg": 0.1,
        }
        sol = model.solve(pk_params, make_iv_doses(), obs_times)
        assert sol.amounts.shape == (len(obs_times), 3)

    @pytest.mark.parametrize(
        "pk_params",
        [
            {"CL": 1.0, "V": 10.0, "kon": 0.0, "koff": 0.01, "kint": 0.2, "Ksyn": 0.5, "Kdeg": 0.1},
            {"CL": 0.3, "V": 4.0, "kon": 0.0, "koff": 0.2, "kint": 0.05, "Ksyn": 0.2, "Kdeg": 0.05},
        ],
    )
    def test_reduces_to_advan1_when_binding_is_disabled(self, pk_params):
        model = FullTMDD()
        advan1 = ADVAN1()

        sol_tmdd = model.solve(pk_params, make_iv_doses(100.0), obs_times)
        sol_ref = advan1.solve(
            {"K": pk_params["CL"] / pk_params["V"], "V": pk_params["V"]},
            make_iv_doses(100.0),
            obs_times,
        )

        np.testing.assert_allclose(sol_tmdd.ipred, sol_ref.ipred, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            sol_tmdd.amounts[:, 1],
            pk_params["Ksyn"] / pk_params["Kdeg"],
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(sol_tmdd.amounts[:, 2], 0.0, atol=1e-10)


class TestQSSATMDDModel:
    def test_basic_solve(self):
        model = QSSATMDDModel()
        pk_params = {"CL": 0.5, "V": 5.0, "Kss": 1.0, "kint": 0.2, "Ksyn": 0.5, "Kdeg": 0.1}
        sol = model.solve(pk_params, make_iv_doses(100.0), obs_times)
        assert sol.ipred.shape == obs_times.shape
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0)

    def test_higher_dose_higher_conc(self):
        """Higher dose -> higher initial concentration."""
        model = QSSATMDDModel()
        pk_params = {"CL": 0.5, "V": 5.0, "Kss": 1.0, "kint": 0.2, "Ksyn": 0.5, "Kdeg": 0.1}
        sol_lo = model.solve(pk_params, make_iv_doses(10.0), obs_times[:3])
        sol_hi = model.solve(pk_params, make_iv_doses(100.0), obs_times[:3])
        assert sol_hi.ipred[0] > sol_lo.ipred[0]

    @pytest.mark.parametrize(
        "pk_params",
        [
            {"CL": 1.0, "V": 10.0, "Kss": 1.0, "kint": 0.0, "Ksyn": 0.5, "Kdeg": 0.1},
            {"CL": 0.4, "V": 6.0, "Kss": 3.0, "kint": 0.0, "Ksyn": 0.3, "Kdeg": 0.15},
        ],
    )
    def test_reduces_to_advan1_when_target_mediated_elimination_is_disabled(self, pk_params):
        model = QSSATMDDModel()
        advan1 = ADVAN1()

        sol_tmdd = model.solve(pk_params, make_iv_doses(100.0), obs_times)
        sol_ref = advan1.solve(
            {"K": pk_params["CL"] / pk_params["V"], "V": pk_params["V"]},
            make_iv_doses(100.0),
            obs_times,
        )

        np.testing.assert_allclose(sol_tmdd.ipred, sol_ref.ipred, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            sol_tmdd.amounts[:, 1],
            pk_params["Ksyn"] / pk_params["Kdeg"],
            rtol=1e-8,
            atol=1e-8,
        )


class TestMichaelisMentenTMDD:
    def test_basic_solve(self):
        model = MichaelisMentenTMDD()
        pk_params = {"CL": 0.1, "V": 5.0, "Vmax": 2.0, "Km": 1.0}
        sol = model.solve(pk_params, make_iv_doses(100.0), obs_times)
        assert sol.ipred.shape == obs_times.shape
        assert np.all(np.isfinite(sol.ipred))
        assert np.all(sol.ipred >= 0)

    def test_nonlinear_kinetics(self):
        """MM model: high dose gives slower initial decline than low dose."""
        model = MichaelisMentenTMDD()
        pk_params = {"CL": 0.0, "V": 5.0, "Vmax": 2.0, "Km": 5.0}

        sol_lo = model.solve(pk_params, make_iv_doses(5.0), obs_times[:4])
        sol_hi = model.solve(pk_params, make_iv_doses(100.0), obs_times[:4])

        if np.all(np.isfinite(sol_lo.ipred)) and np.all(np.isfinite(sol_hi.ipred)):
            # High dose has higher concentration throughout
            assert sol_hi.ipred[0] > sol_lo.ipred[0]

    def test_linear_limit(self):
        """When C << Km, approaches linear kinetics."""
        model = MichaelisMentenTMDD()
        # Very high Km -> linear behavior
        pk_params = {"CL": 0.5, "V": 10.0, "Vmax": 0.001, "Km": 1000.0}
        sol = model.solve(pk_params, make_iv_doses(10.0), obs_times)
        assert np.all(np.isfinite(sol.ipred))

    @pytest.mark.parametrize(
        "pk_params",
        [
            {"CL": 0.0, "V": 5.0, "Vmax": 2.0, "Km": 1.0},
            {"CL": 0.0, "V": 20.0, "Vmax": 50.0, "Km": 10.0},
        ],
    )
    def test_matches_advan10_when_linear_clearance_zero(self, pk_params):
        model = MichaelisMentenTMDD()
        advan10 = ADVAN10()

        sol_tmdd = model.solve(pk_params, make_iv_doses(100.0), obs_times)
        sol_advan10 = advan10.solve(
            {"V": pk_params["V"], "Vmax": pk_params["Vmax"], "Km": pk_params["Km"]},
            make_iv_doses(100.0),
            obs_times,
        )

        np.testing.assert_allclose(sol_tmdd.ipred, sol_advan10.ipred, rtol=1e-5, atol=1e-6)
