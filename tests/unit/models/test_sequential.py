"""Tests for SequentialPKPDWorkflow."""

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.model.parameters import ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.models.pkpd import LinearPDModel, PDData, SequentialPKPDWorkflow
from openpkpd.pk.ode.advan6 import ADVAN6
from tests.regression.diagnostic_helpers import build_pop_model_and_result


class TestSequentialPKPDWorkflow:
    @staticmethod
    def _workflow() -> SequentialPKPDWorkflow:
        class MockPKResult:
            pass

        class MockPKModel:
            pass

        return SequentialPKPDWorkflow(MockPKResult(), MockPKModel())

    def test_fit_matches_direct_linear_pd_fit_with_provided_concentrations(self):
        """Sequential composition should match a direct PD fit."""
        workflow = self._workflow()
        model = LinearPDModel()

        times = np.linspace(0.0, 12.0, 25)
        concs = 5.0 * np.exp(-0.2 * times)
        truth = {"E0": 2.0, "slope": 1.5}
        data = PDData(
            1,
            times,
            model.predict(truth, PDData(1, times, np.zeros(len(times)), concs)),
            concs,
        )
        initial = {"E0": 0.5, "slope": 0.5}

        direct = model.fit(data, initial_params=initial)
        sequential = workflow.fit_pd(data, LinearPDModel, initial_params=initial)

        assert sequential.converged == direct.converged
        assert sequential.ofv == pytest.approx(direct.ofv, abs=1e-10)
        np.testing.assert_allclose(sequential.predicted, direct.predicted, atol=1e-10, rtol=0.0)
        assert sequential.params["E0"] == pytest.approx(direct.params["E0"], abs=1e-10)
        assert sequential.params["slope"] == pytest.approx(direct.params["slope"], abs=1e-10)

    def test_fit_without_concentrations_matches_direct_zero_concentration_fit(self):
        """Missing concentrations should match explicit zero concentrations."""
        workflow = self._workflow()
        model = LinearPDModel()

        times = np.linspace(0.0, 10.0, 15)
        response = np.full(len(times), 4.0)
        zero_concs = np.zeros(len(times))
        initial = {"E0": 1.0, "slope": 0.7}

        direct = model.fit(PDData(1, times, response, zero_concs), initial_params=initial)
        missing = PDData(1, times, response, concentrations=None)
        sequential = workflow.fit_pd(missing, LinearPDModel, initial_params=initial)

        assert missing.concentrations is None
        assert sequential.ofv == pytest.approx(direct.ofv, abs=1e-10)
        np.testing.assert_allclose(sequential.predicted, direct.predicted, atol=1e-10, rtol=0.0)
        assert sequential.params["E0"] == pytest.approx(direct.params["E0"], abs=1e-10)
        assert sequential.params["slope"] == pytest.approx(direct.params["slope"], abs=1e-10)

    def test_fit_without_concentrations_uses_pk_predictions_when_available(self):
        """Missing concentrations should be extracted from the fitted PK workflow when possible."""
        from openpkpd.models.pkpd import EmaxModel

        pk_model, pk_result = build_pop_model_and_result(n_subjects=1, seed=77)
        workflow = SequentialPKPDWorkflow(pk_result, pk_model)

        sid = pk_model.subject_ids()[0]
        indiv = pk_model.individual_model(sid)
        concs, _obs_mask, _f = indiv.evaluate(
            pk_result.theta_final,
            pk_result.post_hoc_etas[sid],
            pk_model.params.sigma,
            trans=pk_model.trans,
        )
        times = indiv.subject_events.obs_times
        truth = {"E0": 1.5, "Emax": 12.0, "EC50": 3.0}
        response = EmaxModel().predict(truth, PDData(sid, times, np.zeros(len(times)), concs))
        response = response + np.random.default_rng(123).normal(0.0, 0.08, size=len(times))
        initial = {"E0": 0.5, "Emax": 8.0, "EC50": 1.0}

        direct = EmaxModel().fit(PDData(sid, times, response, concs), initial_params=initial)
        missing = PDData(sid, times, response, concentrations=None)
        sequential = workflow.fit_pd(missing, EmaxModel, initial_params=initial)

        assert missing.concentrations is None
        assert sequential.ofv == pytest.approx(direct.ofv, abs=1e-10)
        np.testing.assert_allclose(sequential.predicted, direct.predicted, atol=1e-10, rtol=0.0)
        assert sequential.params["E0"] == pytest.approx(direct.params["E0"], abs=1e-10)
        assert sequential.params["Emax"] == pytest.approx(direct.params["Emax"], abs=1e-10)
        assert sequential.params["EC50"] == pytest.approx(direct.params["EC50"], abs=1e-10)

    def test_fit_without_concentrations_uses_time_varying_ode_pk_predictions_on_new_grid(self):
        """Sequential extraction should honor time-varying ODE covariates on non-matching PD grids."""
        rows = [
            {
                "ID": 1,
                "TIME": 0.0,
                "AMT": 100.0,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0.0,
                "SS": 0,
                "WT": 70.0,
            },
            {
                "ID": 1,
                "TIME": 1.0,
                "AMT": 0.0,
                "DV": 0.0,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0.0,
                "SS": 0,
                "WT": 70.0,
            },
            {
                "ID": 1,
                "TIME": 3.0,
                "AMT": 0.0,
                "DV": np.nan,
                "EVID": 2,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0.0,
                "SS": 0,
                "WT": 140.0,
            },
            {
                "ID": 1,
                "TIME": 5.0,
                "AMT": 0.0,
                "DV": 0.0,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0.0,
                "SS": 0,
                "WT": 140.0,
            },
            {
                "ID": 1,
                "TIME": 9.0,
                "AMT": 0.0,
                "DV": 0.0,
                "EVID": 0,
                "MDV": 0,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0.0,
                "SS": 0,
                "WT": 140.0,
            },
        ]
        dataset = NONMEMDataset.from_dataframe(pd.DataFrame(rows))
        params = ParameterSet.from_specs(
            [
                ThetaSpec(init=0.1, lower=0.01, upper=1.0),
                ThetaSpec(init=10.0, lower=1.0, upper=50.0),
            ],
            [],
            [SigmaSpec(block_size=1, values=[0.01])],
        )

        def pk_callable(theta, eta, t=0.0, covariates=None):
            covariates = covariates or {}
            wt = float(covariates.get("WT", 70.0))
            return {"K": float(theta[0]) * (wt / 70.0), "V": float(theta[1])}

        def des_callable(t, a, pk_params, theta, eta):
            return [-pk_params["K"] * a[0]]

        pk_model = PopulationModel(
            dataset=dataset,
            pk_subroutine=ADVAN6(n_compartments=1, rtol=1e-8, atol=1e-10),
            params=params,
            pk_callable=pk_callable,
            des_callable=des_callable,
            trans=1,
            advan=6,
            covariate_columns=["WT"],
        )

        class MockPKResult:
            theta_final = params.theta.copy()
            sigma_final = params.sigma.copy()
            post_hoc_etas = {1: np.array([], dtype=float)}

        workflow = SequentialPKPDWorkflow(MockPKResult(), pk_model)
        individual = pk_model.individual_model(1)
        pd_times = np.array([2.0, 4.0, 8.0])
        theta = MockPKResult.theta_final
        eta = MockPKResult.post_hoc_etas[1]
        base_params = pk_callable(
            list(theta), list(eta), covariates=individual.subject_events.covariate_at(0.0)
        )
        explicit = individual.pk_subroutine.solve(
            base_params,
            individual.subject_events.dose_events,
            pd_times,
            pk_callable=None,
            des_callable=individual.des_callable,
            covariate_fn=lambda t: pk_callable(
                list(theta),
                list(eta),
                t=t,
                covariates=individual.subject_events.covariate_at(t),
            ),
            covariate_change_times=individual.subject_events.covariate_change_times(),
        ).ipred
        constant = individual.pk_subroutine.solve(
            base_params,
            individual.subject_events.dose_events,
            pd_times,
            pk_callable=None,
            des_callable=individual.des_callable,
        ).ipred

        assert explicit[1] < constant[1]
        assert explicit[2] < constant[2]

        truth = {"E0": 2.0, "slope": 1.25}
        response = LinearPDModel().predict(
            truth, PDData(1, pd_times, np.zeros(len(pd_times)), explicit)
        )
        response = response + np.random.default_rng(321).normal(0.0, 0.01, size=len(pd_times))
        initial = {"E0": 1.0, "slope": 0.5}

        missing = PDData(1, pd_times, response, concentrations=None)
        extracted = workflow._extract_pk_concentrations(missing)
        direct = LinearPDModel().fit(
            PDData(1, pd_times, response, explicit), initial_params=initial
        )
        sequential = workflow.fit_pd(missing, LinearPDModel, initial_params=initial)

        np.testing.assert_allclose(extracted, explicit, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(sequential.predicted, direct.predicted, atol=1e-10, rtol=0.0)
        assert sequential.ofv == pytest.approx(direct.ofv, abs=1e-10)
        assert sequential.params["E0"] == pytest.approx(direct.params["E0"], abs=1e-10)
        assert sequential.params["slope"] == pytest.approx(direct.params["slope"], abs=1e-10)
