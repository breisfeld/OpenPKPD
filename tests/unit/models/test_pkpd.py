"""Tests for PD and PK/PD models."""

from __future__ import annotations

import numpy as np
import pytest

from openpkpd.models.pkpd import (
    EffectCompartmentModel,
    EmaxModel,
    HillModel,
    IndirectResponseModel,
    InhibEmaxModel,
    LinearPDModel,
    PDData,
    PDResult,
    PlaceboResponseModel,
    TumorGrowthInhibitionModel,
    TurnoverModel,
)


def make_conc_times(n=20, t_max=24.0):
    """Simple PK profile: 1-cmt IV."""
    times = np.linspace(0.1, t_max, n)
    concs = 10.0 * np.exp(-0.3 * times)
    return times, concs


def _relative_error(estimate: float, truth: float) -> float:
    scale = max(abs(truth), 1e-8)
    return abs(estimate - truth) / scale


def _make_direct_pd_dataset(model, true_params, *, seed=42, sigma=0.1):
    times = np.linspace(0.5, 24.0, 24)
    concs = 12.0 * np.exp(-0.18 * times)
    base = PDData(1, times, np.zeros_like(times), concs)
    pred = model.predict(true_params, base)
    obs = pred + np.random.default_rng(seed).normal(0.0, sigma, size=len(times))
    return PDData(1, times, obs, concs)


class TestPDData:
    def test_creation(self):
        data = PDData(subject_id=1, times=np.array([1.0, 2.0]), response=np.array([5.0, 4.0]))
        assert data.subject_id == 1
        assert len(data.times) == 2

    def test_concentrations_optional(self):
        data = PDData(subject_id=1, times=np.array([1.0]), response=np.array([5.0]))
        assert data.concentrations is None


class TestDirectPDModels:
    def test_linear_model_matches_affine_formula(self):
        times = np.array([0.5, 1.0, 2.0])
        concs = np.array([1.0, 3.0, 5.0])
        data = PDData(1, times, np.zeros(len(times)), concs)
        pred = LinearPDModel().predict({"E0": 2.0, "slope": 1.5}, data)
        np.testing.assert_allclose(pred, np.array([3.5, 6.5, 9.5]), atol=1e-10)

    def test_hill_with_gamma_one_matches_emax(self):
        times = np.array([0.5, 1.0, 2.0, 4.0])
        concs = np.array([0.0, 1.0, 3.0, 10.0])
        data = PDData(1, times, np.zeros(len(times)), concs)
        params = {"E0": 1.0, "Emax": 8.0, "EC50": 2.5}
        emax_pred = EmaxModel().predict(params, data)
        hill_pred = HillModel().predict({**params, "gamma": 1.0}, data)
        np.testing.assert_allclose(hill_pred, emax_pred, atol=1e-10)

    def test_inhibitory_emax_is_bounded_and_monotone(self):
        times = np.array([0.5, 1.0, 2.0, 4.0])
        concs = np.array([0.0, 1.0, 3.0, 10.0])
        data = PDData(1, times, np.zeros(len(times)), concs)
        params = {"E0": 12.0, "Imax": 0.7, "IC50": 2.0, "gamma": 1.5}
        pred = InhibEmaxModel().predict(params, data)
        assert np.all(np.diff(pred) <= 1e-12)
        assert np.all(pred <= params["E0"] + 1e-12)
        assert np.all(pred >= params["E0"] * (1.0 - params["Imax"]) - 1e-12)

    @pytest.mark.parametrize(
        ("model", "true_params", "initial_params"),
        [
            (
                LinearPDModel(),
                {"E0": 2.0, "slope": 1.5},
                {"E0": 0.5, "slope": 0.5},
            ),
            (
                EmaxModel(),
                {"E0": 1.0, "Emax": 8.0, "EC50": 3.0},
                {"E0": 0.0, "Emax": 5.0, "EC50": 1.0},
            ),
            (
                HillModel(),
                {"E0": 1.0, "Emax": 10.0, "EC50": 4.0, "gamma": 1.7},
                {"E0": 0.0, "Emax": 8.0, "EC50": 2.0, "gamma": 1.0},
            ),
            (
                InhibEmaxModel(),
                {"E0": 12.0, "Imax": 0.7, "IC50": 2.5, "gamma": 1.4},
                {"E0": 10.0, "Imax": 0.5, "IC50": 1.0, "gamma": 1.0},
            ),
        ],
    )
    def test_direct_pd_fit_recovers_parameters_reasonably(
        self,
        model,
        true_params,
        initial_params,
    ):
        data = _make_direct_pd_dataset(model, true_params)
        result = model.fit(data, initial_params=initial_params, sigma2=0.01)
        rel_errors = np.array(
            [
                _relative_error(float(result.params[name]), truth)
                for name, truth in true_params.items()
            ]
        )
        assert result.converged
        assert np.median(rel_errors) < 0.12
        assert np.max(rel_errors) < 0.35


class TestIndirectResponseModel:
    def test_all_types_predict(self):
        """All 4 IDR types produce predictions."""
        times, concs = make_conc_times()
        for idr_type in (1, 2, 3, 4):
            model = IndirectResponseModel(idr_type=idr_type)
            if idr_type in (1, 2):
                params = {"Kin": 5.0, "Kout": 0.5, "EC50": 3.0, "Emax": 1.0}
            else:
                params = {"Kin": 5.0, "Kout": 0.5, "IC50": 3.0, "Imax": 0.8}
            data = PDData(1, times, np.zeros(len(times)), concs)
            pred = model.predict(params, data)
            assert pred.shape == times.shape, f"Type {idr_type} failed"
            assert np.all(np.isfinite(pred)), f"Type {idr_type} has NaNs"

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            IndirectResponseModel(idr_type=5)

    def test_fit_returns_pdresult(self):
        """Fitting returns a PDResult."""
        times, concs = make_conc_times(n=15, t_max=12.0)
        rng = np.random.default_rng(42)
        response = 10.0 * np.ones(len(times)) + rng.normal(0, 0.5, len(times))
        data = PDData(1, times, response, concs)
        model = IndirectResponseModel(idr_type=1)
        result = model.fit(
            data, initial_params={"Kin": 10.0, "Kout": 1.0, "EC50": 2.0, "Emax": 0.5}
        )
        assert isinstance(result, PDResult)
        assert np.isfinite(result.ofv)
        assert np.isfinite(result.aic)

    @pytest.mark.parametrize(
        ("idr_type", "params"),
        [
            (1, {"Kin": 5.0, "Kout": 0.5, "EC50": 3.0, "Emax": 1.0}),
            (2, {"Kin": 5.0, "Kout": 0.5, "EC50": 3.0, "Emax": 1.0}),
            (3, {"Kin": 5.0, "Kout": 0.5, "IC50": 3.0, "Imax": 0.8}),
            (4, {"Kin": 5.0, "Kout": 0.5, "IC50": 3.0, "Imax": 0.8}),
        ],
    )
    def test_no_drug_baseline_is_preserved_for_all_types(self, idr_type, params):
        times = np.linspace(0.0, 24.0, 49)
        baseline = params["Kin"] / params["Kout"]
        data = PDData(1, times, np.zeros(len(times)), np.zeros(len(times)), baseline=baseline)
        pred = IndirectResponseModel(idr_type=idr_type).predict(params, data)

        np.testing.assert_allclose(pred, baseline, atol=1e-10, rtol=0.0)

    @pytest.mark.parametrize(
        ("idr_type", "params", "expected_direction"),
        [
            (1, {"Kin": 5.0, "Kout": 0.5, "EC50": 2.0, "Emax": 1.0}, "up"),
            (2, {"Kin": 5.0, "Kout": 0.5, "EC50": 2.0, "Emax": 1.0}, "down"),
            (3, {"Kin": 5.0, "Kout": 0.5, "IC50": 2.0, "Imax": 0.8}, "down"),
            (4, {"Kin": 5.0, "Kout": 0.5, "IC50": 2.0, "Imax": 0.8}, "up"),
        ],
    )
    def test_constant_drug_moves_response_in_expected_direction(
        self,
        idr_type,
        params,
        expected_direction,
    ):
        times = np.linspace(0.0, 24.0, 49)
        baseline = params["Kin"] / params["Kout"]
        data = PDData(1, times, np.zeros(len(times)), np.full(len(times), 4.0), baseline=baseline)
        pred = IndirectResponseModel(idr_type=idr_type).predict(params, data)

        if expected_direction == "up":
            assert np.all(np.diff(pred) >= -1e-6)
            assert pred[-1] > baseline + 1.0
        else:
            assert np.all(np.diff(pred) <= 1e-6)
            assert pred[-1] < baseline - 1.0


class TestEffectCompartmentModel:
    def test_zero_concentration_gives_zero_effect(self):
        times = np.linspace(0.0, 12.0, 25)
        data = PDData(1, times, np.zeros(len(times)), np.zeros(len(times)))
        params = {"Ke0": 0.5, "Emax": 100.0, "EC50": 5.0, "n": 1.0}
        pred = EffectCompartmentModel().predict(params, data)

        np.testing.assert_allclose(pred, 0.0, atol=1e-12, rtol=0.0)

    def test_constant_concentration_matches_closed_form_effect_site_solution(self):
        times = np.linspace(0.0, 12.0, 49)
        conc = 3.0
        params = {"Ke0": 0.7, "Emax": 15.0, "EC50": 2.0, "n": 1.0}
        data = PDData(1, times, np.zeros(len(times)), np.full(len(times), conc))
        pred = EffectCompartmentModel().predict(params, data)

        ce = conc * (1.0 - np.exp(-params["Ke0"] * times))
        expected = params["Emax"] * ce / (params["EC50"] + ce)

        np.testing.assert_allclose(pred, expected, atol=1e-6, rtol=1e-6)
        assert np.all(pred >= 0)
        assert np.all(pred <= params["Emax"] + 1e-9)


class TestTurnoverModel:
    def test_no_drug_preserves_baseline(self):
        times = np.linspace(0.0, 24.0, 49)
        params = {
            "Kin": 10.0,
            "Kout": 1.0,
            "EC50_in": 5.0,
            "Emax_in": 0.5,
            "EC50_out": 5.0,
            "Emax_out": 0.5,
        }
        baseline = params["Kin"] / params["Kout"]
        data = PDData(1, times, np.zeros(len(times)), np.zeros(len(times)), baseline=baseline)
        pred = TurnoverModel().predict(params, data)

        np.testing.assert_allclose(pred, baseline, atol=1e-10, rtol=0.0)

    def test_constant_concentration_matches_closed_form_turnover_solution(self):
        times = np.linspace(0.0, 24.0, 97)
        conc = 4.0
        params = {
            "Kin": 10.0,
            "Kout": 1.0,
            "EC50_in": 2.0,
            "Emax_in": 0.5,
            "EC50_out": 5.0,
            "Emax_out": 0.3,
        }
        baseline = params["Kin"] / params["Kout"]
        data = PDData(1, times, np.zeros(len(times)), np.full(len(times), conc), baseline=baseline)
        pred = TurnoverModel().predict(params, data)

        s_in = params["Emax_in"] * conc / (params["EC50_in"] + conc)
        s_out = params["Emax_out"] * conc / (params["EC50_out"] + conc)
        r_ss = params["Kin"] * (1.0 + s_in) / (params["Kout"] * (1.0 + s_out))
        expected = r_ss + (baseline - r_ss) * np.exp(-params["Kout"] * (1.0 + s_out) * times)

        np.testing.assert_allclose(pred, expected, atol=1e-5, rtol=1e-6)


class TestPlaceboResponseModel:
    def test_matches_closed_form_and_long_time_limit(self):
        times = np.array([0.0, 1.0, 5.0, 100.0])
        data = PDData(1, times, np.zeros(len(times)))
        model = PlaceboResponseModel()
        params = {"E0": 10.0, "kdeg": 0.1, "Eplacebo": 3.0, "kpl": 0.05}
        pred = model.predict(params, data)

        expected = params["E0"] * np.exp(-params["kdeg"] * times) + params["Eplacebo"] * (
            1.0 - np.exp(-params["kpl"] * times)
        )
        np.testing.assert_allclose(pred, expected, atol=1e-12, rtol=0.0)
        assert pred[0] == pytest.approx(params["E0"])
        assert pred[-1] == pytest.approx(params["Eplacebo"], abs=0.03)

    def test_fit(self):
        """PlaceboResponseModel fits synthetic data."""
        times = np.linspace(0, 20, 20)
        true_params = {"E0": 10.0, "kdeg": 0.1, "Eplacebo": 3.0, "kpl": 0.05}
        model = PlaceboResponseModel()
        response_true = model.predict(true_params, PDData(1, times, np.zeros(len(times))))
        rng = np.random.default_rng(7)
        response_obs = response_true + rng.normal(0, 0.2, len(times))
        data = PDData(1, times, response_obs)
        result = model.fit(data, initial_params=true_params)
        assert isinstance(result, PDResult)
        assert result.converged or np.isfinite(result.ofv)


class TestTumorGrowthInhibitionModel:
    def test_k2_zero_makes_predictions_independent_of_concentration(self):
        times = np.linspace(0.1, 20.0, 40)
        params = {"lambda0": 0.2, "lambda1": 2.0, "K1": 0.1, "K2": 0.0, "psi": 20.0, "X0": 1.0}
        data_zero = PDData(1, times, np.zeros(len(times)), np.zeros(len(times)))
        data_drug = PDData(1, times, np.zeros(len(times)), 5.0 * np.exp(-0.1 * times))

        pred_zero = TumorGrowthInhibitionModel().predict(params, data_zero)
        pred_drug = TumorGrowthInhibitionModel().predict(params, data_drug)

        np.testing.assert_allclose(pred_zero, pred_drug, atol=1e-10, rtol=0.0)

    def test_no_drug_growth(self):
        """Without drug, tumor should grow."""
        times = np.linspace(0.1, 20, 20)
        concs = np.zeros(len(times))  # no drug
        data = PDData(1, times, np.zeros(len(times)), concs)
        model = TumorGrowthInhibitionModel()
        params = {"lambda0": 0.2, "lambda1": 2.0, "K1": 0.1, "K2": 0.0, "psi": 20.0, "X0": 1.0}
        pred = model.predict(params, data)
        assert pred.shape == times.shape
        assert np.all(np.isfinite(pred))
        # Tumor should grow
        assert pred[-1] > pred[0]

    def test_with_drug_inhibits_growth(self):
        """With drug, tumor should grow slower."""
        times = np.linspace(0.1, 20, 20)
        concs_no_drug = np.zeros(len(times))
        concs_with_drug = 5.0 * np.exp(-0.1 * times)

        model = TumorGrowthInhibitionModel()
        params = {"lambda0": 0.2, "lambda1": 2.0, "K1": 0.1, "K2": 0.05, "psi": 20.0, "X0": 1.0}

        data_no = PDData(1, times, np.zeros(len(times)), concs_no_drug)
        data_yes = PDData(1, times, np.zeros(len(times)), concs_with_drug)

        pred_no = model.predict(params, data_no)
        pred_yes = model.predict(params, data_yes)

        # With drug, tumor should be smaller at end
        if np.all(np.isfinite(pred_no)) and np.all(np.isfinite(pred_yes)):
            assert pred_yes[-1] <= pred_no[-1] + 1e-3
