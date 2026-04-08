from __future__ import annotations

import io
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation import get_estimation_method
from openpkpd.estimation.fo import FOMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.utils.constants import LOG2PI


class _DummySubjectEvents:
    def __init__(self, dv: float) -> None:
        self.obs_dv = np.array([dv], dtype=float)

    def observation_mask(self) -> np.ndarray:
        return np.array([True])


class _LinearPredictionIndividualModel:
    def __init__(self, dv: float, slope: float = 1.5, resid_var: float = 0.1) -> None:
        self.subject_events = _DummySubjectEvents(dv)
        self.slope = slope
        self.resid_var = resid_var

    def evaluate_observation_model(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        structural = np.array([99.0])
        pred = np.array([float(theta[0] + self.slope * eta[0])])
        var = np.array([self.resid_var])
        return structural, structural, structural, pred, var


class _NativeJacobianLinearPredictionIndividualModel(_LinearPredictionIndividualModel):
    def __init__(self, dv: float, slope: float = 1.5, resid_var: float = 0.1) -> None:
        super().__init__(dv, slope=slope, resid_var=resid_var)
        self.native_jacobian_calls = 0

    def supports_prediction_eta_jacobian(self, trans: int = 2) -> bool:
        return True

    def prediction_eta_jacobian(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        sigma: np.ndarray,
        trans: int = 2,
    ) -> np.ndarray:
        self.native_jacobian_calls += 1
        return np.array([[self.slope]], dtype=float)


class _LinearPredictionPopulationModel:
    trans = 2

    def __init__(self, dv_by_subject: dict[int, float]) -> None:
        self._dv_by_subject = dict(dv_by_subject)

    def n_subjects(self) -> int:
        return len(self._dv_by_subject)

    def subject_ids(self) -> list[int]:
        return sorted(self._dv_by_subject)

    def individual_model(self, subject_id: int) -> _LinearPredictionIndividualModel:
        return _LinearPredictionIndividualModel(self._dv_by_subject[subject_id])


class _NativeJacobianPopulationModel(_LinearPredictionPopulationModel):
    def __init__(self, dv_by_subject: dict[int, float]) -> None:
        super().__init__(dv_by_subject)
        self._individuals = {
            subject_id: _NativeJacobianLinearPredictionIndividualModel(dv)
            for subject_id, dv in dv_by_subject.items()
        }

    def individual_model(self, subject_id: int) -> _NativeJacobianLinearPredictionIndividualModel:
        return self._individuals[subject_id]


class _ConstantPrior:
    def __init__(self, penalty_value: float) -> None:
        self.penalty_value = penalty_value

    def penalty(self, theta: np.ndarray, omega: np.ndarray) -> float:
        return self.penalty_value


def _make_params(theta_init: float = 1.2) -> ParameterSet:
    return ParameterSet.from_specs(
        [ThetaSpec(init=theta_init, lower=0.1, upper=5.0)],
        [OmegaSpec(block_size=1, values=[0.2], fixed=True)],
        [SigmaSpec(block_size=1, values=[0.1], fixed=True)],
    )


def test_fo_individual_ofv_matches_closed_form_and_uses_observation_prediction() -> None:
    method = FOMethod(maxeval=1)
    params = _make_params(theta_init=1.2)
    pop_model = _LinearPredictionPopulationModel({1: 1.0})

    ofv = method._fo_ofv_individual(pop_model, params, subj_id=1, eta_zero=np.zeros(1))

    expected_var = (1.5**2) * 0.2 + 0.1
    expected = LOG2PI + np.log(expected_var) + (1.0 - 1.2) ** 2 / expected_var
    assert ofv == pytest.approx(expected, abs=1e-8)


def test_compute_fo_ofv_adds_prior_penalty() -> None:
    method = FOMethod(maxeval=1)
    params = _make_params(theta_init=1.2)
    pop_model = _LinearPredictionPopulationModel({1: 1.0})
    pop_model.prior = _ConstantPrior(7.5)

    ofv = method._compute_fo_ofv(pop_model, params)

    base = method._fo_ofv_individual(pop_model, params, subj_id=1, eta_zero=np.zeros(1))
    assert ofv == pytest.approx(base + 7.5, abs=1e-8)


def test_fo_individual_uses_native_prediction_eta_jacobian_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = FOMethod(maxeval=1)
    params = _make_params(theta_init=1.2)
    pop_model = _NativeJacobianPopulationModel({1: 1.0})

    def _unexpected_jacobian(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("finite-difference jacobian should not be used")

    monkeypatch.setattr("openpkpd.estimation.fo.jacobian", _unexpected_jacobian)

    ofv = method._fo_ofv_individual(pop_model, params, subj_id=1, eta_zero=np.zeros(1))

    expected_var = (1.5**2) * 0.2 + 0.1
    expected = LOG2PI + np.log(expected_var) + (1.0 - 1.2) ** 2 / expected_var
    assert ofv == pytest.approx(expected, abs=1e-8)
    assert pop_model.individual_model(1).native_jacobian_calls == 1


def test_fo_individual_reuses_base_prediction_for_fd_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = FOMethod(maxeval=1)
    params = _make_params(theta_init=1.2)
    pop_model = _LinearPredictionPopulationModel({1: 1.0})
    captured: dict[str, object] = {}

    def _tracking_jacobian(*args: object, **kwargs: object) -> np.ndarray:
        captured["f0"] = kwargs.get("f0")
        captured["method"] = kwargs.get("method")
        return np.array([[1.5]], dtype=float)

    monkeypatch.setattr("openpkpd.estimation.fo.jacobian", _tracking_jacobian)

    method._fo_ofv_individual(pop_model, params, subj_id=1, eta_zero=np.zeros(1))

    np.testing.assert_allclose(np.asarray(captured["f0"]), np.array([1.2]))
    assert captured["method"] == "forward"


def test_estimate_returns_zero_post_hoc_etas_for_all_subjects() -> None:
    method = FOMethod(maxeval=25, print_interval=999)
    params = _make_params(theta_init=0.8)
    pop_model = _LinearPredictionPopulationModel({1: 1.0, 2: 1.4})
    pop_model.dataset = SimpleNamespace(n_observations=lambda: 2)

    result = method.estimate(pop_model, params)

    assert result.method == "FO"
    assert result.converged
    assert np.isfinite(result.ofv)
    assert result.n_observations == 2
    assert result.n_subjects == 2
    assert result.n_parameters == 1
    assert np.isfinite(result.bic)
    assert set(result.post_hoc_etas) == {1, 2}
    for eta in result.post_hoc_etas.values():
        np.testing.assert_allclose(eta, np.zeros(1), atol=0.0)


def test_get_estimation_method_fo_kwargs_are_forwarded() -> None:
    method = get_estimation_method("FO", maxeval=123, print_interval=17)

    assert isinstance(method, FOMethod)
    assert method.maxeval == 123
    assert method.print_interval == 17


@pytest.mark.parametrize(
    "method",
    ["FO", "FOCE", "FOCEI", "LAPLACIAN", "SAEM", "IMP", "IMPMAP", "BAYES", "NONPARAMETRIC"],
)
def test_get_estimation_method_strips_gui_only_options(method: str) -> None:
    """base_method and blq_method are GUI-level options stored in the shared options
    dict regardless of which estimation method is selected.  They must never reach
    any EstimationMethod constructor — previously FOCEMethod raised TypeError when
    base_method leaked through.
    """
    from openpkpd.estimation.base import EstimationMethod

    # These two keys are always present in the GUI options dict; every branch
    # in get_estimation_method must pop them (or pass them only where accepted).
    inst = get_estimation_method(method, base_method="FOCE", blq_method="M1")

    assert isinstance(inst, EstimationMethod)


def test_fo_fit_penalizes_invalid_transform_proposals_without_transform_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ds = NONMEMDataset.from_dataframe(
        pd.read_csv(
            io.StringIO(
                """ID,TIME,AMT,DV,EVID,MDV
1,0,4.02,0,1,1
1,1.02,0,7.91,0,0
1,3.5,0,8.33,0,0
1,7.03,0,6.08,0,0
1,12.05,0,4.55,0,0
1,24.37,0,1.25,0,0
2,0,4.4,0,1,1
2,1.07,0,4.71,0,0
2,3.5,0,9.02,0,0
2,7.02,0,5.68,0,0
2,12.1,0,3.01,0,0
2,25.0,0,0.9,0,0
"""
            )
        )
    )
    built = (
        ModelBuilder()
        .problem("Theophylline FO transform guard")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk(
            """
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
"""
        )
        .error("Y = F*(1 + EPS(1))")
        .theta([1.5, 0.08, 30.0])
        .omega([0.3, 0.2, 0.2])
        .sigma(0.1)
        .estimation(method="FO", maxeval=200)
        .build()
    )

    with caplog.at_level(logging.WARNING):
        result = built.fit()

    assert np.isfinite(result.ofv)
    assert "failed at pk_param transform" not in caplog.text
