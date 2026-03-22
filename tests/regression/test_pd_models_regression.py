"""Regression tests for deterministic PD model fits."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from openpkpd.models.pkpd import EmaxModel
from tests.regression.pd_model_helpers import (
    fit_direct_pd_case,
    fit_indirect_pd_case,
    fit_mechanistic_pd_case,
    fit_population_pd_case,
    fit_sequential_from_pk_pd_case,
    fit_sequential_multi_subject_from_pk_pd_case,
    fit_sequential_pd_case,
    fit_turnover_pd_case,
)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference_runs")


def load_reference(name: str) -> dict:
    path = os.path.join(REFERENCE_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.mark.regression
@pytest.mark.parametrize("case_name", ["emax_direct_pd", "hill_direct_pd"])
def test_direct_pd_fit_matches_reference(case_name: str):
    ref = load_reference(case_name)
    case, data, result = fit_direct_pd_case(case_name)
    param_names = list(case["true_params"])

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))
    assert np.all(np.diff(result.predicted) <= 1e-10)

    rmse = float(np.sqrt(np.mean((data.response - result.predicted) ** 2)))
    assert rmse < 0.11

    rel_errors = np.array(
        [
            abs(float(result.params[name]) - case["true_params"][name])
            / max(abs(case["true_params"][name]), 1e-12)
            for name in param_names
        ]
    )
    assert np.median(rel_errors) < 0.05
    assert np.max(rel_errors) < 0.10

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_effect_compartment_fit_matches_reference():
    ref = load_reference("effect_compartment_pd")
    case, data, result = fit_mechanistic_pd_case("effect_compartment_pd")
    param_names = list(case["true_params"])

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))

    peak_idx = int(np.argmax(result.predicted))
    assert peak_idx > 0
    assert peak_idx < len(result.predicted) - 1
    assert result.predicted[0] == pytest.approx(0.0, abs=1e-12)
    assert result.predicted[-1] < result.predicted[peak_idx]

    rmse = float(np.sqrt(np.mean((data.response - result.predicted) ** 2)))
    assert rmse < 0.18

    rel_errors = np.array(
        [
            abs(float(result.params[name]) - case["true_params"][name])
            / max(abs(case["true_params"][name]), 1e-12)
            for name in param_names
        ]
    )
    assert np.median(rel_errors) < 0.10
    assert np.max(rel_errors) < 0.20

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_placebo_response_fit_matches_reference():
    ref = load_reference("placebo_response_pd")
    case, data, result = fit_mechanistic_pd_case("placebo_response_pd")
    param_names = list(case["true_params"])

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))
    assert result.predicted[0] == pytest.approx(float(result.params["E0"]), abs=1e-10)
    assert np.all(np.diff(result.predicted) <= 1e-10)
    assert result.predicted[-1] < result.predicted[0] - 20.0

    rmse = float(np.sqrt(np.mean((data.response - result.predicted) ** 2)))
    assert rmse < 1.2

    rel_errors = np.array(
        [
            abs(float(result.params[name]) - case["true_params"][name])
            / max(abs(case["true_params"][name]), 1e-12)
            for name in param_names
        ]
    )
    assert np.median(rel_errors) < 0.12
    assert np.max(rel_errors) < 0.30

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_tumor_growth_inhibition_fit_matches_reference():
    ref = load_reference("tumor_growth_inhibition_pd")
    case, data, result = fit_mechanistic_pd_case("tumor_growth_inhibition_pd")
    param_names = list(case["true_params"])

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))
    assert np.all(result.predicted > 0.0)
    assert result.predicted[0] == pytest.approx(float(result.params["X0"]), abs=1e-10)
    assert result.predicted[-1] > result.predicted[0]

    no_drug_model = case["model_cls"]()
    no_drug_data = data.__class__(
        subject_id=data.subject_id,
        times=data.times,
        response=data.response,
        concentrations=np.zeros_like(data.times),
    )
    no_drug_pred = no_drug_model.predict(result.params, no_drug_data)
    assert result.predicted[-1] < no_drug_pred[-1]

    rmse = float(np.sqrt(np.mean((data.response - result.predicted) ** 2)))
    assert rmse < 2.0

    rel_errors = np.array(
        [
            abs(float(result.params[name]) - case["true_params"][name])
            / max(abs(case["true_params"][name]), 1e-12)
            for name in param_names
        ]
    )
    assert np.median(rel_errors) < 0.01
    assert np.max(rel_errors) < 0.05

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_population_emax_fit_matches_reference():
    ref = load_reference("population_emax_pd")
    case, subjects, result = fit_population_pd_case("population_emax_pd")
    theta_names = list(case["true_theta"])

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(
        result.ofv + 2.0 * (len(theta_names) + result.omega.size + 1), abs=1e-10
    )
    assert set(result.post_hoc_etas) == {data.subject_id for data in subjects}
    assert all(eta.shape == (1,) for eta in result.post_hoc_etas.values())
    assert np.isfinite(result.omega[0, 0])
    assert result.omega[0, 0] > 0.0
    assert np.isfinite(result.sigma2)
    assert result.sigma2 > 0.0

    theta_errors = np.array(
        [
            abs(float(result.theta[name]) - case["true_theta"][name])
            / max(abs(case["true_theta"][name]), 1e-12)
            for name in theta_names
        ]
    )
    assert np.median(theta_errors) < 0.06
    assert np.max(theta_errors) < 0.10
    assert abs(float(result.sigma2) - case["true_sigma2"]) / case["true_sigma2"] < 0.35

    eta_values = np.array(
        [float(result.post_hoc_etas[sid][0]) for sid in sorted(result.post_hoc_etas)]
    )
    assert np.all(np.isfinite(eta_values))
    assert abs(float(np.mean(eta_values))) < 0.05

    np.testing.assert_allclose(
        [float(result.theta[name]) for name in theta_names],
        [ref["theta"][name] for name in theta_names],
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(result.omega, ref["omega"], atol=1e-6, rtol=1e-6)
    assert result.sigma2 == pytest.approx(ref["sigma2"], abs=1e-6)
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(
        eta_values,
        [ref["post_hoc_etas"][str(sid)][0] for sid in sorted(result.post_hoc_etas)],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.regression
def test_indirect_response_fit_matches_reference():
    ref = load_reference("indirect_response_pd")
    case, data, result = fit_indirect_pd_case("indirect_response_pd")
    param_names = list(case["true_params"])
    baseline = case["true_params"]["Kin"] / case["true_params"]["Kout"]

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))
    assert result.predicted[0] == pytest.approx(baseline, abs=1e-12)
    assert np.all(np.diff(result.predicted) >= -1e-10)
    assert result.predicted[-1] > baseline + 4.0

    rmse = float(np.sqrt(np.mean((data.response - result.predicted) ** 2)))
    assert rmse < 0.06

    rel_errors = np.array(
        [
            abs(float(result.params[name]) - case["true_params"][name])
            / max(abs(case["true_params"][name]), 1e-12)
            for name in param_names
        ]
    )
    assert np.median(rel_errors) < 0.03
    assert np.max(rel_errors) < 0.25

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_sequential_emax_workflow_matches_reference_and_direct_fit():
    ref = load_reference("sequential_emax_pd")
    case, data, result = fit_sequential_pd_case("sequential_emax_pd")
    param_names = list(case["true_params"])
    direct = EmaxModel().fit(data, initial_params=case["initial_params"])

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))
    assert np.all(np.diff(result.predicted) <= 1e-10)

    rmse = float(np.sqrt(np.mean((data.response - result.predicted) ** 2)))
    assert rmse < 0.11

    rel_errors = np.array(
        [
            abs(float(result.params[name]) - case["true_params"][name])
            / max(abs(case["true_params"][name]), 1e-12)
            for name in param_names
        ]
    )
    assert np.median(rel_errors) < 0.05
    assert np.max(rel_errors) < 0.10

    np.testing.assert_allclose(result.predicted, direct.predicted, atol=1e-10, rtol=0.0)
    assert result.ofv == pytest.approx(direct.ofv, abs=1e-10)
    assert result.aic == pytest.approx(direct.aic, abs=1e-10)
    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [float(direct.params[name]) for name in param_names],
        atol=1e-10,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_turnover_fit_matches_reference():
    ref = load_reference("turnover_pd")
    case, data, result = fit_turnover_pd_case("turnover_pd")
    param_names = list(case["true_params"])
    baseline = case["true_params"]["Kin"] / case["true_params"]["Kout"]

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))
    assert result.predicted[0] == pytest.approx(baseline, abs=1e-12)
    peak_idx = int(np.argmax(result.predicted))
    assert 0 < peak_idx < len(result.predicted) - 1
    assert result.predicted[-1] < result.predicted[peak_idx]

    rmse = float(np.sqrt(np.mean((data.response - result.predicted) ** 2)))
    assert rmse < 0.35

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_sequential_emax_from_pk_matches_reference_and_explicit_concentrations():
    ref = load_reference("sequential_emax_from_pk_pd")
    case, explicit_data, result = fit_sequential_from_pk_pd_case("sequential_emax_from_pk_pd")
    param_names = list(case["true_params"])
    direct = EmaxModel().fit(explicit_data, initial_params=case["initial_params"])

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
    assert np.all(np.isfinite(result.predicted))
    peak_idx = int(np.argmax(result.predicted))
    assert 0 < peak_idx < len(result.predicted) - 1
    assert np.all(np.diff(result.predicted[: peak_idx + 1]) >= -1e-10)
    assert np.all(np.diff(result.predicted[peak_idx:]) <= 1e-10)

    rmse = float(np.sqrt(np.mean((explicit_data.response - result.predicted) ** 2)))
    assert rmse < 0.10

    np.testing.assert_allclose(result.predicted, direct.predicted, atol=1e-10, rtol=0.0)
    assert result.ofv == pytest.approx(direct.ofv, abs=1e-10)
    assert result.aic == pytest.approx(direct.aic, abs=1e-10)
    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [float(direct.params[name]) for name in param_names],
        atol=1e-10,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        [float(result.params[name]) for name in param_names],
        [ref["params"][name] for name in param_names],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(result.predicted, ref["predicted"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_sequential_multi_subject_from_pk_matches_reference_and_explicit_concentrations():
    ref = load_reference("sequential_multi_subject_from_pk_pd")
    case, explicit_by_sid, results = fit_sequential_multi_subject_from_pk_pd_case(
        "sequential_multi_subject_from_pk_pd"
    )
    param_names = list(case["true_params"])
    subject_ids = sorted(results)

    assert subject_ids == sorted(explicit_by_sid)
    assert subject_ids == [int(sid) for sid in sorted(ref["subjects"], key=int)]
    assert all(abs(case["post_hoc_etas"][sid][0]) > 0.0 for sid in subject_ids)

    conc_profiles = [explicit_by_sid[sid].concentrations for sid in subject_ids]
    assert any(
        not np.allclose(conc_profiles[0], conc_profiles[idx], atol=1e-10, rtol=0.0)
        for idx in range(1, len(conc_profiles))
    )
    np.testing.assert_allclose(
        [case["post_hoc_etas"][sid][0] for sid in subject_ids],
        [ref["post_hoc_etas"][str(sid)][0] for sid in subject_ids],
        atol=1e-12,
        rtol=0.0,
    )

    for sid in subject_ids:
        explicit_data = explicit_by_sid[sid]
        result = results[sid]
        direct = EmaxModel().fit(explicit_data, initial_params=case["initial_params"])
        subject_ref = ref["subjects"][str(sid)]

        assert result.converged
        assert bool(result.converged) is subject_ref["converged"]
        assert result.aic == pytest.approx(result.ofv + 2.0 * len(param_names), abs=1e-10)
        assert np.all(np.isfinite(result.predicted))
        peak_idx = int(np.argmax(result.predicted))
        assert 0 < peak_idx < len(result.predicted) - 1
        assert np.all(np.diff(result.predicted[: peak_idx + 1]) >= -1e-10)
        assert np.all(np.diff(result.predicted[peak_idx:]) <= 1e-10)

        rmse = float(np.sqrt(np.mean((explicit_data.response - result.predicted) ** 2)))
        assert rmse < 0.10

        np.testing.assert_allclose(result.predicted, direct.predicted, atol=1e-10, rtol=0.0)
        assert result.ofv == pytest.approx(direct.ofv, abs=1e-10)
        assert result.aic == pytest.approx(direct.aic, abs=1e-10)
        np.testing.assert_allclose(
            [float(result.params[name]) for name in param_names],
            [float(direct.params[name]) for name in param_names],
            atol=1e-10,
            rtol=0.0,
        )

        np.testing.assert_allclose(
            [float(result.params[name]) for name in param_names],
            [subject_ref["params"][name] for name in param_names],
            atol=1e-6,
            rtol=1e-6,
        )
        assert result.ofv == pytest.approx(subject_ref["ofv"], abs=1e-6)
        assert result.aic == pytest.approx(subject_ref["aic"], abs=1e-6)
        np.testing.assert_allclose(result.predicted, subject_ref["predicted"], atol=1e-6, rtol=1e-6)
