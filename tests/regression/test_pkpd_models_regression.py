"""Regression tests for deterministic joint PK/PD model fits."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.regression.pd_model_helpers import fit_joint_pkpd_case

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference_runs")


def load_reference(name: str) -> dict:
    path = os.path.join(REFERENCE_DIR, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def joint_emax_case():
    return fit_joint_pkpd_case("joint_emax_pkpd")


@pytest.mark.regression
def test_joint_emax_pkpd_matches_reference(joint_emax_case):
    ref = load_reference("joint_emax_pkpd")
    case, _built, result, predicted, pk_f = joint_emax_case

    assert result.converged
    assert bool(result.converged) is ref["converged"]
    assert result.aic == pytest.approx(result.ofv + 2.0 * result.n_parameters, abs=1e-10)

    np.testing.assert_allclose(
        [float(result.theta_final[idx]) for idx, _ in enumerate(case["param_names"])],
        [ref["params"][name] for name in case["param_names"]],
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        [float(result.omega_final[idx, idx]) for idx, _ in enumerate(case["omega_param_names"])],
        [ref["omega_diag"][name] for name in case["omega_param_names"]],
        atol=1e-6,
        rtol=1e-6,
    )
    assert result.ofv == pytest.approx(ref["ofv"], abs=1e-6)
    assert result.aic == pytest.approx(ref["aic"], abs=1e-6)
    np.testing.assert_allclose(predicted, ref["predicted"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(pk_f, ref["pk_f"], atol=1e-6, rtol=1e-6)


@pytest.mark.regression
def test_joint_emax_pkpd_recovers_truth_within_tolerance(joint_emax_case):
    case, _built, result, _predicted, _pk_f = joint_emax_case
    truth = case["true_params"]

    rel_errors = {
        name: abs(float(result.theta_final[idx]) - truth[name]) / max(abs(truth[name]), 1e-12)
        for idx, name in enumerate(case["param_names"])
    }
    assert rel_errors["K"] < 0.10
    assert rel_errors["V"] < 0.10
    assert rel_errors["E0"] < 0.20
    assert rel_errors["EMAX"] < 0.05
    assert rel_errors["EC50"] < 0.12
    assert rel_errors["W"] < 0.30

    omega_truth = case["true_omega_diag"]
    omega_errors = {
        name: abs(float(result.omega_final[idx, idx]) - omega_truth[name]) / omega_truth[name]
        for idx, name in enumerate(case["omega_param_names"])
    }
    assert omega_errors["K"] < 0.60
    assert omega_errors["V"] < 0.35


@pytest.mark.regression
def test_joint_emax_pkpd_predictions_are_reasonable(joint_emax_case):
    _case, built, result, predicted, pk_f = joint_emax_case
    obs_rows = built.population_model.dataset.observation_rows().copy()
    dose_rows = built.population_model.dataset.df[built.population_model.dataset.df["EVID"] == 1][
        ["ID", "AMT"]
    ]

    assert np.all(np.isfinite(predicted))
    assert np.all(np.isfinite(pk_f))
    assert np.all(pk_f > 0.0)

    rmse = float(np.sqrt(np.mean((obs_rows["DV"].to_numpy(float) - predicted) ** 2)))
    assert rmse < 1.5

    pk_by_subject = pk_f.reshape(built.population_model.n_subjects(), -1)
    assert np.all(np.diff(pk_by_subject, axis=1) < 0.0)

    merged = obs_rows.merge(dose_rows.rename(columns={"AMT": "DOSE"}), on="ID", how="left")
    merged["PRED"] = predicted
    early_means = (
        merged[merged["TIME"].isin([1.0, 2.0, 4.0])].groupby("DOSE")["PRED"].mean().to_dict()
    )
    assert early_means[50.0] < early_means[100.0] < early_means[200.0]
    assert float(np.mean(predicted[obs_rows["TIME"] <= 4.0])) > float(
        np.mean(predicted[obs_rows["TIME"] >= 12.0])
    )
    assert result.ofv_history[-1] < result.ofv_history[0]
