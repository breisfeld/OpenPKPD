"""
Diagnostic coverage for the remaining warfarin FOCEI parity gap.

These tests are not trying to release-gate exact parity against nlmixr2.
Instead they pin the current diagnosis:

- OpenPKPD's own FOCEI objective prefers its current fitted basin to the frozen
  nlmixr2 reference basin on the warfarin PK-only benchmark.
- Therefore the remaining gap is not a simple "outer optimizer failed to reach
  a clearly better point under the current objective".
- The objective breakdown is internally consistent, which narrows future work
  toward objective/parity-definition investigation rather than ad hoc tolerance
  changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation import get_estimation_method
from openpkpd.estimation.foce import _compute_G_i
from openpkpd.math.matrix import repair_pd
from openpkpd.model.parameters import ParameterSet
from openpkpd.utils.constants import LOG2PI


DATA_PATH = Path("tests/external_validation/data/warfarin_pk.csv")
REF_PATH = Path("tests/external_validation/nlmixr2/reference/warfarin_pk_foce.json")


def _load_reference() -> dict:
    with REF_PATH.open() as handle:
        return json.load(handle)


def _build_warfarin_model(maxeval: int = 40):
    ds = NONMEMDataset.from_csv(str(DATA_PATH))
    return (
        ModelBuilder()
        .problem("Warfarin PK-only 1-cmt oral — FOCEI diagnostics")
        .dataset(ds)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV  = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 0.9, 20), (0.001, 0.13, 5), (0.1, 8.7, 200)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(method="FOCEI", maxeval=maxeval)
        .build()
    )


def _parameter_set_from_result(built_model, result) -> ParameterSet:
    return ParameterSet(
        theta=np.asarray(result.theta_final, dtype=float),
        omega=np.asarray(result.omega_final, dtype=float),
        sigma=np.asarray(result.sigma_final, dtype=float),
        theta_specs=list(built_model.params.theta_specs),
        omega_specs=list(built_model.params.omega_specs),
        sigma_specs=list(built_model.params.sigma_specs),
    )


def _parameter_set_from_reference(built_model, ref: dict) -> ParameterSet:
    theta = np.array(
        [
            float(ref["theta"]["KA"]),
            float(ref["theta"]["CL"]),
            float(ref["theta"]["V"]),
        ],
        dtype=float,
    )
    omega = np.diag(
        [
            float(ref["omega_diag"]["KA"]),
            float(ref["omega_diag"]["CL"]),
            float(ref["omega_diag"]["V"]),
        ]
    )
    sigma = np.array([[float(ref["sigma_prop_err_variance"])]], dtype=float)
    return ParameterSet(
        theta=theta,
        omega=omega,
        sigma=sigma,
        theta_specs=list(built_model.params.theta_specs),
        omega_specs=list(built_model.params.omega_specs),
        sigma_specs=list(built_model.params.sigma_specs),
    )


def _subject_objective_terms(population_model, params: ParameterSet, eta_hat: dict[int, np.ndarray]):
    """Replicate the FOCEI outer objective subject by subject for diagnostics."""
    omega_rep = repair_pd(params.omega)
    omega_inv = np.linalg.inv(omega_rep)
    _sign_omega, log_det_omega = np.linalg.slogdet(omega_rep)
    subject_rows: list[dict[str, float | int]] = []
    total = 0.0

    for sid in population_model.subject_ids():
        indiv = population_model.individual_model(sid)
        subj_ev = indiv.subject_events
        obs_mask = subj_ev.observation_mask()
        eta_i = eta_hat[sid]
        _, _, _, pred, var = indiv.evaluate_observation_model(
            params.theta,
            eta_i,
            params.sigma,
            trans=population_model.trans,
        )
        dv_obs = subj_ev.obs_dv[obs_mask]
        pred_obs = pred[obs_mask]
        residuals = dv_obs - pred_obs
        var_obs = var[obs_mask]
        n_obs = int(len(dv_obs))

        G = _compute_G_i(
            indiv,
            params.theta,
            eta_i,
            params.sigma,
            population_model.trans,
            obs_mask,
            pred_obs,
        )
        g_t_rinv = G.T / var_obs
        m = omega_inv + g_t_rinv @ G
        sign_m, log_det_m = np.linalg.slogdet(m)
        assert sign_m > 0

        quad_r = float(np.sum(residuals**2 / var_obs))
        log_det_r = float(np.sum(np.log(var_obs)))
        eta_penalty = float(eta_i @ omega_inv @ eta_i)
        ofv_i = (
            n_obs * LOG2PI
            + log_det_r
            + log_det_omega
            + float(log_det_m)
            + quad_r
            + eta_penalty
            - len(eta_i) * LOG2PI
        )
        total += ofv_i
        subject_rows.append(
            {
                "subject_id": int(sid),
                "n_obs": n_obs,
                "quad_r": quad_r,
                "log_det_r": log_det_r,
                "log_det_omega": float(log_det_omega),
                "log_det_m": float(log_det_m),
                "eta_penalty": eta_penalty,
                "ofv_i": float(ofv_i),
            }
        )

    return subject_rows, float(total)


@pytest.fixture(scope="module")
def warfarin_diagnostic_bundle():
    built = _build_warfarin_model(maxeval=40)
    result = built.fit()
    ref = _load_reference()
    est = get_estimation_method("FOCEI", maxeval=40)

    current_params = _parameter_set_from_result(built, result)
    ref_params = _parameter_set_from_reference(built, ref)

    current_eta = est._inner_loop(built.population_model, current_params)
    ref_eta = est._inner_loop(built.population_model, ref_params)
    current_ofv = est._outer_ofv(built.population_model, current_params, current_eta)
    ref_ofv = est._outer_ofv(built.population_model, ref_params, ref_eta)

    current_rows, current_total = _subject_objective_terms(
        built.population_model, current_params, current_eta
    )
    ref_rows, ref_total = _subject_objective_terms(built.population_model, ref_params, ref_eta)

    return {
        "built": built,
        "result": result,
        "reference": ref,
        "current_params": current_params,
        "ref_params": ref_params,
        "current_rows": current_rows,
        "ref_rows": ref_rows,
        "current_total": current_total,
        "ref_total": ref_total,
        "current_ofv": float(current_ofv),
        "ref_ofv": float(ref_ofv),
    }


@pytest.mark.external_validation
@pytest.mark.slow
def test_current_and_reference_warfarin_points_are_nearly_tied_under_openpkpd_objective(
    warfarin_diagnostic_bundle,
):
    current_ofv = warfarin_diagnostic_bundle["current_ofv"]
    ref_ofv = warfarin_diagnostic_bundle["ref_ofv"]

    assert current_ofv <= ref_ofv + 0.5, (
        f"Current and frozen-reference points diverged more than expected under the current "
        f"OpenPKPD objective: current={current_ofv:.4f}, reference={ref_ofv:.4f}"
    )


@pytest.mark.external_validation
@pytest.mark.slow
def test_warfarin_diagnostic_objective_matches_reported_fit_ofv(warfarin_diagnostic_bundle):
    result = warfarin_diagnostic_bundle["result"]
    current_ofv = warfarin_diagnostic_bundle["current_ofv"]

    assert abs(float(result.ofv) - current_ofv) < 1e-4, (
        f"Diagnostic objective total {current_ofv:.4f} diverges from reported OFV {result.ofv:.4f}"
    )


@pytest.mark.external_validation
@pytest.mark.slow
def test_warfarin_diagnostic_breakdown_has_expected_shape(warfarin_diagnostic_bundle):
    current_rows = warfarin_diagnostic_bundle["current_rows"]
    reference = warfarin_diagnostic_bundle["reference"]

    assert len(current_rows) == int(reference["n_subjects"])
    assert all(row["n_obs"] >= 1 for row in current_rows)
    assert all("quad_r" in row and "log_det_m" in row and "eta_penalty" in row for row in current_rows)
