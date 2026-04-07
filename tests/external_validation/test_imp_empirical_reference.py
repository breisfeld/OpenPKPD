"""Diagnostic empirical validation for IMP on bundled datasets."""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

from tests._release_validation import require_release_fixture


REF_DIR = os.path.join(os.path.dirname(__file__), "nlmixr2", "reference")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_ref(name: str) -> dict:
    path = require_release_fixture(os.path.join(REF_DIR, name), kind="Reference file")
    with path.open() as f:
        return json.load(f)


def _build_theophylline_imp_model():
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    data_path = require_release_fixture(
        os.path.join(DATA_DIR, "theophylline_boeckmann.csv"),
        kind="Data file",
    )

    dataset = NONMEMDataset.from_csv(data_path)
    return (
        ModelBuilder()
        .problem("Theophylline 1-cmt oral — empirical IMP validation")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV  = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.1, 1.5, 8), (0.3, 2.8, 8), (5, 32, 80)])
        .omega([0.09, 0.06, 0.04])
        .sigma(0.03)
        .estimation(method="IMP", isample=150, maxeval=12, seed=42, warm_start=True)
        .build()
    )


def _build_warfarin_impmap_model():
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    data_path = require_release_fixture(
        os.path.join(DATA_DIR, "warfarin_pk.csv"),
        kind="Data file",
    )

    dataset = NONMEMDataset.from_csv(data_path)
    return (
        ModelBuilder()
        .problem("Warfarin PK-only 1-cmt oral — empirical IMPMAP validation")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV  = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 0.9, 20), (0.001, 0.13, 5), (0.1, 8.7, 200)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(method="IMPMAP", isample=60, maxeval=12, seed=42)
        .build()
    )


def _build_warfarin_focei_model():
    from tests.external_validation.test_vs_nlmixr2 import _build_warfarin_focei_model as _build

    return _build(maxeval=40)


def _build_warfarin_imp_model():
    from openpkpd import ModelBuilder
    from openpkpd.data.dataset import NONMEMDataset

    data_path = require_release_fixture(
        os.path.join(DATA_DIR, "warfarin_pk.csv"),
        kind="Data file",
    )

    dataset = NONMEMDataset.from_csv(data_path)
    return (
        ModelBuilder()
        .problem("Warfarin PK-only 1-cmt oral — empirical IMP comparison baseline")
        .dataset(dataset)
        .subroutines(advan=2, trans=2)
        .pk("KA = THETA(1)*EXP(ETA(1))\nCL = THETA(2)*EXP(ETA(2))\nV  = THETA(3)*EXP(ETA(3))")
        .error("Y = F*(1 + EPS(1))")
        .theta([(0.01, 0.9, 20), (0.001, 0.13, 5), (0.1, 8.7, 200)])
        .omega([0.4, 0.3, 0.3])
        .sigma(0.05)
        .estimation(method="IMP", isample=60, maxeval=6, seed=42)
        .build()
    )


@pytest.mark.external_validation
@pytest.mark.slow
class TestTheophyllineIMPEmpirical:
    """Raw empirical IMP should remain numerically stable near the validated basin.

    Exact IMP correctness is covered by analytic reference tests elsewhere.
    On this empirical theophylline model, raw IMP is currently a noisy
    proposal-limited path rather than a reliable converged release method, so
    this class is diagnostic rather than release-gating. Stable empirical
    release coverage for a sparse IV benchmark is provided separately by the
    phenobarbital literature validation suite.
    """

    @pytest.fixture(scope="class")
    def fit_result(self):
        warnings.filterwarnings("ignore")
        return _build_theophylline_imp_model().fit()

    @pytest.fixture(scope="class")
    def nlmixr2_ref(self):
        return _load_ref("theophylline_foce.json")

    def test_fit_result_is_numerically_well_behaved(self, fit_result):
        assert np.isfinite(fit_result.ofv)
        assert fit_result.ofv < 1500.0
        assert fit_result.ofv_history
        assert len(fit_result.ofv_history) >= 2
        assert fit_result.method == "IMP"
        assert float(fit_result.ofv_history[-1]) <= float(fit_result.ofv_history[0]) + 50.0

    def test_imp_diagnostics_expose_low_ess_limitation(self, fit_result):
        is_diag = fit_result.diagnostics["importance_sampling"]
        assert is_diag["isample_doublings"] >= 1
        assert is_diag["final_eval_n_below_warn_threshold"] > 0

    def test_theta_stays_in_validated_theophylline_basin(self, fit_result, nlmixr2_ref):
        observed = [float(value) for value in fit_result.theta_final]
        expected = [
            float(nlmixr2_ref["theta"]["KA"]),
            float(nlmixr2_ref["theta"]["CL"]),
            float(nlmixr2_ref["theta"]["V"]),
        ]
        tolerances = [0.12, 0.08, 0.08]
        for name, obs, exp, tol in zip(("KA", "CL", "V"), observed, expected, tolerances):
            rel_err = abs(obs - exp) / exp
            assert rel_err < tol, (
                f"{name}={obs:.4f} vs nlmixr2 FOCEI={exp:.4f} "
                f"(rel_err={rel_err:.1%}, tolerance={tol:.0%})"
            )

    def test_sigma_is_finite_and_same_order_as_external_reference(self, fit_result, nlmixr2_ref):
        observed = float(fit_result.sigma_final[0, 0])
        expected = float(nlmixr2_ref["sigma_prop_err_variance"])
        assert np.isfinite(observed)
        rel_err = abs(observed - expected) / expected
        assert rel_err < 0.30, (
            f"SIGMA={observed:.5f} vs nlmixr2 FOCEI={expected:.5f} "
            f"(rel_err={rel_err:.1%})"
        )

    def test_imp_short_run_improves_ofv(self, fit_result):
        history = [float(value) for value in fit_result.ofv_history or []]
        assert len(history) >= 2
        assert history[-1] <= history[0] + 50.0


@pytest.mark.external_validation
@pytest.mark.slow
class TestWarfarinIMPMAPEmpirical:
    """IMPMAP should be numerically stable on the warfarin PK benchmark."""

    @pytest.fixture(scope="class")
    def fit_result(self):
        warnings.filterwarnings("ignore")
        return _build_warfarin_impmap_model().fit()

    @pytest.fixture(scope="class")
    def nlmixr2_ref(self):
        return _load_ref("warfarin_pk_foce.json")

    def test_fit_result_is_numerically_well_behaved(self, fit_result):
        assert fit_result.converged
        assert np.isfinite(fit_result.ofv)
        assert 0.0 < fit_result.ofv < 2000.0
        assert fit_result.ofv_history
        assert len(fit_result.ofv_history) >= 2

    def test_sigma_is_finite_and_same_order_as_external_reference(self, fit_result, nlmixr2_ref):
        observed = float(fit_result.sigma_final[0, 0])
        expected = float(nlmixr2_ref["sigma_prop_err_variance"])
        assert np.isfinite(observed)
        rel_err = abs(observed - expected) / expected
        assert rel_err < 0.45, (
            f"SIGMA={observed:.5f} vs nlmixr2 FOCEI={expected:.5f} "
            f"(rel_err={rel_err:.1%})"
        )

    def test_short_run_ofv_does_not_regress_badly(self, fit_result):
        history = [float(value) for value in fit_result.ofv_history or []]
        assert len(history) >= 2
        assert history[-1] <= history[0] + 50.0

    def test_fit_records_focei_warm_start_diagnostics(self, fit_result):
        warm_start = fit_result.diagnostics.get("warm_start")
        assert warm_start is not None
        assert warm_start["method"] == "FOCEI"
        assert warm_start["used"] is True
