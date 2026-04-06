"""Diagnostic-only coverage for the remaining Warfarin basin/parity gap."""

from __future__ import annotations

import warnings

import pytest

from tests.external_validation.test_imp_empirical_reference import (
    _build_warfarin_focei_model,
    _build_warfarin_imp_model,
    _build_warfarin_impmap_model,
    _load_ref,
)


@pytest.mark.external_validation
@pytest.mark.slow
class TestWarfarinFOCEIBasinDiagnostics:
    """Diagnostic-only: track the known FOCEI KA basin gap on Warfarin PK."""

    @pytest.fixture(scope="class")
    def focei_result(self):
        warnings.filterwarnings("ignore")
        return _build_warfarin_focei_model().fit()

    @pytest.fixture(scope="class")
    def ref_foce(self):
        return _load_ref("warfarin_pk_foce.json")

    @pytest.fixture(scope="class")
    def ref_fo(self):
        return _load_ref("warfarin_pk_fo.json")

    def test_ka_remains_in_known_practical_basin(self, focei_result, ref_fo, ref_foce):
        ka = float(focei_result.theta_final[0])
        ka_ref_fo = float(ref_fo["theta"]["KA"])
        ka_ref_foce = float(ref_foce["theta"]["KA"])
        assert ka > ka_ref_fo, (
            f"FOCEI KA={ka:.4f} unexpectedly fell below the FO reference {ka_ref_fo:.4f}"
        )
        assert ka < 2.2 * ka_ref_foce, (
            f"FOCEI KA={ka:.4f} left the known practical basin implied by the diagnostic fit"
        )


@pytest.mark.external_validation
@pytest.mark.slow
class TestWarfarinIMPRecommendationDiagnostics:
    """Diagnostic-only: raw IMP stays fragile while IMPMAP follows FOCEI."""

    @pytest.fixture(scope="class")
    def imp_result(self):
        warnings.filterwarnings("ignore")
        return _build_warfarin_imp_model().fit()

    @pytest.fixture(scope="class")
    def impmap_result(self):
        warnings.filterwarnings("ignore")
        return _build_warfarin_impmap_model().fit()

    @pytest.fixture(scope="class")
    def focei_result(self):
        warnings.filterwarnings("ignore")
        return _build_warfarin_focei_model().fit()

    def test_impmap_is_closer_than_raw_imp_to_openpkpd_focei_on_ka(
        self, imp_result, impmap_result, focei_result
    ):
        exp_ka = float(focei_result.theta_final[0])
        imp_ka_rel_err = abs(float(imp_result.theta_final[0]) - exp_ka) / exp_ka
        impmap_ka_rel_err = abs(float(impmap_result.theta_final[0]) - exp_ka) / exp_ka

        assert impmap_ka_rel_err < 0.08, (
            f"IMPMAP KA should stay near the OpenPKPD FOCEI basin; got {impmap_ka_rel_err:.1%}"
        )
        assert imp_ka_rel_err > 0.20, (
            f"Raw IMP should remain basin-sensitive in this benchmark; got {imp_ka_rel_err:.1%}"
        )
        assert impmap_ka_rel_err + 0.10 < imp_ka_rel_err, (
            f"Expected IMPMAP to materially improve KA on warfarin: "
            f"IMP={imp_ka_rel_err:.1%}, IMPMAP={impmap_ka_rel_err:.1%}"
        )

    def test_impmap_has_real_warm_start_while_raw_imp_does_not(self, imp_result, impmap_result):
        assert imp_result.diagnostics.get("warm_start") is None
        warm_start = impmap_result.diagnostics.get("warm_start")
        assert warm_start is not None
        assert warm_start["used"] is True
        assert warm_start["method"] == "FOCEI"

    def test_impmap_is_the_converged_practical_path_for_this_budget(self, imp_result, impmap_result):
        assert impmap_result.converged is True
        assert imp_result.converged is False

        imp_opt = imp_result.diagnostics.get("optimizer", {})
        impmap_opt = impmap_result.diagnostics.get("optimizer", {})
        assert imp_opt.get("iterations", 0) <= 3
        assert impmap_opt.get("success") is True
        assert impmap_opt.get("function_evals", 0) >= 50
