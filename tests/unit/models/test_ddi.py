"""Tests for DDI index calculations."""

import pytest

from openpkpd.models.ddi import (
    DDIResult,
    DDIStudyAnalysis,
    competitive_inhibition_r,
    induction_r,
    time_dependent_inhibition_r,
)


class TestCompetitiveInhibitionR:
    def test_no_inhibitor(self):
        """Zero inhibitor -> AUC ratio = 1."""
        r = competitive_inhibition_r(inhibitor_conc=0.0, ki=1.0, fm=1.0)
        assert r == pytest.approx(1.0)

    def test_inhibition_increases_auc(self):
        """Inhibitor present -> AUC ratio > 1."""
        r = competitive_inhibition_r(inhibitor_conc=1.0, ki=1.0, fm=1.0)
        assert r > 1.0

    def test_partial_fm(self):
        """Partial fm reduces the magnitude of the effect."""
        r_full = competitive_inhibition_r(inhibitor_conc=1.0, ki=1.0, fm=1.0)
        r_half = competitive_inhibition_r(inhibitor_conc=1.0, ki=1.0, fm=0.5)
        assert r_half < r_full

    def test_invalid_ki(self):
        with pytest.raises(ValueError):
            competitive_inhibition_r(1.0, ki=0.0)

    def test_known_value(self):
        """R = 1 + I/Ki = 2 when I=Ki; AUC = 1/(1 - fm*(1 - 0.5)) with fm=1 -> 2."""
        r = competitive_inhibition_r(inhibitor_conc=1.0, ki=1.0, fm=1.0)
        assert r == pytest.approx(2.0)

    def test_large_inhibitor_concentration_matches_asymptotic_limit(self):
        """As I -> inf, competitive inhibition tends to 1 / (1 - fm)."""
        fm = 0.8
        approx_limit = competitive_inhibition_r(inhibitor_conc=1e6, ki=1.0, fm=fm)
        assert approx_limit == pytest.approx(1.0 / (1.0 - fm), rel=1e-5)


class TestTDIR:
    def test_no_inhibitor(self):
        """Zero inhibitor -> AUC ratio = 1."""
        r = time_dependent_inhibition_r(0.0, kinact=0.1, ki_app=1.0, fm=1.0)
        assert r == pytest.approx(1.0)

    def test_tdi_increases_auc(self):
        """TDI with inhibitor -> AUC ratio > 1."""
        r = time_dependent_inhibition_r(1.0, kinact=0.1, ki_app=1.0, fm=1.0)
        assert r > 1.0

    def test_higher_conc_more_inhibition(self):
        """Higher inhibitor concentration -> larger AUC ratio."""
        r1 = time_dependent_inhibition_r(1.0, kinact=0.1, ki_app=1.0, fm=1.0)
        r2 = time_dependent_inhibition_r(5.0, kinact=0.1, ki_app=1.0, fm=1.0)
        assert r2 > r1

    def test_large_concentration_matches_saturation_limit(self):
        """As I -> inf, TDI tends to the kinact-saturated static limit."""
        kinact = 0.12
        kdeg = 0.03
        fm = 0.7
        r_sat = (kdeg + kinact) / kdeg
        expected = 1.0 / (1.0 - fm * (1.0 - 1.0 / r_sat))
        approx_limit = time_dependent_inhibition_r(
            1e6, kinact=kinact, ki_app=1.0, degradation_rate=kdeg, fm=fm
        )
        assert approx_limit == pytest.approx(expected, rel=1e-5)

    def test_invalid_degradation_rate_raises(self):
        with pytest.raises(ValueError, match="degradation_rate"):
            time_dependent_inhibition_r(
                1.0,
                kinact=0.1,
                ki_app=1.0,
                degradation_rate=0.0,
                fm=1.0,
            )


class TestInductionR:
    def test_no_inducer(self):
        """Zero inducer -> AUC ratio = 1 (no effect)."""
        r = induction_r(0.0, emax_ind=5.0, ec50_ind=1.0, fm=1.0)
        assert r == pytest.approx(1.0)

    def test_induction_decreases_auc(self):
        """Induction -> AUC ratio < 1 (reduced exposure)."""
        r = induction_r(10.0, emax_ind=5.0, ec50_ind=1.0, fm=1.0)
        assert r < 1.0

    def test_partial_fm_induction(self):
        """Partial fm reduces induction effect."""
        r_full = induction_r(10.0, emax_ind=5.0, ec50_ind=1.0, fm=1.0)
        r_half = induction_r(10.0, emax_ind=5.0, ec50_ind=1.0, fm=0.5)
        # With fm=0.5, induction effect is smaller (ratio closer to 1)
        assert abs(r_half - 1.0) < abs(r_full - 1.0)

    def test_large_concentration_matches_saturation_limit(self):
        """As I -> inf, induction tends to the Emax-saturated limit."""
        emax_ind = 4.0
        fm = 0.75
        baseline_enzyme = 1.2
        r_ind_sat = baseline_enzyme * (1.0 + emax_ind)
        expected = 1.0 - fm * (1.0 - 1.0 / r_ind_sat)
        approx_limit = induction_r(
            1e6,
            emax_ind=emax_ind,
            ec50_ind=1.0,
            baseline_enzyme=baseline_enzyme,
            fm=fm,
        )
        assert approx_limit == pytest.approx(expected, rel=1e-5)

    def test_induction_monotone_over_concentration_grid(self):
        """Induction AUC ratio should decrease as inducer concentration rises."""
        grid = [
            induction_r(i, emax_ind=3.0, ec50_ind=2.0, fm=0.8) for i in [0.0, 1.0, 2.0, 5.0, 20.0]
        ]
        assert all(a >= b for a, b in zip(grid, grid[1:], strict=False))


class TestDDIStudyAnalysis:
    def test_reversible_ki_roundtrip(self):
        """Back-calculated Ki should reproduce the observed AUC ratio."""
        i_conc = 1.0
        ki_true = 2.0
        fm = 0.8
        auc_ratio_obs = competitive_inhibition_r(i_conc, ki_true, fm)

        analysis = DDIStudyAnalysis()
        result = analysis.fit_reversible_ki(auc_ratio_obs, i_conc, fm)

        assert result.perpetrator_params["Ki"] == pytest.approx(ki_true, rel=1e-4)

    def test_tdi_ki_roundtrip(self):
        """Back-calculated KI_app should reproduce the observed AUC ratio."""
        i_conc = 1.0
        ki_app_true = 0.5
        kinact = 0.1
        kdeg = 0.03
        fm = 1.0

        auc_ratio_obs = time_dependent_inhibition_r(i_conc, kinact, ki_app_true, kdeg, fm)

        analysis = DDIStudyAnalysis()
        result = analysis.fit_tdi_ki(auc_ratio_obs, i_conc, kinact, kdeg, fm)

        assert result.perpetrator_params["KI_app"] == pytest.approx(ki_app_true, rel=0.01)

    def test_result_type(self):
        """Returns DDIResult."""
        analysis = DDIStudyAnalysis()
        result = analysis.fit_reversible_ki(2.0, 1.0, 1.0)
        assert isinstance(result, DDIResult)
        assert result.mechanism == "competitive"

    def test_invalid_auc_ratio(self):
        """AUC ratio <= 1 raises ValueError."""
        analysis = DDIStudyAnalysis()
        with pytest.raises(ValueError):
            analysis.fit_reversible_ki(0.5, 1.0, 1.0)

    def test_reversible_ki_roundtrip_over_grid(self):
        """Reversible Ki back-calculation should hold across a small parameter grid."""
        analysis = DDIStudyAnalysis()
        for i_conc in [0.5, 2.0]:
            for ki_true in [0.4, 2.0]:
                for fm in [0.5, 0.9]:
                    auc_ratio = competitive_inhibition_r(i_conc, ki_true, fm)
                    result = analysis.fit_reversible_ki(auc_ratio, i_conc, fm)
                    assert result.perpetrator_params["Ki"] == pytest.approx(ki_true, rel=1e-4)

    def test_tdi_ki_roundtrip_over_grid(self):
        """TDI KI_app back-calculation should hold across a small grid."""
        analysis = DDIStudyAnalysis()
        for i_conc in [0.5, 2.0]:
            for ki_app_true in [0.4, 1.5]:
                auc_ratio = time_dependent_inhibition_r(
                    i_conc, kinact=0.1, ki_app=ki_app_true, degradation_rate=0.03, fm=0.8
                )
                result = analysis.fit_tdi_ki(
                    auc_ratio_observed=auc_ratio,
                    inhibitor_conc=i_conc,
                    kinact=0.1,
                    degradation_rate=0.03,
                    fm=0.8,
                )
                assert result.perpetrator_params["KI_app"] == pytest.approx(ki_app_true, rel=0.02)

    def test_fit_tdi_ki_invalid_degradation_rate_raises(self):
        analysis = DDIStudyAnalysis()
        with pytest.raises(ValueError, match="degradation_rate"):
            analysis.fit_tdi_ki(
                auc_ratio_observed=2.0,
                inhibitor_conc=1.0,
                kinact=0.1,
                degradation_rate=0.0,
                fm=1.0,
            )
