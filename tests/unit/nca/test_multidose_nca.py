"""Tests for multidose and partial AUC NCA functionality."""

import numpy as np
import pytest

from openpkpd.nca import NCAEngine


class TestPartialAUC:
    def test_partial_auc_subset(self):
        """Partial AUC over subset of interval."""
        engine = NCAEngine(auc_method="linear-trapezoidal")
        times = np.array([0, 1, 2, 4, 8, 12], dtype=float)
        conc = np.array([0, 10, 8, 5, 2, 0.5], dtype=float)
        pauc = engine.compute_partial_auc(times, conc, t1=1.0, t2=4.0)
        # Should be positive and less than total AUC
        assert np.isfinite(pauc)
        assert pauc > 0
        total = engine._compute_auc(times, conc)
        assert pauc < total

    def test_partial_auc_invalid_range(self):
        """t1 >= t2 returns NaN."""
        engine = NCAEngine()
        times = np.array([0, 1, 2, 4], dtype=float)
        conc = np.array([0, 10, 8, 5], dtype=float)
        assert np.isnan(engine.compute_partial_auc(times, conc, t1=3.0, t2=2.0))

    def test_partial_auc_full_range(self):
        """Partial AUC over full range equals total AUC."""
        engine = NCAEngine(auc_method="linear-trapezoidal")
        times = np.array([0.0, 1.0, 2.0, 4.0], dtype=float)
        conc = np.array([10.0, 8.0, 5.0, 2.0], dtype=float)
        pauc = engine.compute_partial_auc(times, conc, t1=0.0, t2=4.0)
        total = engine._compute_auc(times, conc)
        assert abs(pauc - total) < 1e-10


class TestMultidoseNCA:
    def setup_method(self):
        self.engine = NCAEngine(auc_method="linear-trapezoidal")

    def test_basic_multidose(self):
        """Multidose NCA returns standard plus multidose parameters."""
        times = np.array(
            [
                0,
                0.5,
                1,
                2,
                4,
                8,
                12,  # interval 1
                12.5,
                13,
                14,
                16,
                20,
                24,
            ],
            dtype=float,
        )  # interval 2
        conc = np.array([0, 5, 8, 6, 4, 2, 0.5, 5.5, 9, 7, 4.5, 2.1, 0.6], dtype=float)
        params = self.engine.compute_multidose_subject(times, conc, dose=100.0, tau=12.0, ss=True)
        assert np.isfinite(params.auc_tau)
        assert params.auc_tau > 0
        assert np.isfinite(params.c_min)
        assert np.isfinite(params.c_avg)
        assert np.isfinite(params.fluctuation)

    def test_dose_normalized(self):
        """Dose-normalized params computed when dose > 0."""
        times = np.array([0, 1, 2, 4, 8, 12], dtype=float)
        conc = np.array([0, 10, 8, 5, 2, 0.5], dtype=float)
        params = self.engine.compute_multidose_subject(times, conc, dose=100.0, tau=12.0)
        if np.isfinite(params.cmax):
            assert np.isfinite(params.norm_cmax)
            assert abs(params.norm_cmax - params.cmax / 100.0) < 1e-10

    def test_r_ac_with_sd_auc(self):
        """r_ac computed when single-dose AUC_inf provided."""
        times = np.array([0, 1, 2, 4, 8, 12], dtype=float)
        conc = np.array([0, 10, 8, 5, 2, 0.5], dtype=float)
        params = self.engine.compute_multidose_subject(
            times, conc, dose=100.0, tau=12.0, sd_auc_inf=30.0
        )
        assert np.isfinite(params.r_ac)


class TestBLQRule:
    def test_blq_zero(self):
        """BLQ values set to 0."""
        engine = NCAEngine()
        times = np.array([0, 1, 2], dtype=float)
        conc = np.array([0.5, 0.1, 0.01], dtype=float)
        result = engine.apply_predose_blq_rule(times, conc, lloq=0.05, rule="zero")
        assert result[2] == 0.0
        assert result[0] == 0.5

    def test_blq_lloq_half(self):
        """BLQ values set to LLOQ/2."""
        engine = NCAEngine()
        times = np.array([0, 1, 2], dtype=float)
        conc = np.array([5.0, 0.02, 0.01], dtype=float)
        result = engine.apply_predose_blq_rule(times, conc, lloq=0.05, rule="lloq_half")
        assert result[1] == pytest.approx(0.025)
        assert result[2] == pytest.approx(0.025)

    def test_blq_exclude(self):
        """BLQ values set to NaN."""
        engine = NCAEngine()
        times = np.array([0, 1, 2], dtype=float)
        conc = np.array([5.0, 0.5, 0.01], dtype=float)
        result = engine.apply_predose_blq_rule(times, conc, lloq=0.05, rule="exclude")
        assert np.isnan(result[2])
        assert result[0] == 5.0

    @pytest.mark.parametrize("rule", ["zero", "lloq_half", "exclude"])
    def test_exactly_at_lloq_is_not_treated_as_blq(self, rule):
        """Values exactly equal to LLOQ should be preserved."""
        engine = NCAEngine()
        times = np.array([-0.5, 0.0, 1.0], dtype=float)
        conc = np.array([0.04, 0.05, 0.06], dtype=float)

        result = engine.apply_predose_blq_rule(times, conc, lloq=0.05, rule=rule)

        expected_first = {
            "zero": 0.0,
            "lloq_half": 0.025,
            "exclude": np.nan,
        }[rule]
        if np.isnan(expected_first):
            assert np.isnan(result[0])
        else:
            assert result[0] == pytest.approx(expected_first)
        assert result[1] == pytest.approx(0.05)
        assert result[2] == pytest.approx(0.06)

    def test_nan_values_are_preserved_and_input_not_mutated(self):
        """NaNs stay NaN and the input array is not modified in place."""
        engine = NCAEngine()
        times = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
        conc = np.array([np.nan, 0.01, 0.07, np.nan], dtype=float)
        original = conc.copy()

        result = engine.apply_predose_blq_rule(times, conc, lloq=0.05, rule="zero")

        assert np.isnan(result[0])
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.07)
        assert np.isnan(result[3])
        assert np.array_equal(conc, original, equal_nan=True)

    def test_invalid_rule_raises_value_error(self):
        """Unknown BLQ rules should fail explicitly."""
        engine = NCAEngine()

        with pytest.raises(ValueError, match="Unknown BLQ rule"):
            engine.apply_predose_blq_rule(
                np.array([0.0, 1.0], dtype=float),
                np.array([0.01, 0.5], dtype=float),
                lloq=0.05,
                rule="drop",
            )
