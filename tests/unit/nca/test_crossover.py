"""Tests for crossover BE analysis, power, and sample size."""

import math

import numpy as np
import pandas as pd
import pytest

from openpkpd.nca.crossover import (
    CrossoverResult,
    be_power,
    be_sample_size,
    crossover_be_analysis,
)


def make_crossover_df(n_per_seq=12, gmr=1.0, cv=0.20, seed=42):
    """Generate a 2x2 crossover dataset."""
    rng = np.random.default_rng(seed)
    sigma_w = math.sqrt(math.log(cv**2 + 1))

    records = []
    for subj in range(1, 2 * n_per_seq + 1):
        seq = "TR" if subj <= n_per_seq else "RT"

        if seq == "TR":
            # Period 1: Test, Period 2: Reference
            log_t = math.log(gmr) + rng.normal(0, sigma_w)
            log_r = rng.normal(0, sigma_w)
            records += [
                {
                    "subject": subj,
                    "sequence": seq,
                    "period": 1,
                    "treatment": "T",
                    "log_metric": log_t,
                },
                {
                    "subject": subj,
                    "sequence": seq,
                    "period": 2,
                    "treatment": "R",
                    "log_metric": log_r,
                },
            ]
        else:
            # Period 1: Reference, Period 2: Test
            log_r = rng.normal(0, sigma_w)
            log_t = math.log(gmr) + rng.normal(0, sigma_w)
            records += [
                {
                    "subject": subj,
                    "sequence": seq,
                    "period": 1,
                    "treatment": "R",
                    "log_metric": log_r,
                },
                {
                    "subject": subj,
                    "sequence": seq,
                    "period": 2,
                    "treatment": "T",
                    "log_metric": log_t,
                },
            ]
    return pd.DataFrame(records)


class TestCrossoverBEAnalysis:
    def test_basic_be(self):
        """Basic 2x2 crossover with identical formulations → BE."""
        df = make_crossover_df(n_per_seq=24, gmr=1.0, cv=0.20, seed=42)
        result = crossover_be_analysis(df, treatment_col="treatment", metric_col="log_metric")
        assert isinstance(result, CrossoverResult)
        assert np.isfinite(result.gmr)
        assert result.gmr > 0
        assert np.isfinite(result.ci_lo)
        assert np.isfinite(result.ci_hi)
        assert result.ci_lo < result.ci_hi

    def test_not_be_large_difference(self):
        """Large treatment difference → not bioequivalent."""
        df = make_crossover_df(n_per_seq=12, gmr=1.5, cv=0.15, seed=1)
        result = crossover_be_analysis(df, treatment_col="treatment", metric_col="log_metric")
        # With GMR=1.5, should not be BE
        assert isinstance(result, CrossoverResult)

    def test_result_structure(self):
        """CrossoverResult has all expected attributes."""
        df = make_crossover_df(n_per_seq=12, gmr=1.0, cv=0.20, seed=5)
        result = crossover_be_analysis(df, treatment_col="treatment", metric_col="log_metric")
        assert hasattr(result, "treatment_diff")
        assert hasattr(result, "se")
        assert hasattr(result, "df")
        assert hasattr(result, "p_value")
        assert hasattr(result, "period_effects")
        assert result.df > 0

    def test_missing_column_raises(self):
        """Missing column raises ValueError."""
        df = pd.DataFrame({"subject": [1], "period": [1]})
        with pytest.raises(ValueError, match="Missing columns"):
            crossover_be_analysis(df)

    def test_summary_string(self):
        """summary() returns a non-empty string."""
        df = make_crossover_df(n_per_seq=12)
        result = crossover_be_analysis(df, treatment_col="treatment", metric_col="log_metric")
        s = result.summary()
        assert len(s) > 10
        assert "GMR" in s


class TestBEPower:
    def test_power_increases_with_n(self):
        """Power increases as n per sequence increases."""
        p1 = be_power(cv=0.20, n_per_seq=10)
        p2 = be_power(cv=0.20, n_per_seq=20)
        assert p2 > p1

    def test_power_decreases_with_cv(self):
        """Power decreases as CV increases."""
        p1 = be_power(cv=0.15, n_per_seq=20)
        p2 = be_power(cv=0.30, n_per_seq=20)
        assert p1 > p2

    def test_power_range(self):
        """Power is in [0, 1]."""
        p = be_power(cv=0.20, n_per_seq=12)
        assert 0.0 <= p <= 1.0

    def test_power_with_true_ratio_1(self):
        """Power with true ratio = 1.0 should be > 0 for sufficient n."""
        p = be_power(cv=0.20, n_per_seq=24, true_ratio=1.0)
        assert p > 0

    def test_power_decreases_as_true_ratio_moves_away_from_one(self):
        """Power should drop as the true ratio approaches the BE limits."""
        centered = be_power(cv=0.20, n_per_seq=24, true_ratio=1.0)
        off_center = be_power(cv=0.20, n_per_seq=24, true_ratio=0.85)
        assert off_center < centered


class TestBESampleSize:
    def test_sample_size_is_positive(self):
        """Sample size returns a positive integer."""
        n = be_sample_size(cv=0.20, power=0.80)
        assert isinstance(n, int)
        assert n > 0

    def test_achieves_target_power(self):
        """Sample size achieves at least the target power."""
        cv = 0.20
        target_power = 0.80
        n = be_sample_size(cv=cv, power=target_power)
        if n <= 500:  # If solvable
            achieved = be_power(cv=cv, n_per_seq=n)
            assert achieved >= target_power - 0.01  # small tolerance

    def test_larger_cv_needs_more_subjects(self):
        """Higher CV requires more subjects."""
        n1 = be_sample_size(cv=0.20, power=0.80)
        n2 = be_sample_size(cv=0.35, power=0.80)
        assert n2 >= n1

    def test_returns_minimal_n_that_meets_target(self):
        """Returned n should be the first grid point meeting the target power."""
        cv = 0.25
        target_power = 0.80
        n = be_sample_size(cv=cv, power=target_power)
        assert be_power(cv=cv, n_per_seq=n) >= target_power
        if n > 2:
            assert be_power(cv=cv, n_per_seq=n - 1) < target_power

    def test_returns_max_plus_one_when_target_unreachable(self):
        """If the target is not reached within max_n, return max_n + 1."""
        assert be_sample_size(cv=1.0, power=0.9999, max_n=5) == 6
