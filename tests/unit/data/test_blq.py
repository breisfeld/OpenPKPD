"""Unit tests for openpkpd.data.blq — BLQ/LLOQ handling."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from scipy.stats import norm

from openpkpd.data.blq import (
    apply_m5_imputation,
    apply_m7_imputation,
    blq_log_likelihood,
    flag_blq_observations,
    is_blq,
)
from openpkpd.utils.constants import BLQMethod

# ---------------------------------------------------------------------------
# is_blq
# ---------------------------------------------------------------------------


def test_is_blq_true_when_dv_below_lloq():
    assert is_blq(0.05, lloq=0.1)


def test_is_blq_false_when_dv_equals_lloq():
    # Strictly less than, so equal is NOT considered BLQ
    assert not is_blq(0.1, lloq=0.1)


def test_is_blq_false_when_dv_above_lloq():
    assert not is_blq(0.2, lloq=0.1)


def test_is_blq_zero_dv():
    assert is_blq(0.0, lloq=0.05)


# ---------------------------------------------------------------------------
# blq_log_likelihood — M1 (exclude)
# ---------------------------------------------------------------------------


def test_m1_returns_zero():
    ll = blq_log_likelihood(0.05, mu=0.2, sigma2=0.01, lloq=0.1, method=BLQMethod.M1)
    assert ll == 0.0


# ---------------------------------------------------------------------------
# blq_log_likelihood — M2 / M3 (censored likelihood)
# ---------------------------------------------------------------------------


def test_m3_censored_likelihood_is_negative():
    """M3: log P(Y < LLOQ) should be < 0 (log probability)."""
    ll = blq_log_likelihood(0.05, mu=0.2, sigma2=0.01, lloq=0.1, method=BLQMethod.M3)
    assert ll < 0


def test_m2_equals_m3():
    """M2 and M3 share the same formula."""
    sigma2 = 0.01
    mu = 0.2
    lloq = 0.1
    ll_m2 = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M2)
    ll_m3 = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M3)
    assert ll_m2 == pytest.approx(ll_m3, rel=1e-10)


def test_m3_matches_normal_logcdf_formula():
    """M3 should equal log Phi((LLOQ - mu) / sigma)."""
    sigma2 = 0.04
    mu = 0.18
    lloq = 0.1
    sigma = math.sqrt(sigma2)

    ll = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M3)
    expected = norm.logcdf((lloq - mu) / sigma)

    assert ll == pytest.approx(expected, abs=1e-12)


def test_m3_higher_when_prediction_near_lloq():
    """
    When the prediction is near LLOQ, P(Y < LLOQ) should be close to 0.5,
    giving a log-likelihood near log(0.5) ≈ -0.693.
    """
    # mu = lloq means the normal CDF at (lloq - mu) / sigma = 0 → CDF = 0.5
    ll = blq_log_likelihood(0.05, mu=0.1, sigma2=0.04, lloq=0.1, method=BLQMethod.M3)
    assert ll == pytest.approx(math.log(0.5), rel=1e-6)


def test_m3_approaches_zero_when_mu_well_above_lloq():
    """When prediction >> LLOQ, P(Y < LLOQ) → 1 and log(P) → 0."""
    # mu = -10 * lloq: effectively all probability mass is below LLOQ
    ll = blq_log_likelihood(0.05, mu=-1.0, sigma2=0.01, lloq=0.1, method=BLQMethod.M3)
    assert ll == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# blq_log_likelihood — M4 (truncated normal)
# ---------------------------------------------------------------------------


def test_m4_returns_finite_value():
    ll = blq_log_likelihood(0.05, mu=0.2, sigma2=0.01, lloq=0.1, method=BLQMethod.M4)
    assert math.isfinite(ll)


def test_m4_is_negative():
    ll = blq_log_likelihood(0.05, mu=0.2, sigma2=0.01, lloq=0.1, method=BLQMethod.M4)
    assert ll < 0


def test_m4_close_to_m3_when_mu_positive():
    """
    M4 = P(0 <= Y < LLOQ) / P(Y >= 0); its numerator excludes P(Y < 0)
    relative to M3's P(Y < LLOQ), so M4 <= M3 always.  When mu >> sigma
    the two are numerically close.
    """
    sigma2 = 0.01
    mu = 0.5  # well above 0
    lloq = 0.1
    ll_m3 = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M3)
    ll_m4 = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M4)
    # M4 subtracts the P(Y < 0) region from numerator → M4 <= M3
    assert ll_m4 <= ll_m3 + 1e-9
    # When mu >> sigma, the difference is negligible
    assert abs(ll_m4 - ll_m3) < 0.1


def test_m4_matches_truncated_normal_formula():
    """M4 should equal the exact truncated-normal window probability formula."""
    sigma2 = 0.04
    mu = 0.15
    lloq = 0.25
    sigma = math.sqrt(sigma2)
    z_lloq = (lloq - mu) / sigma
    z_0 = -mu / sigma
    expected = math.log(norm.cdf(z_lloq) - norm.cdf(z_0)) - math.log(1.0 - norm.cdf(z_0))

    ll = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M4)

    assert ll == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# blq_log_likelihood — M5 (impute LLOQ/2)
# ---------------------------------------------------------------------------


def test_m5_imputation():
    """M5 replaces DV with LLOQ/2; at mu = LLOQ/2 likelihood should be finite."""
    ll = blq_log_likelihood(0.05, mu=0.05, sigma2=0.01, lloq=0.1, method=BLQMethod.M5)
    assert math.isfinite(ll)
    assert ll > -100  # reasonable value


def test_m5_maximum_at_lloq_half():
    """
    When mu = LLOQ/2 = 0.05, M5 should give the maximum likelihood
    (residual = 0), so ll should be maximized here.
    """
    sigma2 = 0.01
    lloq = 0.1
    ll_at_mu = blq_log_likelihood(0.05, mu=0.05, sigma2=sigma2, lloq=lloq, method=BLQMethod.M5)
    ll_offset = blq_log_likelihood(0.05, mu=0.20, sigma2=sigma2, lloq=lloq, method=BLQMethod.M5)
    assert ll_at_mu > ll_offset


def test_m5_matches_normal_log_likelihood_at_lloq_half():
    """M5 should equal the Gaussian log-likelihood at y = LLOQ / 2."""
    sigma2 = 0.09
    mu = 0.12
    lloq = 0.2
    y = lloq / 2.0
    expected = -0.5 * (math.log(2.0 * math.pi) + math.log(sigma2) + (y - mu) ** 2 / sigma2)

    ll = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M5)

    assert ll == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# blq_log_likelihood — M6 (first BLQ → LLOQ/2)
# ---------------------------------------------------------------------------


def test_m6_same_as_m5():
    """M6 formula is identical to M5 for the first BLQ observation."""
    sigma2 = 0.01
    mu = 0.08
    lloq = 0.1
    ll_m5 = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M5)
    ll_m6 = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M6)
    assert ll_m5 == pytest.approx(ll_m6, rel=1e-10)


# ---------------------------------------------------------------------------
# blq_log_likelihood — M7 (impute 0)
# ---------------------------------------------------------------------------


def test_m7_imputation():
    """M7 replaces DV with 0; at mu = 0 likelihood should be maximized."""
    sigma2 = 0.01
    lloq = 0.1
    ll_at_zero = blq_log_likelihood(0.05, mu=0.0, sigma2=sigma2, lloq=lloq, method=BLQMethod.M7)
    ll_offset = blq_log_likelihood(0.05, mu=0.5, sigma2=sigma2, lloq=lloq, method=BLQMethod.M7)
    assert ll_at_zero > ll_offset


def test_m7_matches_normal_log_likelihood_at_zero():
    """M7 should equal the Gaussian log-likelihood at y = 0."""
    sigma2 = 0.04
    mu = 0.15
    expected = -0.5 * (math.log(2.0 * math.pi) + math.log(sigma2) + mu**2 / sigma2)

    ll = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=0.1, method=BLQMethod.M7)

    assert ll == pytest.approx(expected, abs=1e-12)


def test_m3_lower_than_normal_when_mu_above_lloq():
    """
    BLQ censored likelihood should be lower than observed likelihood when
    the prediction is well above LLOQ (information is lost by censoring).
    """
    sigma2 = 0.01
    mu = 0.5  # prediction well above LLOQ
    lloq = 0.1
    # Censored likelihood
    ll_m3 = blq_log_likelihood(0.05, mu=mu, sigma2=sigma2, lloq=lloq, method=BLQMethod.M3)
    assert ll_m3 < 0


# ---------------------------------------------------------------------------
# blq_log_likelihood — invalid sigma2
# ---------------------------------------------------------------------------


def test_invalid_sigma2_raises():
    with pytest.raises(ValueError, match="sigma2 must be positive"):
        blq_log_likelihood(0.05, mu=0.2, sigma2=0.0, lloq=0.1, method=BLQMethod.M3)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unrecognized BLQ method"):
        blq_log_likelihood(0.05, mu=0.2, sigma2=0.01, lloq=0.1, method="M99")


# ---------------------------------------------------------------------------
# apply_m5_imputation
# ---------------------------------------------------------------------------


def test_apply_m5_imputation_replaces_blq():
    df = pd.DataFrame({"DV": [0.05, 0.5, 0.03], "LLOQ": [0.1, 0.1, 0.1]})
    result = apply_m5_imputation(df)
    assert result["DV"].iloc[0] == pytest.approx(0.05)  # LLOQ/2 = 0.1/2 = 0.05
    assert result["DV"].iloc[1] == pytest.approx(0.5)  # above LLOQ, unchanged
    assert result["DV"].iloc[2] == pytest.approx(0.05)  # LLOQ/2


def test_apply_m5_imputation_returns_copy():
    df = pd.DataFrame({"DV": [0.05, 0.5], "LLOQ": [0.1, 0.1]})
    apply_m5_imputation(df)
    # Original should be unchanged
    assert df["DV"].iloc[0] == pytest.approx(0.05)  # unchanged from original


def test_apply_m5_imputation_missing_dv_col_raises():
    df = pd.DataFrame({"OBS": [0.05], "LLOQ": [0.1]})
    with pytest.raises(KeyError, match="DV column"):
        apply_m5_imputation(df)


def test_apply_m5_imputation_missing_lloq_col_raises():
    df = pd.DataFrame({"DV": [0.05]})
    with pytest.raises(KeyError, match="LLOQ column"):
        apply_m5_imputation(df)


# ---------------------------------------------------------------------------
# apply_m7_imputation
# ---------------------------------------------------------------------------


def test_apply_m7_imputation_replaces_blq_with_zero():
    df = pd.DataFrame({"DV": [0.05, 0.5, 0.03], "LLOQ": [0.1, 0.1, 0.1]})
    result = apply_m7_imputation(df)
    assert result["DV"].iloc[0] == pytest.approx(0.0)
    assert result["DV"].iloc[1] == pytest.approx(0.5)
    assert result["DV"].iloc[2] == pytest.approx(0.0)


def test_apply_m7_imputation_returns_copy():
    df = pd.DataFrame({"DV": [0.05, 0.5], "LLOQ": [0.1, 0.1]})
    apply_m7_imputation(df)
    # Original should be unchanged
    assert df["DV"].iloc[0] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# flag_blq_observations
# ---------------------------------------------------------------------------


def test_flag_blq_observations():
    df = pd.DataFrame({"DV": [0.05, 0.5, 0.03, float("nan")], "LLOQ": [0.1, 0.1, 0.1, 0.1]})
    result = flag_blq_observations(df)
    assert "BLQ" in result.columns
    assert result["BLQ"].iloc[0] == 1  # 0.05 < 0.1
    assert result["BLQ"].iloc[1] == 0  # 0.5 >= 0.1
    assert result["BLQ"].iloc[2] == 1  # 0.03 < 0.1
    assert result["BLQ"].iloc[3] == 0  # NaN → 0
