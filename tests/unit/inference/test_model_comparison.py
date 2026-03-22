"""Unit tests for openpkpd.inference.model_comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.inference.model_comparison import LRTResult, aic_weights, compare_models, lrt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(ofv: float, n_params: int, n_obs: int = 100) -> EstimationResult:
    """Build a minimal EstimationResult for testing."""
    result = EstimationResult(
        theta_final=np.ones(n_params),
        omega_final=np.eye(1),
        sigma_final=np.eye(1),
        ofv=ofv,
    )
    result.n_observations = n_obs
    result._n_parameters = n_params
    return result


# ---------------------------------------------------------------------------
# LRTResult dataclass
# ---------------------------------------------------------------------------


def test_lrt_result_fields():
    r = LRTResult(
        ofv_full=100.0,
        ofv_reduced=115.0,
        delta_ofv=15.0,
        df=1,
        p_value=0.0001,
        significant=True,
        alpha=0.05,
    )
    assert r.delta_ofv == pytest.approx(15.0)
    assert r.significant is True


# ---------------------------------------------------------------------------
# lrt — significant test
# ---------------------------------------------------------------------------


def test_lrt_significant():
    full = make_result(ofv=100.0, n_params=5)
    reduced = make_result(ofv=115.0, n_params=4)
    result = lrt(full, reduced)

    assert result.delta_ofv == pytest.approx(15.0)
    assert result.df == 1
    assert result.p_value < 0.001
    assert result.significant
    assert result.ofv_full == pytest.approx(100.0)
    assert result.ofv_reduced == pytest.approx(115.0)


def test_lrt_not_significant():
    full = make_result(ofv=100.0, n_params=5)
    reduced = make_result(ofv=101.5, n_params=4)
    result = lrt(full, reduced)

    # delta_OFV = 1.5; chi2(1) at p=0.05 is ~3.84, so not significant
    assert not result.significant
    assert result.p_value > 0.05


def test_lrt_p_value_range():
    """P-value must be in [0, 1]."""
    full = make_result(ofv=100.0, n_params=3)
    reduced = make_result(ofv=108.0, n_params=2)
    result = lrt(full, reduced)
    assert 0.0 <= result.p_value <= 1.0


def test_lrt_large_delta_ofv_very_small_p():
    """A very large delta OFV should give an essentially zero p-value."""
    full = make_result(ofv=100.0, n_params=3)
    reduced = make_result(ofv=200.0, n_params=2)
    result = lrt(full, reduced)
    assert result.p_value < 1e-20
    assert result.significant


def test_lrt_df_zero_raises():
    """df <= 0 should raise ValueError."""
    full = make_result(ofv=100.0, n_params=4)
    reduced = make_result(ofv=105.0, n_params=4)  # same n_params → df=0
    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        lrt(full, reduced)


def test_lrt_df_negative_raises():
    """df < 0 (full has fewer params than reduced) should raise."""
    full = make_result(ofv=100.0, n_params=3)
    reduced = make_result(ofv=105.0, n_params=5)
    with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
        lrt(full, reduced)


def test_lrt_custom_alpha():
    """Custom alpha threshold affects significance decision."""
    full = make_result(ofv=100.0, n_params=5)
    reduced = make_result(ofv=103.0, n_params=4)
    # delta_OFV=3.0, chi2(1) p ≈ 0.083
    result_05 = lrt(full, reduced, alpha=0.05)
    result_10 = lrt(full, reduced, alpha=0.10)
    assert not result_05.significant  # p > 0.05
    assert result_10.significant  # p < 0.10


def test_lrt_negative_delta_ofv_gives_p_one():
    """
    When full model OFV > reduced (numerically inverted), delta < 0.
    p_value should be 1.0 (no evidence against the null).
    """
    full = make_result(ofv=110.0, n_params=5)  # worse than reduced
    reduced = make_result(ofv=100.0, n_params=4)
    result = lrt(full, reduced)
    assert result.p_value == pytest.approx(1.0)
    assert not result.significant


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


def test_compare_models_basic():
    results = [
        make_result(ofv=110.0, n_params=5),
        make_result(ofv=100.0, n_params=4),
        make_result(ofv=105.0, n_params=3),
    ]
    df = compare_models(results, labels=["M3", "M2", "M1"])

    assert len(df) == 3
    assert "AIC" in df.columns
    assert "BIC" in df.columns
    assert "OFV" in df.columns
    assert "n_params" in df.columns
    assert "dOFV" in df.columns
    assert "dAIC" in df.columns


def test_compare_models_sorted_by_aic():
    results = [make_result(110, 5), make_result(100, 4), make_result(105, 3)]
    df = compare_models(results, labels=["M3", "M2", "M1"])
    aics = df["AIC"].tolist()
    assert aics == sorted(aics)


def test_compare_models_best_model_daic_zero():
    """The best model should have dAIC = 0."""
    results = [make_result(110, 5), make_result(100, 4), make_result(105, 3)]
    df = compare_models(results)
    assert df["dAIC"].iloc[0] == pytest.approx(0.0)


def test_compare_models_dofv_non_negative():
    """All dOFV values should be >= 0 (relative to model with best OFV)."""
    results = [make_result(110, 5), make_result(100, 4), make_result(105, 3)]
    df = compare_models(results)
    assert (df["dOFV"] >= -1e-9).all()


def test_compare_models_aic_values_correct():
    """AIC = OFV + 2 * n_params."""
    res = make_result(ofv=100.0, n_params=4, n_obs=100)
    df = compare_models([res], labels=["M1"])
    assert df["AIC"].iloc[0] == pytest.approx(100.0 + 2 * 4)


def test_compare_models_bic_values_correct():
    """BIC = OFV + ln(n_obs) * n_params."""
    n_obs = 100
    res = make_result(ofv=100.0, n_params=4, n_obs=n_obs)
    expected_bic = 100.0 + math.log(n_obs) * 4
    df = compare_models([res], labels=["M1"])
    assert df["BIC"].iloc[0] == pytest.approx(expected_bic, rel=1e-6)


def test_compare_models_default_labels():
    """When labels is None, default labels like 'Model_1' are used."""
    results = [make_result(100, 4), make_result(105, 3)]
    df = compare_models(results)
    assert set(df["Model"]) == {"Model_1", "Model_2"}


def test_compare_models_label_length_mismatch_raises():
    results = [make_result(100, 4), make_result(105, 3)]
    with pytest.raises(ValueError, match="Length of labels"):
        compare_models(results, labels=["only_one"])


def test_compare_models_empty_returns_empty_df():
    df = compare_models([])
    assert len(df) == 0
    assert "AIC" in df.columns


# ---------------------------------------------------------------------------
# aic_weights
# ---------------------------------------------------------------------------


def test_aic_weights_sum_to_one():
    results = [make_result(100, 4), make_result(105, 3), make_result(110, 5)]
    weights = aic_weights(results)
    assert weights.sum() == pytest.approx(1.0, rel=1e-9)


def test_aic_weights_best_model_highest_weight():
    # Model with AIC=108 (100 + 2*4), Model with AIC=111 (105+2*3), Model with AIC=120 (110+2*5)
    results = [make_result(100, 4), make_result(105, 3), make_result(110, 5)]
    weights = aic_weights(results)
    # Best AIC is 108 (first), so it should have the highest weight
    aics = np.array([r.aic for r in results])
    best_idx = int(np.argmin(aics))
    assert int(np.argmax(weights)) == best_idx


def test_aic_weights_equal_models():
    """Models with identical AIC should receive equal weights."""
    results = [make_result(100, 4), make_result(100, 4)]
    weights = aic_weights(results)
    assert weights[0] == pytest.approx(weights[1], rel=1e-9)
    assert weights[0] == pytest.approx(0.5, rel=1e-9)


def test_aic_weights_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        aic_weights([])


def test_aic_weights_all_finite():
    results = [make_result(100, 4), make_result(105, 3)]
    weights = aic_weights(results)
    assert all(math.isfinite(w) for w in weights)


# ---------------------------------------------------------------------------
# EstimationResult — AIC / BIC / n_parameters
# ---------------------------------------------------------------------------


def test_aic_formula():
    res = make_result(ofv=200.0, n_params=6, n_obs=80)
    assert res.aic == pytest.approx(200.0 + 2 * 6)


def test_bic_formula():
    n_obs = 80
    res = make_result(ofv=200.0, n_params=6, n_obs=n_obs)
    expected = 200.0 + math.log(n_obs) * 6
    assert res.bic == pytest.approx(expected, rel=1e-6)


def test_bic_inf_when_no_observations():
    res = make_result(ofv=100.0, n_params=4, n_obs=0)
    assert math.isinf(res.bic)


def test_n_parameters_from_explicit():
    res = make_result(ofv=100.0, n_params=7)
    assert res.n_parameters == 7


def test_n_parameters_inferred_when_zero():
    """When _n_parameters is 0, n_parameters falls back to shape inference."""
    res = EstimationResult(
        theta_final=np.ones(3),
        omega_final=np.eye(2),
        sigma_final=np.eye(1),
        ofv=100.0,
    )
    # _n_parameters = 0 → inferred: 3 theta + 3 omega (2x2 lower tri) + 1 sigma = 7
    assert res.n_parameters == 7


def test_compute_n_parameters_with_specs():
    """compute_n_parameters correctly counts free parameters from specs."""
    res = make_result(ofv=100.0, n_params=0)
    res._n_parameters = 0  # reset

    class MockSpec:
        def __init__(self, fixed: bool) -> None:
            self.fixed = fixed

    theta_specs = [MockSpec(False), MockSpec(False), MockSpec(True)]  # 2 free
    omega_specs = [MockSpec(False), MockSpec(True)]  # 1 free
    sigma_specs = [MockSpec(False)]  # 1 free

    res.compute_n_parameters(theta_specs, omega_specs, sigma_specs)
    assert res._n_parameters == 4  # 2 + 1 + 1
