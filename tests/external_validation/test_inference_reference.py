"""
External validation: LRT, AIC, BIC, Akaike weights vs scipy.stats and R.

Validates openpkpd model comparison utilities against:
  - scipy.stats.chi2 (LRT p-values — exact chi-squared distribution)
  - R AIC() / BIC() formulas (OFV-based information criteria)
  - Burnham & Anderson (2002) Akaike weight formulas

All tests are closed-form (no fitting required) and run in < 0.1 seconds.

Formulas
--------
LRT statistic:   ΔOF V = OFV_reduced - OFV_full ~ χ²(df)
                 df = n_params_full - n_params_reduced
AIC:             OFV + 2 · n_params
BIC:             OFV + ln(n_obs) · n_params
Akaike weight:   w_i = exp(-0.5 · ΔAIC_i) / Σ_j exp(-0.5 · ΔAIC_j)

References
----------
Burnham KP & Anderson DR (2002). Model Selection and Multimodel Inference:
    A Practical Information-Theoretic Approach, 2nd ed. Springer, New York.
Akaike H (1974). A new look at the statistical model identification.
    IEEE Trans Autom Control 19(6):716–723.
Schwarz G (1978). Estimating the dimension of a model.
    Ann Stat 6(2):461–464.
Wilks SS (1938). The large-sample distribution of the likelihood ratio.
    Ann Math Stat 9(1):60–62.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import chi2

from openpkpd.estimation.base import EstimationResult

# ---------------------------------------------------------------------------
# Helpers: build lightweight EstimationResult stubs
# ---------------------------------------------------------------------------


def _make_result(
    ofv: float,
    n_theta: int = 3,
    n_eta: int = 2,
    n_eps: int = 1,
    n_obs: int = 120,
) -> EstimationResult:
    """
    Construct a minimal EstimationResult for testing model comparison utilities.

    Parameters
    ----------
    ofv:      Objective function value (-2 · log-likelihood).
    n_theta:  Number of fixed-effect parameters.
    n_eta:    Number of random-effect variance parameters (diagonal OMEGA).
    n_eps:    Number of residual-error variance parameters (diagonal SIGMA).
    n_obs:    Number of observations (for BIC).
    """
    from openpkpd.estimation.base import EstimationResult

    return EstimationResult(
        theta_final=np.zeros(n_theta),
        omega_final=np.eye(n_eta),
        sigma_final=np.eye(n_eps),
        ofv=ofv,
        converged=True,
        n_observations=n_obs,
    )


# ---------------------------------------------------------------------------
# Layer A: Likelihood Ratio Test vs scipy.stats.chi2
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestLRTVsScipy:
    """
    Validate lrt() p-values against scipy.stats.chi2.sf (exact chi-squared CDF).

    The LRT test statistic ΔOFV = OFV_reduced - OFV_full follows χ²(df)
    under H₀ (Wilks 1938). openpkpd uses chi2.sf internally, so agreement
    should be to floating-point precision.
    """

    def test_lrt_pvalue_matches_chi2_sf(self):
        """LRT p-value = chi2.sf(delta_ofv, df) — exact equality."""
        from openpkpd.inference import lrt

        ofv_full = 200.0
        ofv_reduced = 207.84  # delta_ofv = 7.84

        # full model: 3 theta + 3 eta (diag OMEGA 3×3) + 1 eps = 7 params
        # reduced  : 3 theta + 2 eta (diag OMEGA 2×2) + 1 eps = 6 params → df = 1?
        # Actually n_params_full - n_params_reduced must be > 0.
        # Let's set _n_parameters explicitly to avoid confusion with matrix inference.
        full = _make_result(ofv_full, n_theta=4, n_eta=2, n_eps=1)
        reduced = _make_result(ofv_reduced, n_theta=3, n_eta=2, n_eps=1)

        result = lrt(full, reduced)

        expected_p = float(chi2.sf(ofv_reduced - ofv_full, df=1))
        np.testing.assert_allclose(
            result.p_value, expected_p, rtol=1e-12, err_msg="LRT p-value must match chi2.sf exactly"
        )
        assert result.delta_ofv == pytest.approx(ofv_reduced - ofv_full)
        assert result.df == 1

    def test_lrt_df2_example(self):
        """LRT with df=2 and known delta_OFV matches chi2.sf(delta, df=2)."""
        from openpkpd.inference import lrt

        delta = 5.991  # chi2(2) 95th percentile ≈ 5.991
        full = _make_result(100.0, n_theta=5, n_eta=2, n_eps=1)
        reduced = _make_result(100.0 + delta, n_theta=3, n_eta=2, n_eps=1)

        result = lrt(full, reduced)
        expected_p = float(chi2.sf(delta, df=2))
        np.testing.assert_allclose(result.p_value, expected_p, rtol=1e-10)
        # chi2(2) 95th percentile → p ≈ 0.05 (approximately, verify sign)
        assert result.p_value < 0.06

    def test_lrt_significant_at_alpha05(self):
        """ΔOFV = 6.63 (chi2(1) 99th pctile) → p ≈ 0.01 → significant at α=0.05."""
        from openpkpd.inference import lrt

        full = _make_result(150.0, n_theta=4, n_eta=2, n_eps=1)
        reduced = _make_result(150.0 + 6.63, n_theta=3, n_eta=2, n_eps=1)

        result = lrt(full, reduced, alpha=0.05)
        assert result.significant, (
            f"ΔOFV=6.63 (df=1) should be significant at α=0.05, p={result.p_value:.4f}"
        )

    def test_lrt_not_significant_for_small_delta(self):
        """ΔOFV = 1.0 (df=1) → p ≈ 0.317 → not significant at α=0.05."""
        from openpkpd.inference import lrt

        full = _make_result(200.0, n_theta=4, n_eta=2, n_eps=1)
        reduced = _make_result(201.0, n_theta=3, n_eta=2, n_eps=1)

        result = lrt(full, reduced, alpha=0.05)
        expected_p = float(chi2.sf(1.0, df=1))
        np.testing.assert_allclose(result.p_value, expected_p, rtol=1e-10)
        assert not result.significant, (
            f"ΔOFV=1.0 (df=1) should not be significant; p={result.p_value:.4f}"
        )

    def test_lrt_raises_on_nonpositive_df(self):
        """lrt() must raise ValueError when df ≤ 0 (full ≤ reduced parameters)."""
        from openpkpd.inference import lrt

        full = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        reduced = _make_result(210.0, n_theta=4, n_eta=2, n_eps=1)  # more params

        with pytest.raises(ValueError, match="[Dd]egrees of freedom"):
            lrt(full, reduced)

    def test_lrt_scm_forward_threshold(self):
        """
        SCM forward inclusion threshold: ΔOFV ≥ 3.84 (p ≤ 0.05, df=1).

        This is the standard forward-selection threshold in pharmacometric SCM
        (Jonsson & Karlsson 1998), equivalent to chi2(1) 5% critical value.
        """
        from openpkpd.inference import lrt

        chi2_5pct = float(chi2.ppf(0.95, df=1))  # ≈ 3.841
        full = _make_result(100.0, n_theta=4, n_eta=2, n_eps=1)
        reduced = _make_result(100.0 + chi2_5pct + 0.001, n_theta=3, n_eta=2, n_eps=1)

        result = lrt(full, reduced, alpha=0.05)
        assert result.significant, (
            f"ΔOFV slightly above 3.841 should be significant; got p={result.p_value:.4f}"
        )


# ---------------------------------------------------------------------------
# Layer B: AIC and BIC formula validation
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestAICBICFormulas:
    """
    Validate AIC and BIC properties against exact formulas.

    AIC = OFV + 2 · n_params    (Akaike 1974)
    BIC = OFV + ln(n) · n_params (Schwarz 1978)

    where OFV = -2 · log-likelihood.
    """

    def test_aic_formula_exact(self):
        """AIC = OFV + 2 * n_params (exact)."""
        ofv = 215.7
        res = _make_result(ofv, n_theta=3, n_eta=2, n_eps=1)
        # n_params inferred: 3 + lower-tri(2) + lower-tri(1) = 3 + 3 + 1 = 7
        n_p = res.n_parameters
        expected_aic = ofv + 2.0 * n_p
        np.testing.assert_allclose(res.aic, expected_aic, rtol=1e-12)

    def test_bic_formula_exact(self):
        """BIC = OFV + ln(n_obs) * n_params (exact)."""
        ofv, n_obs = 215.7, 120
        res = _make_result(ofv, n_theta=3, n_eta=2, n_eps=1, n_obs=n_obs)
        n_p = res.n_parameters
        expected_bic = ofv + math.log(n_obs) * n_p
        np.testing.assert_allclose(res.bic, expected_bic, rtol=1e-12)

    def test_bic_larger_than_aic_for_nobs_gt_8(self):
        """BIC > AIC when n_obs > e^2 ≈ 7.39 (Burnham & Anderson 2002, §6)."""
        res = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1, n_obs=100)
        assert res.bic > res.aic, f"BIC={res.bic:.2f} should exceed AIC={res.aic:.2f} for n=100"

    def test_bic_inf_when_no_observations(self):
        """BIC = inf when n_observations = 0 (formula undefined)."""
        res = _make_result(200.0, n_theta=2, n_eta=1, n_eps=1, n_obs=0)
        assert math.isinf(res.bic), "BIC should be inf when n_observations=0"

    def test_n_parameters_inferred_from_matrix_shapes(self):
        """
        n_parameters inferred: n_theta + lower-tri(omega) + lower-tri(sigma).

        For n_theta=3, n_eta=2 (2×2 OMEGA), n_eps=1 (1×1 SIGMA):
            n_params = 3 + (2*3/2) + (1*2/2) = 3 + 3 + 1 = 7
        """
        res = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        # 2×2 OMEGA lower-tri has 3 elements; 1×1 SIGMA has 1
        assert res.n_parameters == 7, f"Expected n_params=7, got {res.n_parameters}"

    def test_lower_ofv_wins_on_aic_for_equal_params(self):
        """With equal n_params, model with lower OFV has lower AIC and BIC."""
        full = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        alt = _make_result(210.0, n_theta=3, n_eta=2, n_eps=1)
        assert full.aic < alt.aic
        assert full.bic < alt.bic

    def test_aic_penalises_extra_params(self):
        """AIC penalises extra parameters: equal OFV + 1 extra param raises AIC by 2."""
        simple = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        complex_ = _make_result(200.0, n_theta=4, n_eta=2, n_eps=1)
        aic_diff = complex_.aic - simple.aic
        assert abs(aic_diff - 2.0) < 1e-10, (
            f"One extra free param should raise AIC by 2; got {aic_diff:.4f}"
        )


# ---------------------------------------------------------------------------
# Layer C: Akaike weights vs Burnham & Anderson (2002) formula
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestAkaikeWeights:
    """
    Validate aic_weights() against the Burnham & Anderson (2002) formula.

    w_i = exp(-0.5 · ΔAIC_i) / Σ_j exp(-0.5 · ΔAIC_j)
    """

    def test_single_model_weight_is_one(self):
        """With a single model the Akaike weight must be 1.0."""
        from openpkpd.inference import aic_weights

        res = [_make_result(200.0)]
        w = aic_weights(res)
        np.testing.assert_allclose(w, [1.0], atol=1e-12)

    def test_two_equal_aic_weights_are_half(self):
        """Two models with identical AIC should each have weight 0.5."""
        from openpkpd.inference import aic_weights

        # same n_theta/n_eta/n_eps → same n_params → same AIC if same OFV
        r1 = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        r2 = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        w = aic_weights([r1, r2])
        np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-12)

    def test_weights_sum_to_one(self):
        """Akaike weights must sum to 1 regardless of number of models."""
        from openpkpd.inference import aic_weights

        results = [
            _make_result(200.0, n_theta=3, n_eta=2, n_eps=1),
            _make_result(205.0, n_theta=3, n_eta=2, n_eps=1),
            _make_result(210.0, n_theta=4, n_eta=2, n_eps=1),
            _make_result(215.0, n_theta=2, n_eta=1, n_eps=1),
        ]
        w = aic_weights(results)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_best_model_has_largest_weight(self):
        """The model with the lowest AIC should have the highest weight."""
        from openpkpd.inference import aic_weights

        # r1 has OFV=200 with same params → lowest AIC
        r1 = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        r2 = _make_result(210.0, n_theta=3, n_eta=2, n_eps=1)
        r3 = _make_result(220.0, n_theta=3, n_eta=2, n_eps=1)
        w = aic_weights([r1, r2, r3])
        assert w[0] == w.max(), "Best model (lowest AIC) should have highest weight"

    def test_weights_formula_exact(self):
        """Weights match manual Burnham & Anderson formula to floating-point precision."""
        from openpkpd.inference import aic_weights

        r1 = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        r2 = _make_result(206.0, n_theta=3, n_eta=2, n_eps=1)

        aic1 = r1.aic
        aic2 = r2.aic
        delta1 = aic1 - min(aic1, aic2)
        delta2 = aic2 - min(aic1, aic2)
        raw1 = math.exp(-0.5 * delta1)
        raw2 = math.exp(-0.5 * delta2)
        total = raw1 + raw2
        expected = np.array([raw1 / total, raw2 / total])

        w = aic_weights([r1, r2])
        np.testing.assert_allclose(w, expected, rtol=1e-12)

    def test_large_delta_aic_gives_near_zero_weight(self):
        """Model with ΔAIC > 20 should have negligible weight (< 0.0001)."""
        from openpkpd.inference import aic_weights

        r_best = _make_result(200.0, n_theta=3, n_eta=2, n_eps=1)
        r_poor = _make_result(220.0, n_theta=3, n_eta=2, n_eps=1)
        w = aic_weights([r_best, r_poor])
        assert w[1] < 0.0001, f"Weight of poor model (ΔAIC≈20) should be < 0.0001; got {w[1]:.6f}"


# ---------------------------------------------------------------------------
# Layer D: compare_models table properties
# ---------------------------------------------------------------------------


@pytest.mark.external_validation
class TestCompareModels:
    """
    Validate compare_models() output properties (sorting, dOFV, dAIC columns).
    """

    def test_sorted_by_aic_ascending(self):
        """compare_models output must be sorted by AIC ascending."""
        from openpkpd.inference.model_comparison import compare_models

        results = [
            _make_result(210.0, n_theta=3, n_eta=2, n_eps=1),
            _make_result(200.0, n_theta=3, n_eta=2, n_eps=1),
            _make_result(220.0, n_theta=3, n_eta=2, n_eps=1),
        ]
        df = compare_models(results, labels=["M3", "M1", "M2"])
        assert list(df["AIC"]) == sorted(df["AIC"].tolist()), (
            "compare_models must return rows sorted by AIC ascending"
        )

    def test_best_model_has_zero_dofv_and_daic(self):
        """Best model should have dOFV=0 and dAIC=0."""
        from openpkpd.inference.model_comparison import compare_models

        results = [
            _make_result(200.0, n_theta=3, n_eta=2, n_eps=1),
            _make_result(210.0, n_theta=3, n_eta=2, n_eps=1),
        ]
        df = compare_models(results)
        best_row = df.iloc[0]
        np.testing.assert_allclose(best_row["dOFV"], 0.0, atol=1e-10)
        np.testing.assert_allclose(best_row["dAIC"], 0.0, atol=1e-10)

    def test_returns_expected_columns(self):
        """compare_models must return the documented column set."""
        from openpkpd.inference.model_comparison import compare_models

        results = [_make_result(200.0)]
        df = compare_models(results)
        expected_cols = {"Model", "OFV", "n_params", "AIC", "BIC", "dOFV", "dAIC"}
        assert expected_cols.issubset(df.columns), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_empty_list_returns_empty_dataframe(self):
        """compare_models([]) should return an empty DataFrame without error."""
        from openpkpd.inference.model_comparison import compare_models

        df = compare_models([])
        assert len(df) == 0
