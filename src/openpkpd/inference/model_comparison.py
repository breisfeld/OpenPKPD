"""
Model comparison utilities: LRT, AIC/BIC comparison tables, Akaike weights.

Implements likelihood ratio testing (LRT) and tabular comparison of
multiple estimation results by AIC, BIC, and OFV, following standard
model selection practice in population PK/PD analysis.

References:
    Burnham, K.P. & Anderson, D.R. (2002). Model Selection and Multimodel
    Inference: A Practical Information-Theoretic Approach (2nd ed.).
    Springer-Verlag, New York.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from openpkpd.estimation.base import EstimationResult


@dataclass
class LRTResult:
    """
    Result of a likelihood ratio test (LRT) comparing two nested models.

    Attributes:
        ofv_full:    OFV (-2 * log-likelihood) of the full (more complex) model.
        ofv_reduced: OFV of the reduced (simpler/null) model.
        delta_ofv:   Test statistic: OFV_reduced - OFV_full. Under H0 this
                     follows a chi-squared distribution with ``df`` degrees
                     of freedom.
        df:          Degrees of freedom = n_params_full - n_params_reduced.
        p_value:     P-value from the chi-squared distribution.
        significant: True when p_value < alpha.
        alpha:       Significance level used for the decision (default 0.05).
    """

    ofv_full: float
    ofv_reduced: float
    delta_ofv: float
    df: int
    p_value: float
    significant: bool
    alpha: float = 0.05


def lrt(
    result_full: EstimationResult,
    result_reduced: EstimationResult,
    alpha: float = 0.05,
) -> LRTResult:
    """
    Likelihood ratio test comparing a full model against a reduced (nested) model.

    Under the null hypothesis (i.e. the restricted model is the true model),
    the test statistic

        delta_OFV = OFV_reduced - OFV_full

    is asymptotically chi-squared distributed with degrees of freedom equal
    to the difference in the number of estimated parameters.

    The models must be *nested*: the reduced model is a special case of the
    full model obtained by constraining one or more parameters.

    Args:
        result_full:    EstimationResult from the more complex (full) model.
        result_reduced: EstimationResult from the simpler (reduced) model.
        alpha:          Significance level for the binary decision (default 0.05).

    Returns:
        LRTResult containing the test statistic, degrees of freedom, p-value,
        and significance decision.

    Raises:
        ValueError: If delta_OFV is negative (full model should have lower OFV
                    than reduced model for nested comparisons) or if df <= 0.
    """
    delta_ofv = result_reduced.ofv - result_full.ofv
    df = result_full.n_parameters - result_reduced.n_parameters

    if df <= 0:
        raise ValueError(
            f"Degrees of freedom must be positive (got {df}). "
            f"The full model must have more parameters than the reduced model. "
            f"n_params_full={result_full.n_parameters}, "
            f"n_params_reduced={result_reduced.n_parameters}."
        )

    # In practice delta_ofv < 0 can happen when the optimizer converged to a local
    # minimum in the full model; we allow it but the p-value will be 1.
    p_value = 1.0 if delta_ofv < 0 else float(chi2.sf(delta_ofv, df))

    significant = p_value < alpha

    return LRTResult(
        ofv_full=result_full.ofv,
        ofv_reduced=result_reduced.ofv,
        delta_ofv=delta_ofv,
        df=df,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
    )


def compare_models(
    results: list[EstimationResult],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare multiple models by AIC, BIC, and OFV.

    Produces a summary DataFrame sorted by AIC (ascending). The best model
    (lowest AIC) is used as the reference for delta-OFV and delta-AIC columns.

    Args:
        results: List of EstimationResult objects. All results should use the
                 same dataset (same n_observations) for BIC to be comparable.
        labels:  Optional list of model names corresponding to each result.
                 Defaults to ``["Model_1", "Model_2", ...]``.

    Returns:
        A pandas DataFrame with columns:
            - Model:    Model label.
            - OFV:      Objective function value (-2 * log-likelihood).
            - n_params: Number of estimated parameters.
            - AIC:      Akaike Information Criterion.
            - BIC:      Bayesian Information Criterion.
            - dOFV:     OFV difference vs. the model with the lowest OFV.
            - dAIC:     AIC difference vs. the model with the lowest AIC.

        The DataFrame is sorted by AIC (ascending).

    Raises:
        ValueError: If ``labels`` is provided but its length does not match
                    the length of ``results``.
    """
    if not results:
        return pd.DataFrame(columns=["Model", "OFV", "n_params", "AIC", "BIC", "dOFV", "dAIC"])

    if labels is not None and len(labels) != len(results):
        raise ValueError(
            f"Length of labels ({len(labels)}) must match length of results ({len(results)})."
        )

    if labels is None:
        labels = [f"Model_{i + 1}" for i in range(len(results))]

    rows: list[dict] = []
    for label, res in zip(labels, results, strict=False):
        rows.append(
            {
                "Model": label,
                "OFV": res.ofv,
                "n_params": res.n_parameters,
                "AIC": res.aic,
                "BIC": res.bic,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("AIC", ascending=True).reset_index(drop=True)

    best_ofv = df["OFV"].min()
    best_aic = df["AIC"].min()
    df["dOFV"] = df["OFV"] - best_ofv
    df["dAIC"] = df["AIC"] - best_aic

    return df[["Model", "OFV", "n_params", "AIC", "BIC", "dOFV", "dAIC"]]


def aic_weights(results: list[EstimationResult]) -> np.ndarray:
    """
    Compute Akaike weights for model averaging.

    Akaike weights represent the probability that model *i* is the
    Kullback–Leibler best model among the candidate set:

        w_i = exp(-0.5 * delta_AIC_i) / sum_j exp(-0.5 * delta_AIC_j)

    where delta_AIC_i = AIC_i - min(AIC).

    Args:
        results: List of EstimationResult objects.

    Returns:
        NumPy array of Akaike weights, shape (len(results),), summing to 1.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results must be non-empty.")

    aics = np.array([r.aic for r in results], dtype=float)
    delta = aics - aics.min()
    raw = np.exp(-0.5 * delta)
    total = raw.sum()
    if total == 0.0:
        # Degenerate case: return uniform weights
        return np.ones(len(results)) / len(results)
    return raw / total
