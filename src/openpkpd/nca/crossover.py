"""
Mixed-model crossover bioequivalence analysis.

Implements least-squares ANOVA for 2×2 (and higher-order) crossover designs,
along with power and sample size calculations using the non-central t distribution.

References:
    Chow, S.C. & Liu, J.P. (2009). Design and Analysis of Bioavailability and
        Bioequivalence Studies, 3rd ed. CRC Press.
    FDA Guidance: Statistical Approaches to Establishing Bioequivalence (2001).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CrossoverResult:
    """
    Result of a crossover bioequivalence analysis.

    Attributes:
        treatment_diff:   Estimated log-scale treatment difference (test - reference).
        se:               Standard error of the treatment difference.
        ci_lo:            Lower confidence bound (back-transformed to ratio scale).
        ci_hi:            Upper confidence bound (back-transformed to ratio scale).
        df:               Error degrees of freedom.
        p_value:          Two-sided p-value for treatment difference = 0.
        gmr:              Geometric mean ratio = exp(treatment_diff).
        sequence_effect:  Estimated sequence effect (log scale).
        period_effects:   Array of period effects (log scale).
        carryover_effect: Estimated carryover effect or None if not tested.
        bioequivalent:    True if CI lies within [exp(be_lower), exp(be_upper)].
        be_lower:         Log-scale lower BE limit.
        be_upper:         Log-scale upper BE limit.
    """

    treatment_diff: float
    se: float
    ci_lo: float
    ci_hi: float
    df: float
    p_value: float
    gmr: float
    sequence_effect: float
    period_effects: np.ndarray
    carryover_effect: float | None
    bioequivalent: bool
    be_lower: float
    be_upper: float

    def summary(self) -> str:
        status = "BIOEQUIVALENT" if self.bioequivalent else "NOT bioequivalent"
        lines = [
            "Crossover Bioequivalence Analysis",
            f"  Status:       {status}",
            f"  GMR:          {self.gmr:.4f}",
            f"  90% CI:       [{self.ci_lo:.4f}, {self.ci_hi:.4f}]",
            f"  BE limits:    [{math.exp(self.be_lower):.2f}, {math.exp(self.be_upper):.2f}]",
            f"  log diff:     {self.treatment_diff:.4f} ± {self.se:.4f}",
            f"  df:           {self.df:.1f}",
            f"  p-value:      {self.p_value:.4f}",
        ]
        return "\n".join(lines)


def crossover_be_analysis(
    df: pd.DataFrame,
    subject_col: str = "subject",
    sequence_col: str = "sequence",
    period_col: str = "period",
    treatment_col: str = "treatment",
    metric_col: str = "log_metric",
    ci_level: float = 0.90,
    be_lower: float = math.log(0.80),
    be_upper: float = math.log(1.25),
    test_carryover: bool = True,
) -> CrossoverResult:
    """
    Least-squares ANOVA crossover bioequivalence analysis.

    Fits a linear model:
        log_metric ~ sequence + subject(sequence) + period + treatment [+ carryover]

    Subject is treated as a random blocking factor within sequence.
    Uses ordinary least squares with subject-mean centering.

    The standard 2×2 crossover design has:
    - 2 sequences: TR (test then reference), RT (reference then test)
    - 2 periods
    - Test bioequivalent if 90% CI for GMR lies in [0.80, 1.25]

    Args:
        df:            DataFrame with columns for subject, sequence, period,
                       treatment, and the log-transformed metric.
        subject_col:   Column name for subject identifier.
        sequence_col:  Column name for sequence group.
        period_col:    Column name for period (integer, 1-based).
        treatment_col: Column name for treatment (values used as test/reference).
        metric_col:    Column name for log-transformed PK metric.
        ci_level:      Confidence level (default 0.90 for 90% CI).
        be_lower:      Log-scale lower BE limit (default log(0.80)).
        be_upper:      Log-scale upper BE limit (default log(1.25)).
        test_carryover: If True, include carryover effect in the model.

    Returns:
        CrossoverResult with treatment difference, CI, and BE decision.

    Raises:
        ValueError: If required columns are missing or data is insufficient.
    """
    required = [subject_col, sequence_col, period_col, treatment_col, metric_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df[metric_col] = df[metric_col].astype(float)

    subjects = df[subject_col].unique()
    sequences = sorted(df[sequence_col].unique())
    periods = sorted(df[period_col].unique())
    treatments = sorted(df[treatment_col].unique())

    if len(treatments) < 2:
        raise ValueError("Need at least 2 treatments.")

    n_subj = len(subjects)
    n_seq = len(sequences)
    n_period = len(periods)

    subj_idx = {s: i for i, s in enumerate(subjects)}
    seq_idx = {s: i for i, s in enumerate(sequences)}
    period_idx = {p: i for i, p in enumerate(periods)}

    # Reference treatment is the first alphabetically / sorted
    treatments[0]
    test_treatment = treatments[-1]

    n_obs = len(df)

    # Build design matrix columns:
    # intercept, sequence (n_seq-1 dummies), period (n_period-1 dummies),
    # treatment (1 dummy: test vs ref), carryover (optional),
    # subject fixed effects (n_subj - 1) as blocking within sequence

    # We use subject mean-centering (within-subject design):
    # For a 2x2 crossover, the standard ANOVA approach:
    # - Intra-subject effect = treatment difference from within-subject contrast

    y = df[metric_col].values.astype(float)

    # Build design matrix for fixed effects: intercept, period, treatment, [carryover]
    n_period_dum = n_period - 1
    n_seq_dum = n_seq - 1
    n_treat_dum = 1  # binary: test=1, ref=0
    n_carryover = 1 if test_carryover and n_period > 1 else 0

    n_cols = 1 + n_seq_dum + n_period_dum + n_treat_dum + n_carryover + (n_subj - 1)
    X = np.zeros((n_obs, n_cols))

    # Intercept
    X[:, 0] = 1.0

    col = 1
    # Sequence dummies (drop first)
    for i, row in enumerate(df.itertuples(index=False)):
        seq = getattr(row, sequence_col)
        si = seq_idx[seq]
        if si > 0:
            X[i, col + si - 1] = 1.0
    col += n_seq_dum

    # Period dummies (drop first)
    for i, row in enumerate(df.itertuples(index=False)):
        per = getattr(row, period_col)
        pi = period_idx[per]
        if pi > 0:
            X[i, col + pi - 1] = 1.0
    col += n_period_dum

    # Treatment (test=1, ref=0)
    treat_col_idx = col
    for i, row in enumerate(df.itertuples(index=False)):
        trt = getattr(row, treatment_col)
        X[i, col] = 1.0 if trt == test_treatment else 0.0
    col += 1

    # Carryover (1 if period > 1 and received test in previous period)
    carryover_col_idx = None
    if n_carryover > 0:
        carryover_col_idx = col
        # Build carryover: 1 if previous period treatment was test
        for i, row in enumerate(df.itertuples(index=False)):
            subj = getattr(row, subject_col)
            per = getattr(row, period_col)
            trt = getattr(row, treatment_col)
            if per > min(periods):
                # Find previous period for this subject
                prev_per = periods[period_idx[per] - 1]
                prev_rows = df[(df[subject_col] == subj) & (df[period_col] == prev_per)]
                if len(prev_rows) > 0:
                    prev_trt = prev_rows[treatment_col].iloc[0]
                    X[i, col] = 1.0 if prev_trt == test_treatment else 0.0
        col += 1

    # Subject dummies (drop first subject as reference)
    for i, row in enumerate(df.itertuples(index=False)):
        subj = getattr(row, subject_col)
        si = subj_idx[subj]
        if si > 0:
            X[i, col + si - 1] = 1.0

    # OLS via lstsq
    result_ls = np.linalg.lstsq(X, y, rcond=None)
    beta = result_ls[0]
    y_pred = X @ beta
    residuals = y - y_pred
    df_error = max(n_obs - n_cols, 1)
    mse = float(np.sum(residuals**2) / df_error)

    # SE of treatment effect
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_treat = float(np.sqrt(mse * XtX_inv[treat_col_idx, treat_col_idx]))
    except Exception:
        se_treat = float("nan")

    treatment_diff = float(beta[treat_col_idx])
    gmr = float(np.exp(treatment_diff))

    # Confidence interval
    alpha = 1.0 - ci_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_error))
    if np.isfinite(se_treat):
        ci_lo_log = treatment_diff - t_crit * se_treat
        ci_hi_log = treatment_diff + t_crit * se_treat
    else:
        ci_lo_log = float("-inf")
        ci_hi_log = float("inf")
    ci_lo = float(np.exp(ci_lo_log))
    ci_hi = float(np.exp(ci_hi_log))

    # p-value
    if np.isfinite(se_treat) and se_treat > 0:
        t_stat = treatment_diff / se_treat
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df_error))
    else:
        p_value = float("nan")

    # Sequence effect
    seq_effect = float(beta[1]) if n_seq_dum > 0 else 0.0

    # Period effects
    period_effects = np.zeros(n_period)
    for pi in range(1, n_period):
        period_effects[pi] = float(beta[1 + n_seq_dum + pi - 1])

    # Carryover effect
    carryover_effect = None
    if carryover_col_idx is not None:
        carryover_effect = float(beta[carryover_col_idx])

    # BE decision
    bioequivalent = bool(ci_lo_log >= be_lower and ci_hi_log <= be_upper)

    return CrossoverResult(
        treatment_diff=treatment_diff,
        se=se_treat,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        df=float(df_error),
        p_value=p_value,
        gmr=gmr,
        sequence_effect=seq_effect,
        period_effects=period_effects,
        carryover_effect=carryover_effect,
        bioequivalent=bioequivalent,
        be_lower=be_lower,
        be_upper=be_upper,
    )


def be_power(
    cv: float,
    n_per_seq: int,
    theta_lo: float = -0.223,
    theta_hi: float = 0.223,
    alpha: float = 0.05,
    design: str = "2x2",
    true_ratio: float = 1.0,
) -> float:
    """
    Compute power for a 2-period crossover BE study.

    Uses the exact non-central t distribution approach (TOST).

    Args:
        cv:           Intra-subject coefficient of variation (e.g. 0.20 for 20%).
        n_per_seq:    Number of subjects per sequence.
        theta_lo:     Log-scale lower BE limit (default log(0.80) ≈ -0.2231).
        theta_hi:     Log-scale upper BE limit (default log(1.25) ≈  0.2231).
        alpha:        One-sided significance level (default 0.05).
        design:       '2x2' (only supported currently).
        true_ratio:   True GMR (e.g. 1.0 for identical formulations).

    Returns:
        Statistical power as a float in [0, 1].
    """
    if design != "2x2":
        raise ValueError("Only '2x2' design is currently supported.")

    # Within-subject sigma from CV: sigma = sqrt(log(CV^2 + 1)) ≈ CV for small CV
    sigma_w = float(np.sqrt(np.log(cv**2 + 1.0)))
    true_diff = float(np.log(true_ratio))

    # Total n = 2 * n_per_seq for 2x2 crossover
    n_total = 2 * n_per_seq
    # Standard error of difference = sigma_w * sqrt(2 / n_total)
    se = sigma_w * np.sqrt(2.0 / n_total)
    df = n_total - 2  # df for 2x2 crossover

    if df <= 0 or se <= 0:
        return 0.0

    t_crit = float(stats.t.ppf(1.0 - alpha, df=df))

    # Non-centrality parameters
    ncp_lo = (true_diff - theta_lo) / se
    ncp_hi = (theta_hi - true_diff) / se

    # Power = P(T > t_crit | ncp_lo) + P(T > t_crit | ncp_hi) - 1
    # Using non-central t survival functions
    power_lo = float(stats.nct.sf(t_crit, df=df, nc=ncp_lo))
    power_hi = float(stats.nct.sf(t_crit, df=df, nc=ncp_hi))

    power = float(max(0.0, power_lo + power_hi - 1.0))
    return power


def be_sample_size(
    cv: float,
    power: float = 0.80,
    theta_lo: float = -0.223,
    theta_hi: float = 0.223,
    alpha: float = 0.05,
    design: str = "2x2",
    true_ratio: float = 1.0,
    max_n: int = 500,
) -> int:
    """
    Compute the minimum sample size per sequence for a crossover BE study.

    Grid-searches over n_per_seq until the desired power is achieved.

    Args:
        cv:        Intra-subject CV (e.g. 0.20 for 20%).
        power:     Desired power (e.g. 0.80).
        theta_lo:  Log-scale lower BE limit.
        theta_hi:  Log-scale upper BE limit.
        alpha:     One-sided significance level.
        design:    '2x2' (only supported currently).
        true_ratio: True GMR.
        max_n:     Maximum n per sequence to search.

    Returns:
        Minimum n_per_seq to achieve the desired power. Returns max_n+1
        if power is not achievable within max_n.
    """
    for n in range(2, max_n + 1):
        p = be_power(cv, n, theta_lo, theta_hi, alpha, design, true_ratio)
        if p >= power:
            return n
    return max_n + 1
