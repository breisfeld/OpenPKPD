"""
Bioequivalence analysis using the standard Average Bioequivalence (ABE) approach.

Implements the two one-sided t-tests (TOST) on log-transformed data, which
is the regulatory standard for NCA-based bioequivalence studies (FDA, EMA).

References:
    FDA Guidance: Statistical Approaches to Establishing Bioequivalence (2001).
    Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure
        and the power approach for assessing the equivalence of average
        bioavailability. J Pharmacokinet Biopharm 15(6):657-680.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _ge_with_tol(value: float, lower: float, tol: float = 1e-12) -> bool:
    """Closed lower-bound comparison tolerant to roundoff."""
    return bool(value >= lower or np.isclose(value, lower, rtol=tol, atol=tol))


def _le_with_tol(value: float, upper: float, tol: float = 1e-12) -> bool:
    """Closed upper-bound comparison tolerant to roundoff."""
    return bool(value <= upper or np.isclose(value, upper, rtol=tol, atol=tol))


def _within_closed_interval(value: float, lower: float, upper: float, tol: float = 1e-12) -> bool:
    """Return True when value is within [lower, upper] up to tiny roundoff."""
    return _ge_with_tol(value, lower, tol=tol) and _le_with_tol(value, upper, tol=tol)


def _interval_within_closed_interval(
    lower_value: float,
    upper_value: float,
    lower_bound: float,
    upper_bound: float,
    tol: float = 1e-12,
) -> bool:
    """Return True when [lower_value, upper_value] lies in [lower_bound, upper_bound]."""
    return _ge_with_tol(lower_value, lower_bound, tol=tol) and _le_with_tol(
        upper_value, upper_bound, tol=tol
    )


@dataclass
class BEResult:
    """
    Result of a bioequivalence analysis.

    Attributes:
        metric:       PK metric analysed, e.g. 'AUC' or 'Cmax'.
        gmr:          Geometric Mean Ratio (test / reference).
        gmr_ci_lo:    Lower bound of the confidence interval for GMR.
        gmr_ci_hi:    Upper bound of the confidence interval for GMR.
        bioequivalent: True if the entire CI lies within (be_lower, be_upper).
        n_subjects:   Number of paired observations.
        ci_level:     Nominal confidence level used (e.g. 0.90).
        be_lower:     Lower bioequivalence limit (default 0.80).
        be_upper:     Upper bioequivalence limit (default 1.25).
        p_lower:      p-value for the lower one-sided t-test (H0: GMR <= be_lower).
        p_upper:      p-value for the upper one-sided t-test (H0: GMR >= be_upper).
        df:           Degrees of freedom of the t-distribution.
        log_mean_diff: Mean difference on the log scale (log(test) - log(ref)).
        log_se:       Standard error of the log mean difference.
    """

    metric: str
    gmr: float
    gmr_ci_lo: float
    gmr_ci_hi: float
    bioequivalent: bool
    n_subjects: int
    ci_level: float = 0.90
    be_lower: float = 0.80
    be_upper: float = 1.25
    p_lower: float = float("nan")
    p_upper: float = float("nan")
    df: float = float("nan")
    log_mean_diff: float = float("nan")
    log_se: float = float("nan")

    def summary(self) -> str:
        """Return a formatted text summary of the bioequivalence result."""
        ci_pct = int(round(self.ci_level * 100))
        status = "BIOEQUIVALENT" if self.bioequivalent else "NOT bioequivalent"
        lines = [
            f"Average Bioequivalence — {self.metric}",
            f"  Status:       {status}",
            f"  GMR:          {self.gmr:.4f}",
            f"  {ci_pct}% CI:      [{self.gmr_ci_lo:.4f}, {self.gmr_ci_hi:.4f}]",
            f"  BE limits:    [{self.be_lower:.2f}, {self.be_upper:.2f}]",
            f"  n subjects:   {self.n_subjects}",
            f"  p (lower):    {self.p_lower:.4f}",
            f"  p (upper):    {self.p_upper:.4f}",
            f"  df:           {self.df:.1f}",
        ]
        return "\n".join(lines)


def average_bioequivalence(
    test_values: np.ndarray,
    reference_values: np.ndarray,
    metric: str = "AUC",
    ci_level: float = 0.90,
    be_lower: float = 0.80,
    be_upper: float = 1.25,
) -> BEResult:
    """
    Standard Average Bioequivalence (ABE) test.

    Computes the two one-sided t-tests (TOST) on log-transformed values.
    This is equivalent to constructing a (1 - 2*alpha) confidence interval
    for the geometric mean ratio (GMR) and checking whether it falls
    entirely within [be_lower, be_upper].

    The paired design is assumed: each element in test_values corresponds
    to the same subject in reference_values. For a two-period crossover,
    pass the intra-individual differences directly (test_values[i] is
    subject i's measurement on the test formulation).

    Statistical model (log-scale):
        d_i = log(test_i) - log(ref_i)
        d_i ~ N(mu, sigma^2)
        H0_lower: mu <= log(be_lower)   vs  H1_lower: mu > log(be_lower)
        H0_upper: mu >= log(be_upper)   vs  H1_upper: mu < log(be_upper)
        Bioequivalent if both H0's are rejected at alpha = 1 - ci_level.

    Args:
        test_values:      Array of NCA metric values for the test formulation.
        reference_values: Array of NCA metric values for the reference formulation.
                          Must have the same length as test_values.
        metric:           Label for the PK metric (for reporting).
        ci_level:         Nominal confidence level for the GMR confidence
                          interval (default 0.90 → 90% CI).
        be_lower:         Lower bioequivalence limit (default 0.80 = 80%).
        be_upper:         Upper bioequivalence limit (default 1.25 = 125%).

    Returns:
        BEResult with GMR, CI, and bioequivalence decision.

    Raises:
        ValueError: If arrays have different lengths or fewer than 2 paired
                    observations, or if any value is non-positive.
    """
    test_values = np.asarray(test_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)

    if test_values.shape != reference_values.shape:
        raise ValueError("test_values and reference_values must have the same shape.")
    if test_values.ndim != 1:
        raise ValueError("test_values and reference_values must be 1-D arrays.")

    # Remove NaN pairs
    valid = np.isfinite(test_values) & np.isfinite(reference_values)
    t_vals = test_values[valid]
    r_vals = reference_values[valid]
    n = len(t_vals)

    if n < 2:
        raise ValueError(f"At least 2 valid paired observations are required; got {n}.")

    if np.any(t_vals <= 0) or np.any(r_vals <= 0):
        raise ValueError(
            "All test and reference values must be strictly positive for log-transformation."
        )

    # Log-transform and compute paired differences
    log_diff = np.log(t_vals) - np.log(r_vals)  # log(test/ref)
    mean_diff = float(np.mean(log_diff))
    sd_diff = float(np.std(log_diff, ddof=1))
    se_diff = sd_diff / np.sqrt(n)
    df = float(n - 1)

    # Geometric mean ratio
    gmr = float(np.exp(mean_diff))

    # Confidence interval bounds (on log scale, then back-transform)
    alpha = 1.0 - ci_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci_lo = float(np.exp(mean_diff - t_crit * se_diff))
    ci_hi = float(np.exp(mean_diff + t_crit * se_diff))

    # Two one-sided t-tests (TOST)
    # H0_lower: mu <= log(be_lower)  →  t_lower = (mean_diff - log(be_lower)) / se
    if se_diff > 0:
        t_lower = (mean_diff - np.log(be_lower)) / se_diff
        t_upper = (mean_diff - np.log(be_upper)) / se_diff

        # p-values: both one-sided tests in the favourable direction
        p_lower = float(stats.t.sf(t_lower, df=df))  # P(T > t_lower)
        p_upper = float(stats.t.cdf(t_upper, df=df))  # P(T < t_upper)
    else:
        # Degenerate case: all differences identical
        p_lower = 0.0 if mean_diff > np.log(be_lower) else 1.0
        p_upper = 0.0 if mean_diff < np.log(be_upper) else 1.0

    # Bioequivalence decision: entire CI within [be_lower, be_upper]
    bioequivalent = _interval_within_closed_interval(ci_lo, ci_hi, be_lower, be_upper)

    return BEResult(
        metric=metric,
        gmr=gmr,
        gmr_ci_lo=ci_lo,
        gmr_ci_hi=ci_hi,
        bioequivalent=bioequivalent,
        n_subjects=n,
        ci_level=ci_level,
        be_lower=be_lower,
        be_upper=be_upper,
        p_lower=p_lower,
        p_upper=p_upper,
        df=df,
        log_mean_diff=mean_diff,
        log_se=se_diff,
    )


@dataclass
class RSABEResult:
    """
    Result from Reference-Scaled Average Bioequivalence (RSABE) analysis.

    Attributes:
        metric:           PK metric analysed (e.g. 'AUC', 'Cmax').
        sigma_wr:         Within-subject SD of log-reference.
        scaled_criterion: Scaled criterion value (ln(GMR) scaled).
        upper_bound_ci:   Upper 95% confidence bound of the scaled criterion.
        bioequivalent:    True if upper_bound_ci < 0 (on log scale) and GMR in [0.80, 1.25].
        gmr:              Geometric mean ratio (test / reference).
        gmr_ci_lo:        Lower bound of GMR confidence interval.
        gmr_ci_hi:        Upper bound of GMR confidence interval.
        method:           Regulatory method: 'FDA' or 'EMA'.
        sigma_w0:         Reference sigma threshold used.
        used_abe:         True if sigma_wr <= sigma_w0 (fell back to standard ABE).
        n_subjects:       Number of subjects.
    """

    metric: str
    sigma_wr: float
    scaled_criterion: float
    upper_bound_ci: float
    bioequivalent: bool
    gmr: float
    gmr_ci_lo: float
    gmr_ci_hi: float
    method: str
    sigma_w0: float
    used_abe: bool
    n_subjects: int

    def summary(self) -> str:
        ci_pct = 90
        status = "BIOEQUIVALENT" if self.bioequivalent else "NOT bioequivalent"
        lines = [
            f"Reference-Scaled ABE ({self.method}) — {self.metric}",
            f"  Status:          {status}",
            f"  sigma_wr:        {self.sigma_wr:.4f}  (threshold: {self.sigma_w0:.3f})",
            f"  Used ABE:        {self.used_abe}",
            f"  GMR:             {self.gmr:.4f}",
            f"  {ci_pct}% CI:         [{self.gmr_ci_lo:.4f}, {self.gmr_ci_hi:.4f}]",
            f"  Scaled crit UB:  {self.upper_bound_ci:.4f}",
            f"  n subjects:      {self.n_subjects}",
        ]
        return "\n".join(lines)


def reference_scaled_abe(
    test: np.ndarray,
    reference_1: np.ndarray,
    reference_2: np.ndarray,
    metric: str = "AUC",
    ci_level: float = 0.90,
    sigma_w0: float = 0.25,
    regulatory: str = "FDA",
) -> RSABEResult:
    """
    Reference-Scaled Average Bioequivalence (RSABE) for highly variable drugs.

    Used for replicate crossover designs where subjects receive the reference
    treatment twice (R1, R2). The within-subject variability of the reference
    is used to scale the bioequivalence criterion.

    Algorithm (FDA):
      1. Compute sigma_wr = within-subject SD of log-reference from replicate values.
         sigma_wr = std(log(R1) - log(R2)) / sqrt(2)
      2. If sigma_wr <= sigma_w0 (0.25 for FDA, 0.294 for EMA):
         Fall back to standard ABE (this function calls average_bioequivalence internally).
      3. Else: compute scaled criterion = (ln GMR)^2 - theta^2 * sigma_wr^2
         where theta = ln(1.25) / sigma_w0.
         Upper 95% CI of this criterion must be < 0.
      4. Also enforce GMR in [0.80, 1.25].

    Args:
        test:        Test formulation values (one per subject), shape (n,).
        reference_1: First reference period values, shape (n,).
        reference_2: Second reference period values, shape (n,).
        metric:      Label for reporting.
        ci_level:    Confidence level (0.90 = 90% CI for scaled criterion).
        sigma_w0:    Reference SD threshold. FDA=0.25, EMA=0.294.
        regulatory:  'FDA' or 'EMA'.

    Returns:
        RSABEResult with decision and all intermediate values.

    Raises:
        ValueError: If arrays have different lengths or contain non-positive values.
    """
    test = np.asarray(test, dtype=float)
    reference_1 = np.asarray(reference_1, dtype=float)
    reference_2 = np.asarray(reference_2, dtype=float)

    if not (test.shape == reference_1.shape == reference_2.shape):
        raise ValueError("test, reference_1, reference_2 must have the same shape.")
    if test.ndim != 1:
        raise ValueError("Arrays must be 1-D.")

    # Validity check
    valid = np.isfinite(test) & np.isfinite(reference_1) & np.isfinite(reference_2)
    t = test[valid]
    r1 = reference_1[valid]
    r2 = reference_2[valid]
    n = len(t)

    if n < 3:
        raise ValueError(f"At least 3 valid subjects required; got {n}.")
    if np.any(t <= 0) or np.any(r1 <= 0) or np.any(r2 <= 0):
        raise ValueError("All values must be strictly positive for log-transform.")

    # Within-subject reference SD
    log_r_diff = np.log(r1) - np.log(r2)
    sigma_wr = float(np.std(log_r_diff, ddof=1) / np.sqrt(2.0))

    # EMA uses different threshold
    if regulatory == "EMA":
        sigma_w0 = 0.294

    # GMR from test vs mean of two references
    log_t = np.log(t)
    log_r_mean = (np.log(r1) + np.log(r2)) / 2.0
    log_diff = log_t - log_r_mean
    mean_log_diff = float(np.mean(log_diff))
    gmr = float(np.exp(mean_log_diff))
    se_log_diff = float(np.std(log_diff, ddof=1) / np.sqrt(n))
    df = float(n - 1)
    alpha = 1.0 - ci_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    gmr_ci_lo = float(np.exp(mean_log_diff - t_crit * se_log_diff))
    gmr_ci_hi = float(np.exp(mean_log_diff + t_crit * se_log_diff))

    # Fall back to ABE if sigma_wr <= sigma_w0
    if sigma_wr <= sigma_w0:
        # Standard 90% CI check
        bioequivalent = _interval_within_closed_interval(gmr_ci_lo, gmr_ci_hi, 0.80, 1.25)
        return RSABEResult(
            metric=metric,
            sigma_wr=sigma_wr,
            scaled_criterion=float("nan"),
            upper_bound_ci=float("nan"),
            bioequivalent=bioequivalent,
            gmr=gmr,
            gmr_ci_lo=gmr_ci_lo,
            gmr_ci_hi=gmr_ci_hi,
            method=regulatory,
            sigma_w0=sigma_w0,
            used_abe=True,
            n_subjects=n,
        )

    # RSABE: scaled criterion
    theta = np.log(1.25) / sigma_w0

    # Scaled criterion point estimate: (ln GMR)^2 - theta^2 * sigma_wr^2
    scaled_criterion = mean_log_diff**2 - theta**2 * sigma_wr**2

    # Upper 95% CI of scaled criterion using Howe's method approximation
    # Var(ln GMR) = se_log_diff^2 * n (total variance component)
    # Var(sigma_wr^2) approximated by chi-squared
    # Use linearization: U = scaled_criterion + t_{alpha,df} * sqrt(Var(U))
    # Simplified: upper bound using method of moments
    var_lnGMR = se_log_diff**2  # variance of mean log diff
    # Approximate variance of sigma_wr^2 using chi-sq: Var(s^2) = 2*s^4/(n-1)
    var_sigmawr2 = 2.0 * sigma_wr**4 / (n - 1)
    # Var(scaled_criterion) ≈ 4*lnGMR^2*Var(lnGMR) + theta^4 * Var(sigma_wr^2)
    var_scaled = 4.0 * mean_log_diff**2 * var_lnGMR + theta**4 * var_sigmawr2
    upper_bound_ci = scaled_criterion + t_crit * float(np.sqrt(max(var_scaled, 0.0)))

    # BE decision: upper bound < 0 AND GMR in [0.80, 1.25]
    bioequivalent = bool(upper_bound_ci < 0.0 and _within_closed_interval(gmr, 0.80, 1.25))

    return RSABEResult(
        metric=metric,
        sigma_wr=sigma_wr,
        scaled_criterion=float(scaled_criterion),
        upper_bound_ci=float(upper_bound_ci),
        bioequivalent=bioequivalent,
        gmr=gmr,
        gmr_ci_lo=gmr_ci_lo,
        gmr_ci_hi=gmr_ci_hi,
        method=regulatory,
        sigma_w0=sigma_w0,
        used_abe=False,
        n_subjects=n,
    )
