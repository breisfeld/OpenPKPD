"""
Tests for scalar LOQ overwrite guard in inject_scalar_lloq (D2).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from openpkpd.data.blq import inject_scalar_lloq


def _make_df(lloq_values=None):
    df = pd.DataFrame({"ID": [1, 2, 3], "DV": [0.5, 1.2, 0.8]})
    if lloq_values is not None:
        df["LLOQ"] = lloq_values
    return df


# ---------------------------------------------------------------------------
# Test 1: No existing LLOQ column -> no warning
# ---------------------------------------------------------------------------

def test_inject_no_existing_lloq_no_warning():
    """Injecting scalar LOQ when no LLOQ column exists produces no warning."""
    df = _make_df(lloq_values=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        result = inject_scalar_lloq(df, lloq_value=1.0)
    assert "LLOQ" in result.columns
    assert (result["LLOQ"] == 1.0).all()


# ---------------------------------------------------------------------------
# Test 2: Uniform LLOQ column -> no warning
# ---------------------------------------------------------------------------

def test_inject_uniform_lloq_no_warning():
    """Injecting scalar LOQ when LLOQ column is uniform (safe) -> no warning."""
    df = _make_df(lloq_values=[1.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = inject_scalar_lloq(df, lloq_value=1.0)
    assert (result["LLOQ"] == 1.0).all()


# ---------------------------------------------------------------------------
# Test 3: Mixed LLOQ values -> UserWarning mentioning "per-observation"
# ---------------------------------------------------------------------------

def test_inject_mixed_lloq_warns():
    """Injecting scalar LOQ over non-uniform LLOQ -> UserWarning."""
    df = _make_df(lloq_values=[0.5, 1.0, 2.0])
    with pytest.warns(UserWarning, match="per-observation"):
        result = inject_scalar_lloq(df, lloq_value=1.0)
    # After injection all values should be the scalar
    assert (result["LLOQ"] == 1.0).all()


# ---------------------------------------------------------------------------
# Test 4: Numerical M3 BLQ log-likelihood
# ---------------------------------------------------------------------------

def test_m3_blq_log_likelihood_numerical():
    """M3 BLQ: log_lik = norm.logcdf((lloq - ipred) / sigma) to within 0.001."""
    obs = 0.5
    ipred = 2.0
    sigma = 0.3
    lloq = 1.0

    expected = norm.logcdf((lloq - ipred) / sigma)

    from openpkpd.data.blq import blq_log_likelihood
    from openpkpd.utils.constants import BLQMethod

    result = blq_log_likelihood(
        y_obs=obs,
        mu=ipred,
        sigma2=sigma ** 2,
        lloq=lloq,
        method=BLQMethod.M3,
    )

    assert abs(result - expected) < 0.001, (
        f"M3 log-likelihood {result:.6f} != expected {expected:.6f}"
    )
