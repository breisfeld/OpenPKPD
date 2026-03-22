"""Unit tests for TRANS parameter transformations."""

from __future__ import annotations

import pytest

from openpkpd.pk.trans import apply_trans
from openpkpd.pk.trans.trans6 import apply_trans6
from openpkpd.utils.errors import PKError


@pytest.mark.unit
def test_apply_trans1_returns_identity_copy():
    params = {"K": 0.1, "KA": 1.2}

    result = apply_trans(params, trans=1, advan=2)

    assert result == params
    assert result is not params


@pytest.mark.unit
def test_apply_trans2_computes_k_from_cl_over_v():
    params = {"CL": 1.5, "V": 12.0, "KA": 0.8}

    result = apply_trans(params, trans=2, advan=2)

    assert result["K"] == pytest.approx(1.5 / 12.0)
    assert result["V"] == pytest.approx(12.0)
    assert result["KA"] == pytest.approx(0.8)


@pytest.mark.unit
def test_apply_trans2_rejects_nonpositive_v():
    with pytest.raises(PKError, match="V must be > 0"):
        apply_trans({"CL": 1.0, "V": 0.0}, trans=2, advan=2)


@pytest.mark.unit
def test_apply_trans3_computes_micro_rates_when_v2_is_provided():
    params = {"CL": 1.0, "V": 10.0, "Q": 2.0, "V2": 20.0}

    result = apply_trans(params, trans=3, advan=3)

    assert result["K"] == pytest.approx(0.1)
    assert result["K12"] == pytest.approx(0.2)
    assert result["K21"] == pytest.approx(0.1)


@pytest.mark.unit
def test_apply_trans3_preserves_explicit_k21_when_v2_is_missing():
    params = {"CL": 1.0, "V": 10.0, "Q": 2.0, "K21": 0.33}

    result = apply_trans(params, trans=3, advan=3)

    assert result["K"] == pytest.approx(0.1)
    assert result["K12"] == pytest.approx(0.2)
    assert result["K21"] == pytest.approx(0.33)


@pytest.mark.unit
def test_apply_trans3_requires_v2_or_k21():
    with pytest.raises(PKError, match="requires V2 or K21"):
        apply_trans({"CL": 1.0, "V": 10.0, "Q": 2.0}, trans=3, advan=3)


@pytest.mark.unit
def test_apply_trans5_returns_identity_copy():
    params = {"K": 0.1, "K12": 0.2, "K21": 0.3, "K13": 0.4, "K31": 0.5}

    result = apply_trans(params, trans=5, advan=11)

    assert result == params
    assert result is not params


@pytest.mark.unit
def test_apply_trans6_accepts_zero_q2_and_computes_zero_micro_rates():
    params = {
        "CL": 1.0,
        "V1": 10.0,
        "Q2": 0.0,
        "V2": 20.0,
        "Q3": 2.0,
        "V3": 30.0,
    }

    result = apply_trans6(params)

    assert result["K"] == pytest.approx(0.1)
    assert result["K12"] == pytest.approx(0.0)
    assert result["K21"] == pytest.approx(0.0)
    assert result["K13"] == pytest.approx(0.2)
    assert result["K31"] == pytest.approx(2.0 / 30.0)


@pytest.mark.unit
def test_apply_trans6_prefers_explicit_zero_q2_over_q_alias():
    params = {
        "CL": 1.0,
        "V1": 10.0,
        "Q2": 0.0,
        "Q": 4.0,
        "V2": 20.0,
        "Q3": 2.0,
        "V3": 30.0,
    }

    result = apply_trans6(params)

    assert result["K12"] == pytest.approx(0.0)
    assert result["K21"] == pytest.approx(0.0)


@pytest.mark.unit
def test_apply_trans_routes_trans6_with_v_alias_and_zero_q2():
    params = {
        "CL": 1.0,
        "V": 10.0,
        "Q2": 0.0,
        "V2": 20.0,
        "Q3": 2.0,
        "V3": 30.0,
    }

    result = apply_trans(params, trans=6, advan=11)

    assert result["K"] == pytest.approx(0.1)
    assert result["K12"] == pytest.approx(0.0)
    assert result["K21"] == pytest.approx(0.0)
    assert result["K13"] == pytest.approx(0.2)
    assert result["K31"] == pytest.approx(2.0 / 30.0)


@pytest.mark.unit
def test_apply_trans4_rejects_explicit_zero_v1_instead_of_falling_back_to_v_alias():
    params = {
        "CL": 1.0,
        "V1": 0.0,
        "V": 10.0,
        "Q": 2.0,
        "V2": 20.0,
    }

    with pytest.raises(PKError, match="V1 must be > 0"):
        apply_trans(params, trans=4, advan=3)


@pytest.mark.unit
def test_apply_trans4_uses_v_alias_when_v1_is_missing():
    params = {
        "CL": 1.0,
        "V": 10.0,
        "Q": 2.0,
        "V2": 20.0,
    }

    result = apply_trans(params, trans=4, advan=3)

    assert result["V1"] == pytest.approx(10.0)
    assert result["K"] == pytest.approx(0.1)
    assert result["K12"] == pytest.approx(0.2)
    assert result["K21"] == pytest.approx(0.1)


@pytest.mark.unit
def test_apply_trans4_supports_advan4_v2_v3_layout():
    params = {
        "KA": 1.2,
        "CL": 1.0,
        "V2": 10.0,
        "Q": 2.0,
        "V3": 20.0,
    }

    result = apply_trans(params, trans=4, advan=4)

    assert result["V2"] == pytest.approx(10.0)
    assert result["V3"] == pytest.approx(20.0)
    assert result["K"] == pytest.approx(0.1)
    assert result["K12"] == pytest.approx(0.2)
    assert result["K21"] == pytest.approx(0.1)
