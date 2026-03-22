"""
Unit tests for the openpkpd.library pre-built model library.

Tests cover:
  - list_models(): returns a non-empty sorted list with expected entries.
  - get_model(): returns a ModelBuilder instance for all registered models.
  - show_model(): returns a meaningful docstring.
  - Individual factory functions: callable without data, return ModelBuilder.
  - Error handling: unknown model name raises KeyError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd.api.model_builder import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.library import get_model, list_models, show_model

# ── list_models ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_list_models_returns_list():
    """list_models() should return a Python list."""
    result = list_models()
    assert isinstance(result, list)


@pytest.mark.unit
def test_list_models_minimum_count():
    """Library must contain at least 5 pre-built models."""
    models = list_models()
    assert len(models) >= 5, f"Expected >= 5 models, got {len(models)}: {models}"


@pytest.mark.unit
def test_list_models_contains_pk_models():
    """Core PK model names must be present in the registry."""
    models = list_models()
    for expected in ("one_cmt_oral", "one_cmt_iv", "two_cmt_iv", "two_cmt_oral"):
        assert expected in models, f"Expected '{expected}' in list_models()"


@pytest.mark.unit
def test_list_models_contains_pd_models():
    """Core PD model names must be present in the registry."""
    models = list_models()
    for expected in ("emax_direct", "sigmoid_emax", "inhibitory_emax"):
        assert expected in models, f"Expected '{expected}' in list_models()"


@pytest.mark.unit
def test_list_models_is_sorted():
    """list_models() should return names in alphabetical order."""
    models = list_models()
    assert models == sorted(models), "list_models() should be sorted alphabetically"


# ── get_model ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_model_returns_model_builder():
    """get_model() for a valid name should return a ModelBuilder."""
    model = get_model("one_cmt_oral")
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_get_unknown_model_raises_key_error():
    """get_model() with an unregistered name should raise KeyError."""
    with pytest.raises(KeyError, match="nonexistent_model"):
        get_model("nonexistent_model")


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name",
    [
        "one_cmt_iv",
        "one_cmt_oral",
        "two_cmt_iv",
        "two_cmt_oral",
        "three_cmt_iv",
        "emax_direct",
        "sigmoid_emax",
        "inhibitory_emax",
        "indirect_response_type_i",
        "effect_compartment",
    ],
)
def test_get_model_all_registered(model_name: str):
    """Every registered model name should return a valid ModelBuilder via get_model()."""
    model = get_model(model_name)
    assert isinstance(model, ModelBuilder), (
        f"get_model('{model_name}') did not return a ModelBuilder"
    )


# ── show_model ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_show_model_returns_string():
    """show_model() should return a non-empty string."""
    doc = show_model("one_cmt_oral")
    assert isinstance(doc, str)
    assert len(doc) > 10, "Docstring is unexpectedly short"


@pytest.mark.unit
def test_show_model_contains_model_description():
    """show_model() for one_cmt_oral should mention key PK concepts."""
    doc = show_model("one_cmt_oral")
    # One of these keywords should appear in the docstring
    keywords = ["one-compartment", "ADVAN2", "absorption", "oral"]
    assert any(kw.lower() in doc.lower() for kw in keywords), (
        f"Expected one of {keywords} in show_model('one_cmt_oral'), got:\n{doc}"
    )


@pytest.mark.unit
def test_show_model_unknown_returns_message():
    """show_model() for an unknown name should return an informative message, not raise."""
    result = show_model("completely_unknown_model")
    assert "unknown" in result.lower() or "not found" in result.lower() or "Unknown" in result


# ── Individual factory functions ──────────────────────────────────────────────


@pytest.mark.unit
def test_one_cmt_iv():
    """one_cmt_iv() returns a ModelBuilder without data."""
    from openpkpd.library import one_cmt_iv

    model = one_cmt_iv()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_one_cmt_oral():
    """one_cmt_oral() returns a ModelBuilder without data."""
    from openpkpd.library import one_cmt_oral

    model = one_cmt_oral()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_two_cmt_iv():
    """two_cmt_iv() returns a ModelBuilder without data."""
    from openpkpd.library import two_cmt_iv

    model = two_cmt_iv()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_two_cmt_oral():
    """two_cmt_oral() returns a ModelBuilder without data."""
    from openpkpd.library import two_cmt_oral

    model = two_cmt_oral()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_two_cmt_oral_library_model_evaluates_predictions() -> None:
    from openpkpd.library import two_cmt_oral

    dataset = NONMEMDataset.from_dataframe(
        pd.DataFrame(
            [
                {"ID": 1, "TIME": 0.0, "AMT": 100.0, "DV": 0.0, "EVID": 1, "MDV": 1},
                {"ID": 1, "TIME": 0.5, "AMT": 0.0, "DV": 5.8, "EVID": 0, "MDV": 0},
                {"ID": 1, "TIME": 1.0, "AMT": 0.0, "DV": 7.1, "EVID": 0, "MDV": 0},
                {"ID": 1, "TIME": 2.0, "AMT": 0.0, "DV": 7.6, "EVID": 0, "MDV": 0},
            ]
        )
    )
    built = two_cmt_oral().dataset(dataset).build()
    indiv = built.population_model.individual_model(1)

    ipred, obs_mask, f = indiv.evaluate(
        np.array([1.2, 1.9, 13.0, 0.65, 17.0, 0.1]),
        np.array([0.04, -0.05, 0.06]),
        np.array([[0.02]]),
        trans=built.population_model.trans,
    )

    assert obs_mask.tolist() == [True, True, True]
    assert np.all(np.isfinite(ipred))
    assert np.all(ipred > 0.0)
    np.testing.assert_allclose(f, ipred)


@pytest.mark.unit
def test_three_cmt_iv():
    """three_cmt_iv() returns a ModelBuilder without data, handling ADVAN fallback."""
    from openpkpd.library import three_cmt_iv

    model = three_cmt_iv()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_emax_direct():
    """emax_direct() returns a ModelBuilder without data."""
    from openpkpd.library import emax_direct

    model = emax_direct()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_sigmoid_emax():
    """sigmoid_emax() returns a ModelBuilder without data."""
    from openpkpd.library import sigmoid_emax

    model = sigmoid_emax()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_inhibitory_emax():
    """inhibitory_emax() returns a ModelBuilder without data."""
    from openpkpd.library import inhibitory_emax

    model = inhibitory_emax()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_indirect_response_type_i():
    """indirect_response_type_i() returns a ModelBuilder without data."""
    from openpkpd.library import indirect_response_type_i

    model = indirect_response_type_i()
    assert isinstance(model, ModelBuilder)


@pytest.mark.unit
def test_effect_compartment():
    """effect_compartment() returns a ModelBuilder without data."""
    from openpkpd.library import effect_compartment

    model = effect_compartment()
    assert isinstance(model, ModelBuilder)


# ── kwarg overrides ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_one_cmt_oral_kwarg_overrides():
    """one_cmt_oral() should accept and use cl_init, v_init kwarg overrides."""
    from openpkpd.library import one_cmt_oral

    model = one_cmt_oral(cl_init=0.05, v_init=15.0, iiv_cl=0.1)
    assert isinstance(model, ModelBuilder)
    # Spot-check that the THETA specs were set with custom values
    assert model._theta_specs[1].init == pytest.approx(0.05, rel=1e-3)
    assert model._theta_specs[2].init == pytest.approx(15.0, rel=1e-3)


@pytest.mark.unit
def test_one_cmt_iv_kwarg_overrides():
    """one_cmt_iv() should accept and use cl_init, v_init kwarg overrides."""
    from openpkpd.library import one_cmt_iv

    model = one_cmt_iv(cl_init=3.0, v_init=12.0)
    assert isinstance(model, ModelBuilder)
    assert model._theta_specs[0].init == pytest.approx(3.0, rel=1e-3)
    assert model._theta_specs[1].init == pytest.approx(12.0, rel=1e-3)


@pytest.mark.unit
def test_get_model_with_kwargs():
    """get_model() should pass **kwargs to the factory function."""
    model = get_model("one_cmt_oral", cl_init=0.2, v_init=25.0)
    assert isinstance(model, ModelBuilder)


# ── Method override ───────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("method", ["FO", "FOCE", "SAEM"])
def test_model_method_override(method: str):
    """All models should accept a 'method' parameter for estimation method."""
    model = get_model("one_cmt_oral", method=method)
    assert isinstance(model, ModelBuilder)
    assert model._estimation_kwargs.get("method") == method
