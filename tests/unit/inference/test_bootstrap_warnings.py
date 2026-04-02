"""
Tests for BootstrapEngine warnings.
Covers:
  BS3: convergence rate warning (< 80% converged)
  BS4: small-n subject warning (< 10 subjects)
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openpkpd.estimation.base import EstimationResult
from openpkpd.inference.bootstrap import BootstrapEngine, BootstrapResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(converged: bool = True) -> EstimationResult:
    """Return a minimal EstimationResult."""
    return EstimationResult(
        theta_final=np.array([1.0]),
        omega_final=np.array([[0.1]]),
        sigma_final=np.array([[0.05]]),
        ofv=100.0,
        converged=converged,
        method="FOCE",
        message="ok",
    )


def _make_engine(n_subjects: int = 20, n_boot: int = 200) -> BootstrapEngine:
    """
    Build a BootstrapEngine with a mock population_model.
    The mock has n_subjects IDs and a trivial dataset.
    """
    pop_model = MagicMock()
    pop_model.dataset.subject_ids.return_value = list(range(1, n_subjects + 1))
    pop_model.subject_ids.return_value = list(range(1, n_subjects + 1))
    pop_model.n_subjects.return_value = n_subjects
    pop_model.pk_subroutine = "ADVAN1"
    pop_model.pk_callable = lambda *a, **kw: None
    pop_model.error_callable = lambda *a, **kw: None
    pop_model.des_callable = None
    pop_model.trans = 1
    pop_model.advan = 1
    pop_model.covariate_columns = []

    # Dataset df mock — needed by _resample_subjects
    import pandas as pd
    rows = []
    for sid in range(1, n_subjects + 1):
        rows.append({"ID": sid, "TIME": 0.0, "DV": 1.0})
    df = pd.DataFrame(rows)
    pop_model.dataset.df = df

    from openpkpd.model.parameters import ParameterSet
    init_params = ParameterSet(
        theta=np.array([1.0]),
        omega=np.array([[0.1]]),
        sigma=np.array([[0.05]]),
    )

    engine = BootstrapEngine(
        population_model=pop_model,
        initial_params=init_params,
        estimation_method="FOCE",
        n_boot=n_boot,
        n_jobs=1,
        seed=42,
    )
    return engine


# ---------------------------------------------------------------------------
# BS3: Convergence rate warning
# ---------------------------------------------------------------------------


def test_bs3_no_warning_90pct_converged():
    """180/200 converged (90%) → no RuntimeWarning."""
    engine = _make_engine(n_subjects=20, n_boot=200)
    n_success = 180
    n_fail = 20

    successful = [_make_result(True)] * n_success
    all_results = [_make_result(True)] * n_success + [None] * n_fail

    with patch.object(engine, "_fit_one", side_effect=all_results):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run()

    rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(rw) == 0
    assert result.n_success == n_success


def test_bs3_warning_75pct_converged():
    """150/200 converged (75%) → RuntimeWarning with percentage."""
    engine = _make_engine(n_subjects=20, n_boot=200)
    n_success = 150

    all_results = [_make_result(True)] * n_success + [None] * 50

    with patch.object(engine, "_fit_one", side_effect=all_results):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run()

    rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(rw) >= 1
    msg = str(rw[0].message)
    assert "150" in msg or "75" in msg


def test_bs3_boundary_80pct_no_warning():
    """160/200 converged (exactly 80%) → no warning (boundary inclusive)."""
    engine = _make_engine(n_subjects=20, n_boot=200)
    n_success = 160

    all_results = [_make_result(True)] * n_success + [None] * 40

    with patch.object(engine, "_fit_one", side_effect=all_results):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run()

    rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(rw) == 0


def test_bs3_zero_success_raises_runtime_error():
    """0/200 converged → RuntimeError still raised (not replaced by warning)."""
    engine = _make_engine(n_subjects=20, n_boot=200)

    all_results = [None] * 200

    with patch.object(engine, "_fit_one", side_effect=all_results):
        with pytest.raises(RuntimeError, match="All bootstrap replicates failed"):
            engine.run()


# ---------------------------------------------------------------------------
# BS4: Small-n subject warning — tested via _resample_subjects directly
# ---------------------------------------------------------------------------


def _call_resample(engine: BootstrapEngine) -> None:
    """
    Call _resample_subjects directly to trigger any warnings.
    The internal NONMEMDataset/PopulationModel are imported locally inside the method,
    so we patch via their original module paths.  We don't care about the return value.
    """
    from unittest.mock import patch as _patch
    with _patch("openpkpd.data.dataset.NONMEMDataset", MagicMock()):
        with _patch("openpkpd.model.population.PopulationModel", MagicMock()):
            try:
                engine._resample_subjects(42)
            except Exception:
                pass  # construction may fail after warning is emitted; that's fine


def test_bs4_no_warning_10_subjects():
    """n_subjects=10 → no RuntimeWarning (boundary: exactly 10 is safe)."""
    engine = _make_engine(n_subjects=10, n_boot=1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _call_resample(engine)

    small_n_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning) and "stratified resampling" in str(x.message)
    ]
    assert len(small_n_warnings) == 0


def test_bs4_warning_9_subjects():
    """n_subjects=9 → RuntimeWarning mentioning 'stratified resampling'."""
    engine = _make_engine(n_subjects=9, n_boot=1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _call_resample(engine)

    small_n_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning) and "stratified resampling" in str(x.message)
    ]
    assert len(small_n_warnings) >= 1


def test_bs4_warning_5_subjects():
    """n_subjects=5 → warning emitted."""
    engine = _make_engine(n_subjects=5, n_boot=1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _call_resample(engine)

    small_n_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning) and "stratified resampling" in str(x.message)
    ]
    assert len(small_n_warnings) >= 1


def test_bs4_warning_once_per_replicate_call():
    """Warning emitted once per _resample_subjects call (once per replicate)."""
    engine = _make_engine(n_subjects=5, n_boot=1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _call_resample(engine)

    small_n_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning) and "stratified resampling" in str(x.message)
    ]
    # Exactly one warning per _resample_subjects call
    assert len(small_n_warnings) == 1
