"""
Tests for _NAN_LLOQ_CACHE thread safety in IndividualModel (M4).
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

import openpkpd.model.individual as ind_module
from openpkpd.model.individual import _NAN_LLOQ_CACHE, _NAN_LLOQ_CACHE_LOCK, _build_lloq_array


def test_concurrent_writes_no_corruption():
    """20 threads writing different keys produce no RuntimeError and final len == 20."""
    # Clear the cache first
    with _NAN_LLOQ_CACHE_LOCK:
        _NAN_LLOQ_CACHE.clear()

    errors = []

    def worker(n):
        try:
            arr = _build_lloq_array(None, n)
            assert arr.shape == (n,)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i + 1,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    with _NAN_LLOQ_CACHE_LOCK:
        assert len(_NAN_LLOQ_CACHE) == 20


def test_concurrent_reads_while_writing():
    """Concurrent reads while writing produce no KeyError or RuntimeError."""
    # Pre-populate with one entry
    with _NAN_LLOQ_CACHE_LOCK:
        _NAN_LLOQ_CACHE.clear()
        _NAN_LLOQ_CACHE[100] = np.full(100, np.nan, dtype=np.float64)

    errors = []

    def reader():
        try:
            for _ in range(50):
                arr = _build_lloq_array(None, 100)
                assert len(arr) == 100
        except Exception as e:
            errors.append(e)

    def writer():
        try:
            for n in range(200, 210):
                _build_lloq_array(None, n)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(10)]
    threads += [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"


def test_cache_correct_lloq_array():
    """Smoke test: _build_lloq_array(None, n) returns all-NaN array of length n."""
    with _NAN_LLOQ_CACHE_LOCK:
        _NAN_LLOQ_CACHE.clear()

    arr = _build_lloq_array(None, 5)
    assert arr.shape == (5,)
    assert np.all(np.isnan(arr))

    # Scalar lloq path
    arr2 = _build_lloq_array(1.0, 5)
    assert arr2.shape == (5,)
    np.testing.assert_allclose(arr2, 1.0)
