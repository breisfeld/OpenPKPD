"""
Unit tests for the parallel execution backends (parallel/__init__.py).

Tests the multiprocessing backend (always available) and the get_backend()
dispatcher. Dask and Ray backends are tested only for ImportError fallback
behaviour.
"""

from __future__ import annotations

import sys
import time

import pytest

from openpkpd.parallel import (
    _MultiprocessingBackend,
    get_backend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _double(x):
    """Simple function for parallel map tests (must be picklable)."""
    return x * 2


def _add_one(x):
    return x + 1


def _square(x):
    return x**2


def _slow_add_one(x):
    time.sleep(0.01)
    return x + 1


# ---------------------------------------------------------------------------
# _MultiprocessingBackend
# ---------------------------------------------------------------------------


class TestMultiprocessingBackend:
    def test_map_basic(self):
        b = _MultiprocessingBackend(n_jobs=1)
        results = b.map(_double, [1, 2, 3, 4, 5])
        assert results == [2, 4, 6, 8, 10]

    def test_map_preserves_order(self):
        """Results must be in the same order as inputs."""
        b = _MultiprocessingBackend(n_jobs=2)
        inputs = list(range(20))
        results = b.map(_square, inputs)
        expected = [x**2 for x in inputs]
        assert results == expected

    def test_map_empty_input(self):
        b = _MultiprocessingBackend(n_jobs=2)
        assert b.map(_double, []) == []

    def test_map_single_element(self):
        b = _MultiprocessingBackend(n_jobs=2)
        assert b.map(_double, [7]) == [14]

    def test_map_n_jobs_1_runs_inline(self):
        """n_jobs=1 should execute inline (no subprocess overhead)."""
        b = _MultiprocessingBackend(n_jobs=1)
        results = b.map(_add_one, [10, 20, 30])
        assert results == [11, 21, 31]

    def test_map_multiprocess(self):
        """n_jobs > 1 should correctly distribute work."""
        b = _MultiprocessingBackend(n_jobs=2)
        results = b.map(_double, list(range(10)))
        assert results == [i * 2 for i in range(10)]

    def test_n_jobs_minus_one_uses_all_cpus(self):
        import os

        b = _MultiprocessingBackend(n_jobs=-1)
        assert b.n_jobs == os.cpu_count()

    def test_n_jobs_zero_uses_all_cpus(self):
        import os

        b = _MultiprocessingBackend(n_jobs=0)
        assert b.n_jobs == os.cpu_count()

    def test_context_manager(self):
        """Backend should work as a context manager."""
        with _MultiprocessingBackend(n_jobs=1) as b:
            results = b.map(_double, [5, 10])
        assert results == [10, 20]

    def test_context_manager_enter_returns_self(self):
        b = _MultiprocessingBackend(n_jobs=1)
        assert b.__enter__() is b

    def test_context_manager_exit_no_exception(self):
        b = _MultiprocessingBackend(n_jobs=1)
        b.__enter__()
        b.__exit__(None, None, None)  # Should not raise

    def test_large_input(self):
        """Performance test: 100 items should complete quickly."""
        b = _MultiprocessingBackend(n_jobs=2)
        inputs = list(range(100))
        results = b.map(_double, inputs)
        assert len(results) == 100
        assert results[50] == 100

    def test_with_generator_input(self):
        """args_iter can be a generator, not just a list."""
        b = _MultiprocessingBackend(n_jobs=1)
        results = b.map(_add_one, (x for x in [1, 2, 3]))
        assert results == [2, 3, 4]

    def test_function_raising_exception_propagates(self):
        """Exceptions from the worker function should propagate."""

        def _raise(x):
            raise ValueError(f"bad input: {x}")

        b = _MultiprocessingBackend(n_jobs=1)
        with pytest.raises(ValueError, match="bad input"):
            b.map(_raise, [42])


# ---------------------------------------------------------------------------
# get_backend dispatcher
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_multiprocessing_backend_explicit(self):
        b = get_backend(n_jobs=2, backend="multiprocessing")
        assert isinstance(b, _MultiprocessingBackend)
        assert b.n_jobs == 2

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend(backend="nonexistent_backend")

    def test_auto_falls_back_to_multiprocessing(self, monkeypatch):
        """When dask and ray are absent, auto selects multiprocessing."""
        monkeypatch.setitem(sys.modules, "dask", None)  # type: ignore
        monkeypatch.setitem(sys.modules, "dask.distributed", None)  # type: ignore
        monkeypatch.setitem(sys.modules, "ray", None)  # type: ignore

        b = get_backend(n_jobs=2, backend="auto")
        assert isinstance(b, _MultiprocessingBackend)

    def test_dask_raises_import_error_without_package(self, monkeypatch):
        from openpkpd.parallel import _DaskBackend

        monkeypatch.setitem(sys.modules, "dask", None)  # type: ignore
        monkeypatch.setitem(sys.modules, "dask.distributed", None)  # type: ignore

        with pytest.raises(ImportError, match="Dask"):
            _DaskBackend(n_jobs=2)

    def test_ray_raises_import_error_without_package(self, monkeypatch):
        from openpkpd.parallel import _RayBackend

        monkeypatch.setitem(sys.modules, "ray", None)  # type: ignore

        with pytest.raises(ImportError, match="Ray"):
            _RayBackend(n_jobs=2)

    def test_multiprocessing_backend_produces_correct_type(self):
        b = get_backend(n_jobs=1, backend="multiprocessing")
        assert hasattr(b, "map")

    def test_backend_map_works_after_get_backend(self):
        b = get_backend(n_jobs=1, backend="multiprocessing")
        results = b.map(_double, [3, 6, 9])
        assert results == [6, 12, 18]

    def test_n_jobs_default_minus_one(self):
        import os

        b = get_backend(backend="multiprocessing")
        assert b.n_jobs == os.cpu_count()


# ---------------------------------------------------------------------------
# ParallelBackend type alias
# ---------------------------------------------------------------------------


class TestParallelBackendTypeAlias:
    def test_multiprocessing_is_parallel_backend(self):
        """ParallelBackend union includes _MultiprocessingBackend."""
        b = _MultiprocessingBackend(n_jobs=1)
        # isinstance check against the union members
        from openpkpd.parallel import _MultiprocessingBackend as MPB

        assert isinstance(b, MPB)

    def test_all_exported(self):
        from openpkpd import parallel

        assert hasattr(parallel, "get_backend")
        assert hasattr(parallel, "ParallelBackend")
        assert hasattr(parallel, "_MultiprocessingBackend")
        assert hasattr(parallel, "_DaskBackend")
        assert hasattr(parallel, "_RayBackend")
        assert hasattr(parallel, "_MPIBackend")


# ---------------------------------------------------------------------------
# _MPIBackend
# ---------------------------------------------------------------------------


class TestMPIBackend:
    def test_mpi_raises_import_error_without_mpi4py(self, monkeypatch):
        from openpkpd.parallel import _MPIBackend

        monkeypatch.setitem(sys.modules, "mpi4py", None)
        with pytest.raises(ImportError, match="mpi4py"):
            _MPIBackend(n_jobs=2)

    def test_get_backend_mpi_raises_without_mpi4py(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mpi4py", None)
        with pytest.raises(ImportError, match="mpi4py"):
            get_backend(backend="mpi", n_jobs=2)

    def test_mpi_backend_exported(self):
        from openpkpd.parallel import _MPIBackend

        assert _MPIBackend is not None

    def test_unknown_backend_error_message_includes_mpi(self):
        with pytest.raises(ValueError, match="mpi"):
            get_backend(backend="bogus_xyz")
