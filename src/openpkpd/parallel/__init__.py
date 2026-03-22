"""
Parallel execution backends for OpenPKPD.

Provides a unified interface for distributing computation across multiple
workers (CPUs, cluster nodes, cloud).  The backend is selected automatically
based on available packages, or can be set explicitly.

Available backends
------------------
``multiprocessing``
    Python ``concurrent.futures.ProcessPoolExecutor``.  Available everywhere.
    Suitable for bootstraps and SCM runs on a single machine.

``dask``
    Dask distributed/local cluster.  Install with ``pip install dask[distributed]``.
    Suitable for large-scale bootstraps on HPC clusters.

``ray``
    Ray distributed runtime.  Install with ``pip install ray``.
    Suitable for cloud and mixed-CPU/GPU workloads.

Usage
-----
::

    from openpkpd.parallel import ParallelBackend, get_backend

    # Auto-select best available backend
    backend = get_backend(n_jobs=4)

    # Run a list of callables in parallel
    results = backend.map(my_function, list_of_args)

    # Context manager that shuts down the backend cleanly
    with get_backend(n_jobs=8) as backend:
        results = backend.map(fit_bootstrap, replicate_args)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


class _MultiprocessingBackend:
    """
    Multiprocessing backend using ``concurrent.futures.ProcessPoolExecutor``.

    Args:
        n_jobs: Number of worker processes.  ``-1`` uses all available CPUs.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs

    def map(
        self,
        fn: Callable,
        args_iter: Iterable[Any],
        *,
        timeout: float | None = None,
    ) -> list[Any]:
        """
        Apply *fn* to each element of *args_iter* in parallel.

        Args:
            fn:         Callable that accepts a single argument.
            args_iter:  Iterable of arguments; each is passed as the sole
                        positional argument to *fn*.
            timeout:    Per-task timeout in seconds (``None`` = no limit).

        Returns:
            List of results in the same order as *args_iter*.
        """
        args = list(args_iter)
        if not args:
            return []
        if self.n_jobs == 1:
            return [fn(a) for a in args]
        results: list[Any | None] = [None] * len(args)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_idx = {executor.submit(fn, a): i for i, a in enumerate(args)}
            for future in as_completed(future_to_idx, timeout=timeout):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def __enter__(self) -> _MultiprocessingBackend:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class _DaskBackend:
    """
    Dask distributed backend.

    Connects to a Dask scheduler (local or remote).  When *scheduler_address*
    is None, a ``LocalCluster`` is started automatically.

    Args:
        n_jobs:            Number of workers for ``LocalCluster`` (ignored when
                           connecting to an existing scheduler).
        scheduler_address: Dask scheduler address, e.g. ``"tcp://scheduler:8786"``.
                           If None, a LocalCluster is started.
        memory_limit:      Per-worker memory limit string, e.g. ``"4GB"``.
    """

    def __init__(
        self,
        n_jobs: int = -1,
        scheduler_address: str | None = None,
        memory_limit: str = "auto",
    ) -> None:
        import importlib.util
        import sys

        if "dask" in sys.modules:
            _dask_found = sys.modules["dask"] is not None
        else:
            try:
                _dask_found = importlib.util.find_spec("dask") is not None
            except (ValueError, ModuleNotFoundError):
                _dask_found = False
        if not _dask_found:
            raise ImportError(
                "Dask is required for the Dask backend. "
                "Install it with: pip install dask[distributed]"
            )
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.scheduler_address = scheduler_address
        self.memory_limit = memory_limit
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from dask.distributed import Client, LocalCluster

        if self.scheduler_address:
            self._client = Client(self.scheduler_address)
        else:
            cluster = LocalCluster(
                n_workers=self.n_jobs,
                memory_limit=self.memory_limit,
                silence_logs=True,
            )
            self._client = Client(cluster)
        return self._client

    def map(
        self,
        fn: Callable,
        args_iter: Iterable[Any],
        *,
        timeout: float | None = None,
    ) -> list[Any]:
        """
        Distribute *fn* over *args_iter* using Dask futures.

        Args:
            fn:         Callable accepting a single argument.
            args_iter:  Iterable of task arguments.
            timeout:    Per-task timeout in seconds.

        Returns:
            List of results in input order.
        """
        args = list(args_iter)
        if not args:
            return []
        client = self._get_client()
        futures = client.map(fn, args)
        return client.gather(futures, direct=True)

    def __enter__(self) -> _DaskBackend:
        self._get_client()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class _MPIBackend:
    """
    MPI parallel backend for HPC clusters via ``mpi4py``.

    Distributes work across MPI ranks using a controller–worker pattern.
    The controller (rank 0) submits tasks; all workers call the same
    function on their assigned argument and return results to rank 0.

    When running outside an MPI context (i.e. as a single process),
    tasks are executed sequentially, just like ``n_jobs=1``.

    Args:
        n_jobs: Hint for number of workers (currently informational;
                the actual concurrency is determined by MPI world size).

    Raises:
        ImportError: If ``mpi4py`` is not installed.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        import importlib.util
        import sys

        if "mpi4py" in sys.modules:
            _mpi_found = sys.modules["mpi4py"] is not None
        else:
            try:
                _mpi_found = importlib.util.find_spec("mpi4py") is not None
            except (ValueError, ModuleNotFoundError):
                _mpi_found = False
        if not _mpi_found:
            raise ImportError(
                "mpi4py is required for the MPI backend. Install it with: pip install mpi4py"
            )
        self.n_jobs = n_jobs

    def map(
        self,
        fn: Callable,
        args_iter: Iterable[Any],
        *,
        timeout: float | None = None,
    ) -> list[Any]:
        """
        Execute *fn* across all MPI ranks and gather results on rank 0.

        Tasks are distributed round-robin across available ranks.  When
        running as a single process (world size == 1) the work is done
        sequentially without any MPI communication.

        Args:
            fn:         Callable accepting a single argument.
            args_iter:  Iterable of arguments.
            timeout:    Ignored (MPI blocking gather has no timeout).

        Returns:
            List of results in the same order as *args_iter*.
        """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        args = list(args_iter)
        if not args:
            return []

        if size == 1:
            # Single-process MPI: run sequentially
            return [fn(a) for a in args]

        n = len(args)
        # Distribute args across ranks
        local_indices = list(range(rank, n, size))
        local_results = [(i, fn(args[i])) for i in local_indices]

        # Gather all results on rank 0
        all_results: list[list[tuple[int, Any]]] = comm.gather(local_results, root=0)
        if rank == 0:
            flat = [item for group in (all_results or []) for item in group]
            flat.sort(key=lambda x: x[0])
            return [v for _, v in flat]
        return []

    def __enter__(self) -> _MPIBackend:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class _RayBackend:
    """
    Ray distributed backend.

    Args:
        n_jobs: Number of CPUs to use.  ``-1`` uses all available.
        address: Ray cluster address (``None`` = start local Ray).
    """

    def __init__(self, n_jobs: int = -1, address: str | None = None) -> None:
        import importlib.util
        import sys

        if "ray" in sys.modules:
            _ray_found = sys.modules["ray"] is not None
        else:
            try:
                _ray_found = importlib.util.find_spec("ray") is not None
            except (ValueError, ModuleNotFoundError):
                _ray_found = False
        if not _ray_found:
            raise ImportError(
                "Ray is required for the Ray backend. Install it with: pip install ray"
            )
        self.n_jobs = n_jobs
        self.address = address
        self._initialized = False

    def _init(self) -> None:
        if self._initialized:
            return
        import ray

        if not ray.is_initialized():
            ray.init(
                address=self.address,
                num_cpus=self.n_jobs if self.n_jobs > 0 else None,
                ignore_reinit_error=True,
                log_to_driver=False,
            )
        self._initialized = True

    def map(
        self,
        fn: Callable,
        args_iter: Iterable[Any],
        *,
        timeout: float | None = None,
    ) -> list[Any]:
        """
        Submit tasks to Ray and gather results.

        Args:
            fn:         Callable accepting a single argument.
            args_iter:  Iterable of task arguments.
            timeout:    Ignored (Ray timeout not yet implemented).

        Returns:
            List of results in input order.
        """
        import ray

        self._init()
        args = list(args_iter)
        if not args:
            return []
        remote_fn = ray.remote(fn)
        refs = [remote_fn.remote(a) for a in args]
        return ray.get(refs)

    def __enter__(self) -> _RayBackend:
        self._init()
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # Ray is typically not shut down per-task


# Public type alias
ParallelBackend = _MultiprocessingBackend | _DaskBackend | _RayBackend | _MPIBackend


def get_backend(
    n_jobs: int = -1,
    backend: str = "auto",
    **kwargs: Any,
) -> ParallelBackend:
    """
    Create and return a parallel execution backend.

    Args:
        n_jobs:   Number of parallel workers.  ``-1`` = all CPUs.
        backend:  Backend name: ``"auto"``, ``"multiprocessing"``,
                  ``"dask"``, ``"ray"``, or ``"mpi"``.
                  When ``"auto"``, tries Dask then Ray then falls back
                  to multiprocessing.
        **kwargs: Additional keyword arguments forwarded to the backend
                  constructor (e.g. ``scheduler_address`` for Dask).

    Returns:
        An instantiated backend with a ``.map()`` method.

    Raises:
        ValueError: If an unknown backend name is given.
    """
    if backend == "mpi":
        return _MPIBackend(n_jobs=n_jobs)
    if backend == "dask":
        return _DaskBackend(n_jobs=n_jobs, **kwargs)
    if backend == "ray":
        return _RayBackend(n_jobs=n_jobs, **kwargs)
    if backend == "multiprocessing":
        return _MultiprocessingBackend(n_jobs=n_jobs)
    if backend == "auto":
        try:
            return _DaskBackend(n_jobs=n_jobs, **kwargs)
        except ImportError:
            pass
        try:
            return _RayBackend(n_jobs=n_jobs)
        except ImportError:
            pass
        return _MultiprocessingBackend(n_jobs=n_jobs)
    raise ValueError(
        f"Unknown backend {backend!r}. "
        f"Choose from: 'auto', 'multiprocessing', 'dask', 'ray', 'mpi'."
    )


__all__ = [
    "ParallelBackend",
    "get_backend",
    "_MultiprocessingBackend",
    "_DaskBackend",
    "_RayBackend",
    "_MPIBackend",
]
