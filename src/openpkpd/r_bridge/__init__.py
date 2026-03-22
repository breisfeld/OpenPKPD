"""
R integration bridge for OpenPKPD via rpy2.

Provides a thin, safe wrapper around rpy2 that:
  - Checks for rpy2 availability at import time (no hard dependency).
  - Converts numpy arrays / pandas DataFrames to R objects and back.
  - Exposes a simple ``RBridge`` class for evaluating R expressions,
    sourcing R scripts, and calling R functions.
  - Captures R stdout/stderr into Python strings.

Requirements
------------
``pip install rpy2``

If rpy2 is not installed, ``is_r_available()`` returns ``False`` and
all ``RBridge`` methods raise ``ImportError`` with a clear message.

References
----------
rpy2 documentation: https://rpy2.github.io/doc/latest/html/index.html
"""

from __future__ import annotations

from typing import Any


def is_r_available() -> bool:
    """Return True if rpy2 and a working R installation can be found."""
    try:
        import rpy2.robjects as _ro

        _ = _ro.r  # confirm R is actually accessible, not just importable
        return True
    except (ImportError, Exception):
        return False


def _require_rpy2() -> None:
    if not is_r_available():
        raise ImportError("rpy2 is required for R integration. Install it with: pip install rpy2")


# ---------------------------------------------------------------------------
# Type conversion helpers
# ---------------------------------------------------------------------------


def numpy_to_r(arr: Any) -> Any:
    """
    Convert a numpy array to an R vector or matrix.

    Scalars are returned as R FloatVector of length 1.

    Args:
        arr: numpy array or scalar.

    Returns:
        rpy2 R object (FloatVector or FloatMatrix).

    Raises:
        ImportError: If rpy2 is not installed.
    """
    _require_rpy2()
    import numpy as np
    import rpy2.robjects as ro

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        return ro.FloatVector([float(arr)])
    if arr.ndim == 1:
        return ro.FloatVector(arr.tolist())
    if arr.ndim == 2:
        # R uses column-major order
        r_matrix = ro.r.matrix(
            ro.FloatVector(arr.flatten("F").tolist()),
            nrow=arr.shape[0],
            ncol=arr.shape[1],
        )
        return r_matrix
    raise ValueError(f"Cannot convert {arr.ndim}-D array to R; max 2-D supported.")


def r_to_numpy(r_obj: Any) -> Any:
    """
    Convert an rpy2 R object to a numpy array.

    Args:
        r_obj: rpy2 R object (vector, matrix, etc.).

    Returns:
        numpy ndarray.

    Raises:
        ImportError: If rpy2 is not installed.
    """
    _require_rpy2()
    import numpy as np
    import rpy2.robjects as ro

    try:
        arr = np.array(list(r_obj), dtype=float)
        # R matrices carry a dim attribute; reshape if present (column-major)
        dim_r = ro.r.dim(r_obj)
        if len(dim_r) > 0:
            shape = tuple(int(x) for x in dim_r)
            arr = arr.reshape(shape, order="F")
        return arr
    except Exception:
        return np.array(r_obj)


def dataframe_to_r(df: Any) -> Any:
    """
    Convert a pandas DataFrame to an R data.frame.

    Args:
        df: pandas DataFrame.

    Returns:
        rpy2 R data.frame.

    Raises:
        ImportError: If rpy2 is not installed.
    """
    _require_rpy2()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def r_to_dataframe(r_df: Any) -> Any:
    """
    Convert an rpy2 R data.frame to a pandas DataFrame.

    Args:
        r_df: rpy2 R data.frame object.

    Returns:
        pandas DataFrame.

    Raises:
        ImportError: If rpy2 is not installed.
    """
    _require_rpy2()
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)


# ---------------------------------------------------------------------------
# RBridge class
# ---------------------------------------------------------------------------


class RBridge:
    """
    Thin Python–R bridge via rpy2.

    Provides methods for evaluating R expressions, sourcing scripts,
    calling R functions, and converting results back to Python objects.

    All methods raise ``ImportError`` when rpy2 is not installed.

    Args:
        capture_output: If True (default), capture R stdout/stderr into
                        the ``last_output`` attribute rather than printing.

    Example::

        from openpkpd.r_bridge import RBridge, is_r_available

        if is_r_available():
            r = RBridge()
            result = r.eval("sqrt(2)")          # → 1.4142…
            r.install_package("nlme")
            df = r.call("read.csv", "data.csv")
    """

    def __init__(self, capture_output: bool = True) -> None:
        _require_rpy2()
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        self._ro = ro
        self._importr = importr
        self._capture = capture_output
        self.last_output: str = ""

    # ------------------------------------------------------------------
    # Core R evaluation
    # ------------------------------------------------------------------

    def eval(self, r_code: str) -> Any:
        """
        Evaluate an R expression and return the result.

        Args:
            r_code: R code string (single expression or block).

        Returns:
            Python object converted from the R result via rpy2 automatic
            conversion, or an rpy2 ``NULL`` / vector object.
        """
        if self._capture:
            captured: list[str] = []

            def _capture_write(text: str) -> None:
                captured.append(text)

            import rpy2.rinterface_lib.callbacks as callbacks

            old_write = callbacks.consolewrite_print
            callbacks.consolewrite_print = _capture_write
            try:
                result = self._ro.r(r_code)
            finally:
                callbacks.consolewrite_print = old_write
                self.last_output = "".join(captured)
        else:
            result = self._ro.r(r_code)
        return result

    def source(self, script_path: str) -> None:
        """
        Source an R script file.

        Args:
            script_path: Absolute or relative path to a ``.R`` file.
        """
        self.eval(f'source("{script_path}")')

    def call(self, r_function_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Call a named R function with Python arguments.

        Positional arguments are passed in order; keyword arguments become
        named R arguments.  numpy arrays are converted via ``numpy_to_r``
        before passing.

        Args:
            r_function_name: Name of the R function (e.g. ``"lm"``).
            *args:           Positional arguments (auto-converted).
            **kwargs:        Named arguments (auto-converted).

        Returns:
            R result converted to the most appropriate Python type.
        """
        import numpy as np

        r_func = self._ro.r[r_function_name]
        r_args = []
        for a in args:
            if hasattr(a, "__array__") or isinstance(a, list):
                r_args.append(numpy_to_r(np.asarray(a)))
            else:
                r_args.append(a)
        r_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "__array__") or isinstance(v, list):
                r_kwargs[k] = numpy_to_r(np.asarray(v))
            else:
                r_kwargs[k] = v
        return r_func(*r_args, **r_kwargs)

    def install_package(
        self, package_name: str, repos: str = "https://cloud.r-project.org"
    ) -> None:
        """
        Install an R package from CRAN if not already installed.

        Args:
            package_name: R package name (e.g. ``"nlme"``).
            repos:        CRAN mirror URL.
        """
        self.eval(
            f"""
            if (!requireNamespace("{package_name}", quietly=TRUE)) {{
                install.packages("{package_name}", repos="{repos}", quiet=TRUE)
            }}
            """
        )

    def library(self, package_name: str) -> Any:
        """
        Import an R package and return the rpy2 package object.

        Args:
            package_name: R package name.

        Returns:
            rpy2 package object with all exported symbols as attributes.
        """
        return self._importr(package_name)

    def set_seed(self, seed: int) -> None:
        """Set the R random seed for reproducibility."""
        self.eval(f"set.seed({seed})")

    def get_version(self) -> str:
        """Return the R version string."""
        result = self.eval("R.version.string")
        try:
            return str(result[0])
        except Exception:
            return str(result)


__all__ = [
    "is_r_available",
    "numpy_to_r",
    "r_to_numpy",
    "dataframe_to_r",
    "r_to_dataframe",
    "RBridge",
]
