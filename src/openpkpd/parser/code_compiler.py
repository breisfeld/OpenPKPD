"""
NMTRANCompiler: Translates $PK, $DES, and $ERROR code blocks to Python callables.

NM-TRAN is FORTRAN-77 style code with NONMEM extensions. This compiler
handles the core subset needed for PK/PD modeling.

Compiled form: exec()-based Python callable for runtime evaluation.

Reserved name mapping:
  THETA(n) → theta[n-1]
  ETA(n)   → eta[n-1]
  EPS(n)   → eps[n-1]
  A(n)     → a[n-1]
  DADT(n)  → dadt[n-1]  (set via assignment)
  T        → t
  F        → f
  Y        → y
  IPRED    → ipred
  DV       → dv

FORTRAN intrinsics mapped to Python equivalents:
  EXP, LOG, SQRT, ABS, MOD, MAX, MIN, INT, FLOAT, ATAN2, SIN, COS, TAN
  DBLE → float cast
"""

from __future__ import annotations

import ast
import keyword
import re
import textwrap
from collections.abc import Callable
from typing import Any

from openpkpd.utils.errors import CompilerError

# ── Intrinsic function mapping ────────────────────────────────────────────────
_INTRINSICS: dict[str, str] = {
    "EXP": "math.exp",
    "LOG": "math.log",
    "LOG10": "math.log10",
    "SQRT": "math.sqrt",
    "ABS": "abs",
    "MOD": "math.fmod",
    "MAX": "max",
    "MIN": "min",
    "INT": "int",
    "FLOAT": "float",
    "DBLE": "float",
    "ATAN2": "math.atan2",
    "SIN": "math.sin",
    "COS": "math.cos",
    "TAN": "math.tan",
    "ASIN": "math.asin",
    "ACOS": "math.acos",
    "ATAN": "math.atan",
    "SIGN": "_nmtran_sign",
    "GAMLN": "math.lgamma",
}

def _nmtran_sign(a: float, b: float) -> float:
    """FORTRAN SIGN(a, b) = abs(a) * sign(b)."""
    import math

    return abs(a) * math.copysign(1.0, b)


# ── Core transformation ────────────────────────────────────────────────────────


def _translate_line(line: str, intrinsics: dict[str, str]) -> str:
    """
    Translate a single NM-TRAN code line to Python.

    Transformations:
      1. Strip inline comments (lines starting with ; or C in col 1)
      2. THETA(n)  → theta[n-1]
      3. ETA(n)    → eta[n-1]
      4. EPS(n)    → eps[n-1]
      5. A(n)      → a[n-1]
      6. DADT(n) = → dadt[n-1] =
      7. FORTRAN intrinsics → Python equivalents
      8. .TRUE. / .FALSE. → True / False
      9. .EQ. .NE. .LT. .LE. .GT. .GE. → == != < <= > >=
      10. .AND. .OR. .NOT. → and or not
      11. ** → **  (already Python-compatible)
      12. Double-precision literals: 1.5D0 → 1.5
    """
    # Strip FORTRAN comment at end of line
    line = re.sub(r"\s*!.*$", "", line)

    # FORTRAN logical operators
    line = re.sub(r"\.EQ\.", "==", line, flags=re.IGNORECASE)
    line = re.sub(r"\.NE\.", "!=", line, flags=re.IGNORECASE)
    line = re.sub(r"\.LT\.", "<", line, flags=re.IGNORECASE)
    line = re.sub(r"\.LE\.", "<=", line, flags=re.IGNORECASE)
    line = re.sub(r"\.GT\.", ">", line, flags=re.IGNORECASE)
    line = re.sub(r"\.GE\.", ">=", line, flags=re.IGNORECASE)
    line = re.sub(r"\.AND\.", "and", line, flags=re.IGNORECASE)
    line = re.sub(r"\.OR\.", "or", line, flags=re.IGNORECASE)
    line = re.sub(r"\.NOT\.", "not ", line, flags=re.IGNORECASE)
    line = re.sub(r"\.TRUE\.", "True", line, flags=re.IGNORECASE)
    line = re.sub(r"\.FALSE\.", "False", line, flags=re.IGNORECASE)

    # Double-precision literals: 1.5D0 or 1.5D-3 → 1.5e0 or 1.5e-3
    line = re.sub(r"(\d)D([+-]?\d)", r"\1e\2", line, flags=re.IGNORECASE)

    # THETA(n), ETA(n), EPS(n), ERR(n), SIGMA(i,j), A(n), DADT(n)
    line = re.sub(
        r"\bTHETA\s*\(\s*(\d+)\s*\)",
        lambda m: f"theta[{int(m.group(1)) - 1}]",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\bETA\s*\(\s*(\d+)\s*\)",
        lambda m: f"eta[{int(m.group(1)) - 1}]",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\bEPS\s*\(\s*(\d+)\s*\)",
        lambda m: f"eps[{int(m.group(1)) - 1}]",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\bERR\s*\(\s*(\d+)\s*\)",
        lambda m: f"eps[{int(m.group(1)) - 1}]",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\bSIGMA\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
        lambda m: f"sigma[{int(m.group(1)) - 1}][{int(m.group(2)) - 1}]",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\bDADT\s*\(\s*(\d+)\s*\)",
        lambda m: f"dadt[{int(m.group(1)) - 1}]",
        line,
        flags=re.IGNORECASE,
    )
    line = re.sub(
        r"\bA\s*\(\s*(\d+)\s*\)", lambda m: f"a[{int(m.group(1)) - 1}]", line, flags=re.IGNORECASE
    )

    # NONMEM reserved scalar names: F→f, Y→y, T→t, DV→dv, IPRED→ipred
    # Only replace as standalone identifiers (not parts of longer names)
    line = re.sub(r"\bF\b", "f", line)
    line = re.sub(r"\bY\b", "y", line)
    line = re.sub(r"\bT\b", "t", line)
    line = re.sub(r"\bIPRED\b", "ipred", line, flags=re.IGNORECASE)
    line = re.sub(r"\bDV\b", "dv", line, flags=re.IGNORECASE)
    line = re.sub(r"\bIRES\b", "ires", line, flags=re.IGNORECASE)
    line = re.sub(r"\bIWRES\b", "iwres", line, flags=re.IGNORECASE)
    line = re.sub(r"\bW\b", "w", line)

    # FORTRAN intrinsics (longest-match first to avoid partial replacements)
    for fname, pyname in sorted(intrinsics.items(), key=lambda x: -len(x[0])):
        line = re.sub(
            rf"\b{re.escape(fname)}\b\s*(?=\()",
            pyname + " ",
            line,
            flags=re.IGNORECASE,
        )

    # IF (...) statement → if (...):\n  ...
    m = re.match(r"^\s*IF\s*\((.+)\)\s+(.+)$", line, re.IGNORECASE)
    if m:
        cond = m.group(1).strip()
        body = m.group(2).strip()
        line = f"if ({cond}):\n    {body}"

    return line


def _is_comment_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(";"):
        return True
    return bool(line and line[0].upper() == "C" and (len(line) == 1 or not line[1].isalnum()))


def _translate_block(code: str, intrinsics: dict[str, str]) -> str:
    """Translate a full NM-TRAN code block to Python."""
    lines = code.splitlines()
    result: list[str] = []
    indent_level = 0
    for line in lines:
        if _is_comment_line(line):
            continue
        stripped = line.strip()

        m_if_then = re.match(r"^IF\s*\((.+)\)\s*THEN\s*$", stripped, re.IGNORECASE)
        if m_if_then:
            cond = _translate_line(m_if_then.group(1).strip(), intrinsics)
            result.append("    " * indent_level + f"if ({cond}):")
            indent_level += 1
            continue

        if re.match(r"^ELSE\s*$", stripped, re.IGNORECASE):
            indent_level = max(indent_level - 1, 0)
            result.append("    " * indent_level + "else:")
            indent_level += 1
            continue

        m_elseif = re.match(r"^ELSE\s*IF\s*\((.+)\)\s*THEN\s*$", stripped, re.IGNORECASE)
        if m_elseif:
            indent_level = max(indent_level - 1, 0)
            cond = _translate_line(m_elseif.group(1).strip(), intrinsics)
            result.append("    " * indent_level + f"elif ({cond}):")
            indent_level += 1
            continue

        if re.match(r"^ENDIF\s*$", stripped, re.IGNORECASE):
            indent_level = max(indent_level - 1, 0)
            continue

        translated = _translate_line(stripped, intrinsics)
        for translated_line in translated.splitlines():
            result.append("    " * indent_level + translated_line)
    return "\n".join(result)


def _collect_assigned_names(code: str) -> tuple[str, ...]:
    """Collect simple assigned variable names from translated Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()

    names: list[str] = []
    seen: set[str] = set()

    def _record_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name) and target.id not in seen:
            seen.add(target.id)
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                _record_target(elt)

    class _Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            for target in node.targets:
                _record_target(target)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
            _record_target(node.target)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            _record_target(node.target)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return tuple(names)


# ── Callable compilation ──────────────────────────────────────────────────────


class CompiledPKCallable:
    """
    A compiled $PK code block.

    Calling signature:
        pk_params = callable(theta, eta, t=0.0)
        → dict mapping PK parameter names (CL, V, KA, etc.) to values
    """

    def __init__(self, code: str, param_names: list[str] | None = None) -> None:
        self._source = code
        self._param_names = param_names or []
        self._output_names = _collect_assigned_names(code)
        self._exec_fn: Callable | None = None
        self._direct_fn_cache: dict[tuple, Callable | None] = {}

    def _compile_exec(self) -> None:
        source_literal = repr(self._source)
        fn_code = (
            "import math\n"
            "def _pk_fn(theta, eta, t=0.0, a=None, covariates=None):\n"
            "    if a is None:\n"
            "        a = [0.0] * 10\n"
            "    if covariates is None:\n"
            "        covariates = {}\n"
            "    _locals = {'theta': theta, 'eta': eta, 't': t, 'a': a, **covariates}\n"
            f"    exec({source_literal}, {{'math': math, '_nmtran_sign': _nmtran_sign}}, _locals)\n"
            "    return _locals\n"
        )

        globs: dict[str, Any] = {
            "math": __import__("math"),
            "_nmtran_sign": _nmtran_sign,
        }
        try:
            exec(compile(fn_code, "<pk_code>", "exec"), globs)  # noqa: S102
            self._exec_fn = globs["_pk_fn"]
        except SyntaxError as exc:
            raise CompilerError(f"Syntax error in compiled $PK block: {exc}\n{fn_code}") from exc

    @staticmethod
    def _is_valid_covariate_name(name: str) -> bool:
        return name.isidentifier() and not keyword.iskeyword(name)

    def _compile_direct(self, covariate_names: tuple[str, ...]) -> Callable | None:
        if any(not self._is_valid_covariate_name(name) for name in covariate_names):
            return None

        assignments = "".join(f"    {name} = covariates[{name!r}]\n" for name in covariate_names)
        body = textwrap.indent(self._source, "    ")
        output_names_literal = repr(self._output_names)
        fn_code = (
            "import math\n"
            "def _pk_fn(theta, eta, t=0.0, a=None, covariates=None):\n"
            "    if a is None:\n"
            "        a = [0.0] * 10\n"
            "    if covariates is None:\n"
            "        covariates = {}\n"
            f"{assignments}"
            f"{body}\n"
            "    _locals = locals()\n"
            "    _result = {}\n"
            f"    for _name in {output_names_literal}:\n"
            "        _value = _locals.get(_name)\n"
            "        if isinstance(_value, (int, float)):\n"
            "            _result[_name] = float(_value)\n"
            "    for _name, _value in covariates.items():\n"
            "        if _name not in _result and isinstance(_value, (int, float)):\n"
            "            _result[_name] = float(_value)\n"
            "    return _result\n"
        )

        globs: dict[str, Any] = {
            "math": __import__("math"),
            "_nmtran_sign": _nmtran_sign,
        }
        try:
            exec(compile(fn_code, "<pk_code>", "exec"), globs)  # noqa: S102
        except SyntaxError:
            return None
        return globs["_pk_fn"]

    def __getstate__(self) -> dict:
        # Drop the compiled function — it's not picklable.
        # The worker process recompiles from _source on first call.
        return {
            "_source": self._source,
            "_param_names": self._param_names,
            "_output_names": self._output_names,
            "_exec_fn": None,
            "_direct_fn_cache": {},
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "_output_names"):
            self._output_names = _collect_assigned_names(self._source)
        self._exec_fn = None
        self._direct_fn_cache = {}

    def __call__(
        self,
        theta: list[float],
        eta: list[float],
        t: float = 0.0,
        a: list[float] | None = None,
        covariates: dict[str, float] | None = None,
    ) -> dict[str, float]:
        import hashlib
        cov_map = covariates or {}
        covariate_names = tuple(sorted(cov_map))
        source_hash = hashlib.md5(self._source.encode()).hexdigest()[:8]
        cache_key = (covariate_names, source_hash)
        if cache_key not in self._direct_fn_cache:
            self._direct_fn_cache[cache_key] = self._compile_direct(covariate_names)

        direct_fn = self._direct_fn_cache[cache_key]
        if direct_fn is not None:
            return direct_fn(theta, eta, t, a, cov_map)

        if self._exec_fn is None:
            self._compile_exec()
        raw = self._exec_fn(theta, eta, t, a, cov_map)  # type: ignore[misc]
        result: dict[str, float] = {}
        skip = {"theta", "eta", "t", "a", "covariates", "eps", "math"}
        for k, v in raw.items():
            if k in skip or k.startswith("_"):
                continue
            if isinstance(v, (int, float)):
                result[k] = float(v)
        return result


class CompiledErrorCallable:
    """
    A compiled $ERROR code block.

    Calling signature:
        result = callable(theta, eta, eps, f, ipred)
        → dict with Y, W, IWRES, etc.
    """

    def __init__(self, code: str) -> None:
        self._source = code
        self._fn: Callable | None = None
        self._supports_full_args = True
        self._uses_amounts = bool(re.search(r"\ba\s*\[", code))

    def __getstate__(self) -> dict:
        return {"_source": self._source, "_fn": None, "_uses_amounts": self._uses_amounts}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def _compile(self) -> None:
        source_literal = repr(self._source)
        fn_code = (
            "import math\n"
            "def _error_fn(theta, eta, eps, f, ipred=None, dv=None, t=0.0, a=None, covariates=None, sigma=None):\n"
            "    if ipred is None:\n"
            "        ipred = f\n"
            "    if a is None:\n"
            "        a = [0.0] * 10\n"
            "    if covariates is None:\n"
            "        covariates = {}\n"
            "    if sigma is None:\n"
            "        sigma = []\n"
            "    _locals = {\n"
            "        'theta': theta,\n"
            "        'eta': eta,\n"
            "        'eps': eps,\n"
            "        'f': f,\n"
            "        'ipred': ipred,\n"
            "        'dv': dv,\n"
            "        't': t,\n"
            "        'a': a,\n"
            "        'sigma': sigma,\n"
            "        'y': f,\n"
            "        'w': None,\n"
            "        'ires': 0.0,\n"
            "        'iwres': 0.0,\n"
            "        **covariates,\n"
            "    }\n"
            f"    exec({source_literal}, {{'math': math, '_nmtran_sign': _nmtran_sign}}, _locals)\n"
            "    return _locals\n"
        )
        globs: dict[str, Any] = {
            "math": __import__("math"),
            "_nmtran_sign": _nmtran_sign,
        }
        try:
            exec(compile(fn_code, "<error_code>", "exec"), globs)  # noqa: S102
            self._fn = globs["_error_fn"]
        except SyntaxError as exc:
            raise CompilerError(f"Syntax error in compiled $ERROR block: {exc}\n{fn_code}") from exc

    def __call__(
        self,
        theta: list[float],
        eta: list[float],
        eps: list[float],
        f: float,
        ipred: float | None = None,
        dv: float | None = None,
        t: float = 0.0,
        a: list[float] | None = None,
        covariates: dict[str, float] | None = None,
        sigma: Any | None = None,
    ) -> dict[str, float]:
        raw = self._call_raw(theta, eta, eps, f, ipred, dv, t, a, covariates, sigma)
        result: dict[str, float] = {}
        skip = {"theta", "eta", "eps", "f", "dv", "t", "a", "covariates", "sigma", "math"}
        # Map lowercase translated names back to NONMEM uppercase names
        lower_to_nm = {"y": "Y", "w": "W", "ires": "IRES", "iwres": "IWRES", "ipred": "IPRED"}
        for k, v in raw.items():
            if k in skip or k.startswith("_"):
                continue
            if isinstance(v, (int, float)):
                nm_key = lower_to_nm.get(k, k.upper() if k.islower() else k)
                result[nm_key] = float(v)
        return result

    def _call_raw(
        self,
        theta: list[float],
        eta: list[float],
        eps: list[float],
        f: float,
        ipred: float | None = None,
        dv: float | None = None,
        t: float = 0.0,
        a: list[float] | None = None,
        covariates: dict[str, float] | None = None,
        sigma: Any | None = None,
    ) -> dict[str, Any]:
        if self._fn is None:
            self._compile()
        return self._fn(theta, eta, eps, f, ipred, dv, t, a, covariates or {}, sigma)  # type: ignore[misc]


class CompiledDESCallable:
    """
    A compiled $DES code block (ODE right-hand side).

    Calling signature:
        dadt = callable(t, a, pk_params, theta, eta)
        → list[float] of dA(n)/dt values

    The pk_params dict values (K, V, K12, K21, etc.) are injected into the
    user's code namespace so that the DES code can reference them by name
    (e.g., ``DADT(1) = -K * A(1)`` which compiles to ``dadt[0] = -K * a[0]``).

    This is implemented by running the translated user code via exec() with
    a mutable locals dict that is pre-populated from pk_params, plus t, a,
    dadt, theta, eta, and math.
    """

    def __init__(self, code: str, n_compartments: int = 10) -> None:
        self._source = code
        self._n_compartments = n_compartments
        # Store the translated source for use in exec; compiled lazily.
        self._source_translated: str = code
        self._fn: Callable | None = None
        # Cached Numba @njit function (compiled on first use when numba is available)
        self._numba_fn: Callable | None = None
        self._numba_param_keys: tuple[str, ...] | None = None
        # Cached Numba @cfunc for use with scipy LowLevelCallable
        self._cfunc_fn: Any | None = None
        self._cfunc_cache_key: tuple | None = None

    def __getstate__(self) -> dict:
        return {
            "_source": self._source,
            "_n_compartments": self._n_compartments,
            "_source_translated": self._source_translated,
            "_fn": None,
            "_numba_fn": None,
            "_numba_param_keys": None,
            "_cfunc_fn": None,
            "_cfunc_cache_key": None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if "_numba_fn" not in self.__dict__:
            self._numba_fn = None
        if "_numba_param_keys" not in self.__dict__:
            self._numba_param_keys = None
        if "_cfunc_fn" not in self.__dict__:
            self._cfunc_fn = None
        if "_cfunc_cache_key" not in self.__dict__:
            self._cfunc_cache_key = None

    @property
    def translated_source(self) -> str:
        """The translated (Python) DES source, for use by JIT compilers."""
        return self._source_translated

    def try_compile_numba(self, param_keys: tuple[str, ...]) -> bool:
        """
        Attempt to compile the DES code to a Numba @njit function.

        The compiled function signature is ``_des_numba(t, a, p)`` where
        ``a`` and ``p`` are 1-D float64 NumPy arrays.  ``p[i]`` corresponds
        to ``param_keys[i]``.  Returns True on success, False otherwise.
        """
        if self._numba_fn is not None and self._numba_param_keys == param_keys:
            return True
        try:
            import numba as _numba  # noqa: PLC0415
            import numpy as _np    # noqa: PLC0415
        except ImportError:
            return False

        n = self._n_compartments
        unpack = "\n".join(
            f"    {key} = p[{i}]" for i, key in enumerate(param_keys)
        )
        body = "\n".join("    " + ln for ln in self._source_translated.splitlines())
        fn_src = (
            "import numpy as _np\n"
            # cache=True requires a real on-disk source file; exec'd strings have
            # no file path so the Numba cache locator raises RuntimeError.  Use
            # cache=False here — the compiled native code is kept in the Dispatcher
            # object for the lifetime of the Python process.
            "@_numba.njit(cache=False)\n"
            "def _des_numba(t, a, p):\n"
            f"    dadt = _np.zeros({n})\n"
            f"{unpack}\n"
            f"{body}\n"
            "    return dadt\n"
        )
        globs: dict[str, Any] = {"_numba": _numba, "_np": _np}
        try:
            exec(compile(fn_src, "<des_numba>", "exec"), globs)  # noqa: S102
            # Trigger first compilation so subsequent calls reuse the compiled code.
            # Use ones (not zeros) to avoid div-by-zero in models like Michaelis-Menten
            # where a parameter appears in a denominator (KM*V + A(1) = 0 + 0 = 0).
            globs["_des_numba"](0.0, _np.ones(n), _np.ones(len(param_keys)))
            self._numba_fn = globs["_des_numba"]
            self._numba_param_keys = param_keys
            return True
        except Exception:
            return False

    def try_compile_cfunc(
        self,
        param_keys: tuple[str, ...],
        max_infusions: int = 10,
    ) -> bool:
        """
        Attempt to compile the DES code to a Numba ``@cfunc`` for use with
        ``scipy.integrate.solve_ivp`` via a ``LowLevelCallable``.

        The cfunc has the C-compatible signature expected by scipy::

            int _des_cfunc(double t, double *y, double *dydt, void *data)

        The ``data`` void pointer must point to a ``float64`` array with the
        layout::

            [p[0], ..., p[n_params-1], n_infusions,
             cmt0, rate0, end_t0, cmt1, rate1, end_t1, ...]

        Parameters are in the same order as ``param_keys``.  Infusion data
        is packed/updated by :func:`openpkpd.pk.ode.jit.make_llc_rhs`.

        Returns ``True`` on success, ``False`` if Numba is unavailable or the
        DES code cannot be compiled (e.g., incompatible constructs).
        """
        cache_key = (param_keys, max_infusions)
        if self._cfunc_fn is not None and self._cfunc_cache_key == cache_key:
            return True

        try:
            import math as _math          # noqa: PLC0415
            import numba as _numba        # noqa: PLC0415
            import numpy as _np           # noqa: PLC0415
        except ImportError:
            return False

        n = self._n_compartments
        n_params = len(param_keys)
        # data layout: params first, then [n_infusions, cmt, rate, end_t, ...]
        data_size = n_params + 1 + max_infusions * 3

        unpack = "\n".join(
            f"    {key} = p[{i}]" for i, key in enumerate(param_keys)
        )
        body = "\n".join("    " + ln for ln in self._source_translated.splitlines())

        fn_src = (
            # cache=True is incompatible with exec'd code (no real file path);
            # the compiled function pointer is retained in the cfunc object.
            "@_numba.cfunc(\n"
            "    _numba.types.int32(\n"
            "        _numba.float64,\n"
            "        _numba.types.CPointer(_numba.float64),\n"
            "        _numba.types.CPointer(_numba.float64),\n"
            "        _numba.types.voidptr,\n"
            "    )\n"
            ")\n"
            "def _des_cfunc(t, y, dydt, data):\n"
            f"    a    = _numba.carray(y,    {n})\n"
            f"    dadt = _numba.carray(dydt,  {n})\n"
            f"    p    = _numba.carray(data,  {data_size}, dtype=_np.float64)\n"
            # zero output array
            f"    for _i in range({n}):\n"
            f"        dadt[_i] = 0.0\n"
            # unpack PK parameters
            f"{unpack}\n"
            # user DES body (writes to dadt[i] which aliases dydt)
            f"{body}\n"
            # apply infusions encoded in data array
            f"    _n_inf = int(p[{n_params}])\n"
            f"    for _j in range(_n_inf):\n"
            f"        _base  = {n_params} + 1 + _j * 3\n"
            f"        _cmt   = int(p[_base])\n"
            f"        _rate  = p[_base + 1]\n"
            f"        _end_t = p[_base + 2]\n"
            f"        if t <= _end_t + 1e-14 and 0 <= _cmt < {n}:\n"
            f"            dadt[_cmt] += _rate\n"
            f"    return 0\n"
        )

        globs: dict[str, Any] = {"_numba": _numba, "_np": _np, "math": _math}

        # Provide a Numba-traceable _nmtran_sign if the DES code uses SIGN()
        if "_nmtran_sign" in self._source_translated:
            import math as _m  # noqa: PLC0415

            def _nmtran_sign_nb(a: float, b: float) -> float:
                return abs(a) * _m.copysign(1.0, b)

            globs["_nmtran_sign"] = _nmtran_sign_nb

        try:
            exec(compile(fn_src, "<des_cfunc>", "exec"), globs)  # noqa: S102
            self._cfunc_fn = globs["_des_cfunc"]
            self._cfunc_cache_key = cache_key
            return True
        except Exception:
            return False

    def as_multidose_probe(
        self,
        param_keys: tuple[str, ...],
        n_states: int,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        fd_eps: float = 1e-5,
    ) -> "tuple[Any, Any, Any, Any] | None":
        """Build multi-dose probe callables backed by the Numba @njit RHS.

        Returns ``(state_probe_fn, sens_probe_fn, infusion_state_probe_fn,
        infusion_sens_probe_fn)`` or ``None`` if Numba compilation fails.

        Probe signatures::

            state_probe_fn(obs_times, dose_times, dose_amts, theta)
                -> list[list[float]]  # shape (n_obs, n_states)
            infusion_state_probe_fn(obs_times, dose_times, dose_amts, dose_rates, theta)
                -> list[list[float]]
            sens_probe_fn(obs_times, dose_times, dose_amts, theta)
                -> (states, sens)     # sens shape (n_obs, n_params, n_states)
            infusion_sens_probe_fn(obs_times, dose_times, dose_amts, dose_rates, theta)
                -> (states, sens)
        """
        if not self.try_compile_numba(param_keys):
            return None

        import numpy as _np  # noqa: PLC0415

        _des_fn = self._numba_fn
        _n = n_states
        _rtol = rtol
        _atol = atol
        _eps = fd_eps
        _n_params = len(param_keys)

        def _integrate(obs_times, dose_times, dose_amts, dose_rates_in, theta):
            """Piecewise multi-dose ODE integration."""
            from openpkpd.pk.ode.jit import numpy_rk45_solve  # noqa: PLC0415

            params = _np.array(theta, dtype=_np.float64)
            obs_arr = _np.asarray(obs_times, dtype=float)
            n_obs = len(obs_arr)
            result = _np.zeros((n_obs, _n), dtype=float)
            if n_obs == 0:
                return result

            sort_idx = _np.argsort(obs_arr, kind="stable")
            obs_sorted = obs_arr[sort_idx]
            t_max = float(obs_sorted[-1])

            d_times = [float(t) for t in dose_times]
            d_amts = [float(a) for a in dose_amts]
            d_rates = [float(r) for r in (dose_rates_in or [0.0] * len(d_times))]

            bp_set = {0.0}
            for t_d, a_d, r_d in zip(d_times, d_amts, d_rates):
                if t_d <= t_max + 1e-12:
                    bp_set.add(t_d)
                    if r_d > 0.0 and a_d > 0.0:
                        et = t_d + a_d / r_d
                        if et <= t_max + 1e-12:
                            bp_set.add(et)
            bp_set.add(t_max)
            breakpoints = sorted(bp_set)

            y = _np.zeros(_n, dtype=float)
            active_inf: dict = {}
            obs_ptr = 0

            def make_rhs(inf):
                def rhs(t, yy):
                    dydt = _des_fn(t, yy, params)
                    for cmt, (rate, et) in inf.items():
                        if t <= et + 1e-14 and 0 <= cmt < _n:
                            dydt[cmt] += rate
                    return dydt
                return rhs

            for t_d, a_d, r_d in zip(d_times, d_amts, d_rates):
                if abs(t_d) < 1e-14:
                    if r_d == 0.0:
                        y[0] += a_d
                    else:
                        active_inf[0] = (r_d, a_d / r_d)

            while obs_ptr < n_obs and obs_sorted[obs_ptr] < 1e-14:
                result[sort_idx[obs_ptr]] = y.copy()
                obs_ptr += 1

            prev_t = 0.0
            for bp in breakpoints:
                if bp <= prev_t + 1e-14:
                    continue

                seg_t: list[float] = []
                seg_i: list[int] = []
                ptr = obs_ptr
                while ptr < n_obs and obs_sorted[ptr] <= bp + 1e-12:
                    if obs_sorted[ptr] > prev_t + 1e-14:
                        seg_t.append(float(obs_sorted[ptr]))
                        seg_i.append(int(sort_idx[ptr]))
                    ptr += 1
                obs_ptr = ptr

                t_eval = _np.array(sorted(set(seg_t))) if seg_t else _np.array([bp])
                tf = float(t_eval[-1])
                seg_states = numpy_rk45_solve(
                    make_rhs(dict(active_inf)), prev_t, tf, y.copy(), t_eval, _rtol, _atol,
                )
                for t_o, orig_i in zip(seg_t, seg_i):
                    idx = min(int(_np.searchsorted(t_eval, t_o)), len(seg_states) - 1)
                    result[orig_i] = seg_states[idx]
                y = seg_states[-1]

                if tf < bp - 1e-14:
                    extra = numpy_rk45_solve(
                        make_rhs(dict(active_inf)), tf, bp, y.copy(), _np.array([bp]), _rtol, _atol,
                    )
                    y = extra[-1]

                prev_t = bp
                for t_d, a_d, r_d in zip(d_times, d_amts, d_rates):
                    if abs(t_d - bp) < 1e-12:
                        if r_d == 0.0:
                            y[0] += a_d
                        else:
                            active_inf[0] = (r_d, bp + a_d / r_d)
                active_inf = {c: v for c, v in active_inf.items() if v[1] > bp + 1e-12}

            return result

        def state_probe_fn(obs_times, dose_times, dose_amts, theta):
            return _integrate(obs_times, dose_times, dose_amts, None, theta).tolist()

        def infusion_state_probe_fn(obs_times, dose_times, dose_amts, dose_rates, theta):
            return _integrate(obs_times, dose_times, dose_amts, dose_rates, theta).tolist()

        def sens_probe_fn(obs_times, dose_times, dose_amts, theta):
            theta_arr = list(theta)
            states = _integrate(obs_times, dose_times, dose_amts, None, theta_arr)
            n_obs = len(obs_times)
            sens = _np.zeros((n_obs, _n_params, _n), dtype=float)
            for j in range(_n_params):
                tp = list(theta_arr); tp[j] += _eps
                tm = list(theta_arr); tm[j] -= _eps
                sens[:, j, :] = (
                    _integrate(obs_times, dose_times, dose_amts, None, tp)
                    - _integrate(obs_times, dose_times, dose_amts, None, tm)
                ) / (2.0 * _eps)
            return states.tolist(), sens.tolist()

        def infusion_sens_probe_fn(obs_times, dose_times, dose_amts, dose_rates, theta):
            theta_arr = list(theta)
            states = _integrate(obs_times, dose_times, dose_amts, dose_rates, theta_arr)
            n_obs = len(obs_times)
            sens = _np.zeros((n_obs, _n_params, _n), dtype=float)
            for j in range(_n_params):
                tp = list(theta_arr); tp[j] += _eps
                tm = list(theta_arr); tm[j] -= _eps
                sens[:, j, :] = (
                    _integrate(obs_times, dose_times, dose_amts, dose_rates, tp)
                    - _integrate(obs_times, dose_times, dose_amts, dose_rates, tm)
                ) / (2.0 * _eps)
            return states.tolist(), sens.tolist()

        return state_probe_fn, sens_probe_fn, infusion_state_probe_fn, infusion_sens_probe_fn

    def _compile(self) -> None:
        import math as _math

        n = self._n_compartments
        translated = self._source_translated

        # ── Key optimisation ──────────────────────────────────────────────────
        # Pre-compile the user's DES code to a code object *once*.
        # The original code placed a raw string in the wrapper's closure and
        # called exec(string, ...) on every RHS evaluation, which caused Python
        # to re-parse and re-compile the user code on *every single call*.
        # Profiling shows this accounts for ~52 % of total ODE solve time.
        # Using a pre-compiled code object eliminates the parse/compile step and
        # reduces the exec overhead to pure bytecode execution.
        try:
            user_code_obj = compile(translated, "<des_user_code>", "exec")
        except SyntaxError as exc:
            raise CompilerError(
                f"Syntax error in translated $DES block: {exc}\n{translated}"
            ) from exc

        # Build the wrapper function.  _user_code_obj is a code object (fast),
        # not a string.  We also pass a fixed globals dict so the exec'd code can
        # access math and _nmtran_sign without an extra lookup.
        fn_code = (
            "def _des_fn(t, a, pk_params, theta, eta):\n"
            f"    dadt = [0.0] * {n}\n"
            "    _env = dict(pk_params)\n"
            "    _env['t'] = t\n"
            "    _env['a'] = a\n"
            "    _env['theta'] = theta\n"
            "    _env['eta'] = eta\n"
            "    _env['dadt'] = dadt\n"
            "    _env['math'] = _math\n"
            "    _env['_nmtran_sign'] = _nmtran_sign\n"
            "    exec(_user_code_obj, _des_globals, _env)\n"  # code object — no re-parse
            "    return _env.get('dadt', dadt)\n"
        )

        _des_globals: dict[str, Any] = {
            "_math": _math,
            "math": _math,
            "_nmtran_sign": _nmtran_sign,
        }
        globs: dict[str, Any] = {
            "_math": _math,
            "_nmtran_sign": _nmtran_sign,
            "_user_code_obj": user_code_obj,   # ← code object, not string
            "_des_globals": _des_globals,
        }
        try:
            exec(compile(fn_code, "<des_code>", "exec"), globs)  # noqa: S102
            self._fn = globs["_des_fn"]
        except SyntaxError as exc:
            raise CompilerError(f"Syntax error in compiled $DES block: {exc}\n{fn_code}") from exc

    def __call__(
        self,
        t: float,
        a: list[float],
        pk_params: dict[str, float],
        theta: list[float],
        eta: list[float],
    ) -> list[float]:
        if self._fn is None:
            self._compile()
        return self._fn(t, a, pk_params, theta, eta)  # type: ignore[misc]


# ── Main compiler class ────────────────────────────────────────────────────────


class NMTRANCompiler:
    """
    Translates NM-TRAN code blocks to Python callables.

    Usage:
        compiler = NMTRANCompiler()
        pk_fn = compiler.compile_pk(pk_code)
        error_fn = compiler.compile_error(error_code)
        des_fn = compiler.compile_des(des_code, n_compartments=4)

    The compiled callables are standard exec-based Python callables.
    """

    def __init__(self) -> None:
        self._intrinsics = _INTRINSICS

    def compile_pk(self, code: str) -> CompiledPKCallable:
        """Compile a $PK code block to a Python callable."""
        translated = _translate_block(code, self._intrinsics)
        return CompiledPKCallable(translated)

    def compile_error(self, code: str) -> CompiledErrorCallable:
        """Compile a $ERROR code block to a Python callable."""
        translated = _translate_block(code, self._intrinsics)
        return CompiledErrorCallable(translated)

    def compile_des(self, code: str, n_compartments: int = 10) -> CompiledDESCallable:
        """Compile a $DES code block to a Python callable."""
        translated = _translate_block(code, self._intrinsics)
        return CompiledDESCallable(translated, n_compartments=n_compartments)

    def compile_pred(self, code: str) -> CompiledPKCallable:
        """Compile a $PRED code block (same interface as $PK)."""
        translated = _translate_block(code, self._intrinsics)
        return CompiledPKCallable(translated)
