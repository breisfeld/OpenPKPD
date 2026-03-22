"""
NMTRANCompiler: Translates $PK, $DES, and $ERROR code blocks to Python callables.

NM-TRAN is FORTRAN-77 style code with NONMEM extensions. This compiler
handles the core subset needed for PK/PD modeling.

Stage 1: exec()-based Python callable (correct results, no autodiff)
Stage 2: JAX-compatible form for gradient computation (optional)

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

FORTRAN intrinsics mapped to Python/NumPy/JAX equivalents:
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

_INTRINSICS_JAX: dict[str, str] = {
    "EXP": "jnp.exp",
    "LOG": "jnp.log",
    "LOG10": "jnp.log10",
    "SQRT": "jnp.sqrt",
    "ABS": "jnp.abs",
    "MOD": "jnp.mod",
    "MAX": "jnp.maximum",
    "MIN": "jnp.minimum",
    "INT": "jnp.int32",
    "FLOAT": "jnp.float32",
    "DBLE": "jnp.float64",
    "ATAN2": "jnp.arctan2",
    "SIN": "jnp.sin",
    "COS": "jnp.cos",
    "TAN": "jnp.tan",
    "ASIN": "jnp.arcsin",
    "ACOS": "jnp.arccos",
    "ATAN": "jnp.arctan",
    "SIGN": "jnp.sign",
    "GAMLN": "jax.scipy.special.gammaln",
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
    for line in lines:
        if _is_comment_line(line):
            continue
        translated = _translate_line(line.strip(), intrinsics)
        result.append(translated)
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
        self._direct_fn_cache: dict[tuple[str, ...], Callable | None] = {}

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
        cov_map = covariates or {}
        covariate_names = tuple(sorted(cov_map))
        if covariate_names not in self._direct_fn_cache:
            self._direct_fn_cache[covariate_names] = self._compile_direct(covariate_names)

        direct_fn = self._direct_fn_cache[covariate_names]
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

    def __getstate__(self) -> dict:
        return {
            "_source": self._source,
            "_n_compartments": self._n_compartments,
            "_source_translated": self._source_translated,
            "_fn": None,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def _compile(self) -> None:
        import math as _math

        n = self._n_compartments
        translated = self._source_translated

        # Build a wrapper function that:
        # 1. Initialises dadt as a zero list.
        # 2. Populates _env with pk_params, t, a, theta, eta, dadt, math.
        # 3. Runs the user's translated DES code via exec() inside _env.
        # 4. Returns _env['dadt'] (mutated by the user code).
        fn_code = (
            "import math as _math\n"
            "def _des_fn(t, a, pk_params, theta, eta):\n"
            f"    dadt = [0.0] * {n}\n"
            "    _env = dict(pk_params)\n"
            "    _env.update({\n"
            "        't': t,\n"
            "        'a': a,\n"
            "        'theta': theta,\n"
            "        'eta': eta,\n"
            "        'dadt': dadt,\n"
            "        'math': _math,\n"
            "        '_nmtran_sign': _nmtran_sign,\n"
            "    })\n"
            "    exec(_user_code, {'math': _math, '_nmtran_sign': _nmtran_sign}, _env)\n"
            "    return _env.get('dadt', dadt)\n"
        )

        globs: dict[str, Any] = {
            "_math": _math,
            "math": _math,
            "_nmtran_sign": _nmtran_sign,
            "_user_code": translated,
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

    The compiled callables are Stage 1 (exec-based). Stage 2 (JAX-compatible)
    is produced by compile_pk_jax() etc. when JAX is available.
    """

    def __init__(self, use_jax: bool = False) -> None:
        self.use_jax = use_jax
        self._intrinsics = _INTRINSICS_JAX if use_jax else _INTRINSICS

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
