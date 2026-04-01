"""Helpers for optional native-extension loading.

This is intentionally conservative:

- if the core extension imports normally, do nothing special
- if a packaged SUNDIALS-backed native path needs bundled shared libraries, try
  package-adjacent locations first
- allow explicit development fallback to local Cargo build outputs only when
  requested
- never raise just because the native path is unavailable
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
from pathlib import Path


def _deduplicate_dirs(dirs: list[Path]) -> list[Path]:
    """Return a deduplicated, order-preserving list of existing directories."""
    seen: set[Path] = set()
    out: list[Path] = []
    for path in dirs:
        resolved = path.resolve()
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _development_sundials_lib_dirs() -> list[Path]:
    dirs: list[Path] = []

    env_dirs = os.environ.get("OPENPKPD_SUNDIALS_LIBDIRS")
    if env_dirs:
        for entry in env_dirs.split(os.pathsep):
            if entry:
                dirs.append(Path(entry))

    project_root = Path(__file__).resolve().parents[2]
    rust_target = project_root / "rust" / "target"
    for mode in ("release", "debug"):
        build_dir = rust_target / mode / "build"
        if not build_dir.exists():
            continue
        dirs.extend(sorted(build_dir.glob("sundials-sys-*/out/lib")))

    return _deduplicate_dirs(dirs)


def _native_development_fallback_enabled() -> bool:
    value = os.environ.get("OPENPKPD_NATIVE_DEV", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _candidate_sundials_lib_dirs() -> list[Path]:
    dirs = _candidate_package_lib_dirs()
    if _native_development_fallback_enabled():
        dirs.extend(_development_sundials_lib_dirs())

    return _deduplicate_dirs(dirs)


def _candidate_package_lib_dirs() -> list[Path]:
    dirs: list[Path] = []

    package_dir = Path(__file__).resolve().parent
    dirs.extend(
        [
            package_dir / ".libs",
            package_dir / "_core.libs",
            package_dir.parent / "openpkpd.libs",
            package_dir.parent / ".libs",
        ]
    )

    spec = importlib.util.find_spec("openpkpd._core")
    origin = getattr(spec, "origin", None)
    if origin:
        ext_dir = Path(origin).resolve().parent
        dirs.extend(
            [
                ext_dir / ".libs",
                ext_dir / "_core.libs",
                ext_dir.parent / "openpkpd.libs",
                ext_dir.parent / ".libs",
            ]
        )

    return dirs


def _preload_sundials_libs() -> None:
    mode = getattr(ctypes, "RTLD_GLOBAL", os.RTLD_GLOBAL)

    for lib_dir in _candidate_sundials_lib_dirs():
        candidates = sorted(lib_dir.glob("libsundials_*.so*"))
        if not candidates:
            continue
        ordered = sorted(
            candidates,
            key=lambda path: (
                0 if path.name.startswith("libsundials_nvecserial") else
                1 if path.name.startswith("libsundials_cvodes") else
                2,
                path.name,
            ),
        )
        loaded_any = False
        for path in ordered:
            try:
                ctypes.CDLL(str(path), mode=mode)
                loaded_any = True
            except OSError:
                pass
        if loaded_any:
            return


def import_core_symbol(name: str):
    """Import a symbol from ``openpkpd._core`` with optional bundled-lib preload."""
    try:
        module = importlib.import_module("openpkpd._core")
    except ImportError as first_error:
        _preload_sundials_libs()
        try:
            module = importlib.import_module("openpkpd._core")
        except ImportError:
            raise first_error
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"openpkpd._core does not export {name!r}") from exc
