"""Smoke-test a native-cvodes wheel.

Two modes:

  --wheel <path>   Extract the wheel zip to a temp directory, shadow the
                   editable install, and import from there.  This is the
                   correct mode for verifying a freshly-built wheel file
                   without installing it into the environment.

  (no --wheel)     Strip the repo root from sys.path and import from
                   whatever is installed in the active environment.  Useful
                   when testing a wheel that has already been installed in a
                   clean virtualenv.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path


REQUIRED_NATIVE_CVODES_SYMBOLS = (
    "native_cvodes_linear_probe",
    "native_cvodes_advan6_mixed_pkpd_probe",
    "native_cvodes_advan6_mixed_pkpd_repeat_probe",
)


def _sanitize_sys_path() -> None:
    """Remove the repo root and scripts dir so the editable install is not used."""
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    blocked = {"", str(repo_root), str(scripts_dir)}
    sys.path[:] = [entry for entry in sys.path if entry not in blocked]


def _shadow_with_wheel(wheel_path: Path) -> Path:
    """Extract *wheel_path* to a temp dir and prepend it to sys.path.

    The patchelf-repaired RPATH inside the .so resolves SUNDIALS libs
    relative to the extension file itself, so extraction preserves the
    bundled-library contract regardless of the temp dir location.

    Returns the temp directory (caller is responsible for cleanup if needed;
    in practice we let the process exit handle it).
    """
    tmp = Path(tempfile.mkdtemp(prefix="openpkpd_wheel_check_"))
    with zipfile.ZipFile(wheel_path) as zf:
        zf.extractall(tmp)

    # Remove any existing openpkpd entries so the extracted copy wins.
    sys.path[:] = [
        entry for entry in sys.path
        if "openpkpd" not in entry and "site-packages" not in entry
    ]
    sys.path.insert(0, str(tmp))
    return tmp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        metavar="PATH",
        help="path to a .whl file to verify (extracts to temp dir, does not install)",
    )
    parser.add_argument(
        "--require-native-cvodes",
        action="store_true",
        help="fail if the wheel does not export native-cvodes symbols",
    )
    args = parser.parse_args()

    if args.wheel:
        wheel_path = Path(args.wheel).resolve()
        if not wheel_path.exists():
            print(f"error: wheel not found: {wheel_path}")
            return 1
        tmp_dir = _shadow_with_wheel(wheel_path)
        print(f"wheel_path={wheel_path}")
        print(f"extracted_to={tmp_dir}")
    else:
        _sanitize_sys_path()

    print(f"OPENPKPD_NATIVE_DEV={os.environ.get('OPENPKPD_NATIVE_DEV', '')!r}")
    print(f"sys_path_entries={len(sys.path)}")

    try:
        core = importlib.import_module("openpkpd._core")
    except Exception as exc:  # pragma: no cover - exercised in packaging smoke runs
        print(f"wheel_core_import=no error={exc!r}")
        return 1

    print("wheel_core_import=yes")
    has_native = all(hasattr(core, name) for name in REQUIRED_NATIVE_CVODES_SYMBOLS)
    print(f"wheel_native_cvodes_symbols_ok={'yes' if has_native else 'no'}")
    if has_native:
        print("wheel_native_cvodes_symbols=" + ",".join(REQUIRED_NATIVE_CVODES_SYMBOLS))
    elif args.require_native_cvodes:
        missing = [name for name in REQUIRED_NATIVE_CVODES_SYMBOLS if not hasattr(core, name)]
        print("wheel_missing_native_cvodes_symbols=" + ",".join(missing))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
