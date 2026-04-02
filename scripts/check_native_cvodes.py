"""Smoke-test the optional native CVODES core path.

This script is intentionally small and packaging-oriented. It verifies:

- the ``openpkpd._core`` extension can be imported
- baseline core symbols are present
- the optional ``native-cvodes`` symbols are present when requested

It does not run a full benchmark; use the spike scripts for that.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys


REQUIRED_CORE_SYMBOLS = ("neg2ll_obs_loop",)
REQUIRED_NATIVE_CVODES_SYMBOLS = (
    "native_cvodes_transit_1cmt_pkpd_probe",
    "native_cvodes_transit_1cmt_pkpd_probe_multidose",
)


def _bool_label(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-native-cvodes",
        action="store_true",
        help="fail if the native-cvodes symbols are not exported",
    )
    args = parser.parse_args()

    print(f"OPENPKPD_NATIVE_DEV={os.environ.get('OPENPKPD_NATIVE_DEV', '')!r}")

    try:
        core = importlib.import_module("openpkpd._core")
    except Exception as exc:  # pragma: no cover - exercised in packaging smoke runs
        print(f"core_import=no error={exc!r}")
        return 1

    print("core_import=yes")

    missing_core = [name for name in REQUIRED_CORE_SYMBOLS if not hasattr(core, name)]
    print(f"core_symbols_ok={_bool_label(not missing_core)}")
    if missing_core:
        print("missing_core_symbols=" + ",".join(missing_core))
        return 1

    has_native = all(hasattr(core, name) for name in REQUIRED_NATIVE_CVODES_SYMBOLS)
    print(f"native_cvodes_symbols_ok={_bool_label(has_native)}")
    if has_native:
        print("native_cvodes_symbols=" + ",".join(REQUIRED_NATIVE_CVODES_SYMBOLS))
    elif args.require_native_cvodes:
        missing = [name for name in REQUIRED_NATIVE_CVODES_SYMBOLS if not hasattr(core, name)]
        print("missing_native_cvodes_symbols=" + ",".join(missing))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
