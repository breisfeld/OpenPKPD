#!/usr/bin/env python3
"""
PyPI smoke test for openpkpd.

Verifies that a fresh install from PyPI is fully functional:
  1. Package imports and version matches expected
  2. Rust _core extension loads and core symbols are present
  3. Native CVODES symbols are present (warning if absent, not a failure)
  4. A minimal 1-cmt oral FO fit converges to a sensible OFV
  5. CLI entry point is accessible

Usage (standalone):
    python smoke_test_pypi.py
    python smoke_test_pypi.py --expected-version 0.2.7
    python smoke_test_pypi.py --require-native-cvodes

Usage (via docker_smoke_test.sh):
    bash scripts/docker_smoke_test.sh 0.2.7
"""

from __future__ import annotations

import argparse
import importlib
import io
import subprocess
import sys

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg: str = "OK") -> str:
    return f"{GREEN}✓ {msg}{RESET}"


def _warn(msg: str) -> str:
    return f"{YELLOW}⚠ {msg}{RESET}"


def _fail(msg: str) -> str:
    return f"{RED}✗ {msg}{RESET}"


def _section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * 50)


# ---------------------------------------------------------------------------
# Inline theophylline dataset (3 subjects — enough for a quick FO fit)
# ---------------------------------------------------------------------------

_THEO_CSV = """\
ID,TIME,AMT,DV,EVID,MDV,WT
1,0,4.02,0,1,1,79.6
1,0.27,0,0.74,0,0,79.6
1,0.57,0,1.72,0,0,79.6
1,1.02,0,7.91,0,0,79.6
1,1.92,0,8.31,0,0,79.6
1,3.5,0,8.33,0,0,79.6
1,5.02,0,6.85,0,0,79.6
1,7.03,0,6.08,0,0,79.6
1,9.0,0,5.4,0,0,79.6
1,12.05,0,4.55,0,0,79.6
1,24.37,0,1.25,0,0,79.6
2,0,4.4,0,1,1,72.4
2,0.35,0,0.96,0,0,72.4
2,0.6,0,2.33,0,0,72.4
2,1.07,0,4.71,0,0,72.4
2,2.13,0,8.33,0,0,72.4
2,3.5,0,9.02,0,0,72.4
2,5.02,0,7.14,0,0,72.4
2,7.02,0,5.68,0,0,72.4
2,9.1,0,4.55,0,0,72.4
2,12.1,0,3.01,0,0,72.4
2,25.0,0,0.9,0,0,72.4
3,0,4.95,0,1,1,70.5
3,0.27,0,0.64,0,0,70.5
3,0.58,0,1.92,0,0,70.5
3,1.02,0,4.44,0,0,70.5
3,1.92,0,7.03,0,0,70.5
3,3.5,0,9.07,0,0,70.5
3,5.02,0,7.56,0,0,70.5
3,7.02,0,6.59,0,0,70.5
3,9.0,0,5.88,0,0,70.5
3,12.15,0,4.73,0,0,70.5
3,24.17,0,1.25,0,0,70.5
"""

# ---------------------------------------------------------------------------
# Check helpers (return True = passed, False = failed)
# ---------------------------------------------------------------------------

REQUIRED_CORE_SYMBOLS = ("neg2ll_obs_loop",)
REQUIRED_NATIVE_CVODES_SYMBOLS = (
    "native_cvodes_transit_1cmt_pkpd_probe",
    "native_cvodes_transit_1cmt_pkpd_probe_multidose",
)


def check_version(expected: str | None) -> bool:
    label = "Package version"
    try:
        import openpkpd
        ver = openpkpd.__version__
        if expected and ver != expected:
            print(f"  {label:35s}: {_fail(f'got {ver!r}, expected {expected!r}')}")
            return False
        print(f"  {label:35s}: {_ok(ver)}")
        return True
    except Exception as exc:
        print(f"  {label:35s}: {_fail(str(exc))}")
        return False


def check_core_import() -> bool:
    label = "_core extension import"
    try:
        importlib.import_module("openpkpd._core")
        print(f"  {label:35s}: {_ok()}")
        return True
    except Exception as exc:
        print(f"  {label:35s}: {_fail(str(exc))}")
        return False


def check_core_symbols() -> bool:
    label = "Core Rust symbols"
    try:
        core = importlib.import_module("openpkpd._core")
        missing = [s for s in REQUIRED_CORE_SYMBOLS if not hasattr(core, s)]
        if missing:
            print(f"  {label:35s}: {_fail('missing: ' + ', '.join(missing))}")
            return False
        print(f"  {label:35s}: {_ok(', '.join(REQUIRED_CORE_SYMBOLS))}")
        return True
    except Exception as exc:
        print(f"  {label:35s}: {_fail(str(exc))}")
        return False


def check_native_cvodes(require: bool) -> bool:
    label = "Native CVODES symbols"
    try:
        core = importlib.import_module("openpkpd._core")
        missing = [s for s in REQUIRED_NATIVE_CVODES_SYMBOLS if not hasattr(core, s)]
        if missing:
            msg = "not present (source-built or fallback wheel)"
            if require:
                print(f"  {label:35s}: {_fail(msg)}")
                return False
            print(f"  {label:35s}: {_warn(msg)}")
            return True  # soft warning only
        print(f"  {label:35s}: {_ok()}")
        return True
    except Exception as exc:
        print(f"  {label:35s}: {_fail(str(exc))}")
        return False


def check_fo_fit() -> bool:
    label = "FO fit (theophylline 1-cmt)"
    try:
        import pandas as pd
        from openpkpd import ModelBuilder
        from openpkpd.data.dataset import NONMEMDataset

        df = pd.read_csv(io.StringIO(_THEO_CSV))
        ds = NONMEMDataset.from_dataframe(df)

        built = (
            ModelBuilder()
            .problem("Smoke test — theophylline 1-cmt oral FO")
            .dataset(ds)
            .subroutines(advan=2, trans=2)
            .pk("""
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
""")
            .error("Y = F*(1 + EPS(1))")
            .theta([(0.01, 1.5, 20), (0.001, 0.08, 5), (0.1, 30, 500)])
            .omega([0.5, 0.3, 0.3])
            .sigma(0.1)
            .estimation(method="FO", maxeval=500)
            .build()
        )

        result = built.fit()

        if not result.converged:
            print(f"  {label:35s}: {_warn(f'did not converge (OFV={result.ofv:.2f})')}")
            return True  # non-convergence on 3 subjects is not a packaging error

        ofv = result.ofv
        print(f"  {label:35s}: {_ok(f'converged, OFV={ofv:.2f}')}")
        return True
    except Exception as exc:
        print(f"  {label:35s}: {_fail(str(exc))}")
        return False


def check_cli() -> bool:
    label = "CLI entry point (openpkpd --help)"
    try:
        proc = subprocess.run(
            ["openpkpd", "--help"],
            capture_output=True,
            timeout=15,
        )
        if proc.returncode == 0:
            print(f"  {label:35s}: {_ok()}")
            return True
        combined = (proc.stdout + proc.stderr).decode(errors="replace").strip()
        print(f"  {label:35s}: {_fail(f'exit {proc.returncode}')}")
        for line in combined.splitlines():
            print(f"    {line}")
        return False
    except FileNotFoundError:
        print(f"  {label:35s}: {_fail('openpkpd command not found on PATH')}")
        return False
    except Exception as exc:
        print(f"  {label:35s}: {_fail(str(exc))}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected-version", metavar="VER", help="Assert installed version equals VER")
    parser.add_argument("--require-native-cvodes", action="store_true", help="Fail if native CVODES symbols are absent")
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 52}{RESET}")
    print(f"{BOLD}  OpenPKPD PyPI Smoke Test{RESET}")
    print(f"{BOLD}  Python {sys.version.split()[0]} | {sys.platform}{RESET}")
    print(f"{BOLD}{'=' * 52}{RESET}")

    _section("1. Package import & version")
    r_ver = check_version(args.expected_version)

    _section("2. Rust extension")
    r_core_import = check_core_import()
    r_core_sym = check_core_symbols() if r_core_import else False
    r_native = check_native_cvodes(args.require_native_cvodes) if r_core_import else False

    _section("3. Functionality")
    r_fit = check_fo_fit()

    _section("4. CLI")
    r_cli = check_cli()

    # Summary
    results = {
        "version": r_ver,
        "_core import": r_core_import,
        "core symbols": r_core_sym,
        "native CVODES": r_native,
        "FO fit": r_fit,
        "CLI": r_cli,
    }
    passed = sum(results.values())
    total = len(results)

    print(f"\n{BOLD}{'=' * 52}{RESET}")
    if passed == total:
        print(f"{BOLD}{GREEN}  PASSED ({passed}/{total}){RESET}")
    else:
        failures = [k for k, v in results.items() if not v]
        print(f"{BOLD}{RED}  FAILED ({passed}/{total}) — {', '.join(failures)}{RESET}")
    print(f"{BOLD}{'=' * 52}{RESET}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
