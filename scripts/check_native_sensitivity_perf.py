"""Serial native-CVODES sensitivity performance gate.

This is an explicit release/validation check for the transit 1-cmt PK/PD
forward-sensitivity probe. It is intentionally separate from the default
pytest suite because wall-clock benchmarks are unstable under xdist worker
contention.
"""

from __future__ import annotations

import argparse
import sys
import time


def _load_required_symbol(name: str):
    try:
        from openpkpd._native import import_core_symbol
    except Exception as exc:  # pragma: no cover - packaging/runtime failure
        raise RuntimeError(f"failed to import openpkpd._native: {exc!r}") from exc

    symbol = import_core_symbol(name)
    if symbol is None:
        raise RuntimeError(f"required native symbol {name!r} is not available")
    return symbol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-reps",
        type=int,
        default=200,
        help="number of repeated benchmark iterations per measurement",
    )
    parser.add_argument(
        "--n-fd-pairs",
        type=int,
        default=10,
        help="number of base-probe pairs used to approximate finite-difference work",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.2,
        help="minimum acceptable FD-equivalent / sensitivity speedup ratio",
    )
    args = parser.parse_args()

    sens_probe = _load_required_symbol(
        "native_cvodes_transit_1cmt_pkpd_sensitivity_probe_multidose"
    )
    base_probe = _load_required_symbol(
        "native_cvodes_transit_1cmt_pkpd_probe_multidose"
    )

    obs = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0]
    dose_times = [0.0, 24.0]
    dose_amts = [1.0, 1.0]
    theta = [0.8, 1.1, 0.3, 2.2, 0.4, 1.4, 0.25, 0.9]

    sens_probe(obs, dose_times, dose_amts, theta)
    for _ in range(args.n_fd_pairs):
        base_probe(obs, dose_times, dose_amts, theta)

    t0 = time.perf_counter()
    for _ in range(args.n_reps):
        sens_probe(obs, dose_times, dose_amts, theta)
    sens_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.n_reps):
        for _ in range(args.n_fd_pairs):
            base_probe(obs, dose_times, dose_amts, theta)
    fd_s = time.perf_counter() - t0

    speedup = fd_s / sens_s
    print(f"sensitivity_ms_per_call={sens_s * 1e3 / args.n_reps:.3f}")
    print(f"fd_equivalent_ms_per_call={fd_s * 1e3 / args.n_reps:.3f}")
    print(f"fd_equivalent_speedup={speedup:.3f}")

    if speedup < args.min_speedup:
        print(
            "performance_gate=fail "
            f"(expected >= {args.min_speedup:.2f}x, got {speedup:.3f}x)"
        )
        return 1

    print("performance_gate=pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
