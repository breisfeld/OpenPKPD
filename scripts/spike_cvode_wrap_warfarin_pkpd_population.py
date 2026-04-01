"""Population-level repeated FO benchmark for the native warfarin PK/PD hook.

Compares the reduced 4-subject warfarin PK/PD FO objective under:

- native mixed-endpoint hook enabled
- native mixed-endpoint hook disabled

This is still a spike artifact, but it measures an estimator-facing loop rather
than a single-subject trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openpkpd.model import individual as individual_mod
from openpkpd.model.parameters import ParameterSet
from tests.external_validation.test_vs_nlmixr2 import (
    REF_DIR,
    _build_warfarin_pkpd_4_fo_model,
    _build_warfarin_pkpd_6_fo_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["4", "6"], default="4")
    parser.add_argument("--repeats", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dataset == "4":
        model = _build_warfarin_pkpd_4_fo_model(maxeval=1)
        ref_name = "warfarin_pkpd_4_fo.json"
        label = "Reduced 4-subject warfarin PK/PD FO repeat benchmark"
    else:
        model = _build_warfarin_pkpd_6_fo_model(maxeval=1)
        ref_name = "warfarin_pkpd_6_fo.json"
        label = "Reduced 6-subject warfarin PK/PD FO repeat benchmark"
    pm = model.population_model

    with open(os.path.join(REF_DIR, ref_name)) as f:
        ref = json.load(f)
    th = ref["theta"]
    params = ParameterSet(
        theta=np.array(
            [
                th["KTR"],
                th["KA"],
                th["CL"],
                th["V"],
                th["EMAX"],
                th["EC50"],
                th["KOUT"],
                th["E0"],
                th["PK_PROP_ERR"],
                th["PK_ADD_ERR"],
                th["PD_ADD_ERR"],
            ],
            dtype=float,
        ),
        omega=np.diag([1e-8]),
        sigma=np.eye(2),
    )

    n_repeats = args.repeats

    eligible = 0
    for sid in pm.subject_ids():
        indiv = pm.individual_model(sid)
        if indiv._try_native_pk_backend(  # noqa: SLF001 - spike script
            {
                "KTR": th["KTR"],
                "KA": th["KA"],
                "CL": th["CL"],
                "V": th["V"],
                "EMAX": th["EMAX"],
                "EC50": th["EC50"],
                "KOUT": th["KOUT"],
                "E0": th["E0"],
                "PCMT": 3.0,
            },
            indiv.subject_events.obs_times.astype(float),
        ) is not None:
            eligible += 1

    eta_zero = np.zeros(params.n_eta())

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for sid in pm.subject_ids():
            pm.individual_model(sid).evaluate_observation_model(
                params.theta, eta_zero, params.sigma, trans=pm.trans
            )
    native_obs_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    native_ofv = None
    for _ in range(n_repeats):
        native_ofv = pm.ofv_fo(params)
    native_seconds = time.perf_counter() - t0

    saved_probe = individual_mod._native_cvodes_transit_1cmt_pkpd_probe_rust
    individual_mod._native_cvodes_transit_1cmt_pkpd_probe_rust = None
    try:
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            for sid in pm.subject_ids():
                pm.individual_model(sid).evaluate_observation_model(
                    params.theta, eta_zero, params.sigma, trans=pm.trans
                )
        python_obs_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        python_ofv = None
        for _ in range(n_repeats):
            python_ofv = pm.ofv_fo(params)
        python_seconds = time.perf_counter() - t0
    finally:
        individual_mod._native_cvodes_transit_1cmt_pkpd_probe_rust = saved_probe

    print(label)
    print(f"repeat_n={n_repeats}")
    print(f"native_hook_eligible_subjects={eligible}/{pm.n_subjects()}")
    print(f"native_obs_seconds={native_obs_seconds:.6f}")
    print(f"python_obs_seconds={python_obs_seconds:.6f}")
    print(f"native_ofv={native_ofv:.10f}")
    print(f"python_ofv={python_ofv:.10f}")
    print(f"ofv_abs_diff={abs(native_ofv - python_ofv):.6g}")
    print(f"native_seconds={native_seconds:.6f}")
    print(f"python_seconds={python_seconds:.6f}")


if __name__ == "__main__":
    main()
