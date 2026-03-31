"""Estimator-facing native benchmark for reduced warfarin PK/PD.

Measures native-hook impact on:

- repeated `IndividualModel.obj_eta` for one subject
- repeated `LaplacianMethod._outer_ofv` for the reduced 4-subject model

This is still a spike script; it is not part of the public API.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openpkpd.estimation.laplacian import LaplacianMethod
from openpkpd.model import individual as individual_mod
from openpkpd.model.parameters import ParameterSet
from tests.external_validation.test_vs_nlmixr2 import REF_DIR, _build_warfarin_pkpd_4_fo_model


def main() -> None:
    model = _build_warfarin_pkpd_4_fo_model(maxeval=1)
    pm = model.population_model
    ind = pm.individual_model(1)

    with open(os.path.join(REF_DIR, "warfarin_pkpd_4_fo.json")) as f:
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
    eta_zero = np.zeros(params.n_eta())
    lap = LaplacianMethod(maxeval=1)
    eta_hat = {sid: np.zeros(params.n_eta()) for sid in pm.subject_ids()}
    n_repeats = 10

    t0 = time.perf_counter()
    native_obj = None
    for _ in range(n_repeats):
        native_obj = ind.obj_eta(eta_zero, params.theta, params.omega, params.sigma, trans=pm.trans)
    native_obj_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    native_lap = None
    for _ in range(n_repeats):
        native_lap = lap._outer_ofv(pm, params, eta_hat)
    native_lap_seconds = time.perf_counter() - t0

    saved_probe = individual_mod._native_cvodes_advan6_mixed_pkpd_probe_rust
    individual_mod._native_cvodes_advan6_mixed_pkpd_probe_rust = None
    try:
        t0 = time.perf_counter()
        python_obj = None
        for _ in range(n_repeats):
            python_obj = ind.obj_eta(eta_zero, params.theta, params.omega, params.sigma, trans=pm.trans)
        python_obj_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        python_lap = None
        for _ in range(n_repeats):
            python_lap = lap._outer_ofv(pm, params, eta_hat)
        python_lap_seconds = time.perf_counter() - t0
    finally:
        individual_mod._native_cvodes_advan6_mixed_pkpd_probe_rust = saved_probe

    print("Reduced 4-subject warfarin PK/PD advanced-estimator repeat benchmark")
    print(f"repeat_n={n_repeats}")
    print(f"native_obj_eta={native_obj:.10f}")
    print(f"python_obj_eta={python_obj:.10f}")
    print(f"obj_eta_abs_diff={abs(native_obj - python_obj):.6g}")
    print(f"native_obj_eta_seconds={native_obj_seconds:.6f}")
    print(f"python_obj_eta_seconds={python_obj_seconds:.6f}")
    print(f"native_laplacian_ofv={native_lap:.10f}")
    print(f"python_laplacian_ofv={python_lap:.10f}")
    print(f"laplacian_ofv_abs_diff={abs(native_lap - python_lap):.6g}")
    print(f"native_laplacian_seconds={native_lap_seconds:.6f}")
    print(f"python_laplacian_seconds={python_lap_seconds:.6f}")


if __name__ == "__main__":
    main()
