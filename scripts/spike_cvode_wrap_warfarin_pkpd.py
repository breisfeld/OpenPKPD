"""Compare the Rust cvode_wrap probe against the current reduced warfarin PK/PD path.

This script is intentionally narrow:

- uses the bundled reduced 4-subject warfarin PK/PD benchmark
- compares subject 1 only
- uses the bundled nlmixr2 FO reference theta values
- reports compartment- and endpoint-level differences between:
  - the feature-gated Rust `_core.cvode_wrap_warfarin_pkpd_probe`
  - the current OpenPKPD mixed-endpoint path

It is a spike artifact, not a user-facing API.
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

import openpkpd._core as core
from tests.external_validation.test_vs_nlmixr2 import REF_DIR, _build_warfarin_pkpd_4_fo_model


def main() -> None:
    model = _build_warfarin_pkpd_4_fo_model(maxeval=1)
    ind = model.population_model.individual_model(1)

    with open(os.path.join(REF_DIR, "warfarin_pkpd_4_fo.json")) as f:
        ref = json.load(f)
    th = ref["theta"]

    theta = np.array(
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
    )
    sigma = np.eye(2)
    eta = np.zeros(1)

    _, _, _, amounts = ind._evaluate_predictions(theta, eta, sigma, trans=1, include_amounts=True)
    _, _, _, pred, _ = ind.evaluate_observation_model(theta, eta, sigma, trans=1)

    times = ind.subject_events.obs_times.astype(float)
    dvid = np.array([float(c["DVID"]) for c in ind.subject_events.obs_covariates], dtype=float)
    dose_amt = float(ind.subject_events.dose_events[0].amount)

    rust_amounts = np.asarray(
        core.cvode_wrap_warfarin_pkpd_probe(times.tolist(), dose_amt, theta[:8].tolist())
    )
    py_amounts = np.asarray(amounts)[:, :4]
    py_pred = np.asarray(pred, dtype=float)
    rust_pred = np.where(dvid == 1.0, rust_amounts[:, 2] / th["V"], th["E0"] + rust_amounts[:, 3])

    abs_diff = np.abs(rust_amounts - py_amounts)
    pred_abs_diff = np.abs(rust_pred - py_pred)

    print("Subject 1 reduced warfarin PK/PD comparison")
    print(f"n_times={len(times)}")
    print(f"compartment_1_max_abs_diff={float(np.max(abs_diff[:, 0])):.6g}")
    print(f"compartment_2_max_abs_diff={float(np.max(abs_diff[:, 1])):.6g}")
    print(f"compartment_3_max_abs_diff={float(np.max(abs_diff[:, 2])):.6g}")
    print(f"compartment_4_max_abs_diff={float(np.max(abs_diff[:, 3])):.6g}")
    print(f"pk_endpoint_max_abs_diff={float(np.max(pred_abs_diff[dvid == 1.0])):.6g}")
    print(f"pd_endpoint_max_abs_diff={float(np.max(pred_abs_diff[dvid == 2.0])):.6g}")
    print("rust_tail=", rust_pred[-6:].tolist())
    print("python_tail=", py_pred[-6:].tolist())

    n_repeats = 100
    t0 = time.perf_counter()
    last_py_amounts = None
    for _ in range(n_repeats):
        _, _, _, last_py_amounts = ind._evaluate_predictions(
            theta, eta, sigma, trans=1, include_amounts=True
        )
    py_seconds = time.perf_counter() - t0
    rust_seconds, last_rust_state = core.cvode_wrap_warfarin_pkpd_repeat_probe(
        times.tolist(), dose_amt, theta[:8].tolist(), n_repeats
    )
    print(f"repeat_n={n_repeats}")
    print(f"python_repeat_seconds={py_seconds:.6f}")
    print(f"rust_repeat_seconds={rust_seconds:.6f}")
    print(
        "repeat_last_state_abs_diff=",
        float(np.max(np.abs(np.asarray(last_rust_state) - np.asarray(last_py_amounts)[-1, :4]))),
    )


if __name__ == "__main__":
    main()
