"""Prototype reduced warfarin PK/PD solve using sundials4py CVODES.

This is a feasibility spike, not production code. It targets the simplest
real mixed-endpoint benchmark slice currently available in-tree:

- reduced 4-subject warfarin PK/PD benchmark
- reference theta from the bundled nlmixr2 FO run
- one subject at a time
- current event pattern in the reduced file: a single oral bolus at t=0

The script compares the CVODES trajectory against the current OpenPKPD ADVAN6
path for the same subject and parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sundials4py import core, cvodes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.external_validation.test_vs_nlmixr2 import _build_warfarin_pkpd_4_fo_model, _load_ref


def _rhs_factory(theta: np.ndarray):
    ktr = float(theta[0])
    ka = float(theta[1])
    cl = float(theta[2])
    v = float(theta[3])
    emax = float(theta[4])
    ec50 = float(theta[5])
    kout = float(theta[6])
    e0 = float(theta[7])

    def rhs(t: float, yv, ypdot, user_data) -> int:
        y = core.N_VGetArrayPointer(yv)
        yp = core.N_VGetArrayPointer(ypdot)
        conc = y[2] / v
        pd = 1.0 - emax * conc / (ec50 + conc)
        yp[0] = -ktr * y[0]
        yp[1] = ktr * y[0] - ka * y[1]
        yp[2] = ka * y[1] - (cl / v) * y[2]
        yp[3] = kout * e0 * (pd - 1.0) - kout * y[3]
        return 0

    return rhs


def _solve_subject_cvodes(theta: np.ndarray, dose_amt: float, obs_times: np.ndarray, method: str) -> np.ndarray:
    unique_times = np.unique(np.asarray(obs_times, dtype=float))
    flag, sunctx = core.SUNContext_Create(core.SUN_COMM_NULL)
    if flag != 0:
        raise RuntimeError(f"SUNContext_Create failed with flag={flag}")

    y0 = np.zeros(4, dtype=float)
    y0[0] += float(dose_amt)
    y = core.N_VMake_Serial(4, y0, sunctx)
    a = core.SUNDenseMatrix(4, 4, sunctx)
    ls = core.SUNLinSol_Dense(y, a, sunctx)
    lmm = cvodes.CV_BDF if method.upper() == "BDF" else cvodes.CV_ADAMS
    cvode_mem = cvodes.CVodeCreate(lmm, sunctx).get()

    rhs = _rhs_factory(theta)
    for name, rc in [
        ("CVodeInit", cvodes.CVodeInit(cvode_mem, rhs, 0.0, y)),
        ("CVodeSStolerances", cvodes.CVodeSStolerances(cvode_mem, 1e-6, 1e-8)),
        ("CVodeSetLinearSolver", cvodes.CVodeSetLinearSolver(cvode_mem, ls, a)),
    ]:
        if rc != 0:
            raise RuntimeError(f"{name} failed with rc={rc}")

    out = core.N_VNew_Serial(4, sunctx)
    states_by_time: dict[float, np.ndarray] = {}
    for t in unique_times:
        flag, tret = cvodes.CVode(cvode_mem, float(t), out, cvodes.CV_NORMAL)
        if flag != cvodes.CV_SUCCESS:
            raise RuntimeError(f"CVode failed at t={t} with flag={flag}")
        states_by_time[float(tret)] = core.N_VGetArrayPointer(out).copy()

    states = np.vstack([states_by_time[float(t)] for t in obs_times])
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", type=int, default=1)
    parser.add_argument("--method", choices=["Adams", "BDF"], default="BDF")
    args = parser.parse_args()

    model = _build_warfarin_pkpd_4_fo_model(maxeval=1)
    population_model = model.population_model
    individual = population_model._individual_models[args.subject_id]
    ref = _load_ref("warfarin_pkpd_4_fo.json")
    theta = np.array(list(ref["theta"].values()), dtype=float)
    sigma = np.eye(2)
    eta = np.zeros(1)

    _, _, f_ref, amounts_ref = individual._evaluate_predictions(theta, eta, sigma, trans=population_model.trans, include_amounts=True)
    dose_amt = individual.subject_events.dose_events[0].amount
    obs_times = individual.subject_events.obs_times

    states_cv = _solve_subject_cvodes(theta, dose_amt, obs_times, method=args.method)
    f_cv = states_cv[:, 2] / float(theta[3])

    max_state_abs_diff = float(np.max(np.abs(states_cv - amounts_ref[:, :4])))
    max_f_abs_diff = float(np.max(np.abs(f_cv - f_ref)))

    print(f"subject_id={args.subject_id}")
    print(f"method={args.method}")
    print(f"obs_count={len(obs_times)}")
    print(f"max_state_abs_diff={max_state_abs_diff:.6e}")
    print(f"max_f_abs_diff={max_f_abs_diff:.6e}")
    print("first_rows_cvodes=")
    print(np.array2string(states_cv[:3], precision=6, suppress_small=False))
    print("first_rows_openpkpd=")
    print(np.array2string(amounts_ref[:3, :4], precision=6, suppress_small=False))


if __name__ == "__main__":
    main()
