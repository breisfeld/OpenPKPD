"""Integration test for mixed-endpoint routing through $ERROR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpkpd import ModelBuilder
from openpkpd.data.dataset import NONMEMDataset
from openpkpd.data.event_processor import DoseEvent
from openpkpd.pk.analytical.advan1 import ADVAN1

_TRUE_K = 0.20
_TRUE_V = 10.0
_TRUE_BASE = 1.0
_TRUE_SLOPE = 0.10
_TRUE_W = 0.15


def _simulate_dataset(n_subj: int = 4, seed: int = 7) -> NONMEMDataset:
    rng = np.random.default_rng(seed)
    advan1 = ADVAN1()
    doses = [50.0, 100.0, 150.0, 200.0]
    obs_times = np.array([1.0, 2.0, 4.0, 8.0])
    rows: list[dict[str, float]] = []

    for sid in range(1, n_subj + 1):
        dose = doses[(sid - 1) % len(doses)]
        sol = advan1.solve(
            {"K": _TRUE_K, "V": _TRUE_V}, [DoseEvent(time=0.0, amount=dose)], obs_times
        )
        conc = sol.ipred
        amount = sol.amounts[:, 0]
        effect = _TRUE_BASE + _TRUE_SLOPE * amount

        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "DVID": 0,
            }
        )
        for t, c, a_eff in zip(obs_times, conc, effect, strict=True):
            rows.append(
                {
                    "ID": sid,
                    "TIME": float(t),
                    "AMT": 0.0,
                    "DV": float(c + rng.normal(0.0, _TRUE_W)),
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "DVID": 1,
                }
            )
            rows.append(
                {
                    "ID": sid,
                    "TIME": float(t),
                    "AMT": 0.0,
                    "DV": float(a_eff + rng.normal(0.0, _TRUE_W)),
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 2,
                    "DVID": 2,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


_PK_CODE = """\
K = THETA(1)
V = THETA(2)
"""

_ERROR_CODE = """\
BASE = THETA(3)
SLOPE = THETA(4)
W = THETA(5)
Y = BASE + SLOPE*A(1)
IF (DVID .EQ. 1) Y = F
IRES = DV - Y
IWRES = IRES / W
Y = Y + W*EPS(1)
"""


@pytest.mark.integration
def test_mixed_endpoint_error_routing_runs_and_distinguishes_same_time_observations():
    dataset = _simulate_dataset()
    built = (
        ModelBuilder()
        .problem("Mixed-endpoint routing integration test")
        .dataset(dataset)
        .covariates(["DVID"])
        .subroutines(advan=1, trans=2)
        .pk(_PK_CODE)
        .error(_ERROR_CODE)
        .theta(
            [
                (0.01, 0.18, 1.0),
                (1.0, 9.0, 20.0),
                (0.0, 0.5, 5.0),
                (0.001, 0.08, 0.5),
                (0.01, 0.2, 2.0),
            ]
        )
        .omega([1e-6, 1e-6], fixed=True)
        .sigma(1.0, fixed=True)
        .estimation(method="FO", maxeval=300)
        .build()
    )

    result = built.fit()

    assert np.isfinite(result.ofv)
    assert result.theta_final[0] == pytest.approx(_TRUE_K, rel=0.30)
    assert result.theta_final[1] == pytest.approx(_TRUE_V, rel=0.25)
    assert result.theta_final[2] == pytest.approx(_TRUE_BASE, rel=0.35)
    assert result.theta_final[3] == pytest.approx(_TRUE_SLOPE, rel=0.30)
    assert result.theta_final[4] > 0.0

    sid = built.population_model.subject_ids()[0]
    indiv = built.population_model.individual_model(sid)
    eta = result.post_hoc_etas.get(sid, np.zeros(2))
    ipred, _obs_mask, f, pred, _var = indiv.evaluate_observation_model(
        result.theta_final,
        eta,
        result.sigma_final,
        trans=built.population_model.trans,
    )

    assert indiv.subject_events.observation_covariates_at(0)["DVID"] == pytest.approx(1.0)
    assert indiv.subject_events.observation_covariates_at(1)["DVID"] == pytest.approx(2.0)
    assert indiv.subject_events.obs_times[0] == pytest.approx(indiv.subject_events.obs_times[1])
    assert pred[0] == pytest.approx(f[0], rel=1e-6)
    assert pred[1] > pred[0]
    assert abs(pred[0] - pred[1]) > 0.5
