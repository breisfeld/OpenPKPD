"""
Part 4b: Unit tests for Laplacian + prior penalty integration.

Verifies that:
  - LaplacianMethod correctly adds the prior penalty to OFV (the fix in Part 1A).
  - Prior increases OFV relative to the no-prior run for both FOCE and Laplacian.
  - OFV is finite in all cases.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from openpkpd.data.dataset import NONMEMDataset
from openpkpd.estimation.foce import FOCEMethod
from openpkpd.estimation.laplacian import LaplacianMethod
from openpkpd.model.parameters import OmegaSpec, ParameterSet, SigmaSpec, ThetaSpec
from openpkpd.model.population import PopulationModel
from openpkpd.pk.analytical.advan2 import ADVAN2
from openpkpd.prior import PriorAugmentedModel, PriorSpec

# ---------------------------------------------------------------------------
# Minimal 2-subject dataset
# ---------------------------------------------------------------------------


def _build_minimal_dataset(n_subjects: int = 4, seed: int = 7) -> NONMEMDataset:
    """Small 1-cmt oral dataset for quick unit tests."""
    rng = np.random.default_rng(seed)
    ka_pop, cl_pop, v_pop = 1.5, 2.0, 25.0
    dose = 200.0
    obs_times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

    rows = []
    for sid in range(1, n_subjects + 1):
        ka = ka_pop * math.exp(rng.normal(0, 0.2))
        cl = cl_pop * math.exp(rng.normal(0, 0.2))
        v = v_pop * math.exp(rng.normal(0, 0.15))
        k = cl / v

        rows.append(
            {
                "ID": sid,
                "TIME": 0.0,
                "AMT": dose,
                "DV": 0.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
                "RATE": 0.0,
                "ADDL": 0,
                "II": 0,
                "SS": 0,
            }
        )
        for t in obs_times:
            if abs(ka - k) < 1e-6:
                c = dose * ka / v * t * math.exp(-k * t)
            else:
                c = dose * ka / (v * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))
            eps = rng.normal(0, 0.08)
            dv = max(c * (1 + eps), 0.001)
            rows.append(
                {
                    "ID": sid,
                    "TIME": t,
                    "AMT": 0.0,
                    "DV": dv,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                    "RATE": 0.0,
                    "ADDL": 0,
                    "II": 0,
                    "SS": 0,
                }
            )

    return NONMEMDataset.from_dataframe(pd.DataFrame(rows))


def _build_params() -> ParameterSet:
    theta_specs = [
        ThetaSpec(init=1.5, lower=0.3, upper=6.0),  # KA
        ThetaSpec(init=2.0, lower=0.5, upper=10.0),  # CL
        ThetaSpec(init=25.0, lower=5.0, upper=60.0),  # V
    ]
    omega_specs = [OmegaSpec(block_size=1, values=[0.04])]
    sigma_specs = [SigmaSpec(block_size=1, values=[0.01])]
    return ParameterSet.from_specs(theta_specs, omega_specs, sigma_specs)


def _build_pop_model(dataset: NONMEMDataset, params: ParameterSet) -> PopulationModel:
    return PopulationModel(
        dataset=dataset,
        pk_subroutine=ADVAN2(),
        params=params,
        trans=2,
        advan=2,
    )


def _build_tight_prior(theta_mean: list[float]) -> PriorSpec:
    """A tight (informative) prior that should strongly penalise off-centre theta."""
    n = len(theta_mean)
    return PriorSpec(
        theta_prior=np.array(theta_mean, dtype=float),
        theta_prior_cov=np.diag([0.01] * n),  # very tight
    )


# ---------------------------------------------------------------------------
# Fixtures: no-prior runs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    return _build_minimal_dataset()


@pytest.fixture(scope="module")
def foce_no_prior(dataset):
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    method = FOCEMethod(interaction=False, maxeval=300, print_interval=999)
    return method.estimate(pop_model, params)


@pytest.fixture(scope="module")
def lap_no_prior(dataset):
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    method = LaplacianMethod(interaction=False, maxeval=300, print_interval=999)
    return method.estimate(pop_model, params)


# ---------------------------------------------------------------------------
# Fixtures: prior runs (tight prior centred far from data optimum)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def foce_with_prior(dataset):
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    # Prior centred far from estimated values (KA=5, CL=8, V=60) to force penalty
    prior = _build_tight_prior([5.0, 8.0, 60.0])
    aug_model = PriorAugmentedModel(population_model=pop_model, prior=prior)
    method = FOCEMethod(interaction=False, maxeval=300, print_interval=999)
    return method.estimate(aug_model, params)


@pytest.fixture(scope="module")
def lap_with_prior(dataset):
    params = _build_params()
    pop_model = _build_pop_model(dataset, params)
    prior = _build_tight_prior([5.0, 8.0, 60.0])
    aug_model = PriorAugmentedModel(population_model=pop_model, prior=prior)
    method = LaplacianMethod(interaction=False, maxeval=300, print_interval=999)
    return method.estimate(aug_model, params)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_foce_no_prior_ofv_finite(foce_no_prior):
    assert np.isfinite(foce_no_prior.ofv), "FOCE (no prior) OFV must be finite"


def test_laplacian_no_prior_ofv_finite(lap_no_prior):
    assert np.isfinite(lap_no_prior.ofv), "Laplacian (no prior) OFV must be finite"


def test_foce_with_prior_ofv_finite(foce_with_prior):
    assert np.isfinite(foce_with_prior.ofv), "FOCE (with prior) OFV must be finite"


def test_laplacian_with_prior_ofv_finite(lap_with_prior):
    assert np.isfinite(lap_with_prior.ofv), "Laplacian (with prior) OFV must be finite"


def test_prior_increases_foce_ofv(foce_no_prior, foce_with_prior):
    """A tight prior far from the data optimum must increase FOCE OFV."""
    assert foce_with_prior.ofv > foce_no_prior.ofv, (
        f"Prior did not increase FOCE OFV: "
        f"no-prior={foce_no_prior.ofv:.4f}, with-prior={foce_with_prior.ofv:.4f}"
    )


def test_prior_increases_laplacian_ofv(lap_no_prior, lap_with_prior):
    """A tight prior far from the data optimum must increase Laplacian OFV."""
    assert lap_with_prior.ofv > lap_no_prior.ofv, (
        f"Prior did not increase Laplacian OFV: "
        f"no-prior={lap_no_prior.ofv:.4f}, with-prior={lap_with_prior.ofv:.4f}"
    )


def test_laplacian_prior_penalty_applied(lap_no_prior, lap_with_prior, foce_with_prior):
    """
    Laplacian with prior OFV should be strictly higher than Laplacian without prior.
    Also verify that both FOCE+prior and Laplacian+prior give finite OFV
    (i.e. the prior penalty path is exercised in both methods).
    """
    assert np.isfinite(lap_with_prior.ofv)
    assert np.isfinite(foce_with_prior.ofv)
    # The delta due to prior should be positive (tight prior, off-centre mean)
    delta = lap_with_prior.ofv - lap_no_prior.ofv
    assert delta > 0, f"Expected positive OFV delta from prior, got {delta:.4f}"
