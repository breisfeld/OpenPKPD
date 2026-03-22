"""Shared helpers for PD regression baselines."""

from __future__ import annotations

import numpy as np

from openpkpd.model.parameters import ParameterSet
from openpkpd.models.pkpd import (
    EffectCompartmentModel,
    EmaxModel,
    HillModel,
    IndirectResponseModel,
    PDData,
    PDModel,
    PlaceboResponseModel,
    SequentialPKPDWorkflow,
    TumorGrowthInhibitionModel,
    TurnoverModel,
)
from openpkpd.models.population_pd import PopulationPDModel
from tests.regression.diagnostic_helpers import build_pop_model_and_result

DIRECT_PD_CASES = {
    "emax_direct_pd": {
        "model_cls": EmaxModel,
        "dataset": "synthetic_direct_pd_emax",
        "true_params": {"E0": 1.0, "Emax": 8.0, "EC50": 3.0},
        "initial_params": {"E0": 0.0, "Emax": 5.0, "EC50": 1.0},
        "sigma2": 0.01,
        "seed": 42,
        "noise_sd": 0.1,
    },
    "hill_direct_pd": {
        "model_cls": HillModel,
        "dataset": "synthetic_direct_pd_hill",
        "true_params": {"E0": 1.0, "Emax": 10.0, "EC50": 4.0, "gamma": 1.7},
        "initial_params": {"E0": 0.0, "Emax": 8.0, "EC50": 2.0, "gamma": 1.0},
        "sigma2": 0.01,
        "seed": 42,
        "noise_sd": 0.1,
    },
}


MECHANISTIC_PD_CASES = {
    "effect_compartment_pd": {
        "model_cls": EffectCompartmentModel,
        "dataset": "synthetic_effect_compartment_pd",
        "true_params": {"Ke0": 0.7, "Emax": 15.0, "EC50": 2.0, "n": 1.2},
        "initial_params": {"Ke0": 0.8, "Emax": 14.0, "EC50": 2.2, "n": 1.1},
        "sigma2": 0.0064,
        "seed": 11,
        "noise_sd": 0.08,
        "times": np.linspace(0.0, 12.0, 49),
        "concentrations": None,
    },
    "placebo_response_pd": {
        "model_cls": PlaceboResponseModel,
        "dataset": "synthetic_placebo_response_pd",
        "true_params": {"E0": 60.0, "kdeg": 0.02, "Eplacebo": 20.0, "kpl": 0.05},
        "initial_params": {"E0": 55.0, "kdeg": 0.03, "Eplacebo": 15.0, "kpl": 0.04},
        "sigma2": 1.0,
        "seed": 4,
        "noise_sd": 1.0,
    },
    "tumor_growth_inhibition_pd": {
        "model_cls": TumorGrowthInhibitionModel,
        "dataset": "synthetic_tumor_growth_inhibition_pd",
        "true_params": {
            "lambda0": 0.29949662080618694,
            "lambda1": 1.9833639932414355,
            "K1": 0.19876584325612448,
            "K2": 0.024586784112521964,
            "psi": 15.0,
            "X0": 148.75748987215476,
        },
        "initial_params": {
            "lambda0": 0.3,
            "lambda1": 2.0,
            "K1": 0.2,
            "K2": 0.02,
            "psi": 15.0,
            "X0": 150.0,
        },
        "sigma2": 2.25,
        "seed": 3,
        "noise_sd": 1.5,
    },
}


POPULATION_PD_CASES = {
    "population_emax_pd": {
        "pd_model_cls": EmaxModel,
        "dataset": "synthetic_population_emax_pd",
        "eta_params": ["Emax"],
        "theta_init": {"E0": 1.0, "Emax": 7.0, "EC50": 2.0},
        "omega_init": np.array([[0.05]]),
        "sigma2_init": 0.05,
        "estimate_sigma2": True,
        "maxeval": 80,
        "seed": 7,
        "n_subjects": 4,
        "true_theta": {"E0": 1.5, "Emax": 9.0, "EC50": 3.5},
        "true_omega": 0.04,
        "true_sigma2": 0.01,
    },
}


INDIRECT_PD_CASES = {
    "indirect_response_pd": {
        "model_kwargs": {"idr_type": 1},
        "dataset": "synthetic_indirect_response_pd",
        "true_params": {"Kin": 5.0, "Kout": 0.5, "EC50": 2.5, "Emax": 0.8},
        "initial_params": {"Kin": 5.2, "Kout": 0.5, "EC50": 3.0, "Emax": 1.0},
        "sigma2": 0.0025,
        "seed": 10,
        "noise_sd": 0.05,
    },
}


TURNOVER_PD_CASES = {
    "turnover_pd": {
        "model_cls": TurnoverModel,
        "dataset": "synthetic_turnover_pd",
        "true_params": {
            "Kin": 2.0,
            "Kout": 0.5,
            "EC50_in": 1.5,
            "Emax_in": 1.0,
            "EC50_out": 1.0,
            "Emax_out": 0.0,
        },
        "initial_params": {
            "Kin": 2.0,
            "Kout": 0.5,
            "EC50_in": 1.5,
            "Emax_in": 0.8,
            "EC50_out": 1.0,
            "Emax_out": 0.0,
        },
        "sigma2": 0.04,
        "seed": 2,
        "noise_sd": 0.2,
    },
}


SEQUENTIAL_PD_CASES = {
    "sequential_emax_pd": {
        "pd_model_cls": EmaxModel,
        "dataset": "synthetic_sequential_emax_pd",
        "true_params": {"E0": 1.0, "Emax": 8.0, "EC50": 3.0},
        "initial_params": {"E0": 0.0, "Emax": 5.0, "EC50": 1.0},
        "seed": 42,
        "noise_sd": 0.1,
    },
    "sequential_emax_from_pk_pd": {
        "pd_model_cls": EmaxModel,
        "dataset": "synthetic_sequential_emax_from_pk_pd",
        "pk_seed": 77,
        "pk_n_subjects": 1,
        "pd_seed": 123,
        "noise_sd": 0.08,
        "true_params": {"E0": 1.5, "Emax": 12.0, "EC50": 3.0},
        "initial_params": {"E0": 0.5, "Emax": 8.0, "EC50": 1.0},
    },
    "sequential_multi_subject_from_pk_pd": {
        "pd_model_cls": EmaxModel,
        "dataset": "synthetic_sequential_multi_subject_from_pk_pd",
        "pk_seed": 91,
        "pk_n_subjects": 3,
        "pd_seed": 321,
        "noise_sd": 0.06,
        "true_params": {"E0": 1.2, "Emax": 10.0, "EC50": 2.8},
        "initial_params": {"E0": 0.5, "Emax": 7.0, "EC50": 1.5},
        "post_hoc_etas": {1: [-0.35], 2: [0.15], 3: [0.45]},
    },
}


JOINT_PKPD_CASES = {
    "joint_emax_pkpd": {
        "dataset": "synthetic_joint_emax_pkpd",
        "n_subjects": 12,
        "seed": 42,
        "maxeval": 200,
        "param_names": ["K", "V", "E0", "EMAX", "EC50", "W"],
        "omega_param_names": ["K", "V"],
        "true_params": {"K": 0.15, "V": 10.0, "E0": 2.0, "EMAX": 15.0, "EC50": 8.0, "W": 1.5},
        "true_omega_diag": {"K": 0.04, "V": 0.04},
        "initial_params": {"K": 0.15, "V": 10.0, "E0": 2.0, "EMAX": 15.0, "EC50": 8.0, "W": 1.5},
        "omega_init": [0.3, 0.3],
    },
}


def build_direct_pd_dataset(
    model: PDModel,
    true_params: dict[str, float],
    *,
    seed: int,
    sigma: float,
) -> PDData:
    """Create a deterministic concentration-effect dataset for regression tests."""
    times = np.linspace(0.5, 24.0, 24)
    concentrations = 12.0 * np.exp(-0.18 * times)
    base = PDData(
        subject_id=1, times=times, response=np.zeros_like(times), concentrations=concentrations
    )
    truth = model.predict(true_params, base)
    obs = truth + np.random.default_rng(seed).normal(0.0, sigma, size=len(times))
    return PDData(subject_id=1, times=times, response=obs, concentrations=concentrations)


def build_effect_compartment_dataset(
    true_params: dict[str, float],
    *,
    seed: int,
    sigma: float,
) -> PDData:
    """Create a deterministic effect-compartment dataset."""
    times = np.linspace(0.0, 12.0, 49)
    concentrations = 6.0 * np.exp(-0.25 * times)
    model = EffectCompartmentModel()
    base = PDData(
        subject_id=1, times=times, response=np.zeros_like(times), concentrations=concentrations
    )
    truth = model.predict(true_params, base)
    obs = truth + np.random.default_rng(seed).normal(0.0, sigma, size=len(times))
    return PDData(subject_id=1, times=times, response=obs, concentrations=concentrations)


def build_placebo_response_dataset(
    true_params: dict[str, float],
    *,
    seed: int,
    sigma: float,
) -> PDData:
    """Create a deterministic placebo-response dataset."""
    times = np.linspace(0.0, 52.0, 20)
    model = PlaceboResponseModel()
    base = PDData(subject_id=1, times=times, response=np.zeros_like(times))
    truth = model.predict(true_params, base)
    obs = truth + np.random.default_rng(seed).normal(0.0, sigma, size=len(times))
    return PDData(subject_id=1, times=times, response=obs)


def build_tumor_growth_inhibition_dataset(
    true_params: dict[str, float],
    *,
    seed: int,
    sigma: float,
) -> PDData:
    """Create a deterministic tumor-growth inhibition dataset."""
    times = np.array([0.0, 3.0, 7.0, 10.0, 14.0, 17.0, 21.0, 25.0, 28.0])
    concentrations = 2.0 * np.exp(-0.15 * times)
    model = TumorGrowthInhibitionModel()
    base = PDData(
        subject_id=1, times=times, response=np.ones_like(times), concentrations=concentrations
    )
    truth = model.predict(true_params, base)
    obs = np.maximum(truth + np.random.default_rng(seed).normal(0.0, sigma, size=len(times)), 1.0)
    return PDData(subject_id=1, times=times, response=obs, concentrations=concentrations)


def build_population_emax_subjects(
    *,
    seed: int,
    n_subjects: int,
    true_theta: dict[str, float],
    true_omega: float,
    true_sigma2: float,
) -> list[PDData]:
    """Create a deterministic small population PD dataset."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0.5, 24.0, 24)
    concentrations = 12.0 * np.exp(-0.2 * times)
    model = EmaxModel()
    subjects: list[PDData] = []

    for sid in range(1, n_subjects + 1):
        eta = rng.normal(0.0, np.sqrt(true_omega))
        subject_params = dict(true_theta)
        subject_params["Emax"] *= np.exp(eta)
        base = PDData(
            subject_id=sid,
            times=times,
            response=np.zeros_like(times),
            concentrations=concentrations,
        )
        truth = model.predict(subject_params, base)
        obs = truth + rng.normal(0.0, np.sqrt(true_sigma2), size=len(times))
        subjects.append(
            PDData(subject_id=sid, times=times, response=obs, concentrations=concentrations)
        )

    return subjects


def build_indirect_response_dataset(
    true_params: dict[str, float],
    *,
    seed: int,
    sigma: float,
    idr_type: int,
) -> PDData:
    """Create a deterministic indirect-response dataset."""
    model = IndirectResponseModel(idr_type=idr_type)
    times = np.linspace(0.0, 24.0, 49)
    concentrations = np.full(len(times), 4.0)
    baseline = true_params["Kin"] / true_params["Kout"]
    base = PDData(
        subject_id=1,
        times=times,
        response=np.zeros_like(times),
        concentrations=concentrations,
        baseline=baseline,
    )
    truth = model.predict(true_params, base)
    obs = truth + np.random.default_rng(seed).normal(0.0, sigma, size=len(times))
    return PDData(
        subject_id=1, times=times, response=obs, concentrations=concentrations, baseline=baseline
    )


def build_sequential_emax_dataset(
    true_params: dict[str, float],
    *,
    seed: int,
    sigma: float,
) -> PDData:
    """Create a deterministic concentration-effect dataset for SequentialPKPDWorkflow."""
    times = np.linspace(0.5, 24.0, 24)
    concentrations = 12.0 * np.exp(-0.18 * times)
    base = PDData(
        subject_id=1, times=times, response=np.zeros_like(times), concentrations=concentrations
    )
    truth = EmaxModel().predict(true_params, base)
    obs = truth + np.random.default_rng(seed).normal(0.0, sigma, size=len(times))
    return PDData(subject_id=1, times=times, response=obs, concentrations=concentrations)


def build_turnover_dataset(
    true_params: dict[str, float],
    *,
    seed: int,
    sigma: float,
) -> PDData:
    """Create a deterministic turnover-model dataset."""
    times = np.linspace(0.0, 24.0, 49)
    concentrations = 5.0 * np.exp(-0.2 * times)
    baseline = true_params["Kin"] / true_params["Kout"]
    base = PDData(
        subject_id=1,
        times=times,
        response=np.zeros_like(times),
        concentrations=concentrations,
        baseline=baseline,
    )
    truth = TurnoverModel().predict(true_params, base)
    obs = truth + np.random.default_rng(seed).normal(0.0, sigma, size=len(times))
    return PDData(
        subject_id=1, times=times, response=obs, concentrations=concentrations, baseline=baseline
    )


def build_sequential_emax_from_pk_dataset(name: str):
    """Create a deterministic PD dataset whose concentrations come from a fitted PK workflow."""
    case = SEQUENTIAL_PD_CASES[name]
    pk_model, pk_result = build_pop_model_and_result(
        n_subjects=case["pk_n_subjects"], seed=case["pk_seed"]
    )
    sid = pk_model.subject_ids()[0]
    individual = pk_model.individual_model(sid)
    concentrations, _obs_mask, _f = individual.evaluate(
        pk_result.theta_final,
        pk_result.post_hoc_etas[sid],
        pk_model.params.sigma,
        trans=pk_model.trans,
    )
    times = individual.subject_events.obs_times
    base = PDData(
        subject_id=sid, times=times, response=np.zeros_like(times), concentrations=concentrations
    )
    truth = case["pd_model_cls"]().predict(case["true_params"], base)
    obs = truth + np.random.default_rng(case["pd_seed"]).normal(
        0.0, case["noise_sd"], size=len(times)
    )
    explicit = PDData(subject_id=sid, times=times, response=obs, concentrations=concentrations)
    missing = PDData(subject_id=sid, times=times, response=obs, concentrations=None)
    return case, pk_model, pk_result, explicit, missing


def build_multi_subject_sequential_from_pk_datasets(name: str):
    """Create deterministic subject-wise PD datasets from a multi-subject fitted PK workflow."""
    case = SEQUENTIAL_PD_CASES[name]
    pk_model, pk_result = build_pop_model_and_result(
        n_subjects=case["pk_n_subjects"], seed=case["pk_seed"]
    )
    pk_result.post_hoc_etas = {
        sid: np.asarray(case["post_hoc_etas"][sid], dtype=float) for sid in pk_model.subject_ids()
    }

    model = case["pd_model_cls"]()
    rng = np.random.default_rng(case["pd_seed"])
    explicit: dict[int, PDData] = {}
    missing: dict[int, PDData] = {}

    for sid in pk_model.subject_ids():
        individual = pk_model.individual_model(sid)
        concentrations, _obs_mask, _f = individual.evaluate(
            pk_result.theta_final,
            pk_result.post_hoc_etas[sid],
            pk_model.params.sigma,
            trans=pk_model.trans,
        )
        times = individual.subject_events.obs_times
        base = PDData(
            subject_id=sid,
            times=times,
            response=np.zeros_like(times),
            concentrations=concentrations,
        )
        truth = model.predict(case["true_params"], base)
        obs = truth + rng.normal(0.0, case["noise_sd"], size=len(times))
        explicit[sid] = PDData(
            subject_id=sid, times=times, response=obs, concentrations=concentrations
        )
        missing[sid] = PDData(subject_id=sid, times=times, response=obs, concentrations=None)

    return case, pk_model, pk_result, explicit, missing


def fit_direct_pd_case(name: str):
    """Fit one named direct PD regression case and return its ingredients."""
    case = DIRECT_PD_CASES[name]
    model = case["model_cls"]()
    data = build_direct_pd_dataset(
        model,
        case["true_params"],
        seed=case["seed"],
        sigma=case["noise_sd"],
    )
    result = model.fit(data, initial_params=case["initial_params"], sigma2=case["sigma2"])
    return case, data, result


def fit_mechanistic_pd_case(name: str):
    """Fit one named mechanistic single-subject PD regression case."""
    case = MECHANISTIC_PD_CASES[name]
    model = case["model_cls"]()
    if case["model_cls"] is EffectCompartmentModel:
        data = build_effect_compartment_dataset(
            case["true_params"],
            seed=case["seed"],
            sigma=case["noise_sd"],
        )
    elif case["model_cls"] is PlaceboResponseModel:
        data = build_placebo_response_dataset(
            case["true_params"],
            seed=case["seed"],
            sigma=case["noise_sd"],
        )
    elif case["model_cls"] is TumorGrowthInhibitionModel:
        data = build_tumor_growth_inhibition_dataset(
            case["true_params"],
            seed=case["seed"],
            sigma=case["noise_sd"],
        )
    else:
        raise KeyError(f"Unsupported mechanistic PD case: {name}")
    result = model.fit(data, initial_params=case["initial_params"], sigma2=case["sigma2"])
    return case, data, result


def fit_population_pd_case(name: str):
    """Fit one named population PD regression case."""
    case = POPULATION_PD_CASES[name]
    subjects = build_population_emax_subjects(
        seed=case["seed"],
        n_subjects=case["n_subjects"],
        true_theta=case["true_theta"],
        true_omega=case["true_omega"],
        true_sigma2=case["true_sigma2"],
    )
    model = PopulationPDModel(
        pd_model=case["pd_model_cls"](),
        eta_params=case["eta_params"],
        theta_init=case["theta_init"],
        omega_init=case["omega_init"],
        sigma2=case["sigma2_init"],
        estimate_sigma2=case["estimate_sigma2"],
        maxeval=case["maxeval"],
    )
    result = model.estimate(subjects)
    return case, subjects, result


def fit_indirect_pd_case(name: str):
    """Fit one named indirect-response regression case."""
    case = INDIRECT_PD_CASES[name]
    data = build_indirect_response_dataset(
        case["true_params"],
        seed=case["seed"],
        sigma=case["noise_sd"],
        idr_type=case["model_kwargs"]["idr_type"],
    )
    model = IndirectResponseModel(**case["model_kwargs"])
    result = model.fit(data, initial_params=case["initial_params"], sigma2=case["sigma2"])
    return case, data, result


def fit_turnover_pd_case(name: str):
    """Fit one named turnover regression case."""
    case = TURNOVER_PD_CASES[name]
    data = build_turnover_dataset(case["true_params"], seed=case["seed"], sigma=case["noise_sd"])
    model = case["model_cls"]()
    result = model.fit(data, initial_params=case["initial_params"], sigma2=case["sigma2"])
    return case, data, result


def fit_sequential_pd_case(name: str):
    """Fit one named sequential PK/PD regression case."""
    case = SEQUENTIAL_PD_CASES[name]
    data = build_sequential_emax_dataset(
        case["true_params"],
        seed=case["seed"],
        sigma=case["noise_sd"],
    )

    class _MockPKResult:
        pass

    class _MockPKModel:
        pass

    workflow = SequentialPKPDWorkflow(_MockPKResult(), _MockPKModel())
    result = workflow.fit_pd(data, case["pd_model_cls"], initial_params=case["initial_params"])
    return case, data, result


def fit_sequential_from_pk_pd_case(name: str):
    """Fit one named sequential PK→PD case using PK-derived concentrations."""
    case, pk_model, pk_result, explicit, missing = build_sequential_emax_from_pk_dataset(name)
    workflow = SequentialPKPDWorkflow(pk_result, pk_model)
    result = workflow.fit_pd(missing, case["pd_model_cls"], initial_params=case["initial_params"])
    return case, explicit, result


def fit_sequential_multi_subject_from_pk_pd_case(name: str):
    """Fit one named multi-subject sequential PK→PD case using subject-specific post-hoc ETAs."""
    case, pk_model, pk_result, explicit, missing = build_multi_subject_sequential_from_pk_datasets(
        name
    )
    workflow = SequentialPKPDWorkflow(pk_result, pk_model)
    results = {
        sid: workflow.fit_pd(
            missing[sid], case["pd_model_cls"], initial_params=case["initial_params"]
        )
        for sid in sorted(missing)
    }
    return case, explicit, results


def fit_joint_pkpd_case(name: str):
    """Fit one named deterministic joint PK/PD regression case."""
    from tests.integration.test_emax_pd import _build_emax_fo_model, _simulate_pkpd_data

    case = JOINT_PKPD_CASES[name]
    dataset = _simulate_pkpd_data(n_subj=case["n_subjects"], seed=case["seed"])
    built = _build_emax_fo_model(
        dataset,
        maxeval=case["maxeval"],
        problem=f"joint PK/PD regression: {name}",
    )
    result = built.fit()

    final_params = ParameterSet(
        theta=result.theta_final.copy(),
        omega=result.omega_final.copy(),
        sigma=result.sigma_final.copy(),
        theta_specs=list(built.params.theta_specs),
        omega_specs=list(built.params.omega_specs),
        sigma_specs=list(built.params.sigma_specs),
    )

    pop_model = built.population_model
    predicted: list[float] = []
    pk_f: list[float] = []
    for sid in pop_model.subject_ids():
        eta = result.post_hoc_etas[sid]
        _ipred, obs_mask, f, pred, _var = pop_model.individual_model(
            sid
        ).evaluate_observation_model(
            final_params.theta,
            eta,
            final_params.sigma,
            trans=pop_model.trans,
        )
        predicted.extend(np.asarray(pred)[obs_mask].tolist())
        pk_f.extend(np.asarray(f)[obs_mask].tolist())

    return case, built, result, np.asarray(predicted, dtype=float), np.asarray(pk_f, dtype=float)
