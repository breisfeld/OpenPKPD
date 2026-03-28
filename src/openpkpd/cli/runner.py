"""
Pipeline runner: orchestrates parsing → estimation → output for a full run.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from openpkpd.covariance.sandwich import SandwichCovariance
from openpkpd.estimation import get_estimation_method
from openpkpd.estimation.base import EstimationResult
from openpkpd.mixture import MixtureModel, MixtureResult
from openpkpd.model.parameters import ParameterSet
from openpkpd.model.problem import Problem
from openpkpd.output.cov_writer import write_cor, write_cov
from openpkpd.output.ext_writer import write_ext
from openpkpd.output.lst_writer import write_lst
from openpkpd.output.mixture_writer import write_mixture_assignments, write_mixture_summary
from openpkpd.output.phi_writer import write_phi
from openpkpd.output.table_writer import write_table
from openpkpd.parser.control_stream import ControlStream
from openpkpd.parser.records.estimation import EstimationRecord
from openpkpd.parser.records.mixture import MixtureRecord
from openpkpd.parser.records.simulation import SimulationRecord
from openpkpd.simulation.engine import SimulationEngine, SimulationResult
from openpkpd.utils.constants import Method
from openpkpd.utils.logging import configure_logging, get_logger

logger = get_logger("cli.runner")


def run_model(
    ctl_path: str,
    dataset_path: str | None = None,
    method_override: str | None = None,
    n_parallel: int = 1,
    verbose: bool = False,
    output_dir: str | None = None,
) -> EstimationResult | SimulationResult | MixtureResult:
    """
    Run a full openpkpd estimation pipeline.

    Args:
        ctl_path:        Path to NONMEM control stream (.ctl/.mod file).
        dataset_path:    Override dataset path (otherwise uses $DATA filename).
        method_override: Override estimation method (otherwise uses $ESTIMATION).
        n_parallel:      Number of parallel processes for inner loop.
        verbose:         Enable verbose logging.
        output_dir:      Directory for output files (default: same as ctl_path).

    Returns:
        EstimationResult from the primary estimation method, SimulationResult for
        supported ``$SIMULATION ONLYSIMULATION`` workflows, or MixtureResult for
        supported ``$MIXTURE`` workflows.
    """
    configure_logging(verbose=verbose)

    # 1. Parse control stream
    logger.info(f"Parsing control stream: {ctl_path}")
    cs = ControlStream.from_file(ctl_path)

    # 2. Assemble problem (load dataset, compile code, build model)
    logger.info("Assembling model...")
    problem = Problem.from_control_stream(cs, dataset_path=dataset_path)
    pop_model = problem.population_model
    params = pop_model.params
    sim_rec = cs.simulation
    mix_rec = cs.mixture

    logger.info(f"  Subjects: {pop_model.n_subjects()}")
    logger.info(
        f"  THETA: {params.n_theta()}, OMEGA: {params.n_eta()}x{params.n_eta()}, SIGMA: {params.n_eps()}x{params.n_eps()}"
    )

    base = _output_base(ctl_path, output_dir)

    if mix_rec is not None:
        mix_result = _run_mixture_record(
            cs=cs,
            pop_model=pop_model,
            params=params,
            mix_rec=mix_rec,
            method_override=method_override,
            n_parallel=n_parallel,
            base=base,
        )
        logger.info(
            "Mixture run complete. OFV = %.4f, converged = %s",
            mix_result.ofv,
            mix_result.converged,
        )
        return mix_result

    if sim_rec is not None and sim_rec.onlysimulation:
        if method_override is not None:
            raise ValueError(
                "--method override is not supported for $SIMULATION ONLYSIMULATION runs."
            )
        sim_result = _run_simulation_record(pop_model, params, sim_rec, base, result=None)
        logger.info(
            "Simulation-only run complete. Seed=%s, replicates=%s",
            sim_result.seed,
            sim_result.n_replicates,
        )
        return sim_result

    # 3. Determine estimation method
    est_recs: list[EstimationRecord | None] = list(cs.estimation_records)
    if not est_recs:
        est_recs = [None]

    final_result: EstimationResult | None = None

    for est_rec in est_recs:
        if method_override:
            method_name = method_override.upper()
            interaction = False
            maxeval = 9999
        elif est_rec is not None:
            method_name = est_rec.method
            interaction = est_rec.interaction
            maxeval = est_rec.maxeval
        else:
            method_name = Method.FOCE
            interaction = False
            maxeval = 9999

        logger.info(f"Running {method_name} estimation...")
        est_kwargs: dict = {
            "interaction": interaction,
            "maxeval": maxeval,
            "n_parallel": n_parallel,
        }
        if est_rec is not None:
            if getattr(est_rec, "n_starts", 1) > 1:
                est_kwargs["n_starts"] = est_rec.n_starts
            if getattr(est_rec, "gtol", 1e-5) != 1e-5:
                est_kwargs["gtol"] = est_rec.gtol
            if getattr(est_rec, "perturbation_scale", 1.0) != 1.0:
                est_kwargs["perturbation_scale"] = est_rec.perturbation_scale
            if getattr(est_rec, "seed", None) is not None:
                est_kwargs["seed"] = est_rec.seed
            if getattr(est_rec, "outer_optimizer", None):
                est_kwargs["outer_optimizer"] = est_rec.outer_optimizer
            if getattr(est_rec, "outer_fallback_optimizer", None):
                est_kwargs["outer_fallback_optimizer"] = est_rec.outer_fallback_optimizer
            if getattr(est_rec, "outer_fallback_maxeval", None) is not None:
                est_kwargs["outer_fallback_maxeval"] = est_rec.outer_fallback_maxeval
            if getattr(est_rec, "retain_best_iterate", None) is not None:
                est_kwargs["retain_best_iterate"] = est_rec.retain_best_iterate
            if getattr(est_rec, "retry_on_abnormal", None) is not None:
                est_kwargs["retry_on_abnormal"] = est_rec.retry_on_abnormal
            if getattr(est_rec, "retry_omega_scales", ()):
                est_kwargs["retry_omega_scales"] = est_rec.retry_omega_scales
        est = get_estimation_method(method_name, **est_kwargs)

        try:
            result = est.estimate(pop_model, params)
        except Exception as exc:
            logger.error(f"Estimation failed: {exc}")
            raise

        final_result = result

        # Update params for next estimation step (multi-est control streams)
        params = ParameterSet(
            theta=result.theta_final,
            omega=result.omega_final,
            sigma=result.sigma_final,
            theta_specs=params.theta_specs,
            omega_specs=params.omega_specs,
            sigma_specs=params.sigma_specs,
        )

    assert final_result is not None

    # 4. Covariance step
    cov_result = None
    if cs.covariance is not None:
        logger.info("Running covariance step...")
        cov_est = SandwichCovariance(matrix=cs.covariance.matrix)
        try:
            cov_result = cov_est.compute(pop_model, params, final_result.post_hoc_etas)
        except Exception as exc:
            logger.warning(f"Covariance step failed: {exc}")

    # 5. Write output files
    _write_outputs(cs, pop_model, final_result, params, cov_result, base)

    if sim_rec is not None:
        sim_result = _run_simulation_record(pop_model, params, sim_rec, base, result=final_result)
        logger.info(
            "Simulation artifact written for fitted model. Seed=%s, replicates=%s",
            sim_result.seed,
            sim_result.n_replicates,
        )

    logger.info(f"Run complete. OFV = {final_result.ofv:.4f}, converged = {final_result.converged}")
    return final_result


def _run_mixture_record(
    *,
    cs: ControlStream,
    pop_model: Any,
    params: ParameterSet,
    mix_rec: MixtureRecord,
    method_override: str | None,
    n_parallel: int,
    base: str,
) -> MixtureResult:
    """Execute the supported runtime subset of $MIXTURE and write artifacts."""
    if cs.simulation is not None:
        raise ValueError(
            "$MIXTURE and $SIMULATION cannot be combined in the current runtime subset."
        )
    if cs.covariance is not None:
        raise ValueError("$COVARIANCE is not supported for $MIXTURE runtime runs yet.")
    if cs.table_records:
        raise ValueError(
            "$TABLE output is not supported for $MIXTURE runtime runs yet; use the generated .mix artifacts instead."
        )
    if len(cs.estimation_records) > 1:
        raise ValueError("$MIXTURE runtime currently supports at most one $ESTIMATION record.")

    method_name, estimation_kwargs = _resolve_mixture_estimation(
        cs.estimation_records, method_override, n_parallel
    )
    if mix_rec.pmix_theta_index is not None:
        logger.warning(
            "$MIXTURE PMIX=THETA(%d) parsed but not yet used by the current runtime subset.",
            mix_rec.pmix_theta_index,
        )

    mixture = MixtureModel(
        population_model=pop_model,
        n_subpop=int(mix_rec.nspop),
        estimation_method=method_name,
        estimation_kwargs=estimation_kwargs,
    )
    result = mixture.fit(init_params=params)
    _write_mixture_outputs(base, result, method=method_name)
    return result


def _resolve_mixture_estimation(
    est_recs: list[EstimationRecord],
    method_override: str | None,
    n_parallel: int,
) -> tuple[str, dict[str, Any]]:
    """Map control-stream estimation settings into the supported mixture subset."""
    est_rec = est_recs[0] if est_recs else None
    raw_method = (
        method_override or (est_rec.method if est_rec is not None else Method.FOCE)
    ).upper()
    method_name = {
        "COND": Method.FOCE,
        "CONDITIONAL": Method.FOCE,
        "LAPLACE": Method.LAPLACIAN,
    }.get(raw_method, raw_method)
    if method_name not in {Method.FO, Method.FOCE, Method.FOCEI, Method.LAPLACIAN}:
        raise ValueError(
            "$MIXTURE runtime currently supports only FO / FOCE / FOCEI / LAPLACIAN as the inner estimation method."
        )

    if method_name == Method.FO:
        kwargs: dict[str, Any] = {
            "maxeval": est_rec.maxeval if est_rec is not None else 9999,
            "sigdig": est_rec.sigdig if est_rec is not None else 3,
            "print_interval": est_rec.print_interval if est_rec is not None else 5,
            "noabort": est_rec.noabort if est_rec is not None else False,
        }
    else:
        kwargs = {
            "interaction": est_rec.interaction
            if est_rec is not None
            else (method_name == Method.FOCEI),
            "maxeval": est_rec.maxeval if est_rec is not None else 9999,
            "n_parallel": n_parallel,
            "sigdig": est_rec.sigdig if est_rec is not None else 3,
            "print_interval": est_rec.print_interval if est_rec is not None else 5,
            "noabort": est_rec.noabort if est_rec is not None else False,
        }
    return method_name, kwargs


def _output_base(ctl_path: str, output_dir: str | None) -> str:
    base = os.path.splitext(ctl_path)[0]
    if output_dir:
        base = os.path.join(output_dir, os.path.basename(base))
    return base


def _run_simulation_record(
    pop_model: Any,
    params: ParameterSet,
    sim_rec: SimulationRecord,
    base: str,
    *,
    result: EstimationResult | None,
) -> SimulationResult:
    """Execute the supported runtime subset of $SIMULATION and write a CSV artifact."""
    n_replicates = int(sim_rec.subproblems)
    if n_replicates < 1:
        raise ValueError("$SIMULATION SUBPROBLEMS must be >= 1 for runtime execution.")

    if len(sim_rec.seeds) > 1:
        logger.warning(
            "$SIMULATION provided multiple seeds; using only the first seed for runtime execution."
        )
    seed = sim_rec.seeds[0] if sim_rec.seeds else 42

    if sim_rec.true_final:
        logger.info(
            "$SIMULATION TRUE=FINAL parsed; current runtime subset uses the active parameter state without additional switching."
        )

    if result is None:
        result = _initial_simulation_result(pop_model, params)

    sim_engine = SimulationEngine(pop_model, result, seed=seed)
    sim_result = sim_engine.simulate(n_replicates=n_replicates)
    _write_simulation_output(sim_result, base)
    return sim_result


def _initial_simulation_result(pop_model: Any, params: ParameterSet) -> EstimationResult:
    """Construct a synthetic result object for ONLYSIMULATION runs using current parameters."""
    post_hoc_etas = {
        int(sid): np.zeros(params.n_eta(), dtype=float) for sid in pop_model.subject_ids()
    }
    return EstimationResult(
        theta_final=params.theta.copy(),
        omega_final=params.omega.copy(),
        sigma_final=params.sigma.copy(),
        ofv=float("nan"),
        converged=True,
        post_hoc_etas=post_hoc_etas,
        method="SIMULATION",
        message="Synthetic parameter state for $SIMULATION ONLYSIMULATION run.",
        n_observations=pop_model.dataset.n_observations(),
        n_subjects=pop_model.n_subjects(),
    )


def _write_simulation_output(sim_result: SimulationResult, base: str) -> None:
    """Write the stacked simulation DataFrame to a default CSV artifact."""
    sim_path = base + ".sim.csv"
    sim_dir = os.path.dirname(sim_path)
    if sim_dir:
        os.makedirs(sim_dir, exist_ok=True)
    logger.info(f"Writing {sim_path}")
    sim_result.simulated_df.to_csv(sim_path, index=False)


def _write_mixture_outputs(base: str, result: MixtureResult, *, method: str) -> None:
    """Write default mixture artifacts for control-stream mixture runs."""
    summary_path = base + ".mix.json"
    assignments_path = base + ".mix_assignments.csv"
    logger.info(f"Writing {summary_path}")
    write_mixture_summary(summary_path, result, method=method)
    logger.info(f"Writing {assignments_path}")
    write_mixture_assignments(assignments_path, result)


def _write_outputs(
    cs: ControlStream,
    pop_model: Any,
    result: EstimationResult,
    params: ParameterSet,
    cov_result: Any | None,
    base: str,
) -> None:
    """Write all output files for this run."""
    title = cs.problem.title if cs.problem else ""
    method = result.method

    # .lst
    lst_path = base + ".lst"
    logger.info(f"Writing {lst_path}")
    write_lst(
        lst_path,
        result,
        params,
        title=title,
        cov_result=cov_result,
        n_subjects=pop_model.n_subjects(),
        n_obs=pop_model.dataset.n_observations(),
        method=method,
    )

    # .ext
    ext_path = base + ".ext"
    logger.info(f"Writing {ext_path}")
    write_ext(ext_path, result, params, method=method)

    # .phi
    phi_path = base + ".phi"
    logger.info(f"Writing {phi_path}")
    write_phi(phi_path, result, params, pop_model.subject_ids(), method=method)

    # .cov / .cor
    if cov_result is not None:
        cov_path = base + ".cov"
        cor_path = base + ".cor"
        logger.info(f"Writing {cov_path}")
        write_cov(cov_path, cov_result)
        write_cor(cor_path, cov_result)

    # $TABLE files
    for i, table_rec in enumerate(cs.table_records, start=1):
        table_path = table_rec.file or f"{base}.tab{i:03d}"
        logger.info(f"Writing table: {table_path}")
        write_table(
            table_path,
            pop_model,
            result,
            params,
            columns=table_rec.columns,
            noprint=table_rec.noprint,
            oneheader=table_rec.oneheader,
            firstonly=table_rec.firstonly,
        )
