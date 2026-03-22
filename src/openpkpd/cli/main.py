"""
openpkpd CLI entry point.

Usage:
    openpkpd run model.ctl
    openpkpd run model.ctl --method FOCE --parallel 4 --verbose
    openpkpd parse model.ctl        # Parse and display control stream info
"""

from __future__ import annotations

import sys
from typing import Any

import click

from openpkpd.utils.logging import configure_logging


@click.group()
@click.version_option(package_name="openpkpd")
def main() -> None:
    """openpkpd — Python reimplementation of NONMEM for population PK/PD analysis."""


@main.command()
@click.argument("ctl_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "-d", default=None, help="Override dataset path from $DATA record")
@click.option(
    "--method",
    "-m",
    default=None,
    type=click.Choice(
        ["FO", "FOCE", "FOCEI", "LAPLACIAN", "SAEM", "IMP", "IMPMAP"], case_sensitive=False
    ),
    help="Override estimation method",
)
@click.option(
    "--parallel", "-p", default=1, show_default=True, help="Parallel processes for inner loop"
)
@click.option("--output-dir", "-o", default=None, help="Output directory for result files")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run(
    ctl_file: str,
    data: str | None,
    method: str | None,
    parallel: int,
    output_dir: str | None,
    verbose: bool,
) -> None:
    """Run supported estimation, simulation, or mixture workflows from a control stream file."""
    configure_logging(verbose=verbose)

    from openpkpd.cli.runner import run_model

    try:
        result: Any = run_model(
            ctl_path=ctl_file,
            dataset_path=data,
            method_override=method,
            n_parallel=parallel,
            verbose=verbose,
            output_dir=output_dir,
        )
        if hasattr(result, "simulated_df") and hasattr(result, "n_replicates"):
            click.echo("\nSimulation complete.")
            click.echo(f"  Replicates: {result.n_replicates}")
            click.echo(f"  Seed:       {result.seed}")
            click.echo(f"  Rows:       {len(result.simulated_df)}")
            sys.exit(0)
        if hasattr(result, "mixture_probs") and hasattr(result, "subpop_results"):
            click.echo("\nMixture estimation complete.")
            click.echo(f"  OFV:        {result.ofv:.4f}")
            click.echo(f"  Converged:  {result.converged}")
            click.echo(f"  Subpops:    {result.n_subpop}")
            click.echo("  Mixing:     " + ", ".join(f"{p:.4f}" for p in result.mixture_probs))
            sys.exit(0 if result.converged else 1)
        click.echo("\nEstimation complete.")
        click.echo(f"  OFV:       {result.ofv:.4f}")
        click.echo(f"  Converged: {result.converged}")
        click.echo(f"  Method:    {result.method}")
        if result.warnings:
            click.secho("\nWarnings:", fg="yellow")
            for w in result.warnings:
                click.secho(f"  {w}", fg="yellow")
        sys.exit(0 if result.converged else 1)
    except Exception as exc:
        click.secho(f"\nERROR: {exc}", fg="red", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)


@main.command()
@click.argument("ctl_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def parse(ctl_file: str, output_json: bool) -> None:
    """Parse a control stream and display its structure."""
    import json

    from openpkpd.parser.control_stream import ControlStream

    cs = ControlStream.from_file(ctl_file)
    if output_json:
        click.echo(json.dumps(cs.to_dict(), indent=2, default=str))
    else:
        click.echo(f"Control stream: {ctl_file}")
        click.echo(f"Records: {[r.record_name for r in cs.records]}")
        if cs.problem:
            click.echo(f"Title: {cs.problem.title}")
        if cs.subroutines:
            click.echo(f"ADVAN: {cs.subroutines.advan}, TRANS: {cs.subroutines.trans}")
        if cs.theta_records:
            n_theta = sum(len(r.specs) for r in cs.theta_records)
            click.echo(f"THETA: {n_theta} parameters")
        if cs.omega_records:
            n_omega = sum(s.block_size for r in cs.omega_records for s in r.specs)
            click.echo(f"OMEGA: {n_omega}x{n_omega} matrix")
        if cs.estimation_records:
            for rec in cs.estimation_records:
                click.echo(
                    f"Estimation: {rec.method} (INTER={rec.interaction}, MAXEVAL={rec.maxeval})"
                )
        if cs.simulation:
            click.echo(
                "Simulation: "
                f"seeds={cs.simulation.seeds or [42]} "
                f"only={cs.simulation.onlysimulation} "
                f"subproblems={cs.simulation.subproblems}"
            )
        if cs.mixture:
            click.echo(
                f"Mixture: nspop={cs.mixture.nspop} pmix_theta_index={cs.mixture.pmix_theta_index}"
            )


@main.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--id-col", default="ID", show_default=True, help="Subject ID column name")
@click.option("--time-col", default="TIME", show_default=True, help="Time column name")
@click.option("--conc-col", default="DV", show_default=True, help="Concentration column name")
@click.option("--dose-col", default="AMT", show_default=True, help="Dose amount column name")
@click.option("--output", "-o", default=None, help="Output CSV file for NCA results")
def nca(
    data_file: str,
    id_col: str,
    time_col: str,
    conc_col: str,
    dose_col: str,
    output: str | None,
) -> None:
    """Non-compartmental analysis of a concentration-time dataset.

    Reads a NONMEM-format CSV file and computes standard NCA parameters
    (AUC, Cmax, Tmax, half-life, clearance) for each subject.

    Example:
        openpkpd nca data.csv --output nca_results.csv
    """
    try:
        import pandas as pd

        from openpkpd.nca.nca import NCAEngine

        click.echo(f"Reading data from: {data_file}")
        df = pd.read_csv(data_file)
        engine = NCAEngine()
        results_df = engine.compute_dataset(
            df,
            id_col=id_col,
            time_col=time_col,
            conc_col=conc_col,
            dose_col=dose_col,
        )
        if output:
            results_df.to_csv(output, index=False)
            click.echo(f"NCA results written to: {output}")
        else:
            click.echo(results_df.to_string(index=False))
    except ImportError as exc:
        click.secho(
            f"ERROR: NCA module not available: {exc}\n"
            "The NCA engine (openpkpd.nca.nca) may not yet be installed.",
            fg="red",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(2)


@main.command()
@click.argument("model_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--n-rep", default=500, show_default=True, help="Number of simulation replicates")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility")
@click.option(
    "--output",
    "-o",
    default="vpc_output.csv",
    show_default=True,
    help="Output CSV file for VPC percentiles",
)
def vpc(
    model_file: str,
    n_rep: int,
    seed: int,
    output: str,
) -> None:
    """Generate a Visual Predictive Check (VPC) for a fitted model.

    Runs a Monte Carlo simulation from the model's fitted parameters and
    computes observed and simulated percentile bands (5th, 50th, 95th).
    Results are written to a CSV file suitable for plotting.

    Example:
        openpkpd vpc model.ctl --n-rep 200 --output vpc_bands.csv
    """
    try:
        from openpkpd.cli.runner import run_model
        from openpkpd.model.problem import Problem
        from openpkpd.parser.control_stream import ControlStream
        from openpkpd.simulation import SimulationEngine, VPCEngine

        click.echo(f"Fitting model: {model_file}")
        vpc_result: Any = run_model(ctl_path=model_file, verbose=False)
        click.echo(f"OFV: {vpc_result.ofv:.4f}  Converged: {vpc_result.converged}")

        # Reconstruct population model for VPC
        cs = ControlStream.from_file(model_file)
        pop_model = Problem.from_control_stream(cs).population_model

        click.echo(f"Running VPC with {n_rep} replicates (seed={seed})...")
        engine = SimulationEngine(pop_model, vpc_result, seed=seed)
        vpc_engine = VPCEngine(engine)
        vpc_res = vpc_engine.compute(n_replicates=n_rep)

        vpc_res.sim_percentiles.to_csv(output, index=False)
        click.echo(f"VPC percentile bands written to: {output}")
    except ImportError as exc:
        click.secho(
            f"ERROR: Required module not available: {exc}\n"
            "Ensure openpkpd.simulation is installed and the runner module is available.",
            fg="red",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(2)


@main.command()
@click.argument("model_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--n-boot", default=200, show_default=True, help="Number of bootstrap replicates")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility")
@click.option(
    "--n-jobs", default=-1, show_default=True, help="Number of parallel jobs (-1 = all CPUs)"
)
@click.option(
    "--output",
    "-o",
    default="bootstrap_results.csv",
    show_default=True,
    help="Output CSV file for bootstrap parameter estimates",
)
def bootstrap(
    model_file: str,
    n_boot: int,
    seed: int,
    n_jobs: int,
    output: str,
) -> None:
    """Bootstrap confidence intervals for model parameters.

    Draws n-boot bootstrap resamples of the subject-level data, refits the
    model to each resample, and saves the distribution of parameter estimates.
    Use the resulting CSV to compute empirical confidence intervals.

    Example:
        openpkpd bootstrap model.ctl --n-boot 100 --output boot.csv
    """
    try:
        from openpkpd.cli.runner import run_model
        from openpkpd.inference.bootstrap import BootstrapEngine
        from openpkpd.model.problem import Problem
        from openpkpd.parser.control_stream import ControlStream

        click.echo(f"Fitting base model: {model_file}")
        boot_base_result: Any = run_model(ctl_path=model_file, verbose=False)
        click.echo(f"Base OFV: {boot_base_result.ofv:.4f}")

        cs = ControlStream.from_file(model_file)
        pop_model = Problem.from_control_stream(cs).population_model

        click.echo(f"Running {n_boot} bootstrap replicates (seed={seed}, n_jobs={n_jobs})...")
        engine = BootstrapEngine(
            pop_model, boot_base_result, n_boot=n_boot, seed=seed, n_jobs=n_jobs
        )
        boot_result = engine.run()
        boot_result.summary().to_csv(output, index=False)
        click.echo(f"Bootstrap results written to: {output}")
    except ImportError as exc:
        click.secho(
            f"ERROR: Required module not available: {exc}\n"
            "Ensure openpkpd.inference.bootstrap is installed.",
            fg="red",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(2)


@main.command("compare")
@click.argument("model_files", nargs=-1, required=True, type=click.Path(exists=True))
def compare_models_cmd(model_files: tuple[str, ...]) -> None:
    """Compare multiple fitted model .lst or .ctl files by AIC, BIC, and OFV.

    Fits each model and displays a comparison table sorted by OFV.
    Useful for model selection based on information criteria.

    Example:
        openpkpd compare model1.ctl model2.ctl model3.ctl
    """
    try:
        from openpkpd.cli.runner import run_model

        results_summary: list[dict] = []
        for model_path in model_files:
            click.echo(f"Fitting: {model_path}  ", nl=False)
            try:
                r: Any = run_model(ctl_path=model_path, verbose=False)
                results_summary.append(
                    {
                        "model": model_path,
                        "OFV": r.ofv,
                        "AIC": r.aic,
                        "BIC": r.bic,
                        "n_params": r.n_parameters,
                        "converged": r.converged,
                        "method": r.method,
                    }
                )
                click.echo(f"OFV={r.ofv:.3f} AIC={r.aic:.3f}")
            except Exception as exc:
                click.secho(f"FAILED: {exc}", fg="yellow")
                results_summary.append(
                    {
                        "model": model_path,
                        "OFV": float("inf"),
                        "AIC": float("inf"),
                        "BIC": float("inf"),
                        "n_params": 0,
                        "converged": False,
                        "method": "FAILED",
                    }
                )

        # Sort by OFV and display
        results_summary.sort(key=lambda x: x["OFV"])
        click.echo("\nModel Comparison (sorted by OFV):")
        click.echo("-" * 80)
        header = f"{'Model':<35} {'OFV':>10} {'AIC':>10} {'BIC':>10} {'n_par':>6} {'Conv':>5}"
        click.echo(header)
        click.echo("-" * 80)
        for rec in results_summary:
            ofv_s = f"{rec['OFV']:.3f}" if rec["OFV"] != float("inf") else "FAILED"
            aic_s = f"{rec['AIC']:.3f}" if rec["AIC"] != float("inf") else "---"
            bic_s = f"{rec['BIC']:.3f}" if rec["BIC"] != float("inf") else "---"
            name = str(rec["model"])[-34:]
            click.echo(
                f"{name:<35} {ofv_s:>10} {aic_s:>10} {bic_s:>10} "
                f"{rec['n_params']:>6} {'Y' if rec['converged'] else 'N':>5}"
            )
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(2)


@main.command()
@click.argument("model_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--n", default=1000, show_default=True, help="Number of subjects to simulate")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility")
@click.option(
    "--output",
    "-o",
    default="simulated_data.csv",
    show_default=True,
    help="Output CSV file for simulated dataset",
)
def simulate(
    model_file: str,
    n: int,
    seed: int,
    output: str,
) -> None:
    """Simulate data from a fitted model.

    Fits the specified model and then simulates a new virtual population of
    n subjects using the final parameter estimates. The simulated data is
    written to a CSV file in NONMEM-compatible format.

    Example:
        openpkpd simulate model.ctl --n 500 --output simdata.csv
    """
    try:
        from openpkpd.cli.runner import run_model
        from openpkpd.model.problem import Problem
        from openpkpd.parser.control_stream import ControlStream
        from openpkpd.simulation import SimulationEngine

        click.echo(f"Fitting model: {model_file}")
        sim_run_result: Any = run_model(ctl_path=model_file, verbose=False)
        click.echo(f"OFV: {sim_run_result.ofv:.4f}  Converged: {sim_run_result.converged}")

        cs = ControlStream.from_file(model_file)
        pop_model = Problem.from_control_stream(cs).population_model

        click.echo(f"Simulating {n} subjects (seed={seed})...")
        engine = SimulationEngine(pop_model, sim_run_result, seed=seed)
        # Simulate one replicate of the observed design (n subjects total)
        sim_result = engine.simulate(n_replicates=1)
        sim_df = sim_result.simulated_df[sim_result.simulated_df["REP"] == 1]
        sim_df = sim_df.drop(columns=["REP"], errors="ignore")
        sim_df.to_csv(output, index=False)
        click.echo(f"Simulated data ({len(sim_df)} rows) written to: {output}")
    except ImportError as exc:
        click.secho(
            f"ERROR: Required module not available: {exc}\n"
            "Ensure openpkpd.simulation is installed.",
            fg="red",
            err=True,
        )
        sys.exit(2)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(2)


@main.command("library")
@click.argument("action", type=click.Choice(["list", "show"]), default="list")
@click.argument("model_name", default="", required=False)
def library_cmd(action: str, model_name: str) -> None:
    """Browse the built-in PK/PD model library.

    Available actions:

    \b
      list        Display all available pre-built model names.
      show NAME   Show documentation for a specific model.

    Examples:

    \b
        openpkpd library list
        openpkpd library show one_cmt_oral
        openpkpd library show emax_direct
    """
    try:
        from openpkpd.library import list_models, show_model

        if action == "list":
            models = list_models()
            click.echo(f"Available models ({len(models)} total):")
            click.echo("-" * 40)
            for name in models:
                click.echo(f"  {name}")
            click.echo("\nUse 'openpkpd library show <name>' for details.")
        elif action == "show":
            if not model_name:
                click.secho(
                    "ERROR: model_name is required for 'show' action.\n"
                    "Usage: openpkpd library show <model_name>",
                    fg="red",
                    err=True,
                )
                sys.exit(1)
            doc = show_model(model_name)
            click.echo(doc)
    except ImportError as exc:
        click.secho(
            f"ERROR: Library module not available: {exc}",
            fg="red",
            err=True,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
