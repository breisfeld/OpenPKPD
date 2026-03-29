"""
openpkpd.plots — Standard PK/PD and diagnostic plots.

Usage:
    from openpkpd.plots.diagnostics import compute_diagnostics, compute_npde
    from openpkpd.plots.gof import diagnostic_panel
    from openpkpd.plots.pk import spaghetti_plot, concentration_time, mean_profile
    from openpkpd.plots.pd import effect_time, emax_curve, hysteresis_loop
    from openpkpd.plots.eta import eta_histograms, eta_pairs, eta_vs_covariate
    from openpkpd.plots.model_perf import ofv_history, vpc
    from openpkpd.plots.simulation import vpc_plot, npde_plot, simulation_panel

    diag_df = compute_diagnostics(population_model, result)
    fig = diagnostic_panel(diag_df)
"""

from openpkpd.plots import simulation
from openpkpd.plots.bayesian import (
    ess_plot,
    mcmc_trace_by_chain_plot,
    mcmc_trace_plot,
    posterior_density_plot,
    posterior_forest_plot,
    rhat_plot,
)
from openpkpd.plots.bootstrap import (
    bootstrap_ci_plot,
    bootstrap_parameter_distributions,
)
from openpkpd.plots.categorical import (
    categorical_probability_plot,
    count_frequency_plot,
    cumulative_probability_plot,
    markov_transition_heatmap,
)
from openpkpd.plots.covariate import covariate_forest_plot
from openpkpd.plots.diagnostics import compute_diagnostics, compute_npde
from openpkpd.plots.eta import (
    eps_shrinkage_plot,
    eta_histograms,
    eta_pairs,
    eta_shrinkage_plot,
    eta_vs_covariate,
    iiv_cv_plot,
    omega_heatmap,
)
from openpkpd.plots.gof import (
    abs_iwres_vs_ipred,
    cwres_histogram,
    cwres_qq,
    cwres_vs_pred,
    cwres_vs_time,
    diagnostic_panel,
    dv_vs_ipred,
    dv_vs_pred,
)
from openpkpd.plots.model_perf import (
    likelihood_profile_plot,
    model_comparison_plot,
    ofv_history,
    parameter_uncertainty_plot,
    residual_trends_plot,
    vpc,
)
from openpkpd.plots.nca import (
    dose_proportionality_plot,
    nca_boxplot,
    nca_distributions,
    nca_profile_plot,
)
from openpkpd.plots.pd import (
    effect_compartment_plot,
    effect_time,
    emax_curve,
    hysteresis_loop,
    indirect_response_plot,
    pd_individual,
)
from openpkpd.plots.pk import (
    concentration_time,
    individual_fit_grid,
    mean_profile,
    spaghetti_plot,
)
from openpkpd.plots.simulation import pcvpc_plot, stratified_vpc_plot

__all__ = [
    "compute_diagnostics",
    "compute_npde",
    # GOF
    "dv_vs_ipred",
    "dv_vs_pred",
    "cwres_vs_time",
    "cwres_vs_pred",
    "cwres_qq",
    "cwres_histogram",
    "abs_iwres_vs_ipred",
    "diagnostic_panel",
    # PK
    "concentration_time",
    "spaghetti_plot",
    "mean_profile",
    "individual_fit_grid",
    # PD
    "effect_time",
    "emax_curve",
    "hysteresis_loop",
    "pd_individual",
    "indirect_response_plot",
    "effect_compartment_plot",
    # ETA / IIV
    "eta_histograms",
    "eta_pairs",
    "eta_vs_covariate",
    "eta_shrinkage_plot",
    "eps_shrinkage_plot",
    "omega_heatmap",
    "iiv_cv_plot",
    # Performance / model comparison
    "ofv_history",
    "vpc",
    "parameter_uncertainty_plot",
    "residual_trends_plot",
    "model_comparison_plot",
    "likelihood_profile_plot",
    # Covariate analysis
    "covariate_forest_plot",
    # NCA
    "nca_distributions",
    "nca_boxplot",
    "nca_profile_plot",
    "dose_proportionality_plot",
    # Bootstrap
    "bootstrap_parameter_distributions",
    "bootstrap_ci_plot",
    # Categorical / count / Markov
    "categorical_probability_plot",
    "cumulative_probability_plot",
    "count_frequency_plot",
    "markov_transition_heatmap",
    # Bayesian / MCMC diagnostics
    "mcmc_trace_plot",
    "mcmc_trace_by_chain_plot",
    "rhat_plot",
    "ess_plot",
    "posterior_density_plot",
    "posterior_forest_plot",
    # Simulation-based
    "simulation",
    "pcvpc_plot",
    "stratified_vpc_plot",
]
