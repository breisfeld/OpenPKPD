# Examples

Selected documentation pages for the example suite. The repository currently
ships 34 self-contained example scripts covering common population PK/PD
workflows, while the docs below cover a curated subset of those scripts in
worked-example form. All shipped scripts can
be run directly:

```bash
python examples/01_theophylline_fo.py
```

Set `OPENPKPD_EXAMPLE_OUTPUT=/tmp/figs` to save figures to disk instead
of displaying them interactively.

## Basic PK modelling

| Script | Topic |
|--------|-------|
| `01_theophylline_fo.py` | One-compartment oral model, FO estimation |
| `02_warfarin_foce.py` | Warfarin FOCE with interaction |
| `03_two_compartment_iv.py` | Two-compartment IV bolus, ADVAN3 |
| `09_three_compartment.py` | Three-compartment IV with ADVAN11 |

## PD and PKPD models

| Script | Topic |
|--------|-------|
| `04_emax_pd_model.py` | Direct Emax PD effect |
| `05_indirect_response.py` | Indirect-response simulation with an approximate direct-effect fit |
| `19_count_categorical_models.py` | Poisson/NegBin/ZIP count models; proportional odds, Markov state models |
| `24_advanced_pd_models.py` | Effect compartment, turnover, TGI (Simeoni), placebo response |

## Special features

| Script | Topic |
|--------|-------|
| `06_from_control_stream.py` | Parse and run an existing NONMEM `.ctl` file |
| `08_ode_transit_absorption.py` | ODE-based transit absorption |
| `10_blq_handling.py` | Below-limit-of-quantification (M1/M3/M5) |
| `11_tte_model.py` | Time-to-event survival model |
| `16_dde_model.py` | Delay differential equation (DDE) PK model |
| `17_sbml_import.py` | SBML/QSP model import |
| `22_pbpk_model.py` | 5-organ physiologically-based PK (PBPK) model |
| `23_iov_model.py` | Inter-occasion variability (IOV) modelling |
| `25_focei_optimizer_controls.py` | FOCEI optimizer robustness controls from the Python API |
| `26_control_stream_optimizer_extensions.py` | Inspect FOCEI optimizer extensions from a control stream |
| `27_phenobarbital_population_pk.py` | Neonatal phenobarbital population PK with weight scaling |
| `28_indometh_nca.py` | NCA on the public Indometh WinNonlin-backed dataset |
| `29_optimal_design.py` | PFIM-backed sampling-time optimization |
| `30_four_compartment_advan5.py` | Four-compartment ADVAN5 example with ADVAN11 consistency check |
| `31_impmap_warm_start.py` | Short IMPMAP warm-start diagnostics on warfarin PK |
| `32_nonparametric_support_points.py` | Synthetic nonparametric support-point estimation demo |
| `33_tmdd_model.py` | Full TMDD / QSSA / Michaelis-Menten approximation comparison |
| `34_multidose_ss_nca.py` | Multi-dose steady-state NCA (AUCtau, Ctrough, accumulation ratio) |

## Estimation methods

| Script | Topic |
|--------|-------|
| `01_theophylline_fo.py` | FO estimation |
| `02_warfarin_foce.py` | FOCE with interaction |
| `15_bayes_laplace.py` | Bayesian Laplace approximation (PyMC optional) |
| `20_saem_estimation.py` | SAEM (stochastic EM) with convergence history |
| `21_laplacian_prior.py` | Laplacian estimation + prior augmentation (MAP) |

## Analysis workflows

| Script | Topic |
|--------|-------|
| `07_diagnostic_plots.py` | GOF, spaghetti, ETA diagnostic panels |
| `12_nca.py` | Non-compartmental analysis (NCA) |
| `13_covariate_search.py` | Stepwise covariate modelling (SCM) |
| `14_simulation_vpc.py` | Visual Predictive Check (VPC) |
| `18_parallel_bootstrap.py` | Parallel bootstrap resampling |

```{toctree}
:maxdepth: 1

01_theophylline_fo
02_warfarin_foce
03_two_compartment_iv
04_emax_pd_model
05_indirect_response
06_from_control_stream
07_diagnostic_plots
08_focei_optimizer_controls
09_focei_control_stream_extensions
10_optimal_design
16_dde_model
20_saem_estimation
22_pbpk_model
31_impmap_warm_start
32_nonparametric_support_points
33_tmdd_model
34_multidose_ss_nca
```
