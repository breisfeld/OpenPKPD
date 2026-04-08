# Coverage and Validation Map

This page is the consolidated source of truth for three related questions:

1. What analysis, estimation, PK, simulation, and workflow surfaces exist?
2. What kinds of tests back each surface?
3. Where are the concrete tests that enforce those claims?

Use this page when you want the test-backed inventory of the project. Use the
other pages around it for narrower purposes:

- [`testing.md`](testing.md): how to run the suite and how the pytest tiers are organized
- [`validation_matrix.md`](validation_matrix.md): what validation level the project claims for each surface
- [`external_validation_benchmarks.md`](external_validation_benchmarks.md): the external datasets and third-party tools used as anchors
- [`analysis_validation_gaps.md`](analysis_validation_gaps.md): what is still weak and what should be added next

## How to read this page

Test types:

- **Unit**: local deterministic checks, formulas, invariants, and boundary behavior
- **Integration**: short end-to-end workflows composed from multiple components
- **Regression**: checked-in numerical baselines used to detect drift
- **External validation**: agreement against independent software, literature values, SciPy, or exact closed forms

Validation character:

- **Analytic/reference-heavy**: the strongest footing; relies on closed forms, exact identities, SciPy, literature tables, or cross-tool references
- **Behavioral/integration-heavy**: useful coverage, but more about consistent behavior than independent truth

## Estimation methods

| Surface | Main implementation | Main tests | Test types present | Validation character |
| --- | --- | --- | --- | --- |
| FO | `estimation/fo.py` | `tests/unit/estimation/test_fo.py`, `tests/external_validation/test_estimation_reference.py`, `tests/external_validation/test_vs_nlmixr2.py`, `tests/regression/test_cross_method_validation.py` | Unit, regression, external | Analytic/reference-heavy |
| FOCE / FOCEI | `estimation/foce.py` | `tests/unit/estimation/test_foce.py`, `tests/external_validation/test_estimation_reference.py`, `tests/external_validation/test_vs_nlmixr2.py`, `tests/external_validation/test_vs_nonmem.py`, `tests/regression/test_cross_method_validation.py` | Unit, regression, external | Analytic/reference-heavy |
| Laplacian | `estimation/laplacian.py` | `tests/unit/estimation/test_laplacian.py`, `tests/external_validation/test_estimation_reference.py`, `tests/regression/test_cross_method_validation.py` | Unit, regression, external | Analytic/reference-heavy |
| SAEM | `estimation/saem.py` | `tests/unit/estimation/test_saem.py`, `tests/external_validation/test_saem_reference.py`, `tests/external_validation/test_vs_monolix.py`, `tests/regression/test_regression.py` | Unit, regression, external | Mixed; credible but thinner than FO/FOCEI |
| IMP / IMPMAP | `estimation/imp.py` | `tests/unit/estimation/test_imp.py`, `tests/external_validation/test_estimation_reference.py`, `tests/external_validation/test_imp_empirical_reference.py`, `tests/regression/test_regression.py` | Unit, regression, external | Mixed; strong analytic core, narrower empirical breadth |
| BAYES(Laplace) | `estimation/bayes.py` | `tests/unit/estimation/test_bayes.py`, `tests/external_validation/test_bayes_empirical_reference.py`, `tests/regression/test_regression.py` | Unit, regression, external | Mixed; strong on local Gaussian path, narrower global parity claims |
| BAYES(NUTS) | `estimation/nuts.py` | `tests/unit/estimation/test_nuts.py`, `tests/external_validation/test_bayes_empirical_reference.py`, `tests/regression/test_regression.py` | Unit, regression, limited external | Behavioral plus some exact-target checks; still second-tier |
| Nonparametric (NPML / NPEM) | `estimation/nonparametric.py` | `tests/unit/estimation/test_nonparametric.py`, `tests/external_validation/test_vs_pharmpy.py`, `tests/regression/test_cross_method_validation.py`, `tests/regression/test_regression.py` | Unit, regression, external | Mixed; solid weight/support checks, still narrower empirical breadth |
| Estimation diagnostics / result summaries | `estimation/base.py`, result helpers | `tests/unit/estimation/test_estimation_base.py`, `tests/unit/estimation/test_shrinkage.py`, `tests/external_validation/test_covariance_reference.py` | Unit, external | Good structural coverage |

## PK subroutines and solver surfaces

| Surface | Main implementation | Main tests | Test types present | Validation character |
| --- | --- | --- | --- | --- |
| ADVAN1 | `pk/analytical/advan1.py` | `tests/unit/pk/test_advan.py`, `tests/unit/model/test_individual_validation.py`, `tests/external_validation/test_pk_subroutines_reference.py`, `tests/integration/test_pk_integration.py` | Unit, integration, external | Analytic/reference-heavy |
| ADVAN2 | `pk/analytical/advan2.py` | `tests/unit/pk/test_advan.py`, `tests/unit/model/test_symbolic_gradient_advan2.py`, `tests/external_validation/test_pk_subroutines_reference.py`, `tests/integration/test_theophylline.py`, `tests/integration/test_pk_integration.py` | Unit, integration, external | Analytic/reference-heavy |
| ADVAN3 | `pk/analytical/advan3.py` | `tests/unit/pk/test_advan3.py`, `tests/unit/validation/test_numerical_accuracy.py`, `tests/external_validation/test_pk_subroutines_reference.py`, `tests/integration/test_two_compartment.py` | Unit, integration, external | Analytic/reference-heavy |
| ADVAN4 | `pk/analytical/advan4.py` | `tests/unit/pk/test_advan4.py`, `tests/integration/test_pk_integration.py` | Unit, integration | Good, but less externally anchored than ADVAN1-3 |
| ADVAN5 | `pk/analytical/advan5.py` | `tests/unit/pk/test_advan5.py`, `tests/integration/test_examples.py` | Unit, integration | Strong within its model family |
| ADVAN7 | `pk/analytical/advan7.py` | `tests/unit/pk/test_advan7.py`, `tests/unit/api/test_model_builder_validation.py` | Unit | Functional expm-backed validation now exists; empirical breadth still minimal |
| ADVAN11 | `pk/analytical/advan11.py` | `tests/unit/pk/test_advan11.py` | Unit | Strong formula-level checks |
| ADVAN12 | `pk/analytical/advan12.py` | `tests/unit/pk/test_advan12.py` | Unit | Strong formula-level checks |
| ADVAN6 general ODE | `pk/ode/advan6.py` | `tests/unit/pk/test_ode_advan6.py`, `tests/unit/test_native_cvodes.py`, `tests/unit/rust/test_rust_python_parity.py` | Unit, dedicated native lane | Strong on mechanics; empirical breadth depends on estimator path |
| ADVAN8 stiff ODE | `pk/ode/advan8.py` | `tests/unit/pk/test_ode_advan6.py`, `tests/unit/pk/test_advan13_sensitivity.py` | Unit | Good numerical/mechanical coverage |
| ADVAN10 Michaelis-Menten | `pk/ode/advan10.py` | `tests/unit/pk/test_ode_advan6.py`, `tests/external_validation/test_extended_models_reference.py` | Unit, external | Good reference footing |
| ADVAN13 sensitivities | `pk/ode/advan13.py` | `tests/unit/pk/test_advan13_sensitivity.py`, `tests/unit/test_native_cvodes.py` | Unit, dedicated native lane | Good mechanics/sensitivity coverage; narrower workflow breadth |
| ADVAN16-style DDE | `pk/ode/dde.py` | `tests/unit/pk/test_dde.py`, `tests/integration/test_examples.py` | Unit, integration | Good functional coverage |
| Transit / parallel / EHC absorption | `pk/absorption/` | `tests/unit/pk/test_absorption.py` | Unit | Good within implemented subset |
| PBPK | `pk/pbpk/` | `tests/unit/pk/test_pbpk.py`, `tests/integration/test_examples.py` | Unit, integration | Functional, but still narrower than core compartmental PK |
| TRANS parameterizations | parser + PK routing | `tests/unit/pk/test_transforms.py`, `tests/unit/api/test_model_builder_validation.py` | Unit | Good selector/parameterization coverage |

## Diagnostics, simulation, and NCA

| Surface | Main implementation | Main tests | Test types present | Validation character |
| --- | --- | --- | --- | --- |
| Simulation engine | `simulation/engine.py` | `tests/unit/simulation/test_engine.py` | Unit | Strong behavioral/core mechanics |
| VPC / pcVPC | `simulation/vpc.py` | `tests/unit/simulation/test_pcvpc.py`, `tests/integration/test_vpc_pipeline.py`, `tests/regression/test_diagnostics_regression.py`, `tests/external_validation/test_diagnostics_reference.py` | Unit, integration, regression, external | One of the strongest surfaces |
| NPDE | `simulation/npde.py` | `tests/unit/simulation/test_npde.py`, `tests/regression/test_diagnostics_regression.py`, `tests/external_validation/test_diagnostics_reference.py` | Unit, regression, external | One of the strongest surfaces |
| NPC | `simulation/npc.py` | `tests/unit/simulation/test_npc.py`, `tests/external_validation/test_diagnostics_reference.py` | Unit, external | Good formula-level footing |
| SSE | `simulation/sse.py` | `tests/unit/simulation/test_sse.py`, `tests/regression/test_diagnostics_regression.py` | Unit, regression | More behavioral than externally anchored |
| Diagnostic tables / GOF helpers | `plots/diagnostics.py`, plotting modules | `tests/unit/plots/test_diagnostics.py`, `tests/unit/plots/test_plots.py`, `tests/unit/plots/test_simulation_plots.py` | Unit | Mixed; many deterministic checks, fewer independent references |
| Core dense-profile NCA | `nca/nca.py` | `tests/unit/nca/test_nca.py`, `tests/regression/test_diagnostics_regression.py`, `tests/external_validation/test_vs_pknca.py`, `tests/external_validation/test_vs_winnonlin_indometh.py` | Unit, regression, external | One of the strongest surfaces |
| Multidose NCA | `nca/nca.py` multidose helpers | `tests/unit/nca/test_multidose_nca.py` | Unit | Good local numerical coverage |
| Sparse NCA | `nca/sparse.py` | `tests/unit/nca/test_sparse_nca.py` | Unit | Good analytical checks within its scope |
| Urine NCA | `nca/urine.py` | `tests/unit/nca/test_urine_nca.py` | Unit | Good analytical checks |
| Crossover BE / power / sample size | `nca/crossover.py` | `tests/unit/nca/test_crossover.py`, `tests/external_validation/test_bioequivalence_reference.py`, `tests/external_validation/test_nca_reference.py` | Unit, external | Good formula/reference coverage |
| CDISC PP export | `nca/cdisc_pp.py` | `tests/unit/nca/test_cdisc_pp.py` | Unit | Structural/export coverage |

## Analysis and model families

| Surface | Main implementation | Main tests | Test types present | Validation character |
| --- | --- | --- | --- | --- |
| Direct and mechanistic PD / PK-PD | `models/pkpd.py` | `tests/unit/models/test_pkpd.py`, `tests/unit/models/test_sequential.py`, `tests/integration/test_emax_pd.py`, `tests/regression/test_pd_models_regression.py`, `tests/regression/test_pkpd_models_regression.py` | Unit, integration, regression | Broad functional coverage |
| Population PD | `models/population_pd.py` | `tests/unit/models/test_population_pd.py`, `tests/regression/test_pd_models_regression.py` | Unit, regression | Good recovery-focused coverage |
| TTE / survival | `models/tte.py` | `tests/unit/models/test_tte.py`, `tests/external_validation/test_extended_models_reference.py` | Unit, external | Strong reference footing |
| Count models | `models/count.py` | `tests/unit/models/test_count.py`, `tests/external_validation/test_extended_models_reference.py` | Unit, external | Strong reference footing |
| Ordered categorical / proportional odds | `models/categorical.py` | `tests/unit/models/test_categorical.py`, `tests/external_validation/test_extended_models_reference.py` | Unit, external | Strong reference footing |
| CTMC / Markov / HMM | `models/categorical.py`, `models/markov.py` | `tests/unit/models/test_categorical.py`, `tests/unit/models/test_markov.py`, `tests/external_validation/test_extended_models_reference.py` | Unit, external | Strong reference footing |
| TMDD | `models/tmdd.py` | `tests/unit/models/test_tmdd.py`, `tests/external_validation/test_extended_models_reference.py` | Unit, external | Strong limit-case/reference checks |
| Static DDI analysis | `models/ddi.py` | `tests/unit/models/test_ddi.py` | Unit | Strong formula-level checks |
| Covariate effect functions | covariate/effect helpers | `tests/unit/covariate/test_effects.py`, `tests/external_validation/test_covariate_effects_reference.py` | Unit, external | Good formula/reference coverage |
| Model comparison and information criteria | result/model-comparison helpers | `tests/unit/inference/test_model_comparison.py`, `tests/external_validation/test_inference_reference.py` | Unit, external | Strong formula/reference coverage |
| Bootstrap / SCM | `simulation/bootstrap.py`, `simulation/scm.py` | `tests/unit/inference/test_bootstrap.py`, `tests/unit/inference/test_bootstrap_bca.py`, `tests/unit/scm/test_scm_base_convergence.py`, `tests/unit/covariate/test_scm.py` | Unit | Good workflow mechanics, lighter independent references |

## Workflow, parsing, outputs, and GUI

| Surface | Main implementation | Main tests | Test types present | Validation character |
| --- | --- | --- | --- | --- |
| Control-stream parsing/runtime | parser + runtime layers | `tests/unit/parser/`, `tests/integration/test_control_stream_mixture.py`, `tests/integration/test_control_stream_prior.py`, `tests/integration/test_control_stream_simulation.py` | Unit, integration | Broad supported-subset coverage |
| NONMEM-style writers/readers | `io/nonmem_output.py`, result readers | `tests/unit/output/test_output_writers.py`, `tests/unit/output/test_nonmem_reader.py` | Unit | Good structural/export coverage |
| Data preprocessing / BLQ / covariate imputation | `data/`, `preprocessing/` | `tests/unit/data/test_preprocessor.py`, `tests/unit/data/test_impute.py`, `tests/unit/data/test_blq.py`, `tests/integration/test_blq_pipeline.py` | Unit, integration | Good workflow coverage |
| GUI workflows and review shell | `src/openpkpd_gui/` | `tests/unit/gui/`, `tests/unit/gui/test_shell_smoke.py`, `tests/unit/gui/test_results_workflow.py` | Unit | Strong workflow-shell coverage, lighter empirical references |

## Dedicated release lanes

These are explicit coverage lanes for routes that should not be inferred only
from the broad suite:

| Lane | Command | Main surfaces covered |
| --- | --- | --- |
| Symbolic route | `just run-tests-symbolic` | SymPy-backed analytical kernels, symbolic ETA gradients, symbolic guards |
| Native CVODES route | `just run-tests-native-cvodes` | Rust/native extension, CVODES wiring, native/sensitivity parity, serial native performance gate |
| Strict release suite | `just run-tests-release` | Release-gated unit/integration/regression/external-validation path with strict fixture enforcement |

## External anchors and citations

This page intentionally keeps citation detail light and defers bibliographic
authority to the dedicated reference pages:

- [`external_validation_benchmarks.md`](external_validation_benchmarks.md) lists
  the concrete external tools, datasets, and benchmark assets used by the tests
- [`citations.md`](citations.md) contains the full bibliographic references for
  the literature and public benchmark sources mentioned across the validation docs

The main external anchors referenced by the coverage map are:

- `nlmixr2`, `NONMEM`, `Monolix`, and Pharmpy for cross-tool estimation checks
- `scipy.stats`, `scipy.linalg`, and `scipy.integrate` for exact numerical references
- public PKNCA and WinNonlin-backed Indometh tables for NCA benchmarks

## Current strongest areas

- NCA, especially dense-profile NCA with public PKNCA and WinNonlin-backed anchors
- VPC / pcVPC / NPDE / NPC
- FO / FOCE / FOCEI / Laplacian estimation formulas and core workflows
- TTE, count, categorical, CTMC/HMM, and TMDD limit-case checks
- ADVAN1/2/3 analytical PK

## Current thinner areas

- ODE-heavy advanced estimators beyond the strongest FO/FOCEI paths
- PBPK and DDE compared with the depth available for core PK subroutines
- Bootstrap / SCM / SSE external anchoring relative to the strongest diagnostic and NCA areas
- BAYES(NUTS) and some mixed-endpoint empirical paths, which remain intentionally second-tier

## Recommended maintenance rule

When a method or workflow changes, update this page together with:

1. the concrete tests
2. [`validation_matrix.md`](validation_matrix.md)
3. [`external_validation_benchmarks.md`](external_validation_benchmarks.md) if the external anchor set changed

That keeps support claims, test inventory, and external evidence synchronized.
