# Test suite

OpenPKPD uses a four-tier test strategy that maps directly onto how pharmacometric
software is validated: fast unit checks for correctness of individual formulas,
end-to-end integration checks for workflow behaviour, regression baselines to catch
numerical drift across releases, and external-validation tests that anchor results
to independent reference implementations.

## Test categories

### Unit tests (`@pytest.mark.unit`)

Unit tests are the largest and fastest tier. They target one class or function in
isolation and assert precise numerical outcomes — closed-form formula checks,
algebraic identities, boundary behaviour, and invariants like monotonicity or
positivity. Most finish in microseconds; the full unit suite completes in under
two minutes.

Tests in this tier live under `tests/unit/` and are organised to mirror the source
tree: `tests/unit/estimation/`, `tests/unit/nca/`, `tests/unit/simulation/`, and
so on. They require no external software and carry no data-loading side effects.

### Integration tests (`@pytest.mark.integration`)

Integration tests compose multiple components into a short end-to-end pipeline and
assert that the observable outcome is correct: OFV decreases, parameter estimates
are physiologically plausible, VPC bands cover the observed data, and so on. They
do not check values against an external reference; they check that the pipeline
produces internally consistent results.

Tests live under `tests/integration/` and typically take 5–60 seconds each. A
representative example is `test_warfarin.py`, which loads a NONMEM-style dataset,
builds a one-compartment FOCE model with `ModelBuilder`, runs `.fit()`, and
asserts that OFV decreased, convergence was reached, and CL and V are inside
their published physiological range.

### Regression tests (`@pytest.mark.regression`)

Regression tests compare numerical output against a fixed saved baseline stored as
JSON under `tests/regression/reference_runs/`. They are run on every commit to
detect accidental numerical drift — for example, a refactor that shifts a final OFV
by 0.1 or changes a shrinkage value in the fourth decimal place. The baselines were
generated on a known-good version of the code and are committed alongside the tests.

This tier covers: FOCE, Laplacian, SAEM, IMP, Bayesian, and nonparametric
estimation on shared reference datasets; cross-method THETA consistency; PD model
parameter recovery; and VPC / NPDE / NPC / NCA / SSE diagnostic summaries.
Regression tests are marked `@pytest.mark.slow` and take 1–10 minutes. Run them
with `pytest -m regression`.

Two explicit release lanes now sit beside the broad suite:

- `just run-tests-symbolic` exercises the SymPy-backed analytical-kernel and
  symbolic-gradient route directly.
- `just run-tests-native-cvodes` rebuilds the optional native CVODES extension,
  runs the dedicated native/rust parity suites, and then runs the serial native
  sensitivity performance gate.

### External-validation tests (`@pytest.mark.external_validation`)

External-validation tests compare openpkpd output against an independent reference:
another software package (nlmixr2, Pharmpy), a published formula with a known exact
value, or a scipy implementation of the same mathematical procedure. Agreement at a
specified tolerance constitutes evidence that the implementation is correct, not
merely self-consistent.

The majority of external-validation tests run in under a second because they test
analytic identities — TOST p-values against `scipy.stats.t`, LRT p-values against
`scipy.stats.chi2`, Karlsson–Sheiner shrinkage against a manual formula, allometric
scaling against Anderson & Holford (2008) — and do not require a model fit.
The slower subgroup (marked additionally `@pytest.mark.slow`) calls into nlmixr2
reference JSON files produced by the R scripts in `tests/external_validation/nlmixr2/`
or runs empirical cross-tool fits on the Boeckmann theophylline dataset
(`nlmixr2`, public Monolix theophylline SAEM outputs), the PK-only
`nlmixr2data::warfarin` subset (`nlmixr2`), a reduced 4-subject mixed-endpoint
joint `warfarin` PK/PD benchmark (`nlmixr2`), a broader 6-subject second-tier
mixed-endpoint `warfarin` PK/PD benchmark (`nlmixr2`), and Pharmpy's bundled
phenobarbital `pheno` example to check that openpkpd follows the same estimator
trends, fitted parameter values, and the current nonparametric support-point
envelope. The fast subgroup now also includes a public
PKNCA/Phoenix-style theophylline NCA parity check and a published WinNonlin-
backed Indometh NCA benchmark.

Run only the fast external-validation tests with `pytest -m "external_validation and not slow"`.

To run the public PKNCA theophylline NCA benchmark only, use:

`uv run python -m pytest -q tests/external_validation/test_vs_pknca.py`

To run the public WinNonlin-backed Indometh NCA benchmark only (zero-start, IV
bolus, IV infusion, and extravascular scenarios), use:

`uv run python -m pytest -q tests/external_validation/test_vs_winnonlin_indometh.py`

To run the public Monolix theophylline SAEM benchmark only, use:

`uv run python -m pytest -q tests/external_validation/test_vs_monolix.py -m "slow and external_validation"`

For the Pharmpy-backed slow checks in a temporary test environment, the working
command is:

`uv run --with pharmpy-core python -m pytest -q tests/external_validation/test_vs_pharmpy.py -m "slow and external_validation"`

Use `python -m pytest` here rather than the bare `pytest` console script so the
ephemeral `pharmpy-core` environment is visible to the test runner.

To run only the new nonparametric empirical Pharmpy benchmark, use:

`uv run --with pharmpy-core python -m pytest -q tests/external_validation/test_vs_pharmpy.py -k NonparametricVsPharmpyPheno -n0`

To regenerate the bundled warfarin `nlmixr2` references after provisioning the R
packages in a repo-local library such as `.r-lib`, run:

`cd tests/external_validation/nlmixr2 && Rscript run_warfarin.R`

To regenerate the bundled mixed-endpoint `warfarin` PK/PD references (full,
reduced 4-subject, and reduced 6-subject assets), run:

`cd tests/external_validation/nlmixr2 && Rscript run_warfarin_pkpd.R`

To run only the reduced mixed-endpoint empirical benchmark (about one minute in
the current validation environment), use:

`uv run python -m pytest -q tests/external_validation/test_vs_nlmixr2.py -k WarfarinPKPDReducedFOvsNlmixr2`

To run the broader second-tier 6-subject mixed-endpoint benchmark (about two to
three minutes in the current validation environment), use:

`uv run python -m pytest -q tests/external_validation/test_vs_nlmixr2.py -k WarfarinPKPDReduced6FOvsNlmixr2`

For a full slow external-validation pass in a release-validation environment,
run the following sequence:

1. Regenerate the bundled `nlmixr2` references if the R reference assets need to
   be refreshed:
   - `cd tests/external_validation/nlmixr2 && Rscript run_theophylline.R`
   - `cd tests/external_validation/nlmixr2 && Rscript run_warfarin.R`
   - `cd tests/external_validation/nlmixr2 && Rscript run_warfarin_pkpd.R`
2. Run the `nlmixr2`-backed slow external-validation suite:
   - `uv run python -m pytest -q tests/external_validation/test_vs_nlmixr2.py -m "slow and external_validation"`
3. Run the Pharmpy-backed slow external-validation suite in an ephemeral test
   environment:
   - `uv run --with pharmpy-core python -m pytest -q tests/external_validation/test_vs_pharmpy.py -m "slow and external_validation"`

This keeps the heavier cross-tool checks isolated from the default fast local-dev
workflow while still making them reproducible for release validation.

### Mixed-endpoint benchmark roadmap

The empirical mixed-endpoint benchmark ladder currently has three levels:

- **Release-gated benchmark:** the reduced 4-subject joint `warfarin` PK/PD FO
  benchmark. This is the primary practical gate for the real ODE + `DVID`-routed
  path.
- **Second-tier benchmark:** the broader reduced 6-subject joint `warfarin`
  PK/PD FO benchmark. This is intended for deeper slow validation rather than the
  primary release gate.
- **Future expansion target:** the full 32-subject mixed-effects joint
  `warfarin` PK/PD benchmark. Full-reference assets exist, but this path is not
  yet practical enough for regular release-gating.

## Recommended validation gates

The tiers serve different purposes, so they should not all be treated equally in
release gating.

| Stage | Mandatory tiers | Purpose |
|---|---|---|
| Local dev / pull request | `unit`, `integration`, `external_validation and not slow` | Fast correctness gate: formula regressions, pipeline breakage, and independent reference checks that finish quickly |
| Pre-release / release candidate | all of the above **plus** `regression` | Drift gate: blocks releases if saved numerical baselines move unexpectedly |
| Reference-package validation job | `slow and external_validation` when the required tools/data are present | Cross-tool confidence gate: nlmixr2 / Pharmpy comparisons are slower and environment-dependent, so they should run in a dedicated release-validation environment |

Two cautions:

- `regression` is mandatory before release, but it remains an **internal-baseline**
  gate rather than an independent scientific reference.
- Optional-package checks should be non-blocking in generic developer
  environments, but blocking in the controlled release-validation environment
  where `nlmixr2`/`pharmpy` are intentionally provisioned.

### Slow marker (`@pytest.mark.slow`)

`slow` is a timing marker, not a separate tier. It is applied to tests in the
regression and external-validation tiers that take more than 30 seconds — mainly
full or partial model fits. Run all slow tests with `pytest -m slow`. To skip them
in CI: `pytest -m "not slow"`.

## Running the test suite

```bash
# full suite (slow tests skipped)
uv run pytest -m "not slow"

# fast external-validation only
uv run pytest -m "external_validation and not slow"

# single tier
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m regression
uv run pytest -m slow

# one area
uv run pytest tests/unit/nca/
uv run pytest tests/external_validation/

# dedicated release lanes
just run-tests-symbolic
just run-tests-native-cvodes
```

## Coverage inventory

The consolidated coverage inventory now lives in
[`analysis_tools.md`](analysis_tools.md).

Use that page for:

- the full list of estimation, PK, simulation, NCA, and workflow surfaces
- the concrete test files backing each surface
- the current mix of unit, integration, regression, and external-validation
  evidence behind those surfaces

This page stays focused on test tiers, release gates, and the commands used to
run them.
