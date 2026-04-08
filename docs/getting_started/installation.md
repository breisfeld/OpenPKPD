# Installation

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installing with uv

```bash
# Core package (includes SymPy analytical-kernel support)
uv add OpenPKPD

# With the desktop GUI (Qt / PySide6 + matplotlib)
uv add "OpenPKPD[gui]"

# With diagnostic plots (requires matplotlib)
uv add "OpenPKPD[plots]"

# Full install — plots + optimagic
uv add "OpenPKPD[full]"

# Development install (tests, linting, docs)
uv sync --all-extras

# Development install focused on the symbolic-kernel path
uv sync --extra dev
```

## Installing with pip

```bash
pip install OpenPKPD
pip install "OpenPKPD[gui]"     # desktop GUI (PySide6 + matplotlib)
pip install "OpenPKPD[plots]"   # with matplotlib
pip install "OpenPKPD[full]"
```

## Optional extras

| Extra | Contents | When to use |
|-------|----------|-------------|
| `gui` | `pyside6>=6.10.2`, `platformdirs>=4.9`, `matplotlib>=3.9` | Desktop GUI (`openpkpd-gui`) with plot output |
| `plots` | `matplotlib>=3.9` | Diagnostic plots: GOF, PK, PD, ETA |
| `bayes` | `pymc>=5.16` | PyMC MCMC backend |
| `optim` | `optimagic>=0.5` | Unified optimizer interface |
| `symbolic` | compatibility alias | Legacy extra name; SymPy is now part of the core install |
| `notebooks` | `marimo>=0.10`, `matplotlib>=3.9` | Interactive marimo notebooks |
| `cluster` | `dask[distributed]>=2024.8` | Distributed cluster-parallel execution |
| `r` | `rpy2>=3.5` | Optional Python-R bridge (`openpkpd.r_bridge`) |
| `full` | plots + optim | General scientific extras without Bayesian or GUI packages |
| `docs` | Sphinx + RTD theme | Building this documentation |

The R-backed external validation scripts use `Rscript` and the repo-local
`.r-lib` library tree directly. They do not require `rpy2`.

SymPy is a core dependency because the analytical-kernel generation and
symbolic derivative path are now part of the explicit release validation lane.

## Why Install An Extra?

Use the core install if you want the library, CLI, standard estimation paths,
and the always-on symbolic analytical route. Add extras when you want a
specific capability beyond that baseline:

| Extra | Main benefit | Worth installing when | Main tradeoff |
|-------|--------------|-----------------------|---------------|
| `gui` | Desktop workflow, artifact browsing, interactive review | You want to work through fits, diagnostics, and reports graphically | Pulls in Qt / PySide6 and matplotlib |
| `plots` | Diagnostic and analysis plotting | You want GOF, VPC, PK/PD, ETA, or simulation figures from Python/CLI workflows | Adds matplotlib |
| `jit` | Faster ODE-heavy fitting and simulation | You repeatedly fit or simulate `$DES` models and want runtime speedups | Adds Numba/LLVM compatibility constraints |
| `bayes` | PyMC-backed Bayesian inference | You want a fuller MCMC backend than the built-in Laplace / native NUTS routes | Larger probabilistic-programming stack |
| `optim` | Optimagic optimizer interface | You want access to the broader optimizer surface and optimizer-specific controls | Additional optimizer dependency surface |
| `cluster` | Distributed execution | You run large bootstrap/SSE/simulation jobs across multiple workers or machines | Additional cluster runtime complexity |
| `r` | Python-side R bridge | You want to call R directly from Python via `openpkpd.r_bridge` | Adds `rpy2`; not needed for the Rscript-based validation harness |
| `notebooks` | Marimo notebook runtime | You want interactive exploratory notebooks bundled with the repo | Adds marimo and plotting runtime |
| `packaging` | Installer creation | You are building end-user installers or release artifacts | Build-only tooling, not needed for normal use |
| `docs` | Documentation build tooling | You are editing or publishing the docs | Docs-only dependency set |
| `full` | Common scientific extras in one install | You want plots plus optimizer tooling without selecting extras one by one | Still does not include GUI or Bayesian extras |

Practical rule of thumb:

- use core only for scripting, CLI fits, and lightweight environments
- add `jit` for ODE-heavy performance work
- add `gui` for interactive desktop use
- add `bayes` when Bayesian workflows are central
- add `full` when you want a convenient scientific-workbench install

## Verifying the install

```bash
python -c "import openpkpd; print(openpkpd.__version__)"
openpkpd --help
```

## Launching the desktop GUI

After installing the `gui` extra, start the desktop application with:

```bash
openpkpd-gui
```

From a source checkout, `uv run openpkpd-gui` also works.

If you are working from the repository checkout, `just run-gui` is also
available.

## Building the documentation locally

```bash
just build-docs-html
# or: just build-docs-and-open
```

The repository `justfile` is intended to work across macOS, Linux, and Windows
for common contributor workflows.

If you are working on analytical PK derivative kernels or want the symbolic
unit tests and cache-prewarm path active locally, run:

```bash
just run-tests-symbolic
just prewarm-symbolic-caches
```
