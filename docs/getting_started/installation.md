# Installation

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installing with uv

```bash
# Core package (no plots)
uv add OpenPKPD

# With the desktop GUI (Qt / PySide6 + matplotlib)
uv add "OpenPKPD[gui]"

# With diagnostic plots (requires matplotlib)
uv add "OpenPKPD[plots]"

# Full install — plots + optimagic + sympy
uv add "OpenPKPD[full]"

# Development install (tests, linting, docs)
uv sync --all-extras

# Development install focused on the symbolic-kernel path
uv sync --extra dev --extra symbolic
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
| `symbolic` | `sympy>=1.13` | Symbolic PK solution derivation |
| `notebooks` | `marimo>=0.10`, `matplotlib>=3.9` | Interactive marimo notebooks |
| `cluster` | `dask[distributed]>=2024.8` | Distributed cluster-parallel execution |
| `r` | `rpy2>=3.5` | Optional Python-R bridge (`openpkpd.r_bridge`) |
| `full` | plots + optim + sympy | General scientific extras without Bayesian or GUI packages |
| `docs` | Sphinx + RTD theme | Building this documentation |

The R-backed external validation scripts use `Rscript` and the repo-local
`.r-lib` library tree directly. They do not require `rpy2`.

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
unit tests and cache-prewarm path active locally, install the `symbolic` extra
and run:

```bash
just prewarm-symbolic-caches
```
