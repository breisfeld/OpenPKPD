# Contributing

Contributions to OpenPKPD are welcome. This page describes how to set up
a development environment, run the test suite, and submit a pull request.

## Development setup

```bash
git clone https://github.com/your-org/OpenPKPD.git
cd OpenPKPD

# Install all extras including dev tools
uv sync --all-extras

# Verify
uv run pytest -q
```

## `just` workflow and platform support

The repository includes a `justfile` for common contributor workflows. The
recipes are intended to work on **macOS**, **Linux**, and **Windows**.

- Recipes now select their own `uv` extras where needed, so commands like docs,
  GUI, plotting examples, and cluster examples do not rely on a pre-synced
  all-extras environment.
- Shell-heavy operations have been moved into Python helpers under
  `scripts/just/` to keep behavior as consistent as possible across platforms.

Typical usage:

```bash
just run-tests-unit
just lint
just build-docs-html
just run-example 01
```

Remaining external prerequisites depend on the recipe:

- `install-hooks` requires `git`
- `build-docs-pdf` requires a working LaTeX installation
- `watch-docs` and the `*-and-open` recipes require browser/open support on the host
- `run-gui` requires a working desktop/Qt environment

## Running tests

```bash
# All tests
uv run pytest -q

# Unit tests only (fast)
uv run pytest tests/unit/ -q

# Integration tests
uv run pytest tests/integration/ -v

# With coverage report
uv run pytest --cov=openpkpd --cov-report=html
```

Test categories:

| Marker | Description |
|--------|-------------|
| `unit` | Fast isolated component tests (<1 s each) |
| `integration` | End-to-end pipeline tests |
| `regression` | Comparison vs golden NONMEM output |
| `slow` | Tests taking >30 seconds |

## Code style

OpenPKPD uses **ruff** for linting and formatting:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

Type annotations are checked with **mypy**:

```bash
uv run mypy src/
```

Pre-commit hooks run both automatically on `git commit`:

```bash
just install-hooks
```

## Building documentation

```bash
just build-docs-html
# or: just build-docs-and-open
```

## Building standalone installers

To package OpenPKPD as a self-contained desktop application (no Python
required on the end-user machine), see the
{doc}`installer build guide </building_installers>`.

## Performance-oriented development

For contributors working on runtime improvements, see the
{doc}`developer performance guide </developer_performance>`.

It summarizes the repository's preferred optimization style, existing profiling
entry points, validation expectations, and current high-value hotspots.

## Project structure

See the {doc}`/user_guide/index` for an overview of major modules. Key files:

| File | Role |
|------|------|
| `src/openpkpd/api/model_builder.py` | Fluent Python API |
| `src/openpkpd/parser/control_stream.py` | NM-TRAN parser |
| `src/openpkpd/parser/code_compiler.py` | `$PK`/`$ERROR` → Python |
| `src/openpkpd/pk/analytical/` | ADVAN1–4 closed-form solutions |
| `src/openpkpd/estimation/` | FO, FOCE, SAEM, IMP implementations |
| `src/openpkpd/plots/` | Diagnostic plot functions |

## Submitting a pull request

1. Fork the repository and create a feature branch.
2. Write tests for any new functionality.
3. Ensure all tests pass and ruff/mypy report no errors.
4. Open a pull request with a clear description of the change.

## Roadmap

See the documentation roadmap in `docs/roadmap.md` for current follow-up work:

- **Phase 2** — ADVAN5–10 (ODE-based models)
- **Phase 3** — NUTS/BAYES, mixture models, IOV, simulation
- **Phase 4** — PBPK, DDE, full regression suite

## Licence

OpenPKPD is released under the GNU Affero General Public Licence v3 (AGPLv3).
