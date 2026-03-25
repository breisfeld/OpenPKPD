# OpenPKPD justfile
# Install: https://just.systems/

set dotenv-load

# Default: list all recipes
default:
    @just --list

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# uv command helpers
uv_base := "uv run"
uv_dev := "uv run --extra dev"
uv_docs := "uv run --extra docs --extra dev"
uv_plots := "uv run --extra plots"
uv_cluster := "uv run --extra cluster"
uv_gui := "uv run --extra gui"
uv_notebooks := "uv run --extra notebooks"

# Install all dev dependencies
install:
    uv sync --all-extras

# Install core only (no dev/docs/plots)
install-core:
    uv sync

# Install with plots support
install-plots:
    uv sync --extra plots --extra dev

# Install docs dependencies
install-docs:
    uv sync --extra docs --extra dev

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

# Run the full test suite
run-tests:
    {{uv_dev}} pytest -q

# Run unit tests only (fast, <1 s each)
run-tests-unit:
    {{uv_dev}} pytest tests/unit/ -q

# Run integration tests
run-tests-integration:
    {{uv_dev}} pytest tests/integration/ -v

# Run regression tests (requires NONMEM reference files)
run-tests-regression:
    {{uv_dev}} pytest tests/regression/ -v -m regression

# Run tests with coverage report
run-tests-cov:
    {{uv_dev}} pytest --cov=openpkpd --cov-report=term-missing --cov-report=html -q
    @echo "HTML report: htmlcov/index.html"

# Run a specific test file or pattern
# Usage: just test-only tests/unit/pk/test_advan4.py
#        just test-only -k "test_peak"
run-test-only *args:
    {{uv_dev}} pytest {{ args }}

# ---------------------------------------------------------------------------
# Linting and formatting
# ---------------------------------------------------------------------------

# Run ruff linter
lint:
    {{uv_dev}} ruff check src/ tests/

# Run ruff formatter (check only)
check-format:
    {{uv_dev}} ruff format --check src/ tests/

# Auto-fix lint issues and format
format:
    {{uv_dev}} ruff check --fix src/ tests/
    {{uv_dev}} ruff format src/ tests/

# Run mypy type checking
typecheck:
    {{uv_dev}} mypy src/

# Run all checks (lint + types)
check: lint typecheck

# ---------------------------------------------------------------------------
# Running analyses
# ---------------------------------------------------------------------------

# Run a NONMEM control stream file
# Usage: just run path/to/model.ctl
run-ctl ctl:
    uv run openpkpd run {{ ctl }} --verbose

# Run with method override
# Usage: just run-foce path/to/model.ctl
run-foce ctl:
    uv run openpkpd run {{ ctl }} --method FOCE --verbose

# Parse a control stream (inspect without fitting)
# Usage: just parse path/to/model.ctl
parse-ctl ctl:
    uv run openpkpd parse {{ ctl }}

# Parse a control stream and emit JSON
# Usage: just parse-json path/to/model.ctl
parse-json ctl:
    uv run openpkpd parse {{ ctl }} --json

# Profile representative analysis routines and write JSON baseline output
# Usage: just profile-analysis
#        just profile-analysis --workloads nca --nca-subjects 5000
#        just profile-analysis --workloads diagnostics_covariate --covariate-subjects 140
profile-analysis *args:
    {{uv_base}} python scripts/profile_analysis.py {{ args }}

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

# Run all examples (per-example optional dependencies are selected automatically)
run-all-examples:
    {{uv_base}} python scripts/just/examples.py run-all

# Run a single example by number (01–24)
# Usage: just example 01
#        just example 12
run-example num:
    {{uv_base}} python scripts/just/examples.py run {{ num }}

# Run examples and save figures to a directory
# Usage: just examples-save /tmp/figs
run-examples-and-save outdir:
    {{uv_base}} python scripts/just/examples.py run-all --output {{ outdir }}

# Capture outputs and figures for all examples (run before building docs)
# Individual example failures are tolerated (output captured in *_output.txt).
capture-examples: install-plots
    {{uv_base}} python scripts/just/examples.py capture

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

# Build HTML documentation
build-docs-html: install-docs
    {{uv_docs}} python scripts/just/docs.py html
    @echo "Open: docs/_build/html/index.html"

# Build and open documentation in the browser
build-docs-and-open: build-docs-html
    {{uv_base}} python scripts/just/system.py open docs/_build/html/index.html

# Build PDF user manual (requires a LaTeX installation, e.g. MacTeX / TeX Live)
build-docs-pdf: install-docs
    {{uv_docs}} python scripts/just/docs.py latexpdf
    @echo "PDF: docs/_build/latex/openpkpd.pdf"

# Build and open PDF user manual
build-docs-pdf-and-open: build-docs-pdf
    {{uv_base}} python scripts/just/system.py open docs/_build/latex/openpkpd.pdf

# Clean documentation build artifacts
clean-docs:
    {{uv_docs}} python scripts/just/docs.py clean

# Watch and auto-rebuild docs (requires sphinx-autobuild)
watch-docs: install-docs
    {{uv_docs}} --with sphinx-autobuild python scripts/just/docs.py watch

# ---------------------------------------------------------------------------
# GUI (Qt desktop app)
# ---------------------------------------------------------------------------

# Install GUI dependencies
install-gui:
    uv sync --extra gui

# Launch the desktop GUI
run-gui: install-gui
    {{uv_gui}} python scripts/just/system.py run-gui

# ---------------------------------------------------------------------------
# Notebooks (marimo)
# ---------------------------------------------------------------------------

# Install notebook dependencies
install-notebooks:
    uv sync --extra notebooks

# Launch a specific marimo notebook in edit mode
# Usage: just notebook notebooks/01_quickstart.py
notebook path:
    {{uv_notebooks}} marimo edit {{ path }}

# Launch the notebook index (lists all notebooks)
notebooks: install-notebooks
    {{uv_notebooks}} marimo edit notebooks/

# Run a notebook as a script (non-interactive, for CI/testing)
# Usage: just run-notebook notebooks/01_quickstart.py
run-notebook path:
    {{uv_notebooks}} marimo run --headless {{ path }}

# Run the notebook integration test suite
test-notebooks: install-notebooks
    {{uv_notebooks}} pytest tests/integration/test_notebooks.py -v -m "slow and integration"

# ---------------------------------------------------------------------------
# Rust compiled extensions
# ---------------------------------------------------------------------------

# Build and install the openpkpd._core Rust extension in-place (development).
# Requires: Rust toolchain (rustup.rs) and maturin (installed via uv sync --extra dev).
# Run once after cloning or after modifying rust/src/lib.rs.
build-core:
    env -u CONDA_PREFIX uv run maturin develop --release

# Build a distributable wheel for the current platform only (not manylinux).
# For production manylinux wheels use the CI pipeline.
build-wheel:
    env -u CONDA_PREFIX uv run maturin build --release --out dist/

# ---------------------------------------------------------------------------
# Git hooks
# ---------------------------------------------------------------------------

# Install the project's git hooks (run once after cloning)
install-hooks:
    {{uv_base}} python scripts/just/system.py install-hooks

# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

# Build the platform-native standalone installer
build-installer:
    uv run --extra packaging --extra gui python scripts/build_installer.py

# Build CLI-only installer (no PySide6 required)
build-installer-cli:
    uv run --extra packaging python scripts/build_installer.py --skip-gui

# Bump the version in all canonical locations
# Usage: just bump-version 0.3.0
bump-version version:
    {{uv_base}} python scripts/bump_version.py {{ version }}

# Show the current version
show-version:
    @{{uv_base}} python scripts/bump_version.py

# Build the source distribution and wheel (platform-native; not suitable for Linux PyPI upload)
build:
    rm -rf dist/
    uv build

# Build a manylinux_2_28 wheel + sdist using the maturin Docker image (Linux publish target).
# Requires Docker. Matches what the CI pipeline produces.
build-manylinux:
    rm -rf dist/
    docker run --rm -v "$(pwd)":/io ghcr.io/pyo3/maturin build --release --compatibility manylinux_2_28 --out dist/
    env -u CONDA_PREFIX uv run maturin sdist --out dist/

# Publish to TestPyPI (reads PYPI_TEST_API_TOKEN from .env)
publish-to-pypi-test: build-manylinux
    uv publish --publish-url https://test.pypi.org/legacy/ --token "$PYPI_TEST_API_TOKEN"

# Publish to PyPI (reads PYPI_API_TOKEN from .env)
publish-to-pypi: build-manylinux
    uv publish --token "$PYPI_API_TOKEN"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

# Clean Python bytecode and test/coverage artifacts
clean:
    {{uv_base}} python scripts/just/system.py clean

# Show project info
show-info:
    @echo "OpenPKPD"
    @{{uv_base}} python -c "import openpkpd; print('Version:', openpkpd.__version__)"
    @{{uv_dev}} python scripts/just/system.py show-info
