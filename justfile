# OpenPKPD justfile
# Install: https://just.systems/

set dotenv-load := true

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

# Install symbolic-kernel dev/test dependencies explicitly
install-symbolic:
    uv sync --extra dev --extra symbolic

# Install local R packages used by external-validation tests
install-r-test-deps:
    Rscript --vanilla scripts/install_r_test_deps.R

# Verify local R packages used by external-validation tests
check-r-test-deps:
    Rscript --vanilla scripts/install_r_test_deps.R --check

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

# Run the full test suite
run-tests:
    {{ uv_dev }} pytest -q

# Run unit tests only (fast, <1 s each)
run-tests-unit:
    {{ uv_dev }} pytest tests/unit/ -q

# Run integration tests
run-tests-integration:
    {{ uv_dev }} pytest tests/integration/ -v

# Run regression tests (requires NONMEM reference files)
run-tests-regression:
    {{ uv_dev }} pytest tests/regression/ -v -m regression

# Run tests with coverage report
run-tests-cov:
    {{ uv_dev }} pytest --cov=openpkpd --cov-report=term-missing --cov-report=html -q
    @echo "HTML report: htmlcov/index.html"

# Run a specific test file or pattern
# Usage: just test-only tests/unit/pk/test_advan4.py

# just test-only -k "test_peak"
run-test-only *args:
    {{ uv_dev }} pytest {{ args }}

# Compile/load the symbolic analytical-kernel cache files into the local cache dir
prewarm-symbolic-caches:
    {{ uv_dev }} python scripts/prewarm_symbolic_caches.py

# ---------------------------------------------------------------------------
# Linting and formatting
# ---------------------------------------------------------------------------

# Run ruff linter
lint:
    {{ uv_dev }} ruff check src/ tests/

# Run ruff formatter (check only)
check-format:
    {{ uv_dev }} ruff format --check src/ tests/

# Auto-fix lint issues and format
format:
    {{ uv_dev }} ruff check --fix src/ tests/
    {{ uv_dev }} ruff format src/ tests/

# Run mypy type checking
typecheck:
    {{ uv_dev }} mypy src/

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

# just profile-analysis --workloads diagnostics_covariate --covariate-subjects 140
profile-analysis *args:
    {{ uv_base }} python scripts/profile_analysis.py {{ args }}

# Benchmark representative estimation workloads and write JSON baseline output
# Usage: just benchmark-estimation
#        just benchmark-estimation --workloads focei imp impmap bayes_laplace

# just benchmark-estimation --n-subjects 8 --bayes-samples 80 --bayes-tune 40
benchmark-estimation *args:
    {{ uv_base }} python scripts/benchmark_estimation.py {{ args }}

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

# Run all examples (per-example optional dependencies are selected automatically)
run-all-examples:
    {{ uv_base }} python scripts/just/examples.py run-all

# Run a single example by number (auto-discovered from examples/)
# Usage: just example 01

# just example 12
run-example num:
    {{ uv_base }} python scripts/just/examples.py run {{ num }}

# Run examples and save figures to a directory

# Usage: just examples-save /tmp/figs
run-examples-and-save outdir:
    {{ uv_base }} python scripts/just/examples.py run-all --output {{ outdir }}

# Capture outputs and figures for all examples (run before building docs)

# Individual example failures are tolerated (output captured in *_output.txt).
capture-examples: install-plots
    {{ uv_base }} python scripts/just/examples.py capture

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

# Build HTML and PDF documentation
build-docs: build-docs-html build-docs-pdf

# Build HTML documentation
build-docs-html: install-docs
    {{ uv_docs }} python scripts/just/docs.py html
    @echo "Open: docs/_build/html/index.html"

# Build and open documentation in the browser
build-docs-and-open: build-docs-html
    {{ uv_base }} python scripts/just/system.py open docs/_build/html/index.html

# Build PDF user manual (requires a LaTeX installation, e.g. MacTeX / TeX Live)
build-docs-pdf: install-docs
    {{ uv_docs }} python scripts/just/docs.py latexpdf
    @echo "PDF: docs/_build/latex/openpkpd.pdf"

# Build and open PDF user manual
build-docs-pdf-and-open: build-docs-pdf
    {{ uv_base }} python scripts/just/system.py open docs/_build/latex/openpkpd.pdf

# Clean documentation build artifacts
clean-docs:
    {{ uv_docs }} python scripts/just/docs.py clean

# Watch and auto-rebuild docs (requires sphinx-autobuild)
watch-docs: install-docs
    {{ uv_docs }} --with sphinx-autobuild python scripts/just/docs.py watch

# ---------------------------------------------------------------------------
# GUI (Qt desktop app)
# ---------------------------------------------------------------------------

# Install GUI dependencies
install-gui:
    uv sync --extra gui

# Launch the desktop GUI
run-gui: install-gui
    {{ uv_gui }} python scripts/just/system.py run-gui

# ---------------------------------------------------------------------------
# Notebooks (marimo)
# ---------------------------------------------------------------------------

# Install notebook dependencies
install-notebooks:
    uv sync --extra notebooks

# Launch a specific marimo notebook in edit mode

# Usage: just notebook notebooks/01_quickstart.py
notebook path:
    {{ uv_notebooks }} marimo edit {{ path }}

# Launch the notebook index (lists all notebooks)
notebooks: install-notebooks
    {{ uv_notebooks }} marimo edit notebooks/

# Run a notebook as a script (non-interactive, for CI/testing)

# Usage: just run-notebook notebooks/01_quickstart.py
run-notebook path:
    {{ uv_notebooks }} marimo run --headless {{ path }}

# Run the notebook integration test suite
test-notebooks: install-notebooks
    {{ uv_notebooks }} pytest tests/integration/test_notebooks.py -v -m "slow and integration"

# ---------------------------------------------------------------------------
# Rust compiled extensions
# ---------------------------------------------------------------------------
# Build and install the openpkpd._core Rust extension in-place, then verify it.
# Requires: Rust toolchain (rustup.rs) and maturin (installed via uv sync --extra dev).

# Run once after cloning or after modifying rust/src/lib.rs.
build-core:
    env -u CONDA_PREFIX uv run maturin develop --release
    {{ uv_dev }} python scripts/check_native_cvodes.py

# Build and install the openpkpd._core Rust extension with the optional native

# CVODES path enabled, then verify all native-cvodes symbols are present.
build-core-native-cvodes:
    OPENPKPD_NATIVE_DEV=1 env -u CONDA_PREFIX uv run maturin develop --release --features native-cvodes
    OPENPKPD_NATIVE_DEV=1 {{ uv_dev }} python scripts/check_native_cvodes.py --require-native-cvodes

# Run the serial native sensitivity performance gate.
# This is kept outside the default pytest suite because wall-clock benchmarks
# are unstable under xdist worker contention.
check-native-sensitivity-perf:
    OPENPKPD_NATIVE_DEV=1 {{ uv_dev }} python scripts/check_native_sensitivity_perf.py

# Build a LOCAL smoke-test wheel for the current platform.
# The resulting wheel is tagged manylinux_<host-glibc>; it is NOT suitable
# for PyPI upload because it will not install on systems with older glibc.
# Use `just build-manylinux` to produce the PyPI release wheel.
# Includes native CVODES support; SUNDIALS is bundled via patchelf + auditwheel.
# Uses a dedicated Cargo target dir (rust/target/wheel) so that patchelf's

# in-place RPATH rewrite does not contaminate the dev build artifacts.
build-wheel:
    env -u CONDA_PREFIX CARGO_TARGET_DIR=rust/target/wheel uv run maturin build --release --features native-cvodes --out dist/
    {{ uv_dev }} python scripts/check_installed_native_cvodes_wheel.py --require-native-cvodes --wheel "$(ls -t dist/openpkpd-*-linux*.whl | head -1)"

# Build a macOS x86_64 wheel suitable for PyPI upload.
# SUNDIALS is compiled from source and linked statically, so no delocate bundling is needed.
# MACOSX_DEPLOYMENT_TARGET=10.13 ensures broad Intel Mac compatibility and avoids tagging
# the wheel with the current OS version (which would restrict installation on older systems).

# Requires: Rust toolchain (rustup.rs); x86_64-apple-darwin target is the default on Intel Macs.
build-wheel-macos:
    env -u CONDA_PREFIX MACOSX_DEPLOYMENT_TARGET=10.13 CARGO_TARGET_DIR=rust/target/macos-x86_64-wheel uv run maturin build --release --features native-cvodes --out dist/
    {{ uv_dev }} python scripts/check_installed_native_cvodes_wheel.py --require-native-cvodes --wheel "$(ls -t dist/openpkpd-*-macosx*x86_64*.whl | head -1)"

# Build a macOS arm64 (Apple Silicon) wheel suitable for PyPI upload, cross-compiled from
# any Mac. SUNDIALS is compiled from source and linked statically; no delocate step needed.
# Note: the resulting .so cannot be loaded on the build machine if it is Intel-only, so
# the symbol-check step is skipped here (run it on Apple Silicon hardware after installing).

# Requires: rustup target add aarch64-apple-darwin
build-wheel-macos-arm64:
    env -u CONDA_PREFIX MACOSX_DEPLOYMENT_TARGET=11.0 CARGO_TARGET_DIR=rust/target/macos-arm64-wheel uv run maturin build --release --target aarch64-apple-darwin --features native-cvodes --out dist/

# Build a macOS universal2 wheel (x86_64 + arm64 fat binary) suitable for PyPI upload.
# Contains both slices in a single wheel; the x86_64 slice can be verified locally on
# Intel hardware.

# Requires: rustup target add aarch64-apple-darwin x86_64-apple-darwin
build-wheel-macos-universal2:
    env -u CONDA_PREFIX MACOSX_DEPLOYMENT_TARGET=11.0 CARGO_TARGET_DIR=rust/target/macos-universal2-wheel uv run maturin build --release --target universal2-apple-darwin --features native-cvodes --out dist/
    {{ uv_dev }} python scripts/check_installed_native_cvodes_wheel.py --require-native-cvodes --wheel "$(ls -t dist/openpkpd-*-macosx*universal2*.whl | head -1)"

# Cross-compile a Windows (win_amd64) wheel from Linux using the MinGW-w64 toolchain.
# Includes native CVODES: sundials-sys/static_libraries links SUNDIALS directly into
# the .pyd so no DLLs need to be bundled and delvewheel is not required.
# Requires: mingw-w64 system package and the x86_64-pc-windows-gnu Rust target.
#   sudo apt install mingw-w64

# rustup target add x86_64-pc-windows-gnu
build-wheel-windows:
    env -u CONDA_PREFIX CARGO_TARGET_DIR=rust/target/windows-wheel CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++ PYO3_CROSS_PYTHON_VERSION=3.12 uv run maturin build --release --target x86_64-pc-windows-gnu --features native-cvodes -i python3.12 --out dist/

# ---------------------------------------------------------------------------
# Git hooks
# ---------------------------------------------------------------------------

# Install the project's git hooks (run once after cloning)
install-hooks:
    {{ uv_base }} python scripts/just/system.py install-hooks

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
    {{ uv_base }} python scripts/bump_version.py {{ version }}

# Show the current version
show-version:
    @{{ uv_base }} python scripts/bump_version.py

# Build the source distribution and wheel (platform-native; not suitable for Linux PyPI upload)
build:
    rm -rf dist/
    uv build

# Build a manylinux_2_28 wheel + sdist using the maturin Docker image.
# Requires Docker. See scripts/build_manylinux_wheel.sh for details on the

# cmake/libclang/BINDGEN workarounds needed inside the container.
build-manylinux:
    rm -rf dist/
    docker run --rm -v "$(pwd)":/io --entrypoint bash ghcr.io/pyo3/maturin \
        /io/scripts/build_manylinux_wheel.sh
    env -u CONDA_PREFIX uv run maturin sdist --out dist/

# Build all PyPI release artefacts: manylinux_2_28 + Windows (cross-compiled) + sdist.
# Requires Docker (for the Linux wheel) and mingw-w64 + x86_64-pc-windows-gnu target.

# Leaves dist/ containing exactly the files to upload.
build-release-wheels:
    rm -rf dist/
    docker run --rm -v "$(pwd)":/io --entrypoint bash ghcr.io/pyo3/maturin \
        /io/scripts/build_manylinux_wheel.sh
    env -u CONDA_PREFIX CARGO_TARGET_DIR=rust/target/windows-wheel CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++ PYO3_CROSS_PYTHON_VERSION=3.12 uv run maturin build --release --target x86_64-pc-windows-gnu --features native-cvodes -i python3.12 --out dist/
    env -u CONDA_PREFIX uv run maturin sdist --out dist/

# Build macOS release artefacts: universal2 wheel (x86_64 + arm64) + sdist.

# Requires: rustup target add aarch64-apple-darwin x86_64-apple-darwin
build-release-wheels-macos:
    rm -rf dist/
    env -u CONDA_PREFIX MACOSX_DEPLOYMENT_TARGET=11.0 CARGO_TARGET_DIR=rust/target/macos-universal2-wheel uv run maturin build --release --target universal2-apple-darwin --features native-cvodes --out dist/
    {{ uv_dev }} python scripts/check_installed_native_cvodes_wheel.py --require-native-cvodes --wheel "$(ls -t dist/openpkpd-*-macosx*universal2*.whl | head -1)"
    env -u CONDA_PREFIX uv run maturin sdist --out dist/

# Publish to TestPyPI (reads PYPI_TEST_API_TOKEN from .env).
# On Linux: builds manylinux_2_28 + Windows wheels + sdist.

# On macOS: builds universal2 wheel + sdist.
publish-to-pypi-test:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{ os() }}" == "macos" ]]; then
        just build-release-wheels-macos
    else
        just build-release-wheels  # linux and windows
    fi
    uv publish --publish-url https://test.pypi.org/legacy/ --token "$PYPI_TEST_API_TOKEN"

# Publish to PyPI (reads PYPI_API_TOKEN from .env).
# On Linux: builds manylinux_2_28 + Windows wheels + sdist.

# On macOS: builds universal2 wheel + sdist.
publish-to-pypi:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{ os() }}" == "macos" ]]; then
        just build-release-wheels-macos
    else
        just build-release-wheels  # linux and windows
    fi
    uv publish --token "$PYPI_API_TOKEN"

# ---------------------------------------------------------------------------
# Smoke testing
# ---------------------------------------------------------------------------

# Run the PyPI smoke test in a clean Docker container.
# Usage: just smoke-test-pypi 0.2.7
#        just smoke-test-pypi 0.2.7 --python-version 3.13
#        just smoke-test-pypi 0.2.7 --require-native-cvodes
smoke-test-pypi version *args:
    bash scripts/docker_smoke_test.sh {{ version }} {{ args }}

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

# Clean Python bytecode and test/coverage artifacts
clean:
    {{ uv_base }} python scripts/just/system.py clean

# Show project info
show-info:
    @echo "OpenPKPD"
    @{{ uv_base }} python -c "import openpkpd; print('Version:', openpkpd.__version__)"
    @{{ uv_dev }} python scripts/just/system.py show-info
