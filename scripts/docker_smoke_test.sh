#!/usr/bin/env bash
# Run the OpenPKPD PyPI smoke test + brief unit tests inside a clean Docker container.
#
# A Rust toolchain is installed inside the container so that:
#   - pip can build from the source distribution if no binary wheel is available
#   - test_rust_core.py (cargo test) can run
#
# Usage:
#   bash scripts/docker_smoke_test.sh <version> [options]
#
# Options:
#   --python-version <ver>    Python image tag to use (default: 3.12)
#   --require-native-cvodes   Fail if native CVODES symbols are absent
#   --smoke-only              Skip the unit tests (smoke test only)
#
# Examples:
#   bash scripts/docker_smoke_test.sh 0.2.7
#   bash scripts/docker_smoke_test.sh 0.2.7 --python-version 3.13
#   bash scripts/docker_smoke_test.sh 0.2.7 --require-native-cvodes
#   bash scripts/docker_smoke_test.sh 0.2.7 --smoke-only

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

OPENPKPD_VERSION=""
PYTHON_VERSION="3.12"
SMOKE_EXTRA_ARGS=()
RUN_UNIT_TESTS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-version)
            PYTHON_VERSION="$2"; shift 2 ;;
        --require-native-cvodes)
            SMOKE_EXTRA_ARGS+=("--require-native-cvodes"); shift ;;
        --smoke-only)
            RUN_UNIT_TESTS=false; shift ;;
        -*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$OPENPKPD_VERSION" ]]; then
                OPENPKPD_VERSION="$1"; shift
            else
                echo "Unexpected positional argument: $1" >&2; exit 1
            fi
            ;;
    esac
done

if [[ -z "$OPENPKPD_VERSION" ]]; then
    echo "Usage: $0 <version> [--python-version 3.12] [--require-native-cvodes] [--smoke-only]" >&2
    exit 1
fi

DOCKER_IMAGE="python:${PYTHON_VERSION}-slim"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_TEST="${REPO_DIR}/scripts/smoke_test_pypi.py"
TESTS_DIR="${REPO_DIR}/tests"
RUST_DIR="${REPO_DIR}/rust"

for path in "$SMOKE_TEST" "$TESTS_DIR" "$RUST_DIR"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path not found: $path" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Unit test paths (relative to /tests inside the container).
# test_rust_core.py runs `cargo test` — cargo is installed inside the container.
# NOTE: test_rust_core.py resolves REPO_ROOT as parents[3] of the test file.
#       With tests at /tests, REPO_ROOT = / and it looks for /rust/Cargo.toml,
#       which is satisfied by the -v rust:/rust mount below.
# ---------------------------------------------------------------------------

UNIT_TEST_PATHS=(
    # Rust extension: cargo unit tests, and Rust/Python numerical parity
    "unit/rust/test_rust_core.py"
    "unit/rust/test_rust_python_parity.py"
    # Native CVODES: BLQ log-likelihood loop, ODE integrator accuracy & speed
    "unit/test_native_cvodes.py"
    # Analytical PK subroutines (ADVAN1-5, ADVAN11-12, transforms)
    "unit/pk/test_advan.py"
    "unit/pk/test_advan4.py"
    "unit/pk/test_transforms.py"
    # Data layer: dataset parsing, event processor, BLQ handling
    "unit/data/test_dataset.py"
    "unit/data/test_event_processor.py"
    # Estimation: FO objective function and parameter handling
    "unit/estimation/test_fo.py"
    "unit/estimation/test_parameters.py"
    # NCA: core metrics
    "unit/nca/test_nca.py"
    # Parser: control stream round-trip
    "unit/parser/test_control_stream.py"
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "==> Docker image  : ${DOCKER_IMAGE}"
echo "==> Package       : openpkpd==${OPENPKPD_VERSION}"
echo "==> Unit tests    : $([ "$RUN_UNIT_TESTS" = true ] && echo "yes" || echo "no (--smoke-only)")"
echo "==> Extra args    : ${SMOKE_EXTRA_ARGS[*]:-none}"
echo ""

UNIT_PATHS_STR=""
for p in "${UNIT_TEST_PATHS[@]}"; do
    UNIT_PATHS_STR+=" /tests/$p"
done

INNER_SCRIPT='set -euo pipefail

echo "==> Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq --no-install-recommends curl build-essential

echo "==> Installing Rust toolchain (stable)..."
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --no-modify-path
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version
cargo --version

echo "==> Installing openpkpd=='"${OPENPKPD_VERSION}"' and pytest from PyPI..."
pip install -q --prefer-binary openpkpd=='"${OPENPKPD_VERSION}"' pytest

echo "==> Running smoke test..."
python /smoke_test.py --expected-version '"${OPENPKPD_VERSION}"' '"${SMOKE_EXTRA_ARGS[*]:-}"
if [[ "$RUN_UNIT_TESTS" == true ]]; then
    INNER_SCRIPT+='

echo ""
echo "==> Running brief unit test suite..."
pytest '"${UNIT_PATHS_STR}"' -q --tb=short'
fi

docker run --rm \
    --pull always \
    -v "${SMOKE_TEST}:/smoke_test.py:ro" \
    -v "${TESTS_DIR}:/tests:ro" \
    -v "${RUST_DIR}:/rust:ro" \
    "${DOCKER_IMAGE}" \
    bash -c "$INNER_SCRIPT"
