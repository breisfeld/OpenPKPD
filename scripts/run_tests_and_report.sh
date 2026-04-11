#!/usr/bin/env bash
# Run the full openpkpd test suite and write a detailed datestamped report
# to the test_status/ directory.
#
# Intended to be run from cron. Writes one report file per run:
#   test_status/YYYY-MM-DD_HH-MM-SS.txt
#
# Usage:
#   bash scripts/run_tests_and_report.sh [--unit-only] [--open]
#
# Options:
#   --unit-only   Run only tests/unit/ (faster, skips integration/regression)
#   --open        Open the report in $PAGER after writing (useful for manual runs)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${REPO_DIR}/test_status"
TIMESTAMP="$(date '+%Y%m%dT%H%M%S')"
REPORT_FILE="${REPORT_DIR}/${TIMESTAMP}.txt"
UNIT_ONLY=false
OPEN_AFTER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --unit-only) UNIT_ONLY=true; shift ;;
        --open)      OPEN_AFTER=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$REPORT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_hr() { printf '%0.s─' {1..72}; echo; }

# ---------------------------------------------------------------------------
# Build the report (tee to file and stdout)
# ---------------------------------------------------------------------------

{
    echo "OpenPKPD Test Report"
    _hr
    echo "Generated  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Host       : $(hostname)"
    echo "Repo       : ${REPO_DIR}"
    echo "Git branch : $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    echo "Git commit : $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo "Python     : $(cd "$REPO_DIR" && uv run python --version 2>&1)"
    echo "uv         : $(uv --version 2>&1)"
    echo ""

    if [[ "$UNIT_ONLY" == true ]]; then
        echo "Scope      : unit tests only (tests/unit/)"
        TEST_PATHS="tests/unit/"
    else
        echo "Scope      : full suite (unit + integration)"
        TEST_PATHS="tests/unit/ tests/integration/"
    fi

    _hr

    # Run pytest, capture output, always succeed so we can write the report
    PYTEST_ARGS=(
        --tb=short
        -q
        --no-header
        -rN
        --color=no
    )

    echo ""
    echo "PYTEST COMMAND"
    _hr
    echo "uv run --extra dev pytest ${PYTEST_ARGS[*]} ${TEST_PATHS}"
    echo ""

    START_EPOCH=$(date +%s)

    echo "TEST OUTPUT"
    _hr

    set +e
    cd "$REPO_DIR"
    uv run --extra dev pytest "${PYTEST_ARGS[@]}" $TEST_PATHS 2>&1
    PYTEST_EXIT=$?
    set -e

    END_EPOCH=$(date +%s)
    ELAPSED=$(( END_EPOCH - START_EPOCH ))
    ELAPSED_FMT="$(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"

    echo ""
    _hr
    echo "SUMMARY"
    _hr
    echo "Exit code  : ${PYTEST_EXIT}  (0 = all passed)"
    echo "Duration   : ${ELAPSED_FMT}"
    echo "Status     : $([ "$PYTEST_EXIT" -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    echo ""
    echo "Report     : ${REPORT_FILE}"

} 2>&1 | tee "$REPORT_FILE"

# Symlink latest report for easy access
ln -sf "${TIMESTAMP}.txt" "${REPORT_DIR}/latest.txt"

if [[ "$OPEN_AFTER" == true ]]; then
    "${PAGER:-less}" "$REPORT_FILE"
fi

exit "${PYTEST_EXIT:-0}"
