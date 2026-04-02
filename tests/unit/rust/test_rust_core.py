"""Run the Rust unit tests (cargo test) as part of the pytest suite.

Each Rust test function is reflected as an individual pytest test via
``@pytest.mark.parametrize``, so failures are reported per-test rather
than as a single monolithic failure.

Skip conditions:
- ``cargo`` not found on PATH
- ``rust/Cargo.toml`` does not exist (e.g. CI without Rust toolchain)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CARGO_TOML = REPO_ROOT / "rust" / "Cargo.toml"

cargo_available = shutil.which("cargo") is not None
cargo_toml_exists = CARGO_TOML.exists()

skip_reason = None
if not cargo_available:
    skip_reason = "cargo not found on PATH"
elif not cargo_toml_exists:
    skip_reason = f"rust/Cargo.toml not found at {CARGO_TOML}"


def _collect_rust_test_names() -> list[str]:
    """Run ``cargo test -- --list`` to discover test names without executing them."""
    if skip_reason:
        return []
    result = subprocess.run(
        ["cargo", "test", "--manifest-path", str(CARGO_TOML), "--", "--list"],
        capture_output=True,
        text=True,
    )
    names = re.findall(r"^(\S+): test$", result.stdout, re.MULTILINE)
    # Strip module prefix (e.g. "tests::test_foo" → keep full qualified name)
    return names if names else ["<unknown>"]


# Collect names at module-import time so parametrize sees them.
_RUST_TEST_NAMES: list[str] = _collect_rust_test_names()


def _run_cargo_test(test_filter: str | None = None) -> subprocess.CompletedProcess[str]:
    cmd = ["cargo", "test", "--manifest-path", str(CARGO_TOML)]
    if test_filter:
        cmd += ["--", test_filter, "--exact"]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.skipif(bool(skip_reason), reason=skip_reason or "")
@pytest.mark.parametrize("rust_test", _RUST_TEST_NAMES)
def test_rust_unit(rust_test: str) -> None:
    """Proxy pytest test that runs one Rust ``#[test]`` function via cargo."""
    proc = _run_cargo_test(rust_test)
    # cargo exits non-zero on any test failure
    if proc.returncode != 0:
        # Extract the relevant failure block from stdout for a clean message
        output = proc.stdout + proc.stderr
        pytest.fail(f"Rust test '{rust_test}' failed:\n{output}")
