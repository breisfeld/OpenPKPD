from __future__ import annotations

from pathlib import Path

import pytest

from openpkpd.release.preflight import strict_release_validation_enabled


def require_release_fixture(path: str | Path, *, kind: str) -> Path:
    """Return an existing fixture path or fail/skip based on release mode."""
    fixture_path = Path(path)
    if fixture_path.exists():
        return fixture_path

    message = f"{kind} not found: {fixture_path}"
    if strict_release_validation_enabled():
        pytest.fail(message)
    pytest.skip(message)

    return fixture_path
