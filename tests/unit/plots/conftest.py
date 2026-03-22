"""Configure matplotlib for headless testing before any test imports."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        pass
