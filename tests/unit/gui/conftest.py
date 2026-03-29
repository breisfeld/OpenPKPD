"""
pytest configuration for the GUI unit-test package.

Collection-time guard
---------------------
All tests in this directory depend on ``openpkpd_gui``, which in turn requires
both ``platformdirs`` and ``PySide6``.

* ``platformdirs`` is declared in the ``[dependency-groups] dev`` section of
  ``pyproject.toml``, so it is always installed by ``uv sync``.

* ``PySide6`` is declared only in the ``[gui]`` optional extra
  (``uv sync --extra gui``).  It is a large Qt binding that should not be
  forced on every developer or CI runner.

This conftest is evaluated during *collection* — before any test module is
imported — so it can skip the entire package cleanly when PySide6 is absent,
instead of raising a ``ModuleNotFoundError`` mid-collection.
"""

from __future__ import annotations

import importlib.util

import pytest


def _pyside6_available() -> bool:
    return importlib.util.find_spec("PySide6") is not None


# Apply a module-level skip to every test in this package when PySide6 is not
# installed.  Using collect_ignore / pytest_collect_file hooks would also work,
# but a conftest-level pytestmark is simpler and shows up clearly in the output.
if not _pyside6_available():
    # Mark every test in this directory as skipped at collection time.
    # This prevents the import of openpkpd_gui sub-modules that pull in Qt
    # bindings (e.g. PySide6.QtCore) from ever executing.
    collect_ignore_glob = ["test_*.py"]

    def pytest_collect_file(parent, file_path):  # noqa: ANN001, ANN201
        """Skip all test files in this directory when PySide6 is missing."""
        if file_path.suffix == ".py" and file_path.name.startswith("test_"):
            # Use path= (pathlib.Path) instead of deprecated fspath= (py.path.local)
            return _SkipItem.from_parent(parent, path=file_path)
        return None

    class _SkipItem(pytest.File):
        def collect(self):  # noqa: ANN201
            yield _SkipTest.from_parent(
                self,
                name=self.path.name,
            )

    class _SkipTest(pytest.Item):
        def runtest(self) -> None:
            pytest.skip("PySide6 not installed — install with: uv sync --extra gui")

        def repr_failure(self, excinfo):  # noqa: ANN001, ANN201
            return str(excinfo.value)

        def reportinfo(self):  # noqa: ANN201
            return self.path, None, f"[skip] {self.path.name}: PySide6 not installed"
