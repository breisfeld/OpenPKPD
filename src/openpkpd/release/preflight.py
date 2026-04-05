"""Checks for release-tree hygiene before building sdists or wheels."""

from __future__ import annotations

from pathlib import Path

_DISALLOWED_PATHS = (
    "rust/target",
    "rust/.venv",
)
_DISALLOWED_GLOBS = (
    "src/openpkpd/_core*.so",
)


def release_tree_issues(root: str | Path) -> list[str]:
    """Return paths that should not be present in a clean release tree."""
    root_path = Path(root).resolve()
    issues: list[str] = []

    for relative in _DISALLOWED_PATHS:
        candidate = root_path / relative
        if candidate.exists():
            issues.append(relative)

    for pattern in _DISALLOWED_GLOBS:
        for match in sorted(root_path.glob(pattern)):
            if match.exists():
                issues.append(str(match.relative_to(root_path)))

    return issues
