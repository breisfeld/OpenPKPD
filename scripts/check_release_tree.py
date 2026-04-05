#!/usr/bin/env python3
"""Fail if the source tree contains local build artifacts that should not ship."""

from __future__ import annotations

import sys
from pathlib import Path

from openpkpd.release import release_tree_issues


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    issues = release_tree_issues(repo_root)
    if not issues:
        print("Release tree is clean.")
        return 0

    print("Release tree contains local artifacts that should not ship:", file=sys.stderr)
    for issue in issues:
        print(f" - {issue}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
