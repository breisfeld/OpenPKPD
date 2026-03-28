"""
Bump the OpenPKPD version across all canonical locations.

Usage:
    uv run python scripts/bump_version.py 0.3.0
    uv run python scripts/bump_version.py          # prints current version and exits

Files updated:
    pyproject.toml                                  (version = "...")
    src/openpkpd/__init__.py                        (__version__ = "...")
    docs/conf.py                                    (release = "..." / version = "...")
    docs/changelog.md                               (promotes matching "Planned" heading)
    rust/Cargo.toml                                 (version = "...")
    scripts/packaging/macos/dmgbuild_settings.py    (fallback default version)
"""

from __future__ import annotations

from datetime import date
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Canonical update rules
# Each entry: (relative_path, [(regex_pattern, replacement_template), ...])
# The replacement template may reference captured groups with \1 etc., or use
# the literal {new} placeholder which is substituted before re.sub is called.
# ---------------------------------------------------------------------------

RULES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "pyproject.toml",
        [(r'^(version\s*=\s*")[^"]+(")', r'\g<1>{new}\2')],
    ),
    (
        "src/openpkpd/__init__.py",
        [(r'^(__version__\s*=\s*")[^"]+(")', r'\g<1>{new}\2')],
    ),
    (
        "docs/conf.py",
        [
            (r'^(release\s*=\s*")[^"]+(")', r'\g<1>{new}\2'),
            (r'^(version\s*=\s*")[^"]+(")', r'\g<1>{new}\2'),
        ],
    ),
    (
        "rust/Cargo.toml",
        [(r'^(version\s*=\s*")[^"]+(")', r'\g<1>{new}\2')],
    ),
    (
        "scripts/packaging/macos/dmgbuild_settings.py",
        [(r'(os\.environ\.get\("OPENPKPD_VERSION",\s*")[^"]+(")', r'\g<1>{new}\2')],
    ),
]


def update_changelog_for_release(path: Path, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    today = date.today().isoformat()

    planned_heading = f"## Planned — {new}"
    release_heading = f"## {new} — {today}"

    if planned_heading in text:
        text = text.replace(planned_heading, release_heading, 1)

        # Clean up a duplicated adjacent planned heading if one exists.
        duplicate_block = f"{release_heading}\n\n## Planned — {new}\n"
        if duplicate_block in text:
            text = text.replace(duplicate_block, f"{release_heading}\n", 1)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def current_version() -> str:
    toml = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', toml, re.MULTILINE)
    if not m:
        raise RuntimeError("Cannot find version in pyproject.toml")
    return m.group(1)


def bump(new: str) -> None:
    old = current_version()
    if old == new:
        print(f"Version is already {new}. Nothing to do.")
        return

    changed: list[str] = []
    skipped: list[str] = []

    for rel_path, patterns in RULES:
        path = ROOT / rel_path
        if not path.exists():
            skipped.append(rel_path)
            continue

        text = path.read_text(encoding="utf-8")
        original = text

        for pattern, repl_template in patterns:
            repl = repl_template.replace("{new}", new)
            text = re.sub(pattern, repl, text, flags=re.MULTILINE)

        if text != original:
            try:
                path.write_text(text, encoding="utf-8")
            except OSError as exc:
                skipped.append(f"{rel_path} (write failed: {exc.strerror or exc})")
            else:
                changed.append(rel_path)
        else:
            skipped.append(f"{rel_path} (no match — check manually)")

    changelog_path = ROOT / "docs/changelog.md"
    if not changelog_path.exists():
        skipped.append("docs/changelog.md")
    else:
        try:
            if update_changelog_for_release(changelog_path, new):
                changed.append("docs/changelog.md")
            else:
                skipped.append("docs/changelog.md (no matching planned heading — check manually)")
        except OSError as exc:
            skipped.append(f"docs/changelog.md (write failed: {exc.strerror or exc})")

    print(f"Bumped {old} → {new}")
    print()
    if changed:
        print("Updated:")
        for f in changed:
            print(f"  {f}")
    if skipped:
        print("Skipped (not found or no pattern matched):")
        for f in skipped:
            print(f"  {f}")
    print()
    print("Next steps:")
    print("  1. Review the changes (git diff)")
    print("  2. Review docs/changelog.md release notes")
    print("  3. Commit: git commit -am 'Bump version to {new}'".replace("{new}", new))
    print("  4. Tag:    git tag v{new}".replace("{new}", new))
    print("  5. Build and publish: just publish-to-pypi")


def main() -> None:
    if len(sys.argv) == 1:
        print(f"Current version: {current_version()}")
        print(f"Usage: python scripts/bump_version.py <new-version>")
        return

    new = sys.argv[1].strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        print(f"Error: version must be X.Y.Z, got: {new!r}", file=sys.stderr)
        sys.exit(1)

    bump(new)


if __name__ == "__main__":
    main()
