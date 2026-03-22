from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def open_path(path: str) -> int:
    target = (ROOT / path).resolve().as_uri()
    return 0 if webbrowser.open(target) else 1


def install_hooks() -> int:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    src = root / "scripts" / "git-hooks" / "pre-commit"
    dst = root / ".git" / "hooks" / "pre-commit"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
        mode = "symlinked"
    except OSError:
        shutil.copy2(src, dst)
        mode = "copied"
    if os.name != "nt":
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed pre-commit hook ({mode}) -> {dst}")
    return 0


def clean() -> int:
    for pycache in ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    for pyc in ROOT.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for path in [".pytest_cache", "htmlcov", ".coverage", ".mypy_cache"]:
        target = ROOT / path
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    print("Clean complete")
    return 0


def run_gui() -> int:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    return subprocess.run(["openpkpd-gui"], cwd=ROOT, env=env).returncode


def show_info() -> int:
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--co", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in collected.stdout.splitlines() if line.strip()]
    if lines:
        print(lines[-1])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    open_parser = sub.add_parser("open")
    open_parser.add_argument("path")
    sub.add_parser("install-hooks")
    sub.add_parser("clean")
    sub.add_parser("run-gui")
    sub.add_parser("show-info")
    args = parser.parse_args()

    if args.cmd == "open":
        return open_path(args.path)
    if args.cmd == "install-hooks":
        return install_hooks()
    if args.cmd == "clean":
        return clean()
    if args.cmd == "run-gui":
        return run_gui()
    return show_info()


if __name__ == "__main__":
    raise SystemExit(main())