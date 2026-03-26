from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def env_for(target: str) -> dict[str, str]:
    env = os.environ.copy()
    if target == "latexpdf":
        env.pop("LD_LIBRARY_PATH", None)
        texbin = Path("/Library/TeX/texbin")
        if texbin.exists():
            env["PATH"] = f"{texbin}{os.pathsep}{env.get('PATH', '')}"
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["html", "latexpdf", "rinoh", "clean", "watch"])
    args = parser.parse_args()

    if args.target == "watch":
        cmd = ["sphinx-autobuild", "docs", "docs/_build/html", "--open-browser"]
    elif args.target == "rinoh":
        # sphinx-design uses grid/card directives that rinohtype cannot render.
        # Build from the user_guide root (clean MyST prose) and drop sphinx_design.
        extensions_without_design = (
            "sphinx.ext.autodoc,"
            "sphinx.ext.autosummary,"
            "sphinx.ext.napoleon,"
            "sphinx.ext.viewcode,"
            "sphinx.ext.intersphinx,"
            "sphinx_autodoc_typehints,"
            "myst_parser,"
            "sphinx_copybutton"
        )
        cmd = [
            "sphinx-build", "-M", "rinoh",
            "docs", "docs/_build",
            "-D", f"extensions={extensions_without_design}",
            "-D", "root_doc=rinoh_index",
        ]
    else:
        cmd = ["sphinx-build", "-M", args.target, "docs", "docs/_build"]

    return subprocess.run(cmd, cwd=ROOT, env=env_for(args.target)).returncode


if __name__ == "__main__":
    raise SystemExit(main())