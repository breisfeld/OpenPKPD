"""Helpers for running R scripts in a reproducible subprocess environment."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


_STRIP_ENV_KEYS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_SHLVL",
    "MAMBA_EXE",
    "MAMBA_ROOT_PREFIX",
    "PYTHONHOME",
)


@lru_cache(maxsize=1)
def find_rscript() -> str | None:
    """Return the absolute path to ``Rscript`` if available."""
    return shutil.which("Rscript")


@lru_cache(maxsize=1)
def detect_r_home() -> str | None:
    """Return the R home directory reported by ``Rscript``."""
    rscript = find_rscript()
    if rscript is None:
        return None

    result = subprocess.run(
        [rscript, "-e", "cat(R.home())"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    value = result.stdout.strip()
    return value or None


def build_rscript_env(
    *,
    base_env: Mapping[str, str] | None = None,
    extra_r_libs: Sequence[str | Path] = (),
) -> dict[str, str]:
    """Build an environment for stable ``Rscript`` subprocess execution."""
    env = dict(os.environ if base_env is None else base_env)

    for key in _STRIP_ENV_KEYS:
        env.pop(key, None)

    r_home = detect_r_home()
    if r_home:
        env["R_HOME"] = r_home

    libs: list[str] = []
    for lib in extra_r_libs:
        value = str(Path(lib).resolve())
        if value not in libs:
            libs.append(value)

    existing = env.get("R_LIBS_USER", "").strip()
    if existing:
        for value in existing.split(os.pathsep):
            if value and value not in libs:
                libs.append(value)

    if libs:
        env["R_LIBS_USER"] = os.pathsep.join(libs)

    return env


def run_rscript(
    args: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    extra_r_libs: Sequence[str | Path] = (),
    capture_output: bool = True,
    text: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``Rscript`` with a normalized environment."""
    rscript = find_rscript()
    if rscript is None:
        raise FileNotFoundError("Rscript was not found on PATH")

    return subprocess.run(
        [rscript, *args],
        cwd=cwd,
        env=build_rscript_env(base_env=env, extra_r_libs=extra_r_libs),
        capture_output=capture_output,
        text=text,
        check=check,
    )

