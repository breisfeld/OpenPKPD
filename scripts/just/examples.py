from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from functools import reduce
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
OUTPUTS = ROOT / "docs" / "_static" / "examples"
PLOTS = {1, 2, 3, 4, 5, 7, 9, 11, 12, 14, 16, 17, 20, 21, 22, 23, 24}
CLUSTER = {18}


def extra_for(num: int) -> str | None:
    if num in PLOTS:
        return "plots"
    if num in CLUSTER:
        return "cluster"
    return None


def example_path(num: int) -> Path:
    matches = sorted(EXAMPLES.glob(f"{num:02d}_*.py"))
    if len(matches) != 1:
        raise SystemExit(f"Could not uniquely resolve example {num:02d}")
    return matches[0]


def uv_cmd(num: int) -> list[str]:
    cmd = ["uv", "run"]
    extra = extra_for(num)
    if extra:
        cmd += ["--extra", extra]
    return cmd


def run_one(num: int, output_dir: str | None = None, capture: bool = False) -> int:
    path = example_path(num)
    env = os.environ.copy()
    if output_dir:
        env["OPENPKPD_EXAMPLE_OUTPUT"] = output_dir
    if capture:
        env["MPLBACKEND"] = "Agg"

    cmd = uv_cmd(num) + ["python", str(path)]
    title = path.stem.split("_", 1)[1].replace("_", " ")
    print(f"Running example {num:02d} — {title}")

    if capture:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUTS / f"{num:02d}_output.txt"
        with output_file.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        return proc.returncode

    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def normalize_outputs() -> None:
    replacements = [
        (r'/[^ ]*openpkpd/src/openpkpd/', 'openpkpd/'),
        (r'/[^ ]*openpkpd/examples/', 'examples/'),
        (r'/[^ ]*openpkpd/\.venv/[^ ]*site-packages/', '<site-packages>/'),
        (r'"[^"]*openpkpd/src/openpkpd/', '"openpkpd/'),
        (r'"[^"]*openpkpd/examples/', '"examples/'),
        (r'"[^"]*openpkpd/\.venv/[^"]*site-packages/', '"<site-packages>/'),
    ]
    for path in OUTPUTS.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        text = reduce(lambda acc, pair: re.sub(pair[0], pair[1], acc), replacements, text)
        path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("num", type=int)
    run_parser.add_argument("--output")

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--output")

    sub.add_parser("capture")
    args = parser.parse_args()

    if args.cmd == "run":
        return run_one(args.num, output_dir=args.output)

    if args.cmd == "run-all":
        for num in range(1, 25):
            rc = run_one(num, output_dir=args.output)
            if rc != 0:
                return rc
        return 0

    failures: list[int] = []
    for num in range(1, 25):
        rc = run_one(num, output_dir=str(OUTPUTS), capture=True)
        if rc != 0:
            failures.append(num)
    normalize_outputs()
    if failures:
        print("Capture completed with failures in examples:", ", ".join(f"{n:02d}" for n in failures))
    else:
        print("Capture completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())