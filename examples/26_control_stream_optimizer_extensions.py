"""Example 26 — Inspect FOCEI optimizer controls from a control stream."""

from __future__ import annotations

from pathlib import Path

from openpkpd.parser.control_stream import ControlStream


def main() -> None:
    control_stream_path = Path("examples/control_streams/37_focei_optimizer_controls.ctl")
    control_stream = ControlStream.from_file(control_stream_path)
    estimation = control_stream.estimation_records[0]

    print(f"Problem: {control_stream.problem.title}")
    print(f"Method: {estimation.method}, interaction={estimation.interaction}")
    print(f"Maxeval: {estimation.maxeval}, n_starts={estimation.n_starts}, gtol={estimation.gtol}")
    print(f"Outer optimizer: {estimation.outer_optimizer}")
    print(f"Fallback optimizer: {estimation.outer_fallback_optimizer}")
    print(f"Fallback maxeval: {estimation.outer_fallback_maxeval}")
    print(f"Retain best iterate: {estimation.retain_best_iterate}")
    print(f"Retry on abnormal: {estimation.retry_on_abnormal}")
    print(f"Retry OMEGA scales: {estimation.retry_omega_scales}")


if __name__ == "__main__":
    main()
