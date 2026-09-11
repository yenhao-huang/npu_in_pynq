"""Reproducible controller scaling simulation; latency is modeled, not board time."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", type=Path, default=repo)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--clock-mhz", type=float, default=100.0)
    parser.add_argument("--baseline", action="store_true", help="expect serial baseline latency")
    args = parser.parse_args()
    if args.clock_mhz <= 0 or any(s < 3 for s in args.sizes):
        parser.error("clock must be positive and sizes >= 3 for partial-tile tests")
    args.output.mkdir(parents=True, exist_ok=True)
    rtl = [args.rtl_root.resolve() / "src/hw/rtl" / p for p in (
        "systolic_array/npu_pe.sv", "systolic_array/npu_systolic_array.sv",
        "npu_matrix/npu_matrix_controller.sv",
    )]
    tb = repo / "src/hw/tb/npu_matrix/tb_npu_matrix_scaling.sv"
    evidence = {
        "scope": "RTL simulation; assumed clock, excludes host/DMA setup and DDR stalls",
        "clock_mhz": args.clock_mhz,
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*rtl, tb]},
        "commands": [], "measurements": [],
    }
    for size in args.sizes:
        executable = args.output.resolve() / f"scaling{size}.vvp"
        commands = [
            ["iverilog", "-g2012", "-s", "tb_npu_matrix_scaling",
             f"-Ptb_npu_matrix_scaling.SIZE={size}", "-o", str(executable),
             f"-Ptb_npu_matrix_scaling.EXPECT_OVERLAP={int(not args.baseline)}",
             *map(str, rtl), str(tb)],
            ["vvp", str(executable)],
        ]
        for stage, command in zip(("compile", "simulation"), commands):
            evidence["commands"].append(command)
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            output = result.stdout + result.stderr
            (args.output / f"{stage}{size}.log").write_text(output)
            if result.returncode or re.search(r"\b(fail|fatal|error|mismatch)\b", output, re.I):
                raise RuntimeError(f"{stage} failed; see {args.output}/{stage}{size}.log")
        if f"PASS tb_npu_matrix_scaling SIZE={size}" not in output:
            raise RuntimeError("simulation did not report PASS")
        for line in output.splitlines():
            if not line.startswith("METRIC "):
                continue
            row = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
            row["latency_us_at_assumed_clock"] = row["cycles"] / args.clock_mhz
            row["gmac_per_second_at_assumed_clock"] = row["macs"] * args.clock_mhz / (1000 * row["cycles"])
            row["pe_utilization_percent"] = 100 * row["macs"] / (size*size*row["cycles"])
            evidence["measurements"].append(row)
            print(line)
    (args.output / "metrics.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
