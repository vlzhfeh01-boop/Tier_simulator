#!/usr/bin/env python3
"""Replay one generated workload per (arrival rate, seed) on Tier and Shared."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent


def rate_label(rate: float) -> str:
    return format(rate, ".12g").replace(".", "p").replace("+", "plus").replace("-", "m")


def run_checked(command: list[str], cwd: Path, timeout: int,
                log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, timeout=timeout, check=True)
        return
    with log_path.open("w") as stream:
        try:
            subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=timeout, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"simulation failed; inspect {log_path}") from exc


def row_for(rate: float, seed: int, policy: str, data: dict) -> dict:
    return {
        "rate_rps": rate, "seed": seed, "policy": policy,
        "total": data["total"], "completed": data["completed"],
        "evicted": data["evicted"], "slo_completed": data["slo_completed"],
        "throughput_rps": data["completed_requests_per_second"],
        "goodput_rps": data["goodput_requests_per_second"],
        "admission_wait_p95_cycles": data["admission_wait_cycles"]["p95"],
        "inter_token_p95_cycles": (data["inter_token_cycles"] or {}).get("p95"),
        "kv_growth": data["kv_growth"], "kv_waits": data["kv_waits"],
        "channel_growth": data["channel_growth"],
        "channel_waits": data["channel_waits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates-rps", nargs="+", required=True, type=float)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--num-requests", required=True, type=int)
    parser.add_argument("--workload", default="qwen3_policy_smoke",
                        help="configs/workload name or path; controls length distribution")
    parser.add_argument("--template", type=Path,
                        default=SCRIPTS / "tier_qwen3_row_boundary_tier.yaml")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "ramulator2/build_desktop/ramulator2")
    parser.add_argument("--library-dir", type=Path,
                        default=ROOT / "ramulator2")
    parser.add_argument("--tck-ps", type=int, default=769)
    parser.add_argument("--max-cycles", type=int, default=100_000_000)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--admission-slo-cycles", type=int, required=True)
    parser.add_argument("--tpot-slo-cycles", type=int, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (not args.rates_rps or not args.seeds or args.num_requests <= 0 or
        len(set(args.seeds)) != len(args.seeds) or
        len(set(args.rates_rps)) != len(args.rates_rps) or
        any(not math.isfinite(rate) or rate <= 0 for rate in args.rates_rps) or
        min(args.tck_ps, args.max_cycles, args.timeout_seconds) <= 0 or
        min(args.admission_slo_cycles, args.tpot_slo_cycles) < 0):
        parser.error("rates, cycle/timeout limits and request count must be positive; SLOs nonnegative")
    labels = [rate_label(rate) for rate in args.rates_rps]
    if len(set(labels)) != len(labels):
        parser.error("arrival rates collide in output directory names")
    binary = args.binary.resolve()
    if not args.prepare_only and not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    libdir = str(args.library_dir.resolve())
    env["LD_LIBRARY_PATH"] = libdir + (":" + env["LD_LIBRARY_PATH"]
                                       if env.get("LD_LIBRARY_PATH") else "")
    rows = []
    for rate, label in zip(args.rates_rps, labels):
        for seed in args.seeds:
            stem = f"rate_{label}_seed_{seed}"
            case = out_dir / stem
            case.mkdir(parents=True, exist_ok=True)
            run_checked([sys.executable, str(SCRIPTS / "prepare_tier_policy_workload.py"),
                         "--workload", args.workload, "--out-dir", str(case),
                         "--name", stem, "--tck-ps", str(args.tck_ps),
                         "--template", str(args.template.resolve()),
                         "--max-cycles", str(args.max_cycles),
                         "--num-requests", str(args.num_requests),
                         "--seed", str(seed), "--rate-rps", str(rate)],
                        ROOT, args.timeout_seconds)
            if args.prepare_only:
                continue
            for policy in ("tier", "shared"):
                log = case / f"{policy}.out"
                run_checked([str(binary), "-f", f"{stem}_{policy}.yaml"],
                            case, args.timeout_seconds, log, env)
            metrics_path = case / "metrics.json"
            run_checked([sys.executable, str(SCRIPTS / "compare_tier_policy_logs.py"),
                         str(case / f"{stem}.csv"), str(case / "tier.out"),
                         str(case / "shared.out"), "--tck-ps", str(args.tck_ps),
                         "--admission-slo-cycles", str(args.admission_slo_cycles),
                         "--tpot-slo-cycles", str(args.tpot_slo_cycles),
                         "--json", str(metrics_path)], case, args.timeout_seconds)
            metrics = json.loads(metrics_path.read_text())
            for policy in ("Tier", "Shared"):
                rows.append(row_for(rate, seed, policy, metrics[policy]))
    if args.prepare_only:
        print(f"Prepared {len(args.rates_rps) * len(args.seeds)} paired cases in {out_dir}")
        return
    summary = out_dir / "summary.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Paired runs complete: {len(rows) // 2} cases; summary: {summary}")


if __name__ == "__main__":
    main()
