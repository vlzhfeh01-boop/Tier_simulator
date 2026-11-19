#!/usr/bin/env python3
"""Audit completed Tier/Shared sweep logs by identical rate-and-seed pairs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from compare_tier_policy_logs import metrics, percentiles, read_run, workload


def case_identity(name: str) -> tuple[float, int]:
    match = re.fullmatch(r"rate_(.+)_seed_(-?\d+)", name)
    if match is None:
        raise ValueError(f"unexpected case directory: {name}")
    rate_text = (match[1].replace("plus", "+").replace("p", ".")
                 .replace("m", "-"))
    rate = float(rate_text)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"invalid rate in {name}")
    return rate, int(match[2])


def diagnose_case(case: Path, tck_ps: int, admission_slo: int,
                  tpot_slo: int) -> dict:
    rate, seed = case_identity(case.name)
    requests = workload(case / f"{case.name}.csv")
    tier = metrics(read_run(case / "tier.out", requests), requests, tck_ps,
                   admission_slo, tpot_slo)
    shared = metrics(read_run(case / "shared.out", requests), requests, tck_ps,
                     admission_slo, tpot_slo)
    arrivals = sorted(req["arrival"] for req in requests.values())
    arrival_span = arrivals[-1] - arrivals[0]
    realized_rate = ((len(arrivals) - 1) * 1e12 / (arrival_span * tck_ps)
                     if len(arrivals) > 1 and arrival_span > 0 else None)
    prompts = [r["prompt"] for r in requests.values()]
    actuals = [r["actual"] for r in requests.values()]
    underpredicted = sum(r["predicted"] < r["actual"] for r in requests.values())
    data = {
        "requested_rate_rps": rate, "seed": seed, "request_count": len(requests),
        "realized_rate_rps": realized_rate,
        "prompt_p95": percentiles(prompts)["p95"],
        "actual_output_p95": percentiles(actuals)["p95"],
        "underpredicted_fraction": underpredicted / len(requests),
    }
    for policy, result in (("tier", tier), ("shared", shared)):
        data.update({
            f"{policy}_slo_completed": result["slo_completed"],
            f"{policy}_slo_fraction": result["slo_fraction_of_arrivals"],
            f"{policy}_finite_goodput_rps": result["goodput_requests_per_second"],
            f"{policy}_evicted": result["evicted"],
            f"{policy}_admission_p95_cycles": result["admission_wait_cycles"]["p95"],
            f"{policy}_inter_token_p95_cycles": (
                result["inter_token_cycles"] or {}).get("p95"),
            f"{policy}_growth_wait_events": result["kv_waits"] + result["channel_waits"],
            f"{policy}_growth_wait_cycles": (
                result["kv_wait_cycles"] + result["channel_wait_cycles"]),
            f"{policy}_evicted_wait_cycles": (
                result["kv_wait_censored_cycles"] +
                result["channel_wait_censored_cycles"]),
        })
    data["tier_minus_shared_slo_fraction"] = (
        data["tier_slo_fraction"] - data["shared_slo_fraction"])
    data["tier_minus_shared_finite_goodput_rps"] = (
        data["tier_finite_goodput_rps"] - data["shared_finite_goodput_rps"])
    # This only measures pressure at a KV growth boundary; it does not
    # measure channel-buffer lease collisions during QK or SV.
    data["growth_pressure_observed"] = int(any((
        data["tier_growth_wait_events"], data["shared_growth_wait_events"],
        data["tier_evicted"], data["shared_evicted"])))
    return data


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["requested_rate_rps"]].append(row)
    result = []
    for rate, group in sorted(groups.items()):
        result.append({
            "requested_rate_rps": rate, "seeds": len(group),
            "mean_realized_rate_rps": (
                mean(x["realized_rate_rps"] for x in group)
                if all(x["realized_rate_rps"] is not None for x in group) else None),
            "tier_wins_slo": sum(x["tier_minus_shared_slo_fraction"] > 0 for x in group),
            "ties_slo": sum(x["tier_minus_shared_slo_fraction"] == 0 for x in group),
            "shared_wins_slo": sum(x["tier_minus_shared_slo_fraction"] < 0 for x in group),
            "mean_tier_minus_shared_slo_fraction": mean(
                x["tier_minus_shared_slo_fraction"] for x in group),
            "mean_tier_minus_shared_finite_goodput_rps": mean(
                x["tier_minus_shared_finite_goodput_rps"] for x in group),
            "growth_pressure_pairs": sum(x["growth_pressure_observed"] for x in group),
            "mean_tier_growth_wait_cycles": mean(
                x["tier_growth_wait_cycles"] for x in group),
            "mean_shared_growth_wait_cycles": mean(
                x["shared_growth_wait_cycles"] for x in group),
            "total_tier_evictions": sum(x["tier_evicted"] for x in group),
            "total_shared_evictions": sum(x["shared_evicted"] for x in group),
        })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--tck-ps", type=int, default=769)
    parser.add_argument("--admission-slo-cycles", required=True, type=int)
    parser.add_argument("--tpot-slo-cycles", required=True, type=int)
    args = parser.parse_args()
    if args.tck_ps <= 0 or min(args.admission_slo_cycles,
                                args.tpot_slo_cycles) < 0:
        parser.error("tCK must be positive and SLOs nonnegative")
    root = args.sweep_dir.resolve()
    cases = sorted(path for path in root.iterdir()
                   if path.is_dir() and path.name.startswith("rate_")) if root.is_dir() else []
    if not cases:
        parser.error("no sweep case directories found")
    try:
        rows = [diagnose_case(case, args.tck_ps, args.admission_slo_cycles,
                              args.tpot_slo_cycles) for case in cases]
        identities = {(x["requested_rate_rps"], x["seed"]) for x in rows}
        if len(identities) != len(rows):
            raise ValueError("two case directories identify the same rate and seed")
        summary = summarize(rows)
        write_csv(root / "paired_diagnostics.csv", rows)
        write_csv(root / "rate_diagnostics.csv", summary)
    except (OSError, ValueError, KeyError, ZeroDivisionError) as exc:
        parser.error(str(exc))
    for item in summary:
        print(f"rate={item['requested_rate_rps']:g} seeds={item['seeds']} "
              f"Tier/Shared/tie={item['tier_wins_slo']}/"
              f"{item['shared_wins_slo']}/{item['ties_slo']} "
              f"growth_pressure_pairs={item['growth_pressure_pairs']}")
    print(f"Saved {root / 'paired_diagnostics.csv'} and {root / 'rate_diagnostics.csv'}")


if __name__ == "__main__":
    main()
