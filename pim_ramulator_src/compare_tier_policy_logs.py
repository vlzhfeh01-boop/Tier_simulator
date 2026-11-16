#!/usr/bin/env python3
"""Check identical workloads and summarize Tier/Shared Ramulator outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def fields(line: str) -> dict[str, int | str]:
    result = {}
    for key, value in re.findall(r"\b([a-z_]+)=([^\s]+)", line):
        result[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return result


def workload(path: Path) -> dict[int, dict]:
    result = {}
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) != 5:
                raise ValueError("workload CSV needs exactly five fields per request")
            rid, arrival, prompt, predicted, actual = map(int, row)
            if rid in result or min(rid, arrival, prompt, predicted) < 0 or actual <= 0:
                raise ValueError(f"invalid or duplicated workload request {rid}")
            result[rid] = {"arrival": arrival, "prompt": prompt,
                           "predicted": predicted, "actual": actual}
    if not result:
        raise ValueError("empty workload")
    return result


def read_run(path: Path, requests: dict[int, dict]) -> dict:
    admitted: dict[int, int] = {}
    tokens: dict[int, list[int]] = {}
    completed: dict[int, dict] = {}
    evicted: dict[int, dict] = {}
    counts = {"kv_growth": 0, "kv_waits": 0, "channel_waits": 0}
    summary = None
    with path.open(errors="replace") as stream:
        for line in stream:
            if not line.startswith("TIER_"):
                continue
            event = line.split(" ", 1)[0]
            data = fields(line)
            if event == "TIER_SUMMARY":
                if summary is not None:
                    raise ValueError(f"{path}: duplicate summary")
                summary = data
                continue
            if event == "TIER_KV_GROW":
                counts["kv_growth"] += 1
            elif event == "TIER_KV_WAIT":
                counts["kv_waits"] += 1
            elif event == "TIER_CHANNEL_WAIT":
                counts["channel_waits"] += 1
            if event not in ("TIER_ADMIT", "TIER_TOKEN", "TIER_REQUEST", "TIER_EVICT"):
                continue
            rid = data.get("request_id")
            if rid not in requests:
                raise ValueError(f"{path}: unexpected request_id {rid}")
            if event == "TIER_ADMIT":
                if rid in admitted:
                    raise ValueError(f"{path}: duplicated admission for {rid}")
                admitted[rid] = data["cycle"]
            elif event == "TIER_TOKEN":
                seen = tokens.setdefault(rid, [])
                if data["token"] != len(seen) + 1:
                    raise ValueError(f"{path}: token sequence broken for {rid}")
                if seen and data["completion_cycle"] <= seen[-1]:
                    raise ValueError(f"{path}: token cycles not increasing for {rid}")
                seen.append(data["completion_cycle"])
            elif event == "TIER_REQUEST":
                if rid in completed or rid in evicted:
                    raise ValueError(f"{path}: repeated terminal event for {rid}")
                completed[rid] = data
            elif event == "TIER_EVICT":
                if rid in completed or rid in evicted:
                    raise ValueError(f"{path}: repeated terminal event for {rid}")
                evicted[rid] = data
    if not summary:
        raise ValueError(f"{path}: missing TIER_SUMMARY (aborted or incomplete run)")
    if (summary["total_requests"] != len(requests) or
            summary["completed"] != len(completed) or
            summary["evicted"] != len(evicted) or
            len(completed) + len(evicted) != len(requests)):
        raise ValueError(f"{path}: summary does not match terminal requests")
    for rid, spec in requests.items():
        if rid not in admitted or admitted[rid] < spec["arrival"]:
            raise ValueError(f"{path}: missing/early admission for {rid}")
        seen = tokens.get(rid, [])
        if rid in completed:
            end = completed[rid]
            if (end["arrival_cycle"] != spec["arrival"] or
                    end["admitted_cycle"] != admitted[rid] or
                    len(seen) != spec["actual"] or
                    end["first_token_cycle"] != seen[0] or
                    end["completion_cycle"] != seen[-1]):
                raise ValueError(f"{path}: inconsistent completion for {rid}")
        elif evicted[rid]["generated_tokens"] != len(seen):
            raise ValueError(f"{path}: inconsistent eviction for {rid}")
        if seen and seen[0] <= admitted[rid]:
            raise ValueError(f"{path}: token completed before admission for {rid}")
    if summary["final_cycle"] < max(
            [d["completion_cycle"] for d in completed.values()] +
            [d["cycle"] for d in evicted.values()]):
        raise ValueError(f"{path}: final cycle precedes terminal event")
    return {"admitted": admitted, "tokens": tokens, "completed": completed,
            "evicted": evicted, "summary": summary, "counts": counts}


def percentiles(samples: list[int]) -> dict[str, float | int] | None:
    if not samples:
        return None
    ordered = sorted(samples)
    return {"mean": sum(ordered) / len(ordered),
            "p50": ordered[math.ceil(.50 * len(ordered)) - 1],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
            "p99": ordered[math.ceil(.99 * len(ordered)) - 1],
            "max": ordered[-1]}


def metrics(run: dict, requests: dict[int, dict], tck_ps: int,
            admission_slo: int | None, tpot_slo: int | None) -> dict:
    admitted = run["admitted"]
    completed = run["completed"]
    evicted = run["evicted"]
    tokens = run["tokens"]
    wait = [admitted[rid] - spec["arrival"] for rid, spec in requests.items()]
    first_token = [tokens[rid][0] - admitted[rid] for rid in completed]
    intervals = [b - a for rid in completed
                 for a, b in zip(tokens[rid], tokens[rid][1:])]
    response = [completed[rid]["completion_cycle"] - requests[rid]["arrival"]
                for rid in completed]
    duration = run["summary"]["final_cycle"] - min(
        r["arrival"] for r in requests.values()) + 1
    result = {"total": len(requests), "completed": len(completed),
              "evicted": len(evicted), "evicted_ids": sorted(evicted),
              "final_cycle": run["summary"]["final_cycle"],
              "observed_cycles": duration,
              "completed_requests_per_second": len(completed) * 1e12 / (duration * tck_ps),
              "admission_wait_cycles": percentiles(wait),
              "first_token_after_admission_cycles": percentiles(first_token),
              "inter_token_cycles": percentiles(intervals),
              "response_cycles": percentiles(response),
              **run["counts"]}
    if admission_slo is not None and tpot_slo is not None:
        good = [rid for rid in completed
                if admitted[rid] - requests[rid]["arrival"] <= admission_slo
                and tokens[rid][0] - admitted[rid] <= tpot_slo
                and all(b - a <= tpot_slo
                        for a, b in zip(tokens[rid], tokens[rid][1:]))]
        result["slo_completed"] = len(good)
        result["goodput_requests_per_second"] = (
            len(good) * 1e12 / (duration * tck_ps))
        result["slo_fraction_of_arrivals"] = len(good) / len(requests)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload_csv", type=Path)
    parser.add_argument("tier_log", type=Path)
    parser.add_argument("shared_log", type=Path)
    parser.add_argument("--tck-ps", type=int, required=True)
    parser.add_argument("--admission-slo-cycles", type=int)
    parser.add_argument("--tpot-slo-cycles", type=int)
    parser.add_argument("--json", type=Path, help="write machine readable metrics")
    args = parser.parse_args()
    if args.tck_ps <= 0 or any(v is not None and v < 0 for v in
                                (args.admission_slo_cycles, args.tpot_slo_cycles)):
        parser.error("clock period and SLO cycle limits must be nonnegative")
    if (args.admission_slo_cycles is None) != (args.tpot_slo_cycles is None):
        parser.error("provide both SLO limits or neither")
    try:
        requests = workload(args.workload_csv)
        results = {
            policy: metrics(read_run(path, requests), requests, args.tck_ps,
                            args.admission_slo_cycles, args.tpot_slo_cycles)
            for policy, path in (("Tier", args.tier_log), ("Shared", args.shared_log))
        }
        if args.json:
            args.json.write_text(json.dumps(results, indent=2) + "\n")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    for policy, data in results.items():
        print(f"{policy}: completed={data['completed']}/{data['total']}, "
              f"evicted={data['evicted']}, final_cycle={data['final_cycle']}, "
              f"throughput={data['completed_requests_per_second']:.3f} req/s")
        print(f"  admission_wait={data['admission_wait_cycles']}; "
              f"inter_token={data['inter_token_cycles']}")
        print(f"  kv_growth={data['kv_growth']}, kv_waits={data['kv_waits']}, "
              f"channel_waits={data['channel_waits']}")
        if "goodput_requests_per_second" in data:
            print(f"  SLO goodput={data['goodput_requests_per_second']:.3f} req/s; "
                  f"passing={data['slo_completed']}/{data['total']}")


if __name__ == "__main__":
    main()
