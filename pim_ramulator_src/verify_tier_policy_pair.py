"""Check matched Tier/shared decode timelines and print descriptive cycle deltas."""

import argparse
import csv
import re
from pathlib import Path


ADMIT = re.compile(r"^TIER_ADMIT request_id=(\d+) cycle=(\d+) channels=(\d+)$", re.M)
TOKEN = re.compile(
    r"^TIER_TOKEN request_id=(\d+) token=(\d+) "
    r"last_sv_issue_cycle=(\d+) completion_cycle=(\d+)$", re.M
)
REQUEST = re.compile(
    r"^TIER_REQUEST request_id=(\d+) arrival_cycle=(\d+) "
    r"admitted_cycle=(\d+) first_token_cycle=(\d+) completion_cycle=(\d+)$", re.M
)
PHASE = re.compile(
    r"^TIER_PHASE head=(\d+) event=(\w+) cycle=(\d+) "
    r"request_id=(\d+) token_index=(\d+)$", re.M
)
PHASE_NAMES = (
    "KV_APPEND_FENCE", "QK_LAST_ISSUE", "SFM_SV_ELIGIBLE",
    "SV_LAST_ISSUE", "CONTEXT_MERGE_ZERO_COST",
)
TAIL_CYCLES = 4 + 2 + 10  # sv_drain + context_merge + non_pim in both YAML files.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_workload(path):
    specs = {}
    with path.open(newline="") as stream:
        for row in csv.reader(line for line in stream if line.strip() and
                              not line.lstrip().startswith("#")):
            require(len(row) == 5, f"bad workload row: {row}")
            request_id, arrival, prompt, predicted, actual = map(int, row)
            require(request_id not in specs and actual > 0,
                    f"duplicate request or invalid output length: {request_id}")
            specs[request_id] = (arrival, prompt, predicted, actual)
    require(specs, "empty workload")
    return specs


def parse_run(path, mode, specs):
    output = path.read_text()
    require("Tier decode workload exceeded max_cycles" not in output,
            f"{mode}: exceeded max_cycles")
    admits, reports, tokens, phases = {}, {}, {}, {}

    for rid, cycle, channels in ADMIT.findall(output):
        rid = int(rid)
        require(rid not in admits, f"{mode}: repeated admission: {rid}")
        admits[rid] = (int(cycle), int(channels))
    for rid, arrival, admitted, first, complete in REQUEST.findall(output):
        rid = int(rid)
        require(rid not in reports, f"{mode}: repeated completion: {rid}")
        reports[rid] = tuple(map(int, (arrival, admitted, first, complete)))
    for rid, number, issue, complete in TOKEN.findall(output):
        key = (int(rid), int(number))
        require(key not in tokens, f"{mode}: repeated token: {key}")
        tokens[key] = (int(issue), int(complete))
    for head, event, cycle, rid, index in PHASE.findall(output):
        key = (int(rid), int(index))
        require(head == "0" and event in PHASE_NAMES,
                f"{mode}: unexpected head or phase: {key}, {head}, {event}")
        phases.setdefault(key, {})
        require(event not in phases[key], f"{mode}: repeated phase: {key}, {event}")
        phases[key][event] = int(cycle)

    require(set(admits) == set(reports) == set(specs),
            f"{mode}: missing or extra request")
    expected_tokens = {(rid, number) for rid, (_, _, _, actual) in specs.items()
                       for number in range(1, actual + 1)}
    expected_phases = {(rid, prompt + number - 1)
                       for rid, (_, prompt, _, actual) in specs.items()
                       for number in range(1, actual + 1)}
    require(set(tokens) == expected_tokens and set(phases) == expected_phases,
            f"{mode}: missing or extra token phase")
    require(("TIER_LEASE request_id=" in output) == (mode == "shared"),
            f"{mode}: unexpected execution policy")

    for rid, (arrival, prompt, _predicted, actual) in specs.items():
        reported_arrival, admitted, first, complete = reports[rid]
        admit_cycle, channel_count = admits[rid]
        require(reported_arrival == arrival and admit_cycle == admitted and
                arrival <= admitted <= first <= complete,
                f"{mode}: inconsistent request timeline: {rid}")
        require(channel_count == (16 if mode == "shared" else 1),
                f"{mode}: unexpected initial channel count: {rid}, {channel_count}")
        previous_completion = -1
        for number in range(1, actual + 1):
            issue, token_completion = tokens[rid, number]
            events = phases[rid, prompt + number - 1]
            expected_events = set(PHASE_NAMES)
            if mode == "tier":
                # One active channel needs no cross-channel context merge.
                expected_events.remove("CONTEXT_MERGE_ZERO_COST")
            require(set(events) == expected_events,
                    f"{mode}: incomplete phase sequence: {rid}, {number}")
            kv, qk, sfm, sv = (events[name] for name in PHASE_NAMES[:4])
            merge = events.get("CONTEXT_MERGE_ZERO_COST", sv)
            require(kv < qk < sfm < sv == merge == issue and
                    token_completion == issue + TAIL_CYCLES and
                    token_completion > previous_completion,
                    f"{mode}: invalid token timing: {rid}, {number}")
            previous_completion = token_completion
        require(first == tokens[rid, 1][1] and
                complete == tokens[rid, actual][1],
                f"{mode}: request completion does not match token logs: {rid}")
    return reports, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", type=Path)
    parser.add_argument("tier_output", type=Path)
    parser.add_argument("shared_output", type=Path)
    args = parser.parse_args()
    specs = read_workload(args.workload)
    tier, tier_tokens = parse_run(args.tier_output, "tier", specs)
    shared, shared_tokens = parse_run(args.shared_output, "shared", specs)

    print("PASS: identical requests and token counts; both phase timelines valid")
    print("delta = shared - tier (cycles; positive means shared finished later)")
    print("request_id  tier_admit_wait  shared_admit_wait  "
          "tier_first_token  shared_first_token  tier_complete  shared_complete  delta")
    for rid, (arrival, _, _, actual) in sorted(specs.items()):
        ta, sa = tier[rid], shared[rid]
        print(f"{rid:10d}  {ta[1] - arrival:15d}  {sa[1] - arrival:17d}  "
              f"{ta[2]:16d}  {sa[2]:18d}  {ta[3]:13d}  {sa[3]:15d}  "
              f"{sa[3] - ta[3]:+d}")
        for number in range(1, actual + 1):
            tc = tier_tokens[rid, number][1]
            sc = shared_tokens[rid, number][1]
            print(f"  token {number}: tier={tc}, shared={sc}, delta={sc - tc:+d}")
    start = min(arrival for arrival, *_ in specs.values())
    end_tier = max(row[3] for row in tier.values())
    end_shared = max(row[3] for row in shared.values())
    print(f"makespan: tier={end_tier - start}, shared={end_shared - start}")
    print("This probe does not establish goodput, tail latency, or policy superiority.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}") from error
