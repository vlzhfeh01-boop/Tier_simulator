"""Check shared KV admission/growth waits under a two-stripe test limit.

Run from ramulator2 after tier_shared_capacity.yaml. This is a deliberately
small resource pool; it does not change the 1 MiB physical row-stripe size.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def require(ok, message):
    if not ok:
        raise SystemExit(f"FAIL: {message}")


if len(sys.argv) != 2:
    raise SystemExit("usage: python3 verify_tier_shared_capacity.py RUN_OUTPUT")

output = Path(sys.argv[1]).read_text()
require("[error]" not in output and "exceeded max_cycles" not in output,
        "simulator reported an error")
require("TIER_EVICT " not in output,
        "a request was evicted while another token could still complete")
require(re.search(r"TIER_SUMMARY total_requests=3 completed=3 "
                  r"evicted=0 final_cycle=\d+", output),
        "successful requests were not included in final accounting")

regions = [tuple(map(int, result)) for result in re.findall(
    r"TIER_KV_REGION request_id=(\d+) row_offset=(\d+) "
    r"rows_per_slice=(\d+) chunks=(\d+)", output)]
require(regions == [(1, 0, 1, 2), (2, 1, 1, 2), (3, 0, 1, 2)],
        f"initial pool occupancy or reused row: {regions}")

admissions = {int(request_id): int(cycle) for request_id, cycle in re.findall(
    r"TIER_ADMIT request_id=(\d+) cycle=(\d+)", output)}
finishes = {int(request_id): (int(arrival), int(admitted), int(cycle))
            for request_id, arrival, admitted, cycle in re.findall(
                r"TIER_REQUEST request_id=(\d+) arrival_cycle=(\d+) "
                r"admitted_cycle=(\d+) first_token_cycle=\d+ "
                r"completion_cycle=(\d+)", output)}
require(set(admissions) == {1, 2, 3} and set(finishes) == {1, 2, 3},
        f"requests did not all complete: {admissions}, {finishes}")
require(admissions[1] == admissions[2] == 1 and
        finishes[1][2] < admissions[3] and finishes[3][0] == 1 and
        finishes[3][1] == admissions[3] > finishes[3][0],
        "new admission bypassed the existing request's growth")

wait = [tuple(map(int, v)) for v in re.findall(
    r"TIER_KV_WAIT request_id=(\d+) token_index=(\d+) cycle=(\d+)", output)]
resume = [tuple(map(int, v)) for v in re.findall(
    r"TIER_KV_RESUME request_id=(\d+) token_index=(\d+) "
    r"cycle=(\d+) wait_cycles=(\d+)", output)]
grow = [tuple(map(int, v)) for v in re.findall(
    r"TIER_KV_GROW request_id=(\d+) logical_row=(\d+) "
    r"row_offset=(\d+) rows_per_slice=(\d+) cycle=(\d+)", output)]
require(len(wait) == len(resume) == len(grow) == 1 and
        wait[0][:2] == resume[0][:2] == (1, 4096) and
        grow[0][:4] == (1, 1, 1, 2) and
        wait[0][2] < resume[0][2] == grow[0][4] and
        resume[0][3] == resume[0][2] - wait[0][2] and
        finishes[2][2] < grow[0][4] < finishes[1][2] < admissions[3],
        f"growth wait, retry, or atomic row reuse: {wait}, {resume}, {grow}")

events = defaultdict(dict)
for phase, cycle, request_id, token_index in re.findall(
        r"TIER_PHASE head=0 event=(QK_LAST_ISSUE|SFM_SV_ELIGIBLE|SV_LAST_ISSUE) "
        r"cycle=(\d+) request_id=(\d+) token_index=(\d+)", output):
    key = (int(request_id), int(token_index))
    require(phase not in events[key], f"duplicate phase {key}: {phase}")
    events[key][phase] = int(cycle)
require(set(events) == {(1, 4096), (2, 33), (3, 33)},
        f"missing decode phase: {events}")
for key, phases in events.items():
    require(set(phases) == {"QK_LAST_ISSUE", "SFM_SV_ELIGIBLE", "SV_LAST_ISSUE"}
            and phases["QK_LAST_ISSUE"] < phases["SFM_SV_ELIGIBLE"]
            < phases["SV_LAST_ISSUE"], f"QK/SFM/SV ordering: {key}: {phases}")

lease_re = re.compile(
    r"TIER_LEASE request_id=(\d+) phase=(QK_LAST_ISSUE|SV_LAST_ISSUE) "
    r"event=(ACQUIRE|RELEASE) cycle=(\d+)"
    r"(?: channels=(\d+)| last_issue_cycle=(\d+))")
leases = lease_re.findall(output)
require(len(leases) == len(re.findall(r"^TIER_LEASE ", output, re.MULTILINE)),
        "unrecognized channel lease event")
active = None
counts = Counter()
for rid, phase, kind, cycle, channel_count, last_issue in leases:
    rid, cycle = int(rid), int(cycle)
    if kind == "ACQUIRE":
        require(active is None and int(channel_count) == (16 if rid == 1 else 3),
                f"overlapping leases or incorrect participants: {active}, {rid}")
        active = (rid, phase, cycle)
    else:
        require(active is not None and active[:2] == (rid, phase)
                and cycle >= active[2] and
                cycle >= int(last_issue) + (4 if phase == "SV_LAST_ISSUE" else 0),
                f"channel released without ownership or before drain: {rid}, {phase}")
        counts[rid, phase] += 1
        active = None
require(active is None and counts == Counter({
    (rid, phase): 1 for rid in (1, 2, 3)
    for phase in ("QK_LAST_ISSUE", "SV_LAST_ISSUE")}),
    f"missing phase leases: {counts}")


def commands(channel):
    path = Path(f"log/tier_shared_capacity/cmd.log.ch{channel}")
    require(path.is_file(), f"missing controller trace: {path}")
    for line in path.read_text().splitlines():
        columns = [x.strip() for x in line.split(",")]
        if len(columns) >= 9:
            yield int(columns[0]), columns[1], int(columns[7])


ch0, ch2 = list(commands(0)), list(commands(2))
require(sum(opcode == "SFM" for _, opcode, _ in ch0) == 3,
        "one buffer-die SFM per output token was not issued")
for row in (1, 8193):
    require(any(grow[0][4] <= cycle < finishes[1][2] and
                opcode == "WR" and physical == row
                for cycle, opcode, physical in ch0),
            f"grown request did not append into physical row {row}")
    require(any(admissions[2] <= cycle < finishes[2][2] and
                opcode == "WR" and physical == row
                for cycle, opcode, physical in ch2),
            f"request 2 did not write its K/V in row {row}")
for row in (0, 8192):
    require(any(admissions[3] <= cycle < finishes[3][2] and
                opcode == "WR" and physical == row
                for cycle, opcode, physical in ch2),
            f"request 3 did not write its K/V in row {row}")

print("PASS: growth before admission, two row slots, safe reuse and leases")
