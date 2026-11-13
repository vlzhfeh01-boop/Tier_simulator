"""Verify shared row growth/reuse and real controller issue ordering.

Run from ramulator2 after running tier_shared_growth.yaml. The prompt K/V
contents are assumed resident; this checks appended rows and timing only.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")


if len(sys.argv) != 2:
    raise SystemExit("usage: python3 verify_tier_shared_growth.py RUN_OUTPUT")

output = Path(sys.argv[1]).read_text()
require("exceeded max_cycles" not in output and "[error]" not in output,
        "simulator reported an error")

regions = [tuple(map(int, match)) for match in re.findall(
    r"TIER_KV_REGION request_id=(\d+) row_offset=(\d+) "
    r"rows_per_slice=(\d+) chunks=(\d+)", output)]
require(regions == [(1, 0, 1, 2), (2, 1, 1, 2), (3, 1, 1, 2)],
        f"initial row mapping or freed-row reuse: {regions}")
growth = [tuple(map(int, match)) for match in re.findall(
    r"TIER_KV_GROW request_id=(\d+) logical_row=(\d+) "
    r"row_offset=(\d+) rows_per_slice=(\d+) cycle=(\d+)", output)]
require(len(growth) == 1 and growth[0][:4] == (1, 1, 2, 2),
        f"4097th token did not allocate the free second row: {growth}")

admitted = {int(request_id): int(cycle) for request_id, cycle in re.findall(
    r"TIER_ADMIT request_id=(\d+) cycle=(\d+)", output)}
completed = {int(request_id): int(cycle) for request_id, cycle in re.findall(
    r"TIER_REQUEST request_id=(\d+) .*?completion_cycle=(\d+)", output)}
require(set(admitted) == {1, 2, 3} and set(completed) == {1, 2, 3},
        f"requests did not all enter and complete: {admitted}, {completed}")
require(admitted[1] == admitted[2] <= growth[0][4]
        and completed[2] < admitted[3] < completed[1],
        "short request did not free row 1 before request 3 joined active request 1")

tokens = [(int(request_id), int(token), int(issue), int(done))
          for request_id, token, issue, done in re.findall(
              r"TIER_TOKEN request_id=(\d+) token=(\d+) "
              r"last_sv_issue_cycle=(\d+) completion_cycle=(\d+)", output)]
require({request_id: [token for rid, token, _, _ in tokens if rid == request_id]
         for request_id in (1, 2, 3)} == {1: [1, 2, 3, 4], 2: [1], 3: [1]},
        f"incorrect per-request token completions: {tokens}")
require(all(issue + 16 <= done <= completed[request_id]
            for request_id, _, issue, done in tokens),
        "SV issue or token completion is earlier than configured 4+2+10 cycles")

events = defaultdict(dict)
for phase, cycle, request_id, token_index in re.findall(
        r"TIER_PHASE head=0 event=(QK_LAST_ISSUE|SFM_SV_ELIGIBLE|SV_LAST_ISSUE) "
        r"cycle=(\d+) request_id=(\d+) token_index=(\d+)", output):
    key = (int(request_id), int(token_index))
    require(phase not in events[key], f"duplicate phase {key}: {phase}")
    events[key][phase] = int(cycle)
require(set(events) == {(1, 4096), (1, 4097), (1, 4098), (1, 4099),
                        (2, 33), (3, 33)}, f"missing phase sequence: {events}")
for key, phases in events.items():
    require(set(phases) == {"QK_LAST_ISSUE", "SFM_SV_ELIGIBLE", "SV_LAST_ISSUE"}
            and phases["QK_LAST_ISSUE"] < phases["SFM_SV_ELIGIBLE"]
            < phases["SV_LAST_ISSUE"], f"QK/SFM/SV order: {key}: {phases}")

lease_re = re.compile(
    r"TIER_LEASE request_id=(\d+) phase=(QK_LAST_ISSUE|SV_LAST_ISSUE) "
    r"event=(ACQUIRE|RELEASE) cycle=(\d+)"
    r"(?: channels=(\d+)| last_issue_cycle=(\d+))")
lease_lines = re.findall(r"^TIER_LEASE .*?$", output, re.MULTILINE)
lease_matches = lease_re.findall(output)
require(len(lease_lines) == len(lease_matches), "unrecognized channel lease record")
active = None
intervals = defaultdict(list)
for request_id, phase, event, cycle, channels, last_issue in lease_matches:
    request_id, cycle = int(request_id), int(cycle)
    key = (request_id, phase)
    if event == "ACQUIRE":
        require(active is None and int(channels) == (16 if request_id == 1 else 3),
                f"overlapping leases or wrong participating channels: {active}, {key}")
        active = (key, cycle)
    else:
        require(active is not None and active[0] == key and cycle >= active[1],
                f"release without matching lease: {key}")
        require(cycle >= int(last_issue) + (4 if phase == "SV_LAST_ISSUE" else 0),
                f"PIM buffer released before final issue/drain: {key}")
        intervals[key].append((active[1], cycle))
        active = None
require(active is None and
        {key: len(value) for key, value in intervals.items()} == {
            (1, "QK_LAST_ISSUE"): 4, (1, "SV_LAST_ISSUE"): 4,
            (2, "QK_LAST_ISSUE"): 1, (2, "SV_LAST_ISSUE"): 1,
            (3, "QK_LAST_ISSUE"): 1, (3, "SV_LAST_ISSUE"): 1},
        f"missing or unfinished shared leases: {dict(intervals)}")
require(intervals[1, "QK_LAST_ISSUE"][0][1]
        <= intervals[2, "QK_LAST_ISSUE"][0][0]
        < intervals[1, "SV_LAST_ISSUE"][0][0],
        "request 2 failed to use channels during request 1's SFM gap")


def commands(channel):
    path = Path(f"log/tier_shared_growth/cmd.log.ch{channel}")
    require(path.is_file(), f"missing real controller trace: {path}")
    for line in path.read_text().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 9:
            yield int(fields[0]), fields[1], int(fields[7])


ch0, ch2 = list(commands(0)), list(commands(2))
require(Counter(opcode for _, opcode, _ in ch0)["SFM"] == 6,
        "one buffer-die SFM per output token was not issued")
require({row for _, opcode, row in ch0 if opcode == "WR"} == {2, 8194}
        and {row for _, opcode, row in ch2 if opcode == "WR"} == {1, 8193},
        "appended K/V writes escaped the requests' mapped rows")
for row in (2, 8194):
    require(any(admitted[1] <= cycle < completed[1]
                for cycle, opcode, physical in ch0 if opcode == "WR" and physical == row),
            f"grown request's K/V append did not reach physical row {row}")
for row in (0, 2, 8192, 8194):
    require(any(opcode == "MACAB" and physical == row for _, opcode, physical in ch0),
            f"QK/SV did not read both old and grown rows: {row}")
for request_id in (2, 3):
    for row in (1, 8193):
        require(any(admitted[request_id] <= cycle < completed[request_id]
                    for cycle, opcode, physical in ch2
                    if opcode == "WR" and physical == row),
                f"request {request_id} did not append K/V in reused row {row}")

print("PASS: 4097-token row growth, safe channel leases, freed-row reuse and six SFM issues")
