"""Verify minimum-occupancy, token-boundary abort on real shared KV exhaustion."""

import re
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")


if len(sys.argv) != 2:
    raise SystemExit("usage: python3 verify_tier_shared_deadlock.py RUN_OUTPUT")

output = Path(sys.argv[1]).read_text()
require("exceeded max_cycles" not in output and "terminate called" not in output,
        "simulator failed")

regions = [tuple(map(int, row)) for row in re.findall(
    r"TIER_KV_REGION request_id=(\d+) row_offset=(\d+) "
    r"rows_per_slice=(\d+) chunks=(\d+)", output)]
require(regions == [(1, 0, 1, 2), (2, 1, 2, 4)],
        f"unexpected initial KV allocations: {regions}")

wait = [(int(rid), int(token), int(cycle)) for rid, token, cycle in re.findall(
    r"TIER_KV_WAIT request_id=(\d+) token_index=(\d+) cycle=(\d+)", output)]
require(len(wait) == 2 and {rid for rid, _, _ in wait} == {1, 2},
        f"both requests must block on growth: {wait}")

evictions = [tuple(map(int, event)) for event in re.findall(
    r"TIER_EVICT request_id=(\d+) cycle=(\d+) released_units=(\d+) "
    r"generated_tokens=(\d+) reason=DEADLOCK", output)]
require(len(evictions) == 1 and evictions[0][0] == 1 and
        evictions[0][2:] == (1, 0) and
        evictions[0][1] > max(cycle for _, _, cycle in wait),
        f"wrong victim or unsafe eviction: {evictions}")

resume = [tuple(map(int, row)) for row in re.findall(
    r"TIER_KV_RESUME request_id=(\d+) token_index=(\d+) "
    r"cycle=(\d+) wait_cycles=(\d+)", output)]
grow = [tuple(map(int, row)) for row in re.findall(
    r"TIER_KV_GROW request_id=(\d+) logical_row=(\d+) "
    r"row_offset=(\d+) rows_per_slice=(\d+) cycle=(\d+)", output)]
require(len(resume) == len(grow) == 1 and
        resume[0] == (2, 8192, evictions[0][1],
                      evictions[0][1] - next(cycle for rid, _, cycle in wait
                                               if rid == 2)) and
        grow[0] == (2, 2, 0, 3, evictions[0][1]),
        f"freed row did not go to survivor's growth: {resume}, {grow}")

phases = {name: int(cycle) for name, cycle in re.findall(
    r"TIER_PHASE head=0 event=(QK_LAST_ISSUE|SFM_SV_ELIGIBLE|SV_LAST_ISSUE) "
    r"cycle=(\d+) request_id=2 token_index=8192", output)}
require(len(phases) == 3 and phases["QK_LAST_ISSUE"] <
        phases["SFM_SV_ELIGIBLE"] < phases["SV_LAST_ISSUE"],
        f"survivor did not run complete attention: {phases}")
require(not re.search(r"TIER_(PHASE|TOKEN|REQUEST) .*request_id=1", output),
        "aborted request continued issuing commands or completed")

finish = re.search(r"TIER_REQUEST request_id=2 .*completion_cycle=(\d+)", output)
summary = re.search(r"TIER_SUMMARY total_requests=(\d+) completed=(\d+) "
                    r"evicted=(\d+) final_cycle=(\d+)", output)
require(finish and summary and tuple(map(int, summary.groups()[:3])) ==
        (2, 1, 1) and int(finish.group(1)) == int(summary.group(4)),
        "evicted request was not counted as a failed arrival")

commands = Path("log/tier_shared_deadlock/cmd.log.ch0")
require(commands.is_file(), f"missing controller trace: {commands}")
trace = [[part.strip() for part in line.split(",")]
         for line in commands.read_text().splitlines()]
require(sum(parts[1] == "SFM" for parts in trace if len(parts) >= 9) == 1,
        "expected only the surviving request's softmax")
for physical_row in (0, 8192):
    require(any(len(parts) >= 9 and parts[1] == "WR" and
                int(parts[7]) == physical_row and
                grow[0][4] <= int(parts[0]) < int(finish.group(1))
                for parts in trace),
            f"survivor did not write its new KV to freed row {physical_row}")

print("PASS: actual deadlock aborts smallest request, discards KV, resumes survivor")
