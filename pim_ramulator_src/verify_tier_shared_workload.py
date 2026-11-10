"""Check row separation and phase leases in the shared-channel probe."""
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: python3 verify_tier_shared_workload.py RUN_OUTPUT")

out = Path(sys.argv[1]).read_text()
regions = {int(request_id): int(row) for request_id, row in re.findall(
    r"TIER_KV_REGION request_id=(\d+) row_offset=(\d+)", out)}
if regions != {1: 0, 2: 8}:
    raise SystemExit(f"FAIL: request row windows overlap or are missing: {regions}")

leases = [(int(request_id), phase, event, int(cycle),
           int(channel_count or 0), int(last_issue or 0))
          for request_id, phase, event, cycle, channel_count, last_issue in re.findall(
              r"TIER_LEASE request_id=(\d+) phase=(QK_LAST_ISSUE|SV_LAST_ISSUE) "
              r"event=(ACQUIRE|RELEASE) cycle=(\d+)"
              r"(?: channels=(\d+)| last_issue_cycle=(\d+))?", out)]
active = None
intervals = {(request_id, phase): [] for request_id in (1, 2)
             for phase in ("QK_LAST_ISSUE", "SV_LAST_ISSUE")}
for request_id, phase, event, cycle, channel_count, last_issue in leases:
    if event == "ACQUIRE":
        if active is not None or channel_count != 3:
            raise SystemExit("FAIL: overlapping phase leases or idle channels leased")
        active = (request_id, phase, cycle)
    else:
        if active is None or active[:2] != (request_id, phase) or cycle < active[2]:
            raise SystemExit("FAIL: invalid shared channel release")
        if cycle < last_issue + (4 if phase == "SV_LAST_ISSUE" else 0):
            raise SystemExit("FAIL: PIM channel released before its modeled drain")
        intervals[request_id, phase].append((active[2], cycle))
        active = None
if active is not None or [len(intervals[request_id, phase]) for request_id, phase
                          in intervals] != [2, 2, 1, 1]:
    raise SystemExit(f"FAIL: incomplete QK/SV phase leases: {intervals}")

first_qk_end = intervals[1, "QK_LAST_ISSUE"][0][1]
second_qk_start = intervals[2, "QK_LAST_ISSUE"][0][0]
first_sv_start = intervals[1, "SV_LAST_ISSUE"][0][0]
if not (first_qk_end <= second_qk_start < first_sv_start):
    raise SystemExit("FAIL: request 2 did not enter before request 1's SV")

rows = {1: set(), 2: set()}
for line in Path("log/tier_shared_workload/cmd.log.ch2").read_text().splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) < 9 or fields[1] != "WR":
        continue
    row = int(fields[7])
    matched = False
    for request_id, offset in regions.items():
        if offset <= row < offset + 8 or 8192 + offset <= row < 8192 + offset + 8:
            rows[request_id].add(row)
            matched = True
    if not matched:
        raise SystemExit(f"FAIL: K/V write outside reserved rows: {row}")
if not rows[1] or not rows[2] or rows[1] & rows[2]:
    raise SystemExit(f"FAIL: KV rows missing or shared: {rows}")

sfm = [line for line in Path("log/tier_shared_workload/cmd.log.ch0")
       .read_text().splitlines() if ", SFM," in line]
tokens = re.findall(r"TIER_TOKEN request_id=(\d+) token=(\d+) .*completion_cycle=(\d+)", out)
if len(sfm) != 3 or len(tokens) != 3 or len(re.findall(r"TIER_REQUEST request_id=", out)) != 2:
    raise SystemExit("FAIL: missing SFM, token, or completed request")
first_completion = next(int(cycle) for request_id, token, cycle in tokens
                        if request_id == "1" and token == "1")
if intervals[1, "SV_LAST_ISSUE"][0][1] >= first_completion:
    raise SystemExit("FAIL: shared channels were held during non-PIM token delay")

print("PASS: two shared requests, isolated K/V rows, QK/SV leases, overlap during idle phases")
