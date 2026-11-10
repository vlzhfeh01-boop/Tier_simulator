"""Check the sample multi-request workload's phase and token timeline."""
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: python3 verify_tier_decode_workload.py RUN_OUTPUT")

out = Path(sys.argv[1]).read_text()
phases = {}
for name, cycle, request_id, token_index in re.findall(
    r"TIER_PHASE head=0 event=(\w+) cycle=(\d+) "
    r"request_id=(\d+) token_index=(\d+)", out
):
    key = (int(request_id), int(token_index))
    phases.setdefault(key, {})[name] = int(cycle)

tokens = {}
for request_id, token, last_issue, complete in re.findall(
    r"TIER_TOKEN request_id=(\d+) token=(\d+) "
    r"last_sv_issue_cycle=(\d+) completion_cycle=(\d+)", out
):
    tokens[(int(request_id), int(token))] = (int(last_issue), int(complete))

expected = {(1, 33), (1, 34), (2, 33), (3, 33)}
if set(phases) != expected or set(tokens) != {(1, 1), (1, 2), (2, 1), (3, 1)}:
    raise SystemExit(f"FAIL: missing request/token phases: {phases.keys()} {tokens.keys()}")

sfm_cycles = []
for request_id, token_index in expected:
    timeline = phases[(request_id, token_index)]
    required = ("QK_LAST_ISSUE", "SFM_SV_ELIGIBLE", "SV_LAST_ISSUE")
    if any(name not in timeline for name in required):
        raise SystemExit(f"FAIL: missing attention phase: {(request_id, token_index)}")
    qk, sfm, sv = (timeline[name] for name in required)
    last_issue, complete = tokens[(request_id, token_index - 32)]
    if not (qk < sfm < sv == last_issue and complete >= sv + 16):
        raise SystemExit(f"FAIL: invalid phase/tail cycle: {(request_id, token_index)}")
    sfm_cycles.append(sfm)

if len(set(sfm_cycles)) != len(sfm_cycles):
    raise SystemExit("FAIL: shared buffer-die SFM issued twice in one cycle")
if tokens[(1, 2)][1] <= tokens[(1, 1)][1]:
    raise SystemExit("FAIL: request 1 tokens completed out of order")

reported = re.findall(
    r"TIER_REQUEST request_id=(\d+) arrival_cycle=(\d+) "
    r"admitted_cycle=(\d+) first_token_cycle=(\d+) completion_cycle=(\d+)",
    out,
)
if len(reported) != 3:
    raise SystemExit("FAIL: some requests did not complete")
for request_id, arrival, admitted, first_token, completion in reported:
    request_id, arrival, admitted = map(int, (request_id, arrival, admitted))
    first_token, completion = int(first_token), int(completion)
    if not (arrival <= admitted <= first_token <= completion):
        raise SystemExit(f"FAIL: invalid request timeline for {request_id}")

print("PASS: three requests, four tokens, head-wide SFM and token completion")
