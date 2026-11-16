"""Check full Qwen3-8B-shaped command order and KV rows in Ramulator logs.

This is a one-token, policy-admitted structural test, not a timing model.
Run from ramulator2/ after tier_qwen3_8b_smoke.yaml has completed.
"""

import re
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


LAYERS = 36
QUERY_HEADS = 32
KV_HEADS = 8
CHANNELS = {0}
TOKEN_INDEX = 33
ROWS_PER_BANK = 16384
ROWS_PER_SLICE = ROWS_PER_BANK // (2 * LAYERS * KV_HEADS)
TRACE_DIR = Path("log/tier_qwen3_8b_smoke")
PHASES = ("QK_LAST_ISSUE", "SFM_SV_ELIGIBLE", "SV_LAST_ISSUE")
EVENTS_PER_LAYER = 1 + len(PHASES) * QUERY_HEADS


def fail(message):
    raise SystemExit(f"FAIL: {message}")


if len(sys.argv) != 2:
    raise SystemExit("usage: python3 verify_tier_qwen3_8b_smoke.py RUN_OUTPUT")

output = Path(sys.argv[1]).read_text()
admissions = re.findall(
    r"^TIER_ADMIT request_id=1 cycle=(\d+) channels=(\d+)$",
    output, re.MULTILINE,
)
# 33 prompt tokens + 128 predicted output tokens = 161 tokens.
# One physical channel holds 28 rows/slice * 16 blocks/row * 16 tokens/block
# = 7168 tokens. The runtime policy must therefore admit exactly one channel.
if len(admissions) != 1 or int(admissions[0][1]) != 1:
    fail(f"admission must use the predicted length (one channel): {admissions}")
phase_re = re.compile(
    r"^TIER_PHASE head=(\d+) event=(\w+) cycle=(\d+) "
    r"request_id=(\d+) token_index=(\d+) layer=(\d+)$", re.MULTILINE,
)
events = [
    (int(layer), int(head), phase, int(cycle))
    for head, phase, cycle, request, token, layer in phase_re.findall(output)
    if int(request) == 1 and int(token) == TOKEN_INDEX
]
expected = []
for layer in range(LAYERS):
    expected.append((layer, 0, "KV_APPEND_FENCE"))
    for head in range(QUERY_HEADS):
        expected.extend((layer, head, phase) for phase in PHASES)

if [event[:3] for event in events] != expected:
    fail(f"layer/head/phase order: got {len(events)}, expected {len(expected)}")
if any(events[i][3] < events[i - 1][3] for i in range(1, len(events))):
    fail("phase callback cycles move backward")

layer_re = re.compile(
    r"^TIER_LAYER_DONE request_id=(\d+) token_index=(\d+) "
    r"layer=(\d+) cycle=(\d+) last_sv_issue_cycle=(\d+)$", re.MULTILINE,
)
done = [tuple(map(int, fields)) for fields in layer_re.findall(output)]
if len(done) != LAYERS or any(
    request != 1 or token != TOKEN_INDEX or layer != index
    for index, (request, token, layer, _, _) in enumerate(done)
):
    fail("expected exactly 36 layer completions, in order")
for layer, (_, _, _, cycle, sv_cycle) in enumerate(done):
    final_sv_event = events[layer * EVENTS_PER_LAYER + len(PHASES) * QUERY_HEADS]
    if cycle < sv_cycle or sv_cycle != final_sv_event[3]:
        fail(f"layer={layer}: SV issue or layer completion cycle is wrong")
    if layer + 1 < LAYERS and cycle >= events[(layer + 1) * EVENTS_PER_LAYER][3]:
        fail(f"layer={layer}: next layer started too soon")

steps = re.findall(r"^TIER_STEP last_sv_issue_cycle=(\d+) request_id=1$",
                   output, re.MULTILINE)
tokens = re.findall(
    r"^TIER_TOKEN request_id=1 token=(\d+) last_sv_issue_cycle=(\d+) "
    r"completion_cycle=(\d+)$", output, re.MULTILINE,
)
if len(steps) != 1 or len(tokens) != 1 or len(re.findall(
        r"^TIER_REQUEST request_id=1\b", output, re.MULTILINE)) != 1:
    fail("expected exactly one finished step, token and request")
if (int(steps[0]) != done[-1][4] or tokens[0][:2] != ("1", steps[0])
        or int(tokens[0][2]) < done[-1][3]):
    fail("token completed before the final layer")
if not re.search(r"^TIER_SUMMARY total_requests=1 completed=1 evicted=0\b",
                 output, re.MULTILINE):
    fail("missing successful request summary")


def base_row(kind, layer, kv_head):
    slice_index = layer * KV_HEADS + kv_head
    if kind == "V":
        slice_index += LAYERS * KV_HEADS
    return slice_index * ROWS_PER_SLICE


macs = []  # (cycle, channel, physical row)
sfms = []  # (cycle, channel)
writes = defaultdict(list)  # physical row -> issued cycles
for channel in CHANNELS:
    path = TRACE_DIR / f"cmd.log.ch{channel}"
    if not path.is_file():
        fail(f"missing trace for channel {channel}: {path}")
    for line in path.read_text().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        cycle, opcode, row = int(fields[0]), fields[1], int(fields[7])
        if opcode == "WR":
            writes[row].append(cycle)
        elif opcode == "MACAB":
            macs.append((cycle, channel, row))
        elif opcode == "SFM":
            sfms.append((cycle, channel))

expected_rows = {
    base_row(kind, layer, kv)
    for kind in ("K", "V")
    for layer in range(LAYERS)
    for kv in range(KV_HEADS)
}
if set(writes) != expected_rows:
    fail(f"KV append row set: got {len(writes)}, expected {len(expected_rows)}")
for layer in range(LAYERS):
    first_qk_cycle = events[layer * EVENTS_PER_LAYER + 1][3]
    for kind in ("K", "V"):
        for kv in range(KV_HEADS):
            row = base_row(kind, layer, kv)
            if max(writes[row]) > first_qk_cycle:
                fail(f"layer={layer}: K/V append issued after QK")

macs.sort()
mac_cycles = [cycle for cycle, _, _ in macs]
for index, (layer, head, phase, cycle) in enumerate(events):
    if phase not in ("QK_LAST_ISSUE", "SV_LAST_ISSUE"):
        continue
    earlier = events[index - 1][3]
    macs_in_phase = macs[
        bisect_right(mac_cycles, earlier):bisect_right(mac_cycles, cycle)
    ]
    kind = "K" if phase == "QK_LAST_ISSUE" else "V"
    expected_row = base_row(kind, layer, head // (QUERY_HEADS // KV_HEADS))
    if not macs_in_phase or {ch for _, ch, _ in macs_in_phase} != CHANNELS:
        fail(f"layer={layer} head={head} phase={phase}: missing channel MAC")
    if any(row != expected_row for _, _, row in macs_in_phase):
        fail(f"layer={layer} head={head} phase={phase}: incorrect KV row")

sfms.sort()
if len(sfms) != LAYERS * QUERY_HEADS or any(ch != 0 for _, ch in sfms):
    fail(f"expected {LAYERS * QUERY_HEADS} SFM commands on channel 0")
for ordinal, (cycle, _) in enumerate(sfms):
    layer, head = divmod(ordinal, QUERY_HEADS)
    qk_index = layer * EVENTS_PER_LAYER + 1 + head * len(PHASES)
    if not events[qk_index][3] < cycle <= events[qk_index + 1][3]:
        fail(f"layer={layer} head={head}: SFM bypassed its QK gate")

print("PASS: predicted-length admission, one channel, 36 layers, 32:8 GQA, 1152 SFM")
