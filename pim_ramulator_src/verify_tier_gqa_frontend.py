"""Verify the real_controller, two-layer, 8:2 GQA decode smoke test."""

import re
import sys
from pathlib import Path

def fail(reason):
    raise SystemExit(f"FAIL: {reason}")

if len(sys.argv) != 2 :
    raise SystemExit("usage: python3 verify_tier_gqa_frontend.py RUN_OUTPUT")

output = Path(sys.argv[1]).read_text()
phase_pattern = re.compile(
    r"^TIER_PHASE head=(\d+) event=(\w+) cycle=(\d+) "
    r"request_id=(\d+) token_index=(\d+) layer=(\d+)$", re.MULTILINE
)

events = [
    (int(layer), int(head), phase, int(cycle))
    for head,phase, cycle, request, token, layer in phase_pattern.findall(output)
    if request=="1" and token =="33"
]
expected = []

for layer in range(2):
    expected.append((layer,0,"KV_APPEND_FENCE"))
    for head in range(8):
        expected += [
            (layer,head,"QK_LAST_ISSUE"),
            (layer,head,"SFM_SV_ELIGIBLE"),
            (layer, head, "SV_LAST_ISSUE"),
            (layer, head, "CONTEXT_MERGE_ZERO_COST"),
        ]
if [event[:3] for event in events] != expected :
    fail(f"layer/head/phase order differs (got {len(events)} events, expected 66)")

layers = [
    tuple(map(int,fields))
    for fields in re.findall(
        r"^TIER_LAYER_DONE request_id=(\d+) token_index=(\d+) "
        r"layer=(\d+) cycle=(\d+) last_sv_issue_cycle=(\d+)$",
        output, re.MULTILINE,
    )
]
if len(layers) != 2 or [(e[0], e[1], e[2]) for e in layers] != [
    (1, 33, 0), (1, 33, 1)
]:
    fail(f"expected two ordered layer completions: {layers}")
first_layer_end, second_layer_end = layers[0][3], layers[1][3]
if first_layer_end >= events[33][3] or second_layer_end < events[-1][3]:
    fail("layer boundary is out of order")

steps = re.findall(r"^TIER_STEP last_sv_issue_cycle=(\d+) request_id=1$",
                   output, re.MULTILINE)
tokens = re.findall(
    r"^TIER_TOKEN request_id=1 token=(\d+) last_sv_issue_cycle=(\d+) "
    r"completion_cycle=(\d+)$", output, re.MULTILINE,
)
requests = re.findall(r"^TIER_REQUEST request_id=1\b", output, re.MULTILINE)
if len(steps) != 1 or len(tokens) != 1 or len(requests) != 1:
    fail("token/request was not completed exactly once after both layers")
last_sv = events[-2][3]  # Head 7's SV; the final event is context merge.
if (int(steps[0]) != last_sv or int(tokens[0][0]) != 1 or
        int(tokens[0][1]) != last_sv or int(tokens[0][2]) < second_layer_end or
        second_layer_end <= first_layer_end):
    fail("final token cycle does not follow the last layer's SV")

trace_dir = Path("log/tier_gqa_frontend")
commands = []
for channel in (2, 5):
    path = trace_dir / f"cmd.log.ch{channel}"
    if not path.is_file():
        fail(f"missing command trace: {path}")
    for line in path.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        commands.append((int(fields[0]), fields[1], channel, int(fields[7])))

# The configured geometry has 16,384 rows/bank. The eight K/V slices
# (two layers x two KV heads x K/V) each get 2,048 consecutive rows.
rows_per_slice = 16384 // (2 * 2 * 2)


def base_row(kind, layer, head):
    kv = head // 4  # 8 query heads / 2 KV heads
    slice_index = layer * 2 + kv + (4 if kind == "V" else 0)
    return slice_index * rows_per_slice


for layer in range(2):
    # Both KV heads' K and V stores must reach their own row slice.
    for kind in ("K", "V"):
        for kv in range(2):
            base = base_row(kind, layer, kv * 4)
            if not any(op == "WR" and base <= row < base + rows_per_slice
                       for _, op, _, row in commands):
                fail(f"no KV append in layer={layer} kind={kind} kv={kv}")

for index, (layer, head, phase, cycle) in enumerate(events):
    if phase not in ("QK_LAST_ISSUE", "SV_LAST_ISSUE"):
        continue
    prior = events[index - 1][3]
    kind = "K" if phase == "QK_LAST_ISSUE" else "V"
    macs = [(ch, row) for issued, op, ch, row in commands
            if op == "MACAB" and prior < issued <= cycle]
    base = base_row(kind, layer, head)
    if not macs or any(not (base <= row < base + rows_per_slice)
                       for _, row in macs):
        fail(f"wrong MAC row slice in layer={layer} head={head} {phase}: {macs}")

sfm = sorted((cycle, ch) for cycle, op, ch, _ in commands if op == "SFM")
if len(sfm) != 16 or any(ch != 2 for _, ch in sfm):
    fail(f"expected 16 SFM commands on routing channel 2, got {sfm}")
for index, (issued, _) in enumerate(sfm):
    qk = events[1 + (index // 8) * 33 + (index % 8) * 4]
    eligible = events[2 + (index // 8) * 33 + (index % 8) * 4]
    if not qk[3] < issued <= eligible[3]:
        fail(f"SFM {index} did not follow its QK gate")

print("PASS: two layers, 8:2 GQA, distinct KV rows, 16 SFM, one token completion")
