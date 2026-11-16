"""Verify Qwen3-shaped Tier/Shared execution across the 4096-token row boundary.

Run from ramulator2/: python3 ../pim_ramulator_src/verify_tier_qwen3_row_boundary.py
    tier_qwen3_row_boundary_tier.out tier_qwen3_row_boundary_shared.out
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


REQUESTS = {1, 2}
TOKENS = {4095: 1, 4096: 2}
LAYERS = 36
QUERY_HEADS = 32
KV_HEADS = 8
SLICES = 2 * LAYERS * KV_HEADS
ROWS_PER_SLICE = 16384 // SLICES
HEAD_PHASES = ("QK_LAST_ISSUE", "SFM_SV_ELIGIBLE", "SV_LAST_ISSUE")


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def records(output, pattern):
    return [tuple(map(int, values)) for values in re.findall(
        pattern, output, re.MULTILINE
    )]


def check_run(policy, path):
    output = Path(path).read_text()
    require("terminate called" not in output and "[error]" not in output and
            "exceeded max_cycles" not in output and "TIER_EVICT " not in output,
            f"{policy}: simulator failed or evicted a request")
    expected_channels = 16 if policy == "shared" else 1
    admissions = records(
        output, r"^TIER_ADMIT request_id=(\d+) cycle=(\d+) channels=(\d+)$"
    )
    require([r for r, _, _ in admissions] == [1, 2] and
            all(c == 1 and n == expected_channels for _, c, n in admissions),
            f"{policy}: unexpected predicted-length admission {admissions}")

    completed_tokens = defaultdict(list)
    for rid, token, issue, finish in records(
        output, r"^TIER_TOKEN request_id=(\d+) token=(\d+) "
                r"last_sv_issue_cycle=(\d+) completion_cycle=(\d+)$"
    ):
        require(rid in REQUESTS and finish >= issue,
                f"{policy}: invalid token completion")
        completed_tokens[rid].append((token, finish))
    require(all([token for token, _ in completed_tokens[rid]] == [1, 2] and
                completed_tokens[rid][0][1] < completed_tokens[rid][1][1]
                for rid in REQUESTS),
            f"{policy}: expected two ordered tokens per request: {completed_tokens}")

    phases = defaultdict(list)
    for head, event, cycle, rid, token, layer in re.findall(
        r"^TIER_PHASE head=(\d+) event=(\w+) cycle=(\d+) "
        r"request_id=(\d+) token_index=(\d+) layer=(\d+)$",
        output, re.MULTILINE,
    ):
        rid, token, layer, head, cycle = map(int, (rid, token, layer, head, cycle))
        require(rid in REQUESTS and token in TOKENS and layer < LAYERS,
                f"{policy}: unexpected phase identity")
        phases[rid, token, layer].append((head, event, cycle))
    require(len(phases) == 2 * 2 * LAYERS,
            f"{policy}: missing token/layer combinations")
    for rid in REQUESTS:
        for token in TOKENS:
            for layer in range(LAYERS):
                sequence = phases[rid, token, layer]
                expected = [(0, "KV_APPEND_FENCE")]
                for head in range(QUERY_HEADS):
                    expected.extend((head, phase) for phase in HEAD_PHASES)
                    if policy == "shared":
                        expected.append((head, "CONTEXT_MERGE_ZERO_COST"))
                require([(head, phase) for head, phase, _ in sequence] == expected,
                        f"{policy}: incorrect phase order: request={rid}, "
                        f"token={token}, layer={layer}")

    layer_done = defaultdict(list)
    for rid, token, layer, cycle, last_issue in records(
        output, r"^TIER_LAYER_DONE request_id=(\d+) token_index=(\d+) "
                r"layer=(\d+) cycle=(\d+) last_sv_issue_cycle=(\d+)$"
    ):
        require(rid in REQUESTS and token in TOKENS and cycle >= last_issue,
                f"{policy}: invalid layer completion")
        layer_done[rid, token].append((layer, cycle, last_issue))
    for rid in REQUESTS:
        for token in TOKENS:
            require([layer for layer, _, _ in layer_done[rid, token]] ==
                    list(range(LAYERS)),
                    f"{policy}: missing or reordered layers for {rid}, {token}")
            for layer, done_cycle, sv_issue in layer_done[rid, token]:
                last_sv = next(c for _, event, c in reversed(phases[rid, token, layer])
                               if event == "SV_LAST_ISSUE")
                require(sv_issue == last_sv and done_cycle >= last_sv,
                        f"{policy}: layer finished before its last SV issue")
                if layer + 1 < LAYERS:
                    require(done_cycle < phases[rid, token, layer + 1][0][2],
                            f"{policy}: next layer started early")
            require(completed_tokens[rid][TOKENS[token] - 1][1] >=
                    layer_done[rid, token][-1][1],
                    f"{policy}: token finished before final layer")

    finished = records(
        output, r"^TIER_REQUEST request_id=(\d+) arrival_cycle=(\d+) "
                r"admitted_cycle=(\d+) first_token_cycle=(\d+) "
                r"completion_cycle=(\d+)$"
    )
    require(len(finished) == 2 and {rid for rid, *_ in finished} == REQUESTS,
            f"{policy}: missing request completion")
    for rid, arrival, admitted, first, last in finished:
        require(arrival == 0 and admitted == 1 and
                first == completed_tokens[rid][0][1] and
                last == completed_tokens[rid][1][1],
                f"{policy}: incorrect request timing")
    final_cycle = max(completed_tokens[rid][-1][1] for rid in REQUESTS)
    require(records(
        output, r"^TIER_SUMMARY total_requests=(\d+) completed=(\d+) "
                r"evicted=(\d+) final_cycle=(\d+)$"
    ) == [(2, 2, 0, final_cycle)],
            f"{policy}: missing successful summary")

    growth_rows = {}
    if policy == "shared":
        regions = records(
            output, r"^TIER_KV_REGION request_id=(\d+) row_offset=(\d+) "
                    r"rows_per_slice=(\d+) chunks=(\d+)$"
        )
        require(regions == [(1, 0, 1, SLICES), (2, 1, 1, SLICES)],
                f"shared: initial stripes overlap or have wrong slice count: {regions}")
        growth = records(
            output, r"^TIER_KV_GROW request_id=(\d+) logical_row=(\d+) "
                    r"row_offset=(\d+) rows_per_slice=(\d+) cycle=(\d+)$"
        )
        require(len(growth) == 2 and {r for r, *_ in growth} == REQUESTS and
                {row for _, _, row, _, _ in growth} == {2, 3},
                f"shared: 4097th token did not allocate two distinct stripes: {growth}")
        for rid, logical_row, physical_row, count, cycle in growth:
            require(logical_row == 1 and count == 2 and
                    completed_tokens[rid][0][1] < cycle <=
                    phases[rid, 4096, 0][0][2],
                    f"shared: request {rid} grew outside token boundary")
            growth_rows[rid] = physical_row
        leases = re.findall(
            r"^TIER_LEASE request_id=(\d+) phase=(\w+) "
            r"event=(ACQUIRE|RELEASE) cycle=(\d+) "
            r"(?:channels=(\d+)|last_issue_cycle=(\d+))$",
            output, re.MULTILINE,
        )
        require(leases, "shared: no channel leases recorded")
        active = None
        for rid, phase, event, cycle, channels, last_issue in leases:
            rid, cycle = int(rid), int(cycle)
            if event == "ACQUIRE":
                require(active is None and int(channels) == 16 and
                        phase in ("QK_LAST_ISSUE", "SV_LAST_ISSUE"),
                        "shared: overlapping or incomplete 16-channel lease")
                active = (rid, phase, cycle)
            else:
                require(active is not None and active[:2] == (rid, phase) and
                        cycle >= active[2] and int(last_issue) <= cycle,
                        "shared: phase lease released without its issue boundary")
                active = None
        require(active is None, "shared: lease held after completion")
    else:
        require("TIER_KV_GROW " not in output and
                "TIER_CHANNEL_WAIT " not in output,
                "tier: 4097 tokens should remain on the admitted channel")

    # A controller trace has no request ID, so distinct physical rows identify
    # each request's append. Count row sets for all 576 K/V slices, not just K0.
    trace_dir = Path("log") / f"tier_qwen3_row_boundary_{policy}"
    expected_writes = {}
    if policy == "shared":
        expected_writes[15] = {s * ROWS_PER_SLICE + r
                               for s in range(SLICES) for r in (0, 1)}
        expected_writes[0] = {s * ROWS_PER_SLICE + r
                              for s in range(SLICES) for r in growth_rows.values()}
    else:
        for channel in (0, 1):
            expected_writes[channel] = {s * ROWS_PER_SLICE + r
                                        for s in range(SLICES) for r in (15, 16)}
    sfm_counts = defaultdict(int)
    for channel in range(16):
        trace = trace_dir / f"cmd.log.ch{channel}"
        require(trace.is_file(), f"{policy}: missing channel trace {trace}")
        writes = set()
        grown_macs = set()
        with trace.open() as stream:
            for line in stream:
                fields = [part.strip() for part in line.split(",")]
                if len(fields) < 9:
                    continue
                opcode = fields[1]
                if opcode == "SFM":
                    sfm_counts[channel] += 1
                if opcode == "WR":
                    writes.add(int(fields[7]))
                if opcode == "MACAB":
                    grown_macs.add(int(fields[7]))
        require(writes == expected_writes.get(channel, set()),
                f"{policy}: channel {channel}: KV writes escaped mapped rows "
                f"({len(writes)} instead of {len(expected_writes.get(channel, set()))})")
        if policy == "shared" and channel == 0:
            require(expected_writes[0] <= grown_macs,
                    "shared: QK/SV did not read grown rows for all slices")
        if policy == "tier" and channel in (0, 1):
            require({s * ROWS_PER_SLICE + 16 for s in range(SLICES)} <= grown_macs,
                    f"tier: channel {channel} did not read the next local row")
    expected_sfm = ({0: 4608} if policy == "shared"
                    else {0: 2304, 1: 2304})
    require({ch: count for ch, count in sfm_counts.items() if count} == expected_sfm,
            f"{policy}: wrong real SFM issue count/routing: {dict(sfm_counts)}")
    return final_cycle, growth_rows


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_tier_qwen3_row_boundary.py TIER_OUT SHARED_OUT")
    tier_cycle, _ = check_run("tier", sys.argv[1])
    shared_cycle, growth = check_run("shared", sys.argv[2])
    print("PASS: both policies cross 4096/4097 with 36 layers, 32:8 GQA, "
          "isolated rows, 16-channel leases, 4608 SFM issues")
    print(f"Tier final_cycle={tier_cycle}; Shared final_cycle={shared_cycle}; "
          f"Shared growth rows={growth}")
