"""Check both policies on the same two-request Qwen3-8B-shaped workload.

Usage: python3 verify_tier_qwen3_pair.py TIER_OUTPUT SHARED_OUTPUT
This is a structural and scheduling sanity check, not a performance claim.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


REQUESTS = {1, 2}
LAYERS = 36
HEADS = 32
TOKEN_INDEX = 33
PHASES = ("QK_LAST_ISSUE", "SFM_SV_ELIGIBLE", "SV_LAST_ISSUE")


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def check_run(label, filename):
    output = Path(filename).read_text()
    require("terminate called" not in output and "[error]" not in output
            and "exceeded max_cycles" not in output,
            f"{label}: simulator failed")
    admissions = re.findall(
        r"^TIER_ADMIT request_id=(\d+) cycle=(\d+) channels=(\d+)$",
        output, re.MULTILINE,
    )
    expected_channels = 16 if label == "Shared" else 1
    require([(int(rid), int(ch)) for rid, _, ch in admissions] ==
            [(1, expected_channels), (2, expected_channels)],
            f"{label}: initial allocations must follow policy and predicted length: "
            f"{admissions}")
    require(not re.search(r"^TIER_(?:EVICT|KV_WAIT|CHANNEL_WAIT) ",
                          output, re.MULTILINE),
            f"{label}: this small workload must complete without capacity pressure")

    phases = defaultdict(list)
    phase_re = re.compile(
        r"^TIER_PHASE head=(\d+) event=(\w+) cycle=(\d+) "
        r"request_id=(\d+) token_index=(\d+) layer=(\d+)$", re.MULTILINE,
    )
    sfm_cycles = []
    for head, event, cycle, rid, token, layer in phase_re.findall(output):
        head, cycle, rid, token, layer = map(
            int, (head, cycle, rid, token, layer)
        )
        require(rid in REQUESTS and token == TOKEN_INDEX and layer < LAYERS,
                f"{label}: invalid phase identity")
        phases[rid, layer].append((head, event, cycle))
        if event == "SFM_SV_ELIGIBLE":
            sfm_cycles.append(cycle)
    require(len(phases) == len(REQUESTS) * LAYERS,
            f"{label}: missing per-request layers")
    for rid in REQUESTS:
        for layer in range(LAYERS):
            observed = phases[rid, layer]
            expected = [(0, "KV_APPEND_FENCE")]
            for head in range(HEADS):
                expected.extend((head, event) for event in PHASES)
                if label == "Shared":
                    expected.append((head, "CONTEXT_MERGE_ZERO_COST"))
            require([(head, event) for head, event, _ in observed] == expected,
                    f"{label}: request {rid}, layer {layer}: phase or head order")
            require(all(observed[i][2] >= observed[i - 1][2]
                        for i in range(1, len(observed))),
                    f"{label}: request {rid}, layer {layer}: callback cycle reversed")
    require(len(sfm_cycles) == len(REQUESTS) * LAYERS * HEADS,
            f"{label}: expected 2,304 head-local SFM completions")
    require(len(set(sfm_cycles)) == len(sfm_cycles),
            f"{label}: two heads completed on the single SFM unit in one cycle")

    trace_dir = Path("log") / ("tier_qwen3_pair_shared" if label == "Shared"
                                else "tier_qwen3_pair_tier")
    traces = sorted(trace_dir.glob("cmd.log.ch*"))
    require(len(traces) == 16, f"{label}: missing controller command traces")
    sfm_by_channel = {
        int(path.name.split("ch")[-1]): sum(
            ", SFM," in line for line in path.read_text().splitlines()
        ) for path in traces
    }
    expected_sfm = ({0: 2304} if label == "Shared" else {0: 1152, 1: 1152})
    require({channel: count for channel, count in sfm_by_channel.items() if count}
            == expected_sfm,
            f"{label}: controller SFM count/routing does not match head count: "
            f"{sfm_by_channel}")

    done = defaultdict(list)
    done_re = re.compile(
        r"^TIER_LAYER_DONE request_id=(\d+) token_index=(\d+) "
        r"layer=(\d+) cycle=(\d+) last_sv_issue_cycle=(\d+)$", re.MULTILINE,
    )
    for rid, token, layer, cycle, sv_cycle in done_re.findall(output):
        rid, token, layer, cycle, sv_cycle = map(
            int, (rid, token, layer, cycle, sv_cycle)
        )
        require(rid in REQUESTS and token == TOKEN_INDEX,
                f"{label}: incorrect layer completion identity")
        done[rid].append((layer, cycle, sv_cycle))
    for rid in REQUESTS:
        require([layer for layer, _, _ in done[rid]] == list(range(LAYERS)),
                f"{label}: request {rid} did not finish 36 ordered layers")
        for layer, cycle, sv_cycle in done[rid]:
            require(sv_cycle == next(
                c for h, p, c in reversed(phases[rid, layer])
                if p == "SV_LAST_ISSUE"
            ) and cycle >= sv_cycle,
                    f"{label}: request {rid}, layer {layer}: wrong SV boundary")
            if layer + 1 < LAYERS:
                require(cycle < phases[rid, layer + 1][0][2],
                        f"{label}: request {rid}: next layer started early")

    tokens = re.findall(
        r"^TIER_TOKEN request_id=(\d+) token=(\d+) "
        r"last_sv_issue_cycle=(\d+) completion_cycle=(\d+)$",
        output, re.MULTILINE,
    )
    completed = {}
    for rid, token, last_sv, cycle in tokens:
        rid, token, last_sv, cycle = map(int, (rid, token, last_sv, cycle))
        require(token == 1 and last_sv == done[rid][-1][2]
                and cycle >= done[rid][-1][1],
                f"{label}: request {rid}: token finished before last layer")
        require(rid not in completed, f"{label}: duplicate token completion")
        completed[rid] = cycle
    require(set(completed) == REQUESTS, f"{label}: missing token completion")
    requests = re.findall(
        r"^TIER_REQUEST request_id=(\d+) arrival_cycle=(\d+) "
        r"admitted_cycle=(\d+) first_token_cycle=(\d+) "
        r"completion_cycle=(\d+)$", output, re.MULTILINE,
    )
    require(len(requests) == len(REQUESTS) and
            {int(rid) for rid, *_ in requests} == REQUESTS,
            f"{label}: missing request completion")
    for rid, arrival, admitted, first, finish in requests:
        rid, arrival, admitted, first, finish = map(
            int, (rid, arrival, admitted, first, finish)
        )
        require(arrival == 0 and admitted >= 1 and
                first == finish == completed[rid],
                f"{label}: inconsistent request timing")
    summary = re.search(
        r"^TIER_SUMMARY total_requests=(\d+) completed=(\d+) "
        r"evicted=(\d+) final_cycle=(\d+)$", output, re.MULTILINE,
    )
    require(summary is not None and tuple(map(int, summary.groups())) ==
            (2, 2, 0, max(completed.values())),
            f"{label}: incorrect summary")

    if label == "Shared":
        regions = re.findall(
            r"^TIER_KV_REGION request_id=(\d+) row_offset=(\d+) "
            r"rows_per_slice=(\d+) chunks=(\d+)$", output, re.MULTILINE,
        )
        require([tuple(map(int, fields)) for fields in regions] ==
                [(1, 0, 1, 576), (2, 1, 1, 576)],
                f"Shared: requests must have separate row stripes: {regions}")
        leases = re.findall(
            r"^TIER_LEASE request_id=(\d+) phase=(\w+) "
            r"event=(ACQUIRE|RELEASE) cycle=(\d+) "
            r"(?:channels=(\d+)|last_issue_cycle=(\d+))$",
            output, re.MULTILINE,
        )
        require(leases, "Shared: no channel leases recorded")
        live = {}
        counts = defaultdict(int)
        for rid, phase, event, cycle, channels, last_issue in leases:
            rid, cycle = int(rid), int(cycle)
            if event == "ACQUIRE":
                channels = int(channels)
                require(phase in ("QK_LAST_ISSUE", "SV_LAST_ISSUE")
                        and channels == 3 and not live,
                        "Shared: overlapping or invalid three-channel lease")
                live[rid] = (phase, cycle)
                counts[rid, "ACQUIRE"] += 1
            else:
                require(rid in live and live[rid][0] == phase
                        and cycle >= live[rid][1]
                        and int(last_issue) <= cycle,
                        "Shared: mismatched or premature lease release")
                del live[rid]
                counts[rid, "RELEASE"] += 1
        require(not live and all(counts[rid, "ACQUIRE"] ==
                                 counts[rid, "RELEASE"] > 0
                                 for rid in REQUESTS),
                "Shared: lease not released at request completion")
    return max(completed.values()), completed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_tier_qwen3_pair.py TIER_OUTPUT SHARED_OUTPUT")
    tier_end, tier_tokens = check_run("Tier", sys.argv[1])
    shared_end, shared_tokens = check_run("Shared", sys.argv[2])
    print("PASS: two requests complete all 36 layers and 32:8 GQA heads "
          "under both policies")
    print(f"Tier: final_cycle={tier_end}, token_cycles={tier_tokens}")
    print(f"Shared: final_cycle={shared_end}, token_cycles={shared_tokens}")
