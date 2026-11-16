"""Verify Tier channel wait/reuse and deadlock eviction through Ramulator logs."""

import re
import sys
from pathlib import Path


def require(ok, message):
    if not ok:
        raise SystemExit(f"FAIL: {message}")


def events(output, pattern):
    return [tuple(map(int, match)) for match in re.findall(pattern, output)]


def phase_cycles(output, request_id, token_index):
    found = re.findall(
        rf"TIER_PHASE head=0 event=(QK_LAST_ISSUE|SFM_SV_ELIGIBLE|SV_LAST_ISSUE) "
        rf"cycle=(\d+) request_id={request_id} token_index={token_index}",
        output,
    )
    phases = {name: int(cycle) for name, cycle in found}
    require(len(found) == 3 and len(phases) == 3,
            f"missing or repeated attention phases for request {request_id}: {found}")
    require(phases["QK_LAST_ISSUE"] < phases["SFM_SV_ELIGIBLE"]
            < phases["SV_LAST_ISSUE"],
            f"wrong QK/SFM/SV order for request {request_id}: {phases}")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("wait", "deadlock"):
        raise SystemExit("usage: verify_tier_channel_growth.py wait|deadlock RUN_OUTPUT")
    mode = sys.argv[1]
    output = Path(sys.argv[2]).read_text()
    require("terminate called" not in output and "exceeded max_cycles" not in output
            and "[error]" not in output, "simulator failed")

    admissions = events(output, r"TIER_ADMIT request_id=(\d+) cycle=\d+ channels=(\d+)")
    expected_admissions = ([(1, 1), (2, 15), (3, 15)] if mode == "wait"
                           else [(1, 1), (2, 15)])
    require(admissions == expected_admissions,
            f"wrong initial channel counts or admission order: {admissions}")
    wait = events(output, r"TIER_CHANNEL_WAIT request_id=(\d+) cycle=(\d+)")
    resume = events(
        output,
        r"TIER_CHANNEL_RESUME request_id=(\d+) cycle=(\d+) wait_cycles=(\d+)",
    )
    require(wait and resume, f"missing channel wait or retry: {wait}, {resume}")
    for request_id, cycle, duration in resume:
        begins = [start for rid, start in wait if rid == request_id]
        require(len(begins) == 1 and cycle >= begins[0]
                and duration == cycle - begins[0],
                f"bad growth wait duration for request {request_id}")
    finished = dict(events(
        output,
        r"TIER_REQUEST request_id=(\d+) arrival_cycle=\d+ admitted_cycle=\d+ "
        r"first_token_cycle=\d+ completion_cycle=(\d+)",
    ))
    summary = re.search(
        r"TIER_SUMMARY total_requests=(\d+) completed=(\d+) "
        r"evicted=(\d+) final_cycle=(\d+)", output,
    )
    require(summary is not None, "missing request summary")

    if mode == "wait":
        require("TIER_EVICT " not in output,
                "evicted while another request could finish and release channels")
        require(wait == [(1, wait[0][1])] and
                len(resume) == 1 and resume[0][0] == 1,
                f"wrong request blocked on channel growth: {wait}, {resume}")
        admission_cycles = dict(events(
            output, r"TIER_ADMIT request_id=(\d+) cycle=(\d+) channels=\d+"))
        require(set(finished) == {1, 2, 3} and
                wait[0][1] < finished[2] < resume[0][1] <= finished[1]
                < admission_cycles[3] <= finished[3],
                "growth must precede new admission after channel release: "
                f"{finished}, {admission_cycles}")
        require(tuple(map(int, summary.groups()[:3])) == (3, 3, 0),
                "all requests must complete without eviction")
        phase_cycles(output, 1, 16)
        phase_cycles(output, 2, 239)
        phase_cycles(output, 3, 239)
        print("PASS: Tier waits, grows before new admission, and avoids eviction")
    else:
        require({rid for rid, _ in wait} == {1, 2} and len(wait) == 2,
                f"both running requests must block: {wait}")
        evicted = events(
            output,
            r"TIER_EVICT request_id=(\d+) cycle=(\d+) released_units=(\d+) "
            r"generated_tokens=(\d+) reason=DEADLOCK",
        )
        require(len(evicted) == 1 and evicted[0][0] == 1 and
                evicted[0][2:] == (1, 0) and
                evicted[0][1] >= max(cycle for _, cycle in wait),
                f"did not evict the smallest quiescent request: {evicted}")
        require(len(resume) == 1 and resume[0][0] == 2 and
                resume[0][1] == evicted[0][1],
                f"survivor did not grow immediately after eviction: {resume}")
        require(set(finished) == {2} and
                tuple(map(int, summary.groups()[:3])) == (2, 1, 1),
                f"evicted request must count as failed: {finished}, {summary.groups()}")
        require(not re.search(r"TIER_(PHASE|TOKEN|REQUEST) .*request_id=1", output),
                "evicted request continued running")
        phase_cycles(output, 2, 240)
        trace_path = Path("log/tier_channel_deadlock/cmd.log.ch0")
        require(trace_path.is_file(), f"missing controller trace: {trace_path}")
        trace = [[part.strip() for part in line.split(",")]
                 for line in trace_path.read_text().splitlines()]
        require(any(len(parts) > 1 and parts[1] in ("WR", "WRGB") and
                    int(parts[0]) >= evicted[0][1] for parts in trace),
                "survivor did not append KV on victim's freed channel 0")
        print("PASS: Tier deadlock evicts one-channel victim and resumes survivor")


if __name__ == "__main__":
    main()
