"""Check the one-head two-channel Ramulator integration log."""
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: python3 verify_tier_decode_step.py RUN_OUTPUT")

out = Path(sys.argv[1]).read_text()
events = {}
for phase, cycle in re.findall(
    r"TIER_PHASE head=0 event=(\w+) cycle=(\d+)", out
):
    events[phase] = int(cycle)
last = re.search(r"TIER_STEP last_sv_issue_cycle=(\d+)", out)
required = ("KV_APPEND_FENCE", "QK_LAST_ISSUE",
            "SFM_SV_ELIGIBLE", "SV_LAST_ISSUE")
if not last or any(name not in events for name in required):
    raise SystemExit("FAIL: missing frontend stage events")
if not (events["KV_APPEND_FENCE"] < events["QK_LAST_ISSUE"]
        < events["SFM_SV_ELIGIBLE"] < events["SV_LAST_ISSUE"]):
    raise SystemExit(f"FAIL: invalid phase order {events}")
if int(last.group(1)) != events["SV_LAST_ISSUE"]:
    raise SystemExit("FAIL: last SV cycle differs from stage callback")

logs = Path("log/tier_decode_step")
commands = {}
for channel in (2, 5):
    path = logs / f"cmd.log.ch{channel}"
    if not path.exists():
        raise SystemExit(f"FAIL: no controller log for channel {channel}")
    pairs = []
    for line in path.read_text().splitlines():
        match = re.match(r"\s*(\d+),\s*(\w+),", line)
        if match:
            pairs.append((int(match.group(1)), match.group(2)))
    commands[channel] = pairs

sfm = [(cycle, channel) for channel, pairs in commands.items()
       for cycle, opcode in pairs if opcode == "SFM"]
if len(sfm) != 1 or sfm[0][1] != 2:
    raise SystemExit(f"FAIL: expected one SFM on routing channel 2, got {sfm}")
if sfm[0][0] <= events["QK_LAST_ISSUE"]:
    raise SystemExit("FAIL: SFM issued before QK gate")
for channel in (2, 5):
    pairs = commands[channel]
    if not any(opcode == "MACAB" and cycle < sfm[0][0]
               for cycle, opcode in pairs):
        raise SystemExit(f"FAIL: channel {channel} did not issue a QK MAC")
    moves = [cycle for cycle, opcode in pairs
             if opcode == "MVSB" and cycle < sfm[0][0]]
    if not moves or max(moves) > events["QK_LAST_ISSUE"]:
        raise SystemExit(f"FAIL: channel {channel} has no gated score move")
    sv_input = [cycle for cycle, opcode in pairs if opcode == "MVGB"]
    sv_mac = [cycle for cycle, opcode in pairs
              if opcode == "MACAB" and cycle > sfm[0][0]]
    if not sv_input or min(sv_input) <= sfm[0][0] or not sv_mac or \
            min(sv_mac) <= min(sv_input):
        raise SystemExit(f"FAIL: channel {channel} SV input/MAC order is invalid")
print("PASS: both QK/SV channels, gated scores, SFM, SV input/MAC order")
