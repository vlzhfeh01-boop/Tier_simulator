"""Check growth wait accounting and paired diagnostics without rerunning DRAM."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_tier_policy_sweep import diagnose_case, summarize
from compare_tier_policy_logs import read_run, workload


class SweepDiagnosticsTest(unittest.TestCase):
    def test_resumed_and_evicted_growth_waits_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "rate_1000000_seed_1"
            case.mkdir()
            csv_path = case / f"{case.name}.csv"
            csv_path.write_text("1,0,15,1,1\n2,0,15,1,1\n")
            (case / "tier.out").write_text(
                "TIER_ADMIT request_id=1 cycle=1 channels=1\n"
                "TIER_ADMIT request_id=2 cycle=1 channels=1\n"
                "TIER_CHANNEL_WAIT request_id=1 cycle=2\n"
                "TIER_CHANNEL_WAIT request_id=2 cycle=3\n"
                "TIER_CHANNEL_RESUME request_id=1 cycle=5 wait_cycles=3\n"
                "TIER_EVICT request_id=2 cycle=6 released_units=1 generated_tokens=0 reason=DEADLOCK\n"
                "TIER_TOKEN request_id=1 token=1 completion_cycle=8\n"
                "TIER_REQUEST request_id=1 arrival_cycle=0 admitted_cycle=1 "
                "first_token_cycle=8 completion_cycle=8\n"
                "TIER_SUMMARY total_requests=2 completed=1 evicted=1 final_cycle=8\n")
            (case / "shared.out").write_text(
                "TIER_ADMIT request_id=1 cycle=1 channels=16\n"
                "TIER_ADMIT request_id=2 cycle=1 channels=16\n"
                "TIER_TOKEN request_id=1 token=1 completion_cycle=7\n"
                "TIER_REQUEST request_id=1 arrival_cycle=0 admitted_cycle=1 "
                "first_token_cycle=7 completion_cycle=7\n"
                "TIER_TOKEN request_id=2 token=1 completion_cycle=8\n"
                "TIER_REQUEST request_id=2 arrival_cycle=0 admitted_cycle=1 "
                "first_token_cycle=8 completion_cycle=8\n"
                "TIER_SUMMARY total_requests=2 completed=2 evicted=0 final_cycle=8\n")
            audit = diagnose_case(case, 769, 10, 10)
            self.assertEqual(audit["tier_growth_wait_cycles"], 6)
            self.assertEqual(audit["tier_evicted_wait_cycles"], 3)
            self.assertEqual(audit["tier_minus_shared_slo_fraction"], -0.5)
            self.assertEqual(audit["growth_pressure_observed"], 1)
            self.assertEqual(summarize([audit])[0]["shared_wins_slo"], 1)
            (case / "tier.out").write_text(
                (case / "tier.out").read_text().replace("wait_cycles=3", "wait_cycles=2"))
            with self.assertRaisesRegex(ValueError, "invalid channel wait duration"):
                read_run(case / "tier.out", workload(csv_path))


if __name__ == "__main__":
    unittest.main()
