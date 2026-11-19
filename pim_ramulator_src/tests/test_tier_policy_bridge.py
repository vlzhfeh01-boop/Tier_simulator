"""Boundary and terminal-state checks for paired policy reporting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_tier_policy_logs import metrics, read_run, workload
from prepare_tier_policy_workload import (cycle_at_or_after, load_template,
                                          prepare_configs, validate_context)
from run_tier_policy_sweep import rate_label


class PolicyBridgeTest(unittest.TestCase):
    def test_prediction_beyond_context_is_not_a_replay_rejection(self):
        from types import SimpleNamespace
        request = SimpleNamespace(request_id=7, prompt_tokens=40955,
                                  predicted_output_tokens=20,
                                  actual_output_tokens=2)
        validate_context([request])
        request.actual_output_tokens = 6
        with self.assertRaisesRegex(ValueError, "request_id=7 exceeds Qwen3 context"):
            validate_context([request])

    def test_comparison_disallows_artificial_capacity_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = Path(__file__).resolve().parents[1] / "tier_qwen3_row_boundary_tier.yaml"
            config = load_template(template)
            config["Frontend"]["shared_row_slot_limit"] = 1
            import yaml
            (root / "limited.yaml").write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "artificial KV capacity limits"):
                prepare_configs(root / "limited.yaml", root, "limited", 100000)

    def test_sweep_rate_labels_do_not_depend_on_float_display_style(self):
        self.assertEqual(rate_label(1000000.0), "1000000")
        self.assertEqual(rate_label(0.5), "0p5")

    def test_hbm_refresh_manager_name_remains_string(self):
        template = Path(__file__).resolve().parents[1] / "tier_qwen3_row_boundary_tier.yaml"
        cfg = load_template(template)
        self.assertEqual(cfg["MemorySystem"]["Controller"]["RefreshManager"]["impl"], "No")

    def test_ns_never_rounds_arrival_early(self):
        self.assertEqual(cycle_at_or_after(0, 769), 0)
        self.assertEqual(cycle_at_or_after(1, 769), 2)
        self.assertEqual(cycle_at_or_after(769, 769), 1000)
        self.assertEqual(cycle_at_or_after(770, 769), 1002)

    def test_eviction_is_not_counted_as_goodput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workload.csv").write_text(
                "# request_id,arrival_cycle,prompt,predicted,actual\n"
                "1,0,33,2,2\n2,0,33,2,2\n")
            (root / "run.out").write_text(
                "TIER_ADMIT request_id=1 cycle=1 channels=1\n"
                "TIER_ADMIT request_id=2 cycle=2 channels=1\n"
                "TIER_CHANNEL_GROW request_id=1 channel=2 channels=2 token_index=34 cycle=11\n"
                "TIER_TOKEN request_id=1 token=1 completion_cycle=10\n"
                "TIER_TOKEN request_id=1 token=2 completion_cycle=20\n"
                "TIER_REQUEST request_id=1 arrival_cycle=0 admitted_cycle=1 "
                "first_token_cycle=10 completion_cycle=20\n"
                "TIER_TOKEN request_id=2 token=1 completion_cycle=9\n"
                "TIER_EVICT request_id=2 cycle=12 released_units=1 "
                "generated_tokens=1 reason=DEADLOCK\n"
                "TIER_SUMMARY total_requests=2 completed=1 evicted=1 "
                "final_cycle=20\n")
            requests = workload(root / "workload.csv")
            run = read_run(root / "run.out", requests)
            report = metrics(run, requests, 769, 2, 10)
            self.assertEqual(report["slo_completed"], 1)
            self.assertEqual(report["evicted_ids"], [2])
            self.assertEqual(report["inter_token_cycles"]["p95"], 10)
            self.assertEqual(report["admission_wait_cycles"]["p95"], 2)
            self.assertEqual(report["channel_growth"], 1)
            strict_first = metrics(run, requests, 769, 2, 8)
            self.assertEqual(strict_first["slo_completed"], 0)
            (root / "run.out").write_text(
                (root / "run.out").read_text().replace(
                    "completed=1 evicted=1", "completed=2 evicted=0"))
            with self.assertRaisesRegex(ValueError, "summary does not match"):
                read_run(root / "run.out", requests)


if __name__ == "__main__":
    unittest.main()
