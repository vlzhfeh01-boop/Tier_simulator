#!/usr/bin/env python3
"""Export tier_workload requests to one CSV and paired Ramulator configurations."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tier_workload import generator
from tier_workload.spec import load_experiment

class YAML12Loader(yaml.SafeLoader):
    """Keep Ramulator implementation name 'No' as a string, not a YAML 1.1 bool."""

YAML12Loader.yaml_implicit_resolvers = {
    char: [(tag,pattern) for tag,pattern in resolvers
           if tag != "tag:yaml.org,2002:bool"]
    for char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
YAML12Loader.add_implicit_resolver("tag:yaml.org,2002:bool",
                                   re.compile(r"^(?:true|false)$", re.I),
                                   list("tTfF"))

def load_template(path:Path) -> dict:
     return yaml.load(path.read_text(), Loader=YAML12Loader)

def cycle_at_or_after(ns:int, tck_ps: int) -> int:
    if ns < 0 or tck_ps <= 0:
        raise ValueError("arrival time must be nonnegative and tCK positive")
    return (ns * 1000 + tck_ps -1) // tck_ps

def export_requests(requests, csv_path: Path, tck_ps: int) -> None:
    if not requests:
        raise ValueError("workload has no requests")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w",newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["# request_id", "arrival_cycle", "prompt_tokens",
                         "predicted_output_tokens","actual_output_tokens"])
        for request in requests:
            writer.writerow([request.request_id,
                             cycle_at_or_after(request.arrival_time_ns, tck_ps),
                             request.prompt_tokens, request.predicted_output_tokens,
                             request.actual_output_tokens])

def prepare_configs(template: Path, output: Path, stem: str, max_cycles: int) -> None:
    config = load_template(template)
    frontend = config["Frontend"]
    memory = config["MemorySystem"]
    if (frontend.get("impl") != "TierDecodeStep" or
            memory["DRAM"]["impl"] != "HBM3-PIM" or
            memory["DRAM"]["org"].get("channel") != 16):
        raise ValueError("template must target TierDecodeStep with 16-channel HBM3-PIM")
    if max_cycles <= 0:
        raise ValueError("max_cycles must be positive")
    for shared, policy in ((False, "tier"), (True, "shared")):
        cfg = load_template(template)
        cfg["Frontend"]["workload_path"] = f"./{stem}.csv"
        cfg["Frontend"]["shared_channels"] = shared
        cfg["Frontend"]["max_cycles"] = max_cycles
        plugins = cfg["MemorySystem"]["Controller"].get("plugins", [])
        for plugin in plugins:
            recorder = plugin.get("ControllerPlugin", {})
            if recorder.get("impl") == "HBM3TraceRecorder":
                recorder["path"] = f"./log/{stem}_{policy}/cmd.log"
        (output / f"{stem}_{policy}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw", default="hbm3_16ch")
    parser.add_argument("--model", default="qwen3_8b_pim")
    parser.add_argument("--workload", default="qwen3_policy_smoke")
    parser.add_argument("--requests-json", type=Path,
                        help="reuse an existing tier_workload generator JSON")
    parser.add_argument("--template", type=Path,
                        default=ROOT / "pim_ramulator_src/tier_qwen3_row_boundary_tier.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="tier_policy_workload")
    parser.add_argument("--tck-ps", type=int, required=True,
                            help="DRAM nCK period; HBM3_5.2Gbps preset here uses 1300 ps")
    parser.add_argument("--max-cycles",type=int, default=100_000_000)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rate-rps", type=float,
                            help="override Poisson arrival rate in requests/second")
    parser.add_argument("--batch", action="store_true", help="all arrivals at cycle 0")
    args = parser.parse_args()
    if args.name in ("",".","..") or "/" in args.name or "\\" in args.name:
        parser.error("--name must be a simple filename stem")
    if args.tck_ps <= 0 or args.max_cycles <= 0:
            parser.error("--tck-ps and --max-cycles must be positive")
    if args.batch and args.rate_rps is not None:
        parser.error("choose either --batch or --rate-rps")
    if args.requests_json and (args.batch or any(v is not None for v in
                                    (args.num_requests, args.seed, args.rate_rps))):
        parser.error("generator overrides cannot be used with --requests-json")
    try:
        exp = load_experiment(args.hw, args.model, args.workload)
        if (exp.runtime.total_channels != 16 or exp.runtime.num_layers != 36 or
                exp.runtime.num_kv_heads != 8 or exp.runtime.head_dim != 128 or
                exp.runtime.element_bytes != 2):
            raise ValueError("this fixture requires the 16-channel Qwen3-8B KV geometry")
        config = load_template(args.template)
        if (config["Frontend"].get("layers_on_hbm") != exp.runtime.num_layers or
                config["Frontend"].get("query_head_count") != 32 or
                config["Frontend"].get("kv_head_count") != exp.runtime.num_kv_heads or
                config["MemorySystem"]["DRAM"]["timing"].get("preset") != "HBM3_5.2Gbps"):
            raise ValueError("template must use Qwen3-8B and HBM3_5.2Gbps")
        if(args.tck_ps != 1300):
             raise ValueError("HBM3_5.2Gbps template specifies tCK_ps=1300")
        params = exp.workload
        updates = {}
        for name in ("num_requests", "seed"):
            value = getattr(args, name)
            if value is not None:
                updates[name] = value
        if args.batch:
            updates["rate_rps"] = None
        elif args.rate_rps is not None:
            updates["rate_rps"] = args.rate_rps
        params = replace(params, **updates)
        if params.num_requests <= 0 or (params.rate_rps is not None and
                                        (not math.isfinite(params.rate_rps) or
                                            params.rate_rps <= 0)):
            raise ValueError("request count and Poisson rate must be positive")
        requests = (generator.load_workload(args.requests_json) if args.requests_json
                    else generator.generate(params, exp.model.max_output_tokens))
        generator.validate(requests, exp.runtime)
        if any(r.prompt_tokens + r.actual_output_tokens > 40960 or
                r.prompt_tokens + r.predicted_output_tokens > 40960
                for r in requests):
            raise ValueError("request exceeds physical KV capacity or Qwen3 context")
        if max(cycle_at_or_after(r.arrival_time_ns, args.tck_ps)
                for r in requests) >= args.max_cycles:
            raise ValueError("latest arrival is at or after max_cycles")
        output = args.out_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        export_requests(requests, output / f"{args.name}.csv", args.tck_ps)
        prepare_configs(args.template, output, args.name, args.max_cycles)
    except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
    print(f"Prepared {len(requests)} requests in {output}")
    print("Run both YAML files from that directory, then compare the two outputs.")

if __name__ == "__main__":
    main()

            
