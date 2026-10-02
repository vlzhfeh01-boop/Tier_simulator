from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import pandas as pd

from tier_workload import analyze, generator
from tier_workload.runner import run_matrix
from tier_workload.spec import (AllocationMode, RuntimeConfig,
                                TierPlanningMode, load_experiment)

# default policies
DEFAULT_POLICIES = [
    (AllocationMode.TCP, None),
    (AllocationMode.Tier, TierPlanningMode.Oracle),
    (AllocationMode.Tier, TierPlanningMode.MaxOutput),
    (AllocationMode.Tier, TierPlanningMode.Predicted),
]


def policy_label(alloc: AllocationMode,
                 planning: TierPlanningMode | None) -> str:
    return "TCP" if alloc is AllocationMode.TCP else f"Tier/{planning.name}"


def resolve_policies(args, base: RuntimeConfig) -> list[tuple[str, RuntimeConfig]]:
    if args.alloc is None:
        combos = DEFAULT_POLICIES
    else:
        alloc = AllocationMode(args.alloc)
        planning = TierPlanningMode(args.planning) if args.planning else None
        combos = [(alloc, planning)]

    return [
        (policy_label(alloc, planning),
         replace(base, allocation_mode=alloc,
                 planning_mode=planning or base.planning_mode,
                 eviction_enabled=True))
        for alloc, planning in combos
    ]


# ------------------------------------------------------------------ sweep

def cmd_sweep(args) -> None:
    exp = load_experiment(args.hw, args.model, args.workload)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv

    # CLI 인자로 workload yaml 파일 덮어쓰기
    params = exp.workload
    for field in ("seed", "pred_sigma", "pred_bias", "num_requests"):
        value = getattr(args, field)
        if value is not None:
            params = replace(params, **{field: value})

    # static batch는 rate_rps = None
    rates = [None] if args.batch else [float(x) for x in args.lam.split(",")]

    for rate in rates:
        p = replace(params, rate_rps=rate)

        requests = generator.generate(p, exp.model.max_output_tokens)
        generator.validate(requests, exp.runtime)

        tag = "batch" if rate is None else f"lam{rate}"
        wl_path = out_dir / f"wl_{args.model}_{args.workload}_seed{p.seed}_{tag}.json"
        generator.save_workload(requests, wl_path)


        rows = run_matrix(resolve_policies(args, exp.runtime), requests,
                          tags={"hw": args.hw, "model": args.model,
                                "workload": args.workload,
                                "rate_rps": rate, "seed": p.seed,
                                "pred_sigma": p.pred_sigma,
                                "pred_bias": p.pred_bias,
                                "num_requests": p.num_requests})
        #analyze.append_csv(rows, csv_path)
        csv_path = Path(csv_path)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, mode="w", header=not csv_path.exists(), index=False)
        
        print(f"{tag:<10} requests={len(requests):<5} -> {wl_path.name}")

    print(f"\n결과: {csv_path}")


# ------------------------------------------------------------------- plot

def cmd_plot(args) -> None:
    out = args.out or str(Path(args.input).with_suffix(".png"))
    analyze.plot_crossover(args.input, out)
    print(f"저장: {out}")


# -------------------------------------------------------------------- CLI

def _add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hw", required=True, help="configs/hw/<이름>.yaml")
    p.add_argument("--model", required=True, help="configs/model/<이름>.yaml")
    p.add_argument("--workload", required=True,
                   help="configs/workload/<이름>.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="워크로드 생성 + 정책 실행")
    _add_target_args(sw)
    sw.add_argument("--lam", default="0.5,1,2,4,8,16",
                    help="도착률 목록(req/s), 쉼표 구분")
    sw.add_argument("--batch", action="store_true",
                    help="static batch 모드(전부 0시각 도착). --lam 무시")
    sw.add_argument("--alloc", choices=["Tier", "TCP"],
                    help="생략하면 기본 비교 세트 전체 실행")
    sw.add_argument("--planning", choices=["Predicted", "Oracle", "MaxOutput"],
                    help="--alloc Tier 일 때 필수")
    sw.add_argument("--seed", type=int)
    sw.add_argument("--pred-sigma", type=float, dest="pred_sigma")
    sw.add_argument("--pred-bias", type=float, dest="pred_bias")
    sw.add_argument("--num-requests", type=int, dest="num_requests")
    sw.add_argument("--out-dir", default="experiments", dest="out_dir")
    sw.add_argument("--csv", default="results.csv")
    sw.set_defaults(func=cmd_sweep)

    pl = sub.add_parser("plot", help="results.csv -> 교차점 그래프")
    pl.add_argument("-i", "--input", required=True)
    pl.add_argument("-o", "--out")
    pl.set_defaults(func=cmd_plot)

    args = parser.parse_args()

    if args.cmd == "sweep":
        if args.alloc == "Tier" and not args.planning:
            parser.error("--alloc Tier 는 --planning 이 필요하다")
        if args.alloc != "Tier" and args.planning:
            parser.error("--planning 은 --alloc Tier 에서만 쓴다")

    args.func(args)


if __name__ == "__main__":
    main()
