from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def append_csv(rows: list[dict], path: str | Path) -> None:
    """sweep이 길어서 중간에 죽어도 지금까지 결과는 남도록 append."""
    path = Path(path)
    df = to_dataframe(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def plot_crossover(csv_path: str | Path, out_path: str | Path) -> None:
    """주 결과 그림: x=도착률, y=처리량, 계열=정책.

    낮은 부하에서는 TCP(전체 채널을 한 요청에 몰아줌)가 유리하고,
    높은 부하에서는 Tier(요청별 채널에서 병렬 실행)가 유리해지는
    교차점이 나오는지를 본다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy, g in df.groupby("policy"):
        g = g.sort_values("rate_rps")
        ax.plot(g["rate_rps"], g["throughput"], marker="o", label=policy)
    ax.set_xlabel("arrival rate (req/s)")
    ax.set_ylabel("throughput (req/s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
