from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .spec import RequestSpec, RuntimeConfig, WorkloadParams


# ------------------------------------------------------- arrival_time_ns

def poisson_arrivals(rng, n: int, rate_rps: float) -> np.ndarray:
    """Poisson arrivals function"""
    gaps_ns = rng.exponential(1e9 / rate_rps, n)
    return np.cumsum(gaps_ns).astype(np.int64)


def batch_arrivals(n: int) -> np.ndarray:
    """static batch"""
    return np.zeros(n, dtype=np.int64)


# --------------------------------------------------------- prompt_tokens

def prompt_tokens(rng, mean: int, sigma: float, n: int,
                      lo: int = 1, hi: int | None = None) -> np.ndarray:
    """prompt_tokens, long_tail"""
    mu = np.log(mean) - sigma ** 2 / 2
    x = np.round(rng.lognormal(mu, sigma, n))
    return np.clip(x, lo, hi if hi is not None else 2 ** 40).astype(np.int64)


# ----------------------------------------------- predicted_output_tokens

def predict_output_tokens(rng, actuals: np.ndarray, bias: float,
                        sigma: float, cap: int) -> np.ndarray:
    """still trach value
    TODO: after actual token~"""
    noise = rng.normal(bias, sigma, len(actuals))
    return np.clip(np.round(actuals * np.exp(noise)), 1, cap).astype(np.int64)

# ----------------------------------------------- actual_output_tokens
def actual_output_tokens(rng, actuals: np.ndarray, bias: float,
                        sigma: float, cap: int) -> np.ndarray:
    """TODO: ~~"""
    return


# --------------------------------------------------------------- 조합

def generate(p: WorkloadParams, max_output: int) -> list[RequestSpec]:
    """generating requests list with workload params(rate_rps(lamda), long_tail(sigma,mean), actual output token)"""
    if max_output <= 0:
        raise ValueError("max_output_tokens 를 모델 config에 지정해야 한다")

    s_arr, s_pr, s_out, s_pred = np.random.SeedSequence(p.seed).spawn(4)
    n = p.num_requests

    # arrival_time : poisson 분포
    arrivals = (batch_arrivals(n) if p.rate_rps is None else
                poisson_arrivals(np.random.default_rng(s_arr), n, p.rate_rps))

    # prompt_tokens : long-tail 분포
    prompts = prompt_tokens(
        np.random.default_rng(s_pr), p.prompt_mean, p.prompt_sigma, n)

    # TODO:
    actuals = prompt_tokens(
        np.random.default_rng(s_out), p.output_mean, p.output_sigma, n,
        hi=max_output)

    preds = predict_output_tokens(
        np.random.default_rng(s_pred), actuals, p.pred_bias, p.pred_sigma,
        max_output)

    return [
        RequestSpec(
            request_id=i,
            arrival_time_ns=int(arrivals[i]),
            prompt_tokens=int(prompts[i]),
            predicted_output_tokens=int(preds[i]),
            actual_output_tokens=int(actuals[i]),
            max_output_tokens=max_output,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------- 검증

def validate(requests: list[RequestSpec], config: RuntimeConfig) -> None:
    """validation rule"""
    if config.total_channels <= 0:
        raise ValueError("total_channels > 0")

    # prevent request_ids duplication
    ids = [r.request_id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("request_id is duplicated")

    prev_key = (-1, -1)
    for r in requests:
        if r.prompt_tokens <= 0:
            raise ValueError(f"req {r.request_id}: prompt_tokens > 0")
        if r.actual_output_tokens > r.max_output_tokens:
            raise ValueError(f"req {r.request_id}: actual > max")
        if r.predicted_output_tokens > r.max_output_tokens:
            raise ValueError(f"req {r.request_id}: predicted > max")

        # validate that arrival time and request id is ascending 
        key = (r.arrival_time_ns, r.request_id)
        if key < prev_key:
            raise ValueError(f"req {r.request_id}: arrival sorting error")
        prev_key = key

def save_workload(requests: list[RequestSpec], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in requests], indent=1))


def load_workload(path: str | Path) -> list[RequestSpec]:
    return [RequestSpec(**d) for d in json.loads(Path(path).read_text())]
