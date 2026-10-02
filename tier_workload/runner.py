from __future__ import annotations

from dataclasses import asdict

from .spec import RequestSpec, RuntimeConfig

USE_MOCK = True

def run_matrix(policies: list[tuple[str, RuntimeConfig]],
               requests: list[RequestSpec],
               tags: dict | None = None) -> list[dict]:
    rows = []
    for name, cfg in policies:
        result = _call_cpp(cfg, requests)
        rows.append({"policy": name, **(tags or {}), **result})
    return rows

def _call_cpp(config: RuntimeConfig, requests: list[RequestSpec]) -> dict:
    """tier_runtime binding fucntion"""
    if USE_MOCK:
        return _mock_result(config, requests)

    import tier_runtime

    fields = asdict(config)
    # enum 둘만 C++ 타입으로 교체. 나머지 int/bool은 그대로 대입된다
    fields["allocation_mode"] = getattr(tier_runtime.AllocationMode,
                                        config.allocation_mode.name)
    fields["planning_mode"] = getattr(tier_runtime.TierPlanningMode,
                                      config.planning_mode.name)

    cfg = _fill(tier_runtime.RuntimeConfig(), fields)
    reqs = [_fill(tier_runtime.RequestSpec(), asdict(r)) for r in requests]

    return _flatten_result(tier_runtime.run_simulation(cfg, reqs))


def _fill(obj, fields: dict):
    for key, value in fields.items():
        setattr(obj, key, value)
    return obj


def _flatten_result(result) -> dict:
    """TODO: tier_runtime result 스키마 적용"""
    raise NotImplementedError("SimulationResult 스키마 미확정")


def _mock_result(config: RuntimeConfig, requests: list[RequestSpec]) -> dict:
    return {
        "mock": True,
        "completed_requests": len(requests),
        "total_cycles": 0,
        "growth_events": 0,
        "eviction_events": 0,
    }
