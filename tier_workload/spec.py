from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class AllocationMode(Enum):
    Tier = "Tier"
    TCP = "TCP"


class TierPlanningMode(Enum):
    Predicted = "Predicted"    
    Oracle = "Oracle"          
    MaxOutput = "MaxOutput"    


@dataclass(frozen=True)
class RuntimeConfig:

    # Hardware
    total_channels: int = 0
    pseudochannels_per_channel: int = 0
    ranks_per_pseudochannel: int = 0
    bank_groups_per_rank: int = 0
    banks_per_bank_group: int = 0
    rows_per_bank: int = 0
    columns_per_row: int = 0
    bytes_per_column: int = 0

    # Model shape
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    element_bytes: int = 0
    tokens_per_block: int = 0

    # Runtime policy
    allocation_mode: AllocationMode = AllocationMode.Tier #default
    planning_mode: TierPlanningMode = TierPlanningMode.Predicted #default
    eviction_enabled: bool = False


@dataclass(frozen=True)
class RequestSpec:

    request_id: int
    arrival_time_ns: int #poisson
    prompt_tokens: int #long-tail
    predicted_output_tokens: int # noise mul
    actual_output_tokens: int #TODO: 
    max_output_tokens: int #model dependency


@dataclass(frozen=True)
class ModelExtras:
    """max_output_tokens 따로처리, requestspec이지 runtimeconfig가 아니라..."""
    max_output_tokens: int = 0


@dataclass(frozen=True)
class WorkloadParams:
    num_requests: int = 64
    seed: int = 0

    # None -> static batch, not none -> Poisson arrival
    rate_rps: float | None = 2.0

    # prompt_tokens 의 long-tail 분포
    prompt_mean: int = 4096
    prompt_sigma: float = 0.8

    # actual_output_tokens 용 임시 파라미터 TODO
    output_mean: int = 2048
    output_sigma: float = 0.8

    pred_bias: float = 0.0     
    pred_sigma: float = 0.3    


@dataclass(frozen=True)
class ExperimentConfig:
    runtime: RuntimeConfig
    model: ModelExtras
    workload: WorkloadParams


# ------------------------------------------------------------ Config parsing
CONFIG_ROOT = Path("configs")


def resolve(kind: str, name: str) -> Path:
    """'hbm3_16ch' -> configs/hw/hbm3_16ch.yaml. 경로를 직접 줘도 된다."""
    p = Path(name)
    if p.suffix in (".yaml", ".yml"):
        return p
    return CONFIG_ROOT / kind / f"{name}.yaml"


def load_runtime(hw: str, model: str) -> tuple[RuntimeConfig, ModelExtras]:
    """hw yaml + model yaml -> (RuntimeConfig, ModelExtras)."""

    hw_raw = yaml.safe_load(resolve("hw", hw).read_text()) or {}
    model_raw = yaml.safe_load(resolve("model", model).read_text()) or {}

    overlap = set(hw_raw) & set(model_raw)
    if overlap:
        raise ValueError(f"하드웨어와 모델 config에 같은 키가 있다: {overlap}")

    #model yaml에 max_output_tokens 있지만 binding과 달라서 따로 extra
    extra_keys = {f for f in ModelExtras.__dataclass_fields__}
    extras = {k: model_raw.pop(k) for k in list(model_raw) if k in extra_keys} 

    hw_raw["tokens_per_block"] = (hw_raw["pseudochannels_per_channel"]
                                  * hw_raw["ranks_per_pseudochannel"]
                                  * hw_raw["bank_groups_per_rank"])

    return RuntimeConfig(**hw_raw, **model_raw), ModelExtras(**extras)


def load_workload_params(name: str) -> WorkloadParams:
    raw = yaml.safe_load(resolve("workload", name).read_text()) or {}
    return WorkloadParams(**raw)


def load_experiment(hw: str, model: str, workload: str) -> ExperimentConfig:
    """configs/*/*.yaml file parsing"""
    runtime, extras = load_runtime(hw, model)
    return ExperimentConfig(runtime=runtime, model=extras,
                            workload=load_workload_params(workload))
