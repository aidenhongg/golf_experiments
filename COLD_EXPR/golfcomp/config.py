from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import yaml
import copy


@dataclass
class ModelConfig:
    arch: str = "transformer"
    num_layers: int = 11
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 3
    vocab_size: int = 8192
    seq_len: int = 1024
    activation: str = "leaky_relu_sq"
    use_xsa: bool = True
    xsa_mode: str = "all"
    use_parallel_residuals: bool = True
    parallel_start_layer: int = 7
    use_skip_gates: bool = True
    qk_gain: float = 5.0
    logit_softcap: float = 30.0
    rope_partial_dim: int = 16
    tie_embeddings: bool = True
    embedding_type: str = "standard"
    use_smear_gate: bool = True
    bigram_hash_buckets: int = 3072
    bigram_hash_dim: int = 128
    use_engramlite: bool = False
    engramlite_orders: tuple = (2, 3, 4)
    engramlite_buckets_per_head: int = 2048
    recurrence_type: str = "depth"
    recurrence_layers: list = field(default_factory=lambda: [3, 4, 5])
    recurrence_start_step: int = 2000
    basis_rank: int = 64
    num_shared_bases: int = 32
    lora_rank: int = 8
    num_shared_blocks: int = 7
    num_loops: int = 2
    gla_num_layers: int = 8
    gla_expand_ratio: int = 1
    mamba_num_layers: int = 8
    mamba_d_state: int = 32
    xlstm_num_layers: int = 8
    rwkv_num_layers: int = 11
    factorized_embed_dim: int = 64
    num_hash_tables: int = 3


@dataclass
class TrainingConfig:
    max_steps: int = 4000
    max_time_seconds: int = 1200
    batch_tokens: int = 128_000
    micro_batch_tokens: int = 32_768
    grad_accum_steps: int = 4
    matrix_lr: float = 0.022
    embed_lr: float = 0.035
    scalar_lr: float = 0.025
    gate_lr: float = 0.01
    weight_decay: float = 0.095
    grad_clip: float = 0.3
    optimizer: str = "muon"
    warmup_steps: int = 200
    warmdown_frac: float = 0.72
    cooldown_shape: str = "1_sqrt"
    momentum_warmup: tuple = (0.92, 0.99, 1500)
    use_ema: bool = True
    ema_decay: float = 0.9965
    use_qat: bool = True
    qat_start_lr_scale: float = 0.15
    qat_bits: int = 6
    data_path: str = "./data/fineweb_sp8192/"
    val_data_path: str = ""
    tokenizer_path: str = "./data/tokenizers/fineweb_8192_bpe.model"
    log_interval_seconds: float = 30.0
    pre_quant_ttt: bool = False
    pre_quant_ttt_epochs: int = 10
    pre_quant_ttt_lr: float = 0.00045
    pre_quant_ttt_freeze_blocks: int = 1
    pre_quant_ttt_seq_len: int = 1024


@dataclass
class QuantConfig:
    method: str = "sdclip"
    int6_k: float = 12.85
    int8_embed_k: float = 20.0
    use_gptq: bool = True
    gptq_actorder: bool = True
    embed_bits: int = 8
    body_bits: int = 6
    mlp_bits: int = 6
    attn_bits: int = 6


@dataclass
class EvalConfig:
    window_size: int = 2048
    stride: int = 64
    use_ttt: bool = False
    ttt_type: str = "lora"
    ttt_epochs: int = 3
    ttt_lr: float = 0.005
    ttt_optimizer: str = "adamw_cosine"
    ttt_lora_rank: int = 8


@dataclass
class CompressionConfig:
    method: str = "brotli"


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    category: str = "baseline"
    compare_to: str = "baseline"
    seeds: list = field(default_factory=lambda: [42])


_SUB_CONFIGS = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "quant": QuantConfig,
    "eval": EvalConfig,
    "compression": CompressionConfig,
}


def _dict_to_config(d: dict) -> ExperimentConfig:
    kwargs = {}
    for k, v in d.items():
        if k in _SUB_CONFIGS and isinstance(v, dict):
            kwargs[k] = _SUB_CONFIGS[k](**v)
        else:
            kwargs[k] = v
    return ExperimentConfig(**kwargs)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> ExperimentConfig:
    p = Path(path)
    with open(p) as f:
        d = yaml.safe_load(f) or {}
    if "_base" in d:
        base_rel = d.pop("_base")
        base_path = p.parent / base_rel
        if not base_path.exists():
            base_path = Path(base_rel)
        with open(base_path) as f:
            base_d = yaml.safe_load(f) or {}
        d = _deep_merge(base_d, d)
    return _dict_to_config(d)


def set_nested(config: ExperimentConfig, dotted_key: str, value):
    parts = dotted_key.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
