# Parameter Golf Cold Experiments — Implementation Plan

**Date:** 2026-04-08
**Target:** 21 cold-start experiments on single RTX 5090 (32GB GDDR7)
**Goal:** Relative ranking of architectures and techniques to identify what to promote to hot (8xH100)

## Context

We're competing in OpenAI's Parameter Golf: train the best LM in 16MB, 10 min on 8xH100, scored by BPB on FineWeb. Current SOTA is 1.0822 BPB (PR #1477). We have 21 cold experiments defined in `experiments/cold-exp.md` that screen techniques on a single 5090 before burning expensive H100 time. This plan implements the full experiment suite as a modular Python package.

## Decisions Made

- **Structure:** Modular Python package (`golfcomp/`)
- **HP Optimization:** Optuna (Bayesian w/ pruning for C1-C5, GridSampler for C11-C14)
- **Alt architectures:** Libraries first (FLA, mamba-ssm), custom where needed
- **Evaluation:** Full competition-accurate BPB evaluator from the start
- **FA version:** FA2 on 5090 as specified
- **Novel techniques:** Correct implementations from papers (Basis Sharing, Relaxed Recursive)
- **C14 scope:** Warmdown sweep only (SP16384 content was copy-paste error; that's C21)
- **Mamba-3 (C2, C5):** Implement Mamba-3 deltas on top of Mamba-2 base (complex gates, trapezoidal discretization, MIMO). mamba-ssm likely only has Mamba-2.
- **Basis Sharing + GPTQ (C6):** Reconstruct per-layer weights, then quantize normally. Reliable GPTQ quality. Optimize artifact size later if results are promising.
- **xLSTM (C3):** Pure PyTorch mLSTM from paper. Log that speed numbers need Triton kernels for accuracy. Sample efficiency measured is architecture-independent of kernel speed.
- **Compute budget:** ~12 hours total (up from 8 due to Optuna). Run overnight on 5090.

---

## 1. Project Structure

```
golfcomp/
├── pyproject.toml                 # Package config, dependencies
├── configs/                       # Experiment YAML configs
│   ├── baseline.yaml
│   ├── c01_gla_hybrid.yaml
│   ├── c02_mamba3_hybrid.yaml
│   ├── c03_xlstm_hybrid.yaml
│   ├── c04_rwkv7.yaml
│   ├── c05_mixed_hybrid.yaml
│   ├── c06_basis_sharing.yaml
│   ├── c07_relaxed_recursive.yaml
│   ├── c08_engramlite.yaml
│   ├── c09_swiglu.yaml
│   ├── c10_no_parallel_res.yaml
│   ├── c11_ema_sweep.yaml
│   ├── c12_wd_sweep.yaml
│   ├── c13_recurrence_sweep.yaml
│   ├── c14_warmdown_sweep.yaml
│   ├── c15_sdclip_sweep.yaml
│   ├── c16_mixed_precision.yaml
│   ├── c17_compressor_sweep.yaml
│   ├── c18_ttt_sweep.yaml
│   ├── c19_slot_vs_lora.yaml
│   ├── c20_sliding_window.yaml
│   └── c21_sp16384.yaml
├── golfcomp/
│   ├── __init__.py
│   ├── config.py                  # ExperimentConfig dataclass + YAML loader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseModel ABC
│   │   ├── transformer.py         # TransformerModel (baseline + C6-C14, C21)
│   │   ├── gla_hybrid.py          # GLAHybridModel (C1)
│   │   ├── mamba_hybrid.py        # MambaHybridModel (C2)
│   │   ├── xlstm_hybrid.py        # XLSTMHybridModel (C3)
│   │   ├── rwkv.py                # RWKVModel (C4)
│   │   ├── mixed_hybrid.py        # MixedHybridModel (C5)
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── attention.py       # GQA, XSA, QK-Norm, QK-Gain, LN Scale
│   │       ├── embeddings.py      # TokenEmbed, SmearGate, BigramHash, EngramLite, Factorized
│   │       ├── activations.py     # LeakyReLUSq, SwiGLU
│   │       ├── recurrence.py      # DepthRecurrence, BasisSharing, RelaxedRecursive
│   │       ├── residuals.py       # ParallelResidual, SkipGate, SerialResidual
│   │       └── position.py        # PartialRoPE
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Trainer class (main loop)
│   │   ├── optimizers.py          # Muon, MuonEqR, build_optimizer()
│   │   ├── schedulers.py          # WarmdownScheduler, 1-sqrt cooldown
│   │   ├── ema.py                 # EMAWrapper
│   │   └── data.py                # FineWebDataset, TokenStream
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── sdclip.py              # SDClip quantizer
│   │   ├── gptq.py                # Full GPTQ + Cholesky + actorder
│   │   ├── mixed.py               # MixedPrecisionQuantizer (int5/6/8 per component)
│   │   └── compression.py         # Brotli, zstd-22, LZMA-9, ANS/Huffman
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # BPBEvaluator (sliding window, SP introspection)
│   │   ├── ttt.py                 # LoRATTT, SLOTTTT, ScoreFirstTTT
│   │   └── metrics.py             # LossTracker, PowerLawFitter
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── runner.py              # ExperimentRunner (orchestrates train → quant → eval)
│   │   ├── optuna_search.py       # OptunaSearcher, search spaces per experiment
│   │   └── analysis.py            # ResultsAnalyzer, comparison tables, plots
│   └── utils/
│       ├── __init__.py
│       ├── artifact.py            # ArtifactPacker (serialize model to ≤16MB)
│       ├── logging.py             # CSVLogger, loss curve writer
│       └── seed.py                # set_seed(), reproducibility
├── scripts/
│   ├── run_experiment.py          # CLI: python scripts/run_experiment.py --config configs/c01.yaml
│   ├── run_sweep.py               # CLI: python scripts/run_sweep.py --config configs/c11.yaml
│   ├── run_optuna.py              # CLI: python scripts/run_optuna.py --config configs/c01.yaml --trials 5
│   ├── run_all_cold.py            # Orchestrate all 21 experiments in correct order
│   ├── train_tokenizer.py         # Train SP16384 tokenizer for C21
│   └── analyze_results.py         # Generate comparison tables + plots
├── results/                       # Auto-populated by experiment runs
│   ├── baseline/
│   ├── c01_gla_hybrid/
│   └── ...
└── checkpoints/                   # Model checkpoints for shared experiments
    └── baseline/                  # Reused by C15-C17, C18-C20
```

---

## 2. Core Infrastructure

### 2.1 `golfcomp/config.py` — ExperimentConfig

Central configuration dataclass that every module reads from. Loaded from YAML.

```python
@dataclass
class ModelConfig:
    arch: str = "transformer"          # transformer|gla_hybrid|mamba_hybrid|xlstm_hybrid|rwkv|mixed_hybrid
    num_layers: int = 11
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4              # GQA
    mlp_mult: int = 3                  # hidden = 1536
    vocab_size: int = 8192
    seq_len: int = 1024                # training seq len (eval may differ)
    activation: str = "leaky_relu_sq"  # leaky_relu_sq | swiglu
    use_xsa: bool = True
    xsa_mode: str = "all"              # all | last_N
    use_parallel_residuals: bool = True
    parallel_start_layer: int = 7
    use_skip_gates: bool = True
    qk_gain: float = 5.0
    logit_softcap: float = 30.0
    rope_partial_dim: int = 16
    tie_embeddings: bool = True
    # Embedding config
    embedding_type: str = "standard"   # standard | factorized | multi_hash | factorized_multi_hash
    use_smear_gate: bool = True
    bigram_hash_buckets: int = 3072
    bigram_hash_dim: int = 128
    use_engramlite: bool = False
    # Recurrence config
    recurrence_type: str = "depth"     # depth | basis_sharing | relaxed_recursive | none
    recurrence_layers: list = field(default_factory=lambda: [3, 4, 5])
    recurrence_start_step: int = 2000
    # Basis sharing (C6)
    basis_rank: int = 64
    num_shared_bases: int = 32
    # Relaxed recursive (C7)
    lora_rank: int = 8
    num_shared_blocks: int = 6
    num_loops: int = 2
    # GLA-specific (C1)
    gla_num_layers: int = 8
    gla_expand_ratio: int = 1
    # Mamba-specific (C2)
    mamba_num_layers: int = 8
    mamba_d_state: int = 32
    # xLSTM-specific (C3)
    xlstm_num_layers: int = 8
    # RWKV-specific (C4)
    rwkv_num_layers: int = 11
    # Factorized embedding (C21)
    factorized_embed_dim: int = 64
    num_hash_tables: int = 3

@dataclass
class TrainingConfig:
    max_steps: int = 4000
    max_time_seconds: int = 1200       # 20 min default for cold
    batch_tokens: int = 128_000        # effective batch via grad accum
    micro_batch_tokens: int = 32_768   # per-GPU micro batch
    grad_accum_steps: int = 4          # 32K * 4 = 128K
    # Optimizer
    matrix_lr: float = 0.022
    embed_lr: float = 0.035
    scalar_lr: float = 0.025
    weight_decay: float = 0.095
    grad_clip: float = 0.3
    optimizer: str = "muon"            # muon | muoneq_r | adamw
    # Schedule
    warmup_steps: int = 200
    warmdown_frac: float = 0.72
    cooldown_shape: str = "1_sqrt"     # linear | cosine | 1_sqrt
    momentum_warmup: tuple = (0.92, 0.99, 1500)
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9965
    # QAT
    use_qat: bool = True
    qat_start_lr_scale: float = 0.15
    qat_bits: int = 6
    # Data
    data_path: str = "./data/fineweb_sp8192/"
    tokenizer_path: str = "./data/tokenizers/fineweb_8192_bpe.model"
    # Logging
    log_interval_seconds: float = 30.0

@dataclass
class QuantConfig:
    method: str = "sdclip"             # sdclip | gptq | sdclip+gptq
    int6_k: float = 12.85
    int8_embed_k: float = 20.0
    use_gptq: bool = True
    gptq_actorder: bool = True
    embed_bits: int = 8
    body_bits: int = 6
    # Mixed precision (C16)
    mlp_bits: int = 6                  # 5 for int5 MLP experiment
    attn_bits: int = 6                 # 8 for int8 attention experiment

@dataclass
class EvalConfig:
    window_size: int = 2048
    stride: int = 64
    use_ttt: bool = False
    ttt_type: str = "lora"             # lora | slot | both
    ttt_epochs: int = 3
    ttt_lr: float = 0.005
    ttt_optimizer: str = "adamw_cosine"
    ttt_lora_rank: int = 8

@dataclass
class CompressionConfig:
    method: str = "brotli"             # brotli | zstd22 | lzma9 | ans_huffman

@dataclass
class ExperimentConfig:
    name: str = "baseline"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    # Experiment metadata
    category: str = "baseline"         # baseline|architecture|ablation|hp_sweep|quantization|eval_time|vocab
    compare_to: str = "baseline"       # which experiment to compare against
    seeds: list = field(default_factory=lambda: [42])
```

### 2.2 `golfcomp/training/data.py` — Data Pipeline

```python
class FineWebDataset(IterableDataset):
    """Streams tokenized FineWeb shards. Supports SP8192 and SP16384."""
    def __init__(self, data_path: str, seq_len: int, seed: int): ...
    def __iter__(self) -> Iterator[dict]:
        """Yields {"input_ids": LongTensor[seq_len], "labels": LongTensor[seq_len]}"""

class TokenStream:
    """Wraps FineWebDataset with gradient accumulation batching."""
    def __init__(self, dataset, micro_batch_tokens, grad_accum_steps): ...
    def __iter__(self) -> Iterator[dict]:
        """Yields accumulated batches of shape [accum_steps, micro_batch_seq, seq_len]"""
```

**Data prep** (one-time, before experiments):
```bash
# Download and tokenize FineWeb for SP8192
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
```

For C21 (SP16384), run `scripts/train_tokenizer.py` to train a new tokenizer:
```python
# scripts/train_tokenizer.py
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="data/fineweb_raw.txt",
    model_prefix="data/tokenizers/fineweb_16384_bpe",
    vocab_size=16384,
    model_type="bpe",
    character_coverage=0.9995,
)
# Then re-tokenize FineWeb with the new tokenizer
```

### 2.3 `golfcomp/training/optimizers.py` — Optimizer Suite

```python
class Muon(Optimizer):
    """SGD + Nesterov momentum + Newton-Schulz orthogonalization.
    
    For matrix-shaped parameters only. Scalars/embeddings use AdamW.
    Momentum warmup: 0.92 → 0.99 over 1500 steps.
    """
    def __init__(self, params, lr=0.022, momentum=0.99, nesterov=True): ...
    def step(self):
        """Apply Newton-Schulz orthogonalization to gradient, then SGD+Nesterov."""
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                # Newton-Schulz iteration (5 steps)
                X = grad.view(grad.shape[0], -1)  # reshape to 2D
                X = X / (X.norm() + 1e-7)
                for _ in range(5):
                    X = 1.5 * X - 0.5 * X @ X.T @ X
                # Apply Nesterov momentum + update
                ...

class MuonEqR(Muon):
    """Equalized variant of Muon (PR #1334)."""
    ...

def build_optimizer(model: nn.Module, config: TrainingConfig) -> tuple[Optimizer, Optimizer]:
    """Returns (matrix_optimizer, scalar_optimizer).
    
    Splits parameters:
    - Matrix params (2D, ≥ 256 elements): Muon/MuonEqR at matrix_lr
    - Embedding params: AdamW at embed_lr
    - Scalar/bias/norm params: AdamW at scalar_lr
    """
    matrix_params, embed_params, scalar_params = [], [], []
    for name, param in model.named_parameters():
        if "embedding" in name or "embed" in name:
            embed_params.append(param)
        elif param.ndim >= 2 and param.numel() >= 256:
            matrix_params.append(param)
        else:
            scalar_params.append(param)
    
    muon = Muon(matrix_params, lr=config.matrix_lr, ...)
    adamw = AdamW([
        {"params": embed_params, "lr": config.embed_lr},
        {"params": scalar_params, "lr": config.scalar_lr},
    ], weight_decay=config.weight_decay)
    return muon, adamw
```

**Mixed optimizer for hybrid architectures (C1-C5):**
```python
def build_hybrid_optimizer(model, config):
    """Splits params 3 ways: Muon for projections, AdamW for gates/SSM, AdamW for scalars."""
    projection_params = []   # Linear projections → Muon
    gate_ssm_params = []     # GLA gates / SSM A,B,C / mLSTM memory → AdamW
    scalar_params = []       # Norms, biases → AdamW
    
    for name, param in model.named_parameters():
        if any(k in name for k in ["gate", "ssm", "dt", "A_", "B_", "C_", "memory"]):
            gate_ssm_params.append(param)
        elif param.ndim >= 2 and param.numel() >= 256:
            projection_params.append(param)
        else:
            scalar_params.append(param)
    
    return (
        Muon(projection_params, lr=config.matrix_lr),
        AdamW([
            {"params": gate_ssm_params, "lr": config.gate_lr},   # from Optuna
            {"params": scalar_params, "lr": config.scalar_lr},
        ], weight_decay=config.weight_decay)
    )
```

### 2.4 `golfcomp/training/schedulers.py` — LR Scheduling

```python
class WarmdownScheduler:
    """Warmup → constant → warmdown.
    
    warmdown_frac: fraction of total steps spent in warmdown (default 0.72)
    cooldown_shape: "linear" | "cosine" | "1_sqrt"
    """
    def __init__(self, optimizer, warmup_steps, total_steps, warmdown_frac, shape): ...
    
    def get_lr_scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        warmdown_start = self.total_steps * (1 - self.warmdown_frac)
        if step < warmdown_start:
            return 1.0
        t = (step - warmdown_start) / (self.total_steps - warmdown_start)
        if self.shape == "1_sqrt":
            return 1 - math.sqrt(t)
        elif self.shape == "cosine":
            return 0.5 * (1 + math.cos(math.pi * t))
        else:  # linear
            return 1 - t

class MomentumWarmup:
    """Warms momentum from 0.92 → 0.99 over 1500 steps for Muon."""
    def __init__(self, optimizer, start=0.92, end=0.99, warmup_steps=1500): ...
    def step(self, current_step): ...
```

### 2.5 `golfcomp/training/ema.py` — EMA

```python
class EMAWrapper:
    """Exponential Moving Average of model weights.
    
    Applied before quantization. Requires XSA to be active (EMA hurts without XSA).
    """
    def __init__(self, model: nn.Module, decay: float = 0.9965):
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}
        self.decay = decay
    
    def update(self, model: nn.Module):
        """Call every step. shadow = decay * shadow + (1-decay) * current."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply(self, model: nn.Module):
        """Copy shadow weights into model (for eval/quantization)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """Restore original weights after eval."""
        ...
```

### 2.6 `golfcomp/training/trainer.py` — Main Training Loop

```python
class Trainer:
    """Orchestrates the full training pipeline for one experiment run.
    
    Pipeline: data → forward → loss → backward → optimizer step → EMA → logging
    Supports:
    - Wall-clock time limit (default 20 min for cold)
    - Late QAT activation (when LR scale < threshold)
    - Late recurrence start (step 2000)
    - Gradient accumulation
    - Loss logging every N seconds
    """
    def __init__(self, model, config: ExperimentConfig, seed: int):
        self.model = model.cuda()
        self.config = config
        self.muon_opt, self.adamw_opt = build_optimizer(model, config.training)
        self.scheduler = WarmdownScheduler(...)
        self.momentum_warmup = MomentumWarmup(...)
        self.ema = EMAWrapper(model, config.training.ema_decay) if config.training.use_ema else None
        self.data_stream = TokenStream(FineWebDataset(...), ...)
        self.loss_tracker = LossTracker(log_interval=config.training.log_interval_seconds)
        self.qat_active = False
    
    def train(self) -> dict:
        """Returns {"final_loss": float, "steps": int, "tokens_seen": int, "loss_curve": [...]}"""
        self.model.train()
        start_time = time.time()
        
        for step, batch in enumerate(self.data_stream):
            if time.time() - start_time > self.config.training.max_time_seconds:
                break
            if step >= self.config.training.max_steps:
                break
            
            # Late recurrence start
            if hasattr(self.model, 'set_recurrence_active'):
                self.model.set_recurrence_active(step >= self.config.model.recurrence_start_step)
            
            # Late QAT activation
            lr_scale = self.scheduler.get_lr_scale(step)
            if not self.qat_active and lr_scale < self.config.training.qat_start_lr_scale:
                self.qat_active = True
                self.model.enable_qat(bits=self.config.training.qat_bits)
            
            # Forward + backward
            loss = self._train_step(batch)
            
            # EMA update
            if self.ema:
                self.ema.update(self.model)
            
            # Logging
            tokens_seen = (step + 1) * self.config.training.batch_tokens
            self.loss_tracker.log(step, loss, tokens_seen, time.time() - start_time)
        
        # Apply EMA before returning
        if self.ema:
            self.ema.apply(self.model)
        
        return self.loss_tracker.summary()
    
    def _train_step(self, batch) -> float:
        """Single gradient accumulation step."""
        total_loss = 0.0
        for micro_batch in batch["micro_batches"]:
            input_ids = micro_batch["input_ids"].cuda()
            labels = micro_batch["labels"].cuda()
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            (loss / self.config.training.grad_accum_steps).backward()
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
        self.muon_opt.step(); self.muon_opt.zero_grad()
        self.adamw_opt.step(); self.adamw_opt.zero_grad()
        self.scheduler.step()
        self.momentum_warmup.step()
        return total_loss / self.config.training.grad_accum_steps
    
    def save_checkpoint(self, path: str):
        """Save model + optimizer + EMA state for reuse (C15-C20)."""
        ...
```

### 2.7 `golfcomp/evaluation/evaluator.py` — BPB Evaluator

```python
class BPBEvaluator:
    """Competition-accurate BPB evaluation.
    
    Implements:
    - Sliding window with configurable stride and window size
    - SentencePiece byte introspection for per-byte BPB
    - Document boundary handling
    - Optional TTT (test-time training) integration
    """
    def __init__(self, model, tokenizer_path: str, config: EvalConfig):
        self.model = model
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.config = config
        self.ttt = self._build_ttt() if config.use_ttt else None
    
    def evaluate(self, val_data_path: str) -> dict:
        """Returns {"bpb": float, "loss": float, "tokens_evaluated": int}"""
        self.model.eval()
        total_bits = 0.0
        total_bytes = 0
        
        for doc_tokens, doc_bytes in self._load_validation(val_data_path):
            # Sliding window evaluation
            for window_start in range(0, len(doc_tokens), self.config.stride):
                window_end = min(window_start + self.config.window_size, len(doc_tokens))
                input_ids = doc_tokens[window_start:window_end]
                
                with torch.no_grad():
                    logits = self.model(input_ids.unsqueeze(0).cuda())
                
                # Only score tokens in the "new" part (stride portion)
                score_start = max(0, self.config.window_size - self.config.stride)
                log_probs = F.log_softmax(logits[0, score_start:], dim=-1)
                target = doc_tokens[window_start + score_start + 1 : window_end + 1]
                
                token_nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
                
                # Convert token NLL to byte-level bits
                for i, tok_id in enumerate(target):
                    byte_count = self._token_byte_length(tok_id.item())
                    total_bits += token_nll[i].item() / math.log(2)
                    total_bytes += byte_count
            
            # TTT adaptation (if enabled, score-first)
            if self.ttt:
                self.ttt.adapt(doc_tokens)
        
        bpb = total_bits / total_bytes
        return {"bpb": bpb, "total_bytes": total_bytes}
    
    def _token_byte_length(self, token_id: int) -> int:
        """Use SentencePiece introspection to get byte count per token."""
        piece = self.sp.id_to_piece(token_id)
        return len(piece.replace("▁", " ").encode("utf-8"))
    
    def _build_ttt(self):
        if self.config.ttt_type == "lora":
            return LoRATTT(self.model, self.config)
        elif self.config.ttt_type == "slot":
            return SLOTTTT(self.model, self.config)
```

### 2.8 `golfcomp/evaluation/ttt.py` — Test-Time Training

```python
class LoRATTT:
    """LoRA-based test-time training. Score-first, backward-looking only.
    
    Adds rank-R LoRA adapters to attention projections. Trains per-document
    on already-scored tokens. AdamW+cosine LR (SGD hurts Full GPTQ models).
    """
    def __init__(self, model, config: EvalConfig):
        self.adapters = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                self.adapters[name] = LoRAAdapter(module, rank=config.ttt_lora_rank)
        self.config = config
    
    def adapt(self, scored_tokens: Tensor):
        """Train LoRA on already-scored tokens for ttt_epochs."""
        opt = AdamW(self._adapter_params(), lr=self.config.ttt_lr)
        sched = CosineAnnealingLR(opt, T_max=self.config.ttt_epochs * len(scored_tokens))
        for epoch in range(self.config.ttt_epochs):
            ...  # standard training loop on scored_tokens
    
    def reset(self):
        """Reset adapters between documents."""
        for adapter in self.adapters.values():
            adapter.reset_parameters()

class SLOTTTT:
    """Single Learnable Output Transform. Lighter than LoRA.
    
    Single 512-dim delta vector at last hidden layer.
    Optimized per-batch via gradient descent.
    """
    def __init__(self, model, config: EvalConfig):
        self.delta = nn.Parameter(torch.zeros(config.model_dim))
    
    def adapt(self, scored_tokens):
        """Optimize delta on scored tokens."""
        ...

class LoRAAdapter(nn.Module):
    """Low-rank adapter: output = Wx + BAx where B is [out, rank], A is [rank, in]."""
    def __init__(self, linear: nn.Linear, rank: int = 8):
        self.linear = linear
        self.A = nn.Parameter(torch.randn(rank, linear.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))
    
    def forward(self, x):
        return self.linear(x) + (x @ self.A.T) @ self.B.T
```

### 2.9 `golfcomp/quantization/sdclip.py` — SDClip

```python
class SDClipQuantizer:
    """SDClip: clip = k * std(row). Per-row symmetric quantization.
    
    k=12.85 for int6 body weights, k=20.0 for int8 embeddings.
    """
    def __init__(self, body_bits=6, embed_bits=8, body_k=12.85, embed_k=20.0):
        self.body_bits = body_bits
        self.embed_bits = embed_bits
        self.body_k = body_k
        self.embed_k = embed_k
    
    def quantize_tensor(self, tensor: Tensor, bits: int, k: float) -> tuple[Tensor, dict]:
        """Returns (quantized_tensor, metadata) for serialization."""
        levels = 2 ** bits
        half = levels // 2
        row_std = tensor.std(dim=-1, keepdim=True)
        clip_val = k * row_std
        clipped = tensor.clamp(-clip_val, clip_val)
        scale = clip_val / half
        quantized = (clipped / scale).round().clamp(-half, half - 1).to(torch.int8)
        return quantized, {"scale": scale, "bits": bits}
    
    def quantize_model(self, model: nn.Module) -> dict:
        """Quantize all parameters. Returns serializable state dict."""
        state = {}
        for name, param in model.named_parameters():
            if "embed" in name:
                q, meta = self.quantize_tensor(param.data, self.embed_bits, self.embed_k)
            else:
                q, meta = self.quantize_tensor(param.data, self.body_bits, self.body_k)
            state[name] = {"quantized": q, **meta}
        return state
```

### 2.10 `golfcomp/quantization/gptq.py` — Full GPTQ

```python
class GPTQQuantizer:
    """Full GPTQ with Cholesky decomposition + activation ordering.
    
    Uses self-generated calibration data (model's own outputs, not val data).
    Processes one layer at a time to manage memory.
    """
    def __init__(self, model, bits=6, actorder=True, use_cholesky=True):
        self.model = model
        self.bits = bits
        self.actorder = actorder
        self.use_cholesky = use_cholesky
    
    def quantize(self, calib_data: list[Tensor]) -> nn.Module:
        """Quantize model using GPTQ algorithm.
        
        1. Collect Hessian (H = X^T X) from calibration data per layer
        2. If actorder: reorder columns by Hessian diagonal (descending)
        3. Cholesky decompose H for numerical stability
        4. Greedily quantize each column, compensating error in remaining columns
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                H = self._collect_hessian(module, calib_data)
                if self.actorder:
                    perm = torch.argsort(H.diag(), descending=True)
                    H = H[perm][:, perm]
                    module.weight.data = module.weight.data[:, perm]
                if self.use_cholesky:
                    H = torch.linalg.cholesky(H + 1e-6 * torch.eye(H.size(0)).cuda())
                self._quantize_layer(module, H)
        return self.model
```

### 2.11 `golfcomp/quantization/compression.py`

```python
class Compressor:
    """Compress quantized weight bytes. Supports brotli, zstd-22, lzma-9, ANS."""
    
    @staticmethod
    def compress(data: bytes, method: str) -> bytes:
        if method == "brotli":
            import brotli
            return brotli.compress(data, quality=11)
        elif method == "zstd22":
            import zstandard as zstd
            return zstd.ZstdCompressor(level=22).compress(data)
        elif method == "lzma9":
            import lzma
            return lzma.compress(data, preset=9)
        elif method == "ans_huffman":
            return _ans_encode(data)  # Custom ANS implementation
    
    @staticmethod
    def decompress(data: bytes, method: str) -> bytes:
        ...
```

### 2.12 `golfcomp/evaluation/metrics.py` — Tracking & Analysis

```python
class LossTracker:
    """Logs loss every N seconds. Fits power law extrapolation."""
    def __init__(self, log_interval: float = 30.0):
        self.records = []  # [(step, loss, tokens_seen, wall_time)]
        self.last_log_time = 0
    
    def log(self, step, loss, tokens, wall_time):
        if wall_time - self.last_log_time >= self.log_interval:
            self.records.append((step, loss, tokens, wall_time))
            self.last_log_time = wall_time
    
    def fit_power_law(self) -> dict:
        """Fit L(T) = a * T^(-b) + L_inf. Returns {a, b, L_inf, predicted_10min_loss}."""
        from scipy.optimize import curve_fit
        tokens = np.array([r[2] for r in self.records])
        losses = np.array([r[1] for r in self.records])
        def power_law(T, a, b, L_inf):
            return a * T ** (-b) + L_inf
        popt, _ = curve_fit(power_law, tokens, losses, p0=[1.0, 0.5, 0.5], maxfev=5000)
        return {"a": popt[0], "b": popt[1], "L_inf": popt[2]}
    
    def summary(self) -> dict:
        return {
            "final_loss": self.records[-1][1],
            "steps": self.records[-1][0],
            "tokens_seen": self.records[-1][2],
            "wall_time": self.records[-1][3],
            "loss_curve": self.records,
            "power_law": self.fit_power_law(),
        }
```

---

## 3. Model Components (Shared)

### 3.1 `golfcomp/models/components/attention.py`

```python
class GQAAttention(nn.Module):
    """Grouped Query Attention with optional XSA, QK-Norm, QK-Gain, LN Scale.
    
    8 query heads, 4 KV heads. FlashAttention 2 backend.
    XSA: cross-sequence attention (borrows context from adjacent sequences).
    """
    def __init__(self, dim, num_heads=8, num_kv_heads=4, qk_gain=5.0,
                 use_xsa=True, layer_idx=0, total_layers=11):
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim // (num_heads // num_kv_heads))
        self.v_proj = nn.Linear(dim, dim // (num_heads // num_kv_heads))
        self.o_proj = nn.Linear(dim, dim)
        self.qk_gain = nn.Parameter(torch.full((num_heads,), qk_gain))
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)  # LN Scale
        self.use_xsa = use_xsa
        self._prev_kv = None  # For XSA: cache from previous sequence
    
    def forward(self, x, use_flash=True):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, -1)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, -1)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, -1)
        
        # QK-Norm + QK-Gain
        q = F.normalize(q, dim=-1) * self.qk_gain.view(1, 1, -1, 1)
        k = F.normalize(k, dim=-1)
        
        # XSA: prepend previous sequence's KV
        if self.use_xsa and self._prev_kv is not None:
            k = torch.cat([self._prev_kv[0], k], dim=1)
            v = torch.cat([self._prev_kv[1], v], dim=1)
        self._prev_kv = (k[:, -S:].detach(), v[:, -S:].detach())
        
        # FlashAttention 2
        if use_flash:
            from flash_attn import flash_attn_func
            attn_out = flash_attn_func(q, k, v, causal=True, softcap=30.0)
        else:
            attn_out = self._manual_attention(q, k, v)
        
        return self.o_proj(attn_out.view(B, S, D))
```

### 3.2 `golfcomp/models/components/embeddings.py`

```python
class TokenEmbedding(nn.Module):
    """Standard FP16 token embedding. Tied with output head."""
    def __init__(self, vocab_size, dim): ...

class SmearGate(nn.Module):
    """~512-param sigmoid gate blending current token with previous.
    Requires OrthoInit on the parent model."""
    def __init__(self, dim):
        self.gate = nn.Linear(dim, 1)  # sigmoid gate
    
    def forward(self, x):
        # x: [B, S, D]
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))  # shift right, pad with zeros
        gate = torch.sigmoid(self.gate(x))
        return gate * x + (1 - gate) * prev

class BigramHash(nn.Module):
    """XOR-hash token pairs → learned embedding table.
    buckets=3072, hash_dim=128, projected to model_dim."""
    def __init__(self, buckets=3072, hash_dim=128, model_dim=512):
        self.table = nn.Embedding(buckets, hash_dim)
        self.proj = nn.Linear(hash_dim, model_dim)
    
    def forward(self, input_ids):
        # Hash: XOR current token with previous
        prev_ids = F.pad(input_ids[:, :-1], (1, 0))
        hashed = (input_ids ^ prev_ids) % self.buckets
        return self.proj(self.table(hashed))

class EngramLite(nn.Module):
    """Multi-head n-gram hashing with context-aware gating (C8).
    Replaces BigramHash with richer n-gram representation.
    
    Architecture: K hash heads, each with its own n-gram order (2,3,4),
    hash table, and learned context gate.
    """
    def __init__(self, vocab_size, num_heads=3, hash_dim=128, model_dim=512,
                 orders=(2, 3, 4), buckets_per_head=2048):
        self.heads = nn.ModuleList([
            EngramHead(vocab_size, order=o, buckets=buckets_per_head, hash_dim=hash_dim)
            for o in orders
        ])
        self.context_gate = nn.Linear(model_dim, num_heads)
        self.proj = nn.Linear(hash_dim * num_heads, model_dim)
    
    def forward(self, input_ids, hidden_state=None):
        head_outputs = [head(input_ids) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)
        if hidden_state is not None:
            gates = torch.sigmoid(self.context_gate(hidden_state))
            combined = combined * gates.unsqueeze(-1).repeat(1, 1, self.hash_dim)
        return self.proj(combined)

class FactorizedEmbedding(nn.Module):
    """ALBERT-style factorized embedding for C21.
    vocab_size × low_rank (FP16) + low_rank × model_dim projection.
    
    Variants:
    - standard: vocab_size × model_dim (8.4 MB for SP8192)
    - factorized: vocab_size × 64 + 64 × 512 (~2.1 MB for SP16384)
    - multi_hash: 3 tables of 5461 × 512, compose via sum (~8.4 MB, zero collisions)
    - factorized_multi_hash: 3 tables of 5461 × 64 + shared projection (~2.1 MB)
    """
    def __init__(self, vocab_size, model_dim, low_rank=64, 
                 embed_type="factorized", num_tables=3):
        if embed_type == "factorized":
            self.low_rank = nn.Embedding(vocab_size, low_rank)
            self.projection = nn.Linear(low_rank, model_dim, bias=False)
        elif embed_type == "multi_hash":
            table_size = (vocab_size + num_tables - 1) // num_tables
            self.tables = nn.ModuleList([
                nn.Embedding(table_size, model_dim) for _ in range(num_tables)
            ])
            self.hash_fns = [self._make_hash(i, table_size) for i in range(num_tables)]
        elif embed_type == "factorized_multi_hash":
            table_size = (vocab_size + num_tables - 1) // num_tables
            self.tables = nn.ModuleList([
                nn.Embedding(table_size, low_rank) for _ in range(num_tables)
            ])
            self.projection = nn.Linear(low_rank, model_dim, bias=False)
            self.hash_fns = [self._make_hash(i, table_size) for i in range(num_tables)]
    
    def forward(self, input_ids):
        if self.embed_type == "factorized":
            return self.projection(self.low_rank(input_ids))
        elif self.embed_type == "multi_hash":
            return sum(table(h(input_ids)) for table, h in zip(self.tables, self.hash_fns))
        elif self.embed_type == "factorized_multi_hash":
            low = sum(table(h(input_ids)) for table, h in zip(self.tables, self.hash_fns))
            return self.projection(low)
    
    def compute_logits(self, hidden):
        """Tied output: logits = hidden @ projection.T @ low_rank.T"""
        if self.embed_type == "factorized":
            projected = hidden @ self.projection.weight  # [B, S, low_rank]
            return projected @ self.low_rank.weight.T     # [B, S, vocab_size]
        elif self.embed_type == "multi_hash":
            # Sum of per-table dot products
            return sum(hidden @ table.weight.T for table in self.tables)  # approximate
        ...
```

### 3.3 `golfcomp/models/components/activations.py`

```python
class LeakyReLUSq(nn.Module):
    """(leaky_relu(x, 0.5))^2. Produces 84-98% natural sparsity."""
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=0.5).square()

class SwiGLU(nn.Module):
    """Gated linear unit: SwiGLU(x) = Swish(xW1) * (xW2).
    For C9. Adjust hidden dim to match param count with LeakyReLU² MLP.
    
    LeakyReLU² MLP: up(512→1536) + down(1536→512) = 2 × 512×1536 = 1.57M params
    SwiGLU MLP: gate(512→1024) + up(512→1024) + down(1024→512) = 3 × 512×1024 = 1.57M params
    So hidden_dim = 1024 for SwiGLU to match param count.
    """
    def __init__(self, dim, hidden_dim=1024):
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 3.4 `golfcomp/models/components/recurrence.py`

```python
class DepthRecurrence(nn.Module):
    """Hard depth recurrence: reuse layers K times with FiLM conditioning.
    
    E.g., layers [3,4,5] run twice → 14 virtual layers from 11 physical.
    Late start: recurrence disabled before recurrence_start_step.
    """
    def __init__(self, layers: nn.ModuleList, recurrence_layers: list, num_loops: int = 2):
        self.layers = layers
        self.recurrence_layers = set(recurrence_layers)
        self.num_loops = num_loops
        self.film_scale = nn.ParameterDict()
        self.film_shift = nn.ParameterDict()
        for idx in recurrence_layers:
            for loop in range(num_loops):
                key = f"l{idx}_loop{loop}"
                self.film_scale[key] = nn.Parameter(torch.ones(layers[idx].dim))
                self.film_shift[key] = nn.Parameter(torch.zeros(layers[idx].dim))
        self.active = False
    
    def forward(self, x, layer_idx: int, loop: int = 0):
        h = self.layers[layer_idx](x)
        if self.active and layer_idx in self.recurrence_layers:
            key = f"l{layer_idx}_loop{loop}"
            h = h * self.film_scale[key] + self.film_shift[key]
        return h

class BasisSharing(nn.Module):
    """SVD basis sharing across layers (ICLR 2025).
    
    Decompose: W_l = U @ diag(s_l) @ V^T
    where U, V are shared across all layers, s_l is per-layer.
    
    Shared bases: U (dim × rank), V (rank × dim) — learned, shared
    Per-layer coefficients: s_l (rank,) — unique per layer
    
    Total unique params: rank × dim × 2 (shared) + num_layers × rank (per-layer)
    vs standard: num_layers × dim × dim (all unique)
    """
    def __init__(self, num_layers, dim, rank=64):
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.ones(rank)) for _ in range(num_layers)
        ])
    
    def get_weight(self, layer_idx: int) -> Tensor:
        """Reconstruct W_l = U @ diag(s_l) @ V"""
        return self.U @ torch.diag(self.coefficients[layer_idx]) @ self.V
    
    def apply_to_model(self, model):
        """Replace linear layer weights with basis-shared reconstructions."""
        # Called each forward pass to reconstruct effective weights
        for idx, layer in enumerate(model.layers):
            for name, param in layer.named_parameters():
                if param.ndim == 2:  # Only matrix weights
                    param.data = self.get_weight(idx)

class RelaxedRecursive(nn.Module):
    """Weight tying + per-pass LoRA adapters (C7).
    
    num_shared_blocks shared transformer blocks, each run num_loops times.
    Per-pass differentiation via rank-R LoRA on attention projections.
    
    E.g., 6 shared × 2 loops = 12 virtual layers + 6 × rank-8 LoRA sets.
    """
    def __init__(self, shared_blocks: nn.ModuleList, num_loops=2, lora_rank=8):
        self.shared_blocks = shared_blocks
        self.num_loops = num_loops
        # Per-block, per-loop LoRA adapters
        self.lora_adapters = nn.ModuleDict()
        for block_idx in range(len(shared_blocks)):
            for loop in range(num_loops):
                key = f"b{block_idx}_l{loop}"
                self.lora_adapters[key] = LoRASet(shared_blocks[block_idx], rank=lora_rank)
    
    def forward(self, x):
        for loop in range(self.num_loops):
            for block_idx, block in enumerate(self.shared_blocks):
                key = f"b{block_idx}_l{loop}"
                x = block(x) + self.lora_adapters[key](x)
        return x

class LoRASet(nn.Module):
    """LoRA adapters for all linear layers in a transformer block."""
    def __init__(self, block, rank=8):
        self.adapters = nn.ModuleDict()
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                self.adapters[name] = LoRAAdapter(module, rank)
```

### 3.5 `golfcomp/models/components/residuals.py`

```python
class ParallelResidual(nn.Module):
    """From layer parallel_start_layer+: attention and MLP on separate lanes.
    Learned merge scalar combines them.
    """
    def __init__(self, attn, mlp, dim):
        self.attn = attn
        self.mlp = mlp
        self.merge = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        a = self.attn(x)
        m = self.mlp(x)
        return x + self.merge * a + (1 - self.merge) * m

class SkipGate(nn.Module):
    """Learned gate on residual skip connection."""
    def __init__(self, dim):
        self.gate = nn.Parameter(torch.ones(dim))
    
    def forward(self, residual, x):
        return x + self.gate * residual

class SerialResidual(nn.Module):
    """Standard serial attention → MLP with residual connections."""
    def __init__(self, attn, mlp, dim, use_skip_gate=True):
        self.attn = attn
        self.mlp = mlp
        self.skip1 = SkipGate(dim) if use_skip_gate else None
        self.skip2 = SkipGate(dim) if use_skip_gate else None
    
    def forward(self, x):
        h = self.attn(x)
        x = self.skip1(h, x) if self.skip1 else x + h
        h = self.mlp(x)
        x = self.skip2(h, x) if self.skip2 else x + h
        return x
```

---

## 4. Model Architectures

### 4.1 `golfcomp/models/transformer.py` — Baseline + Ablations

```python
class TransformerBlock(nn.Module):
    """Single transformer layer: LN → Attention → LN → MLP, with configurable residuals."""
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.attn = GQAAttention(config.model_dim, config.num_heads, config.num_kv_heads,
                                  qk_gain=config.qk_gain, use_xsa=config.use_xsa,
                                  layer_idx=layer_idx, total_layers=config.num_layers)
        
        if config.activation == "leaky_relu_sq":
            self.mlp = nn.Sequential(
                nn.Linear(config.model_dim, config.model_dim * config.mlp_mult),
                LeakyReLUSq(),
                nn.Linear(config.model_dim * config.mlp_mult, config.model_dim),
            )
        elif config.activation == "swiglu":
            self.mlp = SwiGLU(config.model_dim, hidden_dim=1024)  # param-matched
        
        # Residual type
        if config.use_parallel_residuals and layer_idx >= config.parallel_start_layer:
            self.residual = ParallelResidual(self.attn, self.mlp, config.model_dim)
        else:
            self.residual = SerialResidual(self.attn, self.mlp, config.model_dim,
                                           use_skip_gate=config.use_skip_gates)

class TransformerModel(BaseModel):
    """Full transformer model. Used by baseline, C6-C10, C11-C14, C21.
    
    Configurable via ModelConfig:
    - recurrence_type: depth | basis_sharing | relaxed_recursive | none
    - activation: leaky_relu_sq | swiglu
    - use_parallel_residuals: True | False
    - embedding components: SmearGate, BigramHash/EngramLite, factorized
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Embeddings
        if config.embedding_type == "standard":
            self.embed = TokenEmbedding(config.vocab_size, config.model_dim)
        else:
            self.embed = FactorizedEmbedding(config.vocab_size, config.model_dim,
                                              embed_type=config.embedding_type)
        
        self.smear_gate = SmearGate(config.model_dim) if config.use_smear_gate else None
        
        if config.use_engramlite:
            self.ngram_embed = EngramLite(config.vocab_size, model_dim=config.model_dim)
        else:
            self.ngram_embed = BigramHash(config.bigram_hash_buckets, config.bigram_hash_dim, 
                                          config.model_dim)
        
        self.rope = PartialRoPE(config.model_dim // config.num_heads, config.rope_partial_dim)
        
        # Layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])
        
        # Recurrence
        if config.recurrence_type == "depth":
            self.recurrence = DepthRecurrence(self.layers, config.recurrence_layers)
        elif config.recurrence_type == "basis_sharing":
            self.basis_sharing = BasisSharing(config.num_layers, config.model_dim, config.basis_rank)
        elif config.recurrence_type == "relaxed_recursive":
            shared = self.layers[:config.num_shared_blocks]
            self.relaxed = RelaxedRecursive(shared, config.num_loops, config.lora_rank)
        
        self.norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embed.weight  # or factorized equivalent
    
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        if self.smear_gate:
            x = self.smear_gate(x)
        x = x + self.ngram_embed(input_ids)
        
        if self.config.recurrence_type == "relaxed_recursive":
            x = self.relaxed(x)
        else:
            for i, layer in enumerate(self.layers):
                if self.config.recurrence_type == "depth":
                    x = self.recurrence(x, i, loop=0)
                    if i in self.recurrence.recurrence_layers:
                        x = self.recurrence(x, i, loop=1)  # second pass
                else:
                    x = layer(x)
        
        x = self.norm(x)
        return self.output(x)  # [B, S, vocab_size]
```

### 4.2 `golfcomp/models/gla_hybrid.py` — C1

```python
class GLAHybridModel(BaseModel):
    """8 GLA layers (bottom) + 3 attention layers (top).
    
    GLA layers: from FLA library (flash-linear-attention)
    Attention layers: standard GQA with XSA
    XSA only on the 3 attention layers.
    EMA 0.9965 (XSA present on attention layers).
    Muon for projections, AdamW for GLA gates.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed = TokenEmbedding(config.vocab_size, config.model_dim)
        self.smear_gate = SmearGate(config.model_dim)
        self.ngram_embed = BigramHash(config.bigram_hash_buckets, config.bigram_hash_dim,
                                       config.model_dim)
        
        # GLA layers (bottom 8)
        from fla.layers import GatedLinearAttention
        self.gla_layers = nn.ModuleList([
            GLABlock(config.model_dim, expand_ratio=config.gla_expand_ratio)
            for _ in range(config.gla_num_layers)
        ])
        
        # Attention layers (top 3) with XSA
        self.attn_layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=config.gla_num_layers + i)
            for i in range(config.num_layers - config.gla_num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embed.weight
    
    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.smear_gate(x) + self.ngram_embed(input_ids)
        for layer in self.gla_layers:
            x = layer(x)
        for layer in self.attn_layers:
            x = layer(x)
        return self.output(self.norm(x))

class GLABlock(nn.Module):
    """Wrapper around FLA's GLA layer + MLP with LeakyReLU²."""
    def __init__(self, dim, expand_ratio=1):
        from fla.layers import GatedLinearAttention
        self.norm1 = nn.LayerNorm(dim)
        self.gla = GatedLinearAttention(d_model=dim, expand_ratio=expand_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 3), LeakyReLUSq(), nn.Linear(dim * 3, dim)
        )
    
    def forward(self, x):
        x = x + self.gla(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### 4.3 `golfcomp/models/mamba_hybrid.py` — C2

```python
class MambaHybridModel(BaseModel):
    """8 Mamba-3 layers (bottom) + 3 attention layers (top).
    
    Uses mamba-ssm library. Mamba-3: complex-valued gates, MIMO, no conv.
    Int8 for Mamba layers at eval (SSM quant sensitivity).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed = TokenEmbedding(config.vocab_size, config.model_dim)
        self.smear_gate = SmearGate(config.model_dim)
        self.ngram_embed = BigramHash(...)
        
        from mamba_ssm import Mamba2  # Mamba-3 may be Mamba2 with updated config
        self.mamba_layers = nn.ModuleList([
            MambaBlock(config.model_dim, d_state=config.mamba_d_state)
            for _ in range(config.mamba_num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=config.mamba_num_layers + i)
            for i in range(config.num_layers - config.mamba_num_layers)
        ])
        self.norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)
    
    # forward: embed → smear → bigram → mamba layers → attn layers → norm → output

class MambaBlock(nn.Module):
    """Wrapper around Mamba-3 (mamba_ssm.Mamba2 with Mamba-3 config)."""
    def __init__(self, dim, d_state=32):
        from mamba_ssm import Mamba2
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2(d_model=dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*3), LeakyReLUSq(), nn.Linear(dim*3, dim))
    
    def forward(self, x):
        x = x + self.mamba(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### 4.4 `golfcomp/models/xlstm_hybrid.py` — C3

```python
class XLSTMHybridModel(BaseModel):
    """8 mLSTM blocks (bottom) + 3 attention layers (top).
    
    mLSTM: fully parallelizable matrix memory LSTM with exponential gating.
    Claims 3.5x training speed. Verify at 512d on consumer GPU.
    """
    def __init__(self, config: ModelConfig):
        # from xlstm import mLSTMBlock (or custom implementation if lib unavailable)
        self.mlstm_layers = nn.ModuleList([
            MLSTMBlock(config.model_dim) for _ in range(config.xlstm_num_layers)
        ])
        self.attn_layers = nn.ModuleList([...])  # top 3 attention
        ...
    # Same pattern: embed → smear → bigram → mLSTM layers → attn layers → output

class MLSTMBlock(nn.Module):
    """mLSTM with matrix memory + exponential gating.
    If xlstm library unavailable, implement from paper (arxiv 2405.04517)."""
    ...
```

### 4.5 `golfcomp/models/rwkv.py` — C4

```python
class RWKVModel(BaseModel):
    """Pure RWKV-7 backbone. No attention → no XSA possible → no EMA.
    11 RWKV-7 layers, 512d. Tests if RWKV-7 can compete without XSA/EMA.
    """
    def __init__(self, config: ModelConfig):
        # from rwkv.model import RWKV7Block (or custom)
        self.layers = nn.ModuleList([
            RWKV7Block(config.model_dim) for _ in range(config.rwkv_num_layers)
        ])
        # No XSA, no EMA (no attention to anchor it)
        ...
```

### 4.6 `golfcomp/models/mixed_hybrid.py` — C5

```python
class MixedHybridModel(BaseModel):
    """4 Mamba-3 (bottom) + 4 GLA (middle) + 3 attention (top).
    
    Tests if mixing two sub-quadratic architectures helps.
    Mixed optimizer: AdamW for SSM, Muon for projections, AdamW for GLA gates.
    """
    def __init__(self, config: ModelConfig):
        self.mamba_layers = nn.ModuleList([MambaBlock(...) for _ in range(4)])
        self.gla_layers = nn.ModuleList([GLABlock(...) for _ in range(4)])
        self.attn_layers = nn.ModuleList([TransformerBlock(...) for _ in range(3)])
        ...
```

---

## 5. Experiment Configurations

### 5.1 Cold Baseline (`configs/baseline.yaml`)
```yaml
name: cold_baseline
category: baseline
model:
  arch: transformer
  num_layers: 11
  model_dim: 512
  num_heads: 8
  num_kv_heads: 4
  mlp_mult: 3
  vocab_size: 8192
  seq_len: 1024
  activation: leaky_relu_sq
  use_xsa: true
  xsa_mode: all
  use_parallel_residuals: true
  parallel_start_layer: 7
  use_skip_gates: true
  qk_gain: 5.0
  recurrence_type: depth
  recurrence_layers: [3, 4, 5]
  recurrence_start_step: 2000
  use_smear_gate: true
  bigram_hash_buckets: 3072
training:
  max_time_seconds: 1200   # 20 min
  batch_tokens: 128000
  matrix_lr: 0.022
  weight_decay: 0.095
  ema_decay: 0.9965
  warmdown_frac: 0.72
quant:
  method: sdclip+gptq
  int6_k: 12.85
  int8_embed_k: 20.0
eval:
  window_size: 2048
  stride: 64
compression:
  method: brotli
```

### 5.2 Architecture Experiments (C1-C5) — Optuna HP Spaces

Each overrides `model.arch` and uses Optuna for HP search.

| Experiment | `arch` | Optuna Search Space |
|---|---|---|
| **C1** GLA Hybrid | `gla_hybrid` | matrix_lr: [0.01, 0.04], gate_lr: [0.001, 0.02], WD: [0.04, 0.15], EMA: [0.990, 0.999], expand_ratio: {1, 2} |
| **C2** Mamba-3 Hybrid | `mamba_hybrid` | matrix_lr: [0.01, 0.04], ssm_lr: [0.001, 0.02], WD: [0.04, 0.15], d_state: {16, 32, 64} |
| **C3** xLSTM Hybrid | `xlstm_hybrid` | matrix_lr: [0.01, 0.04], mlstm_lr: [0.001, 0.02], WD: [0.04, 0.15] |
| **C4** RWKV-7 | `rwkv` | lr: [0.005, 0.04], WD: [0.04, 0.15] |
| **C5** Mixed Hybrid | `mixed_hybrid` | Combined C1+C2 spaces |

**Optuna config for C1-C5:**
- Sampler: TPESampler (Bayesian)
- Pruner: MedianPruner(n_startup_trials=1, n_warmup_steps=100)
- Trials: 5 per architecture (first trial uses baseline-derived HPs)
- Time per trial: 10 min (pruned trials shorter)
- Best trial re-run: 20 min with best HPs

### 5.3 Technique Ablations (C6-C10)

Each inherits baseline config and changes ONE thing:

| Experiment | Change from baseline |
|---|---|
| **C6** Basis Sharing | `recurrence_type: basis_sharing`, `basis_rank: 64`, `recurrence_start_step: 0` (no late start needed) |
| **C7** Relaxed Recursive | `recurrence_type: relaxed_recursive`, `num_shared_blocks: 6`, `num_loops: 2`, `lora_rank: 8` |
| **C8** EngramLite | `use_engramlite: true`, `bigram_hash_buckets: 0` (EngramLite replaces BigramHash) |
| **C9** SwiGLU | `activation: swiglu` (hidden dim auto-adjusted to 1024 for param match) |
| **C10** No Parallel Residuals | `use_parallel_residuals: false` |

### 5.4 HP Sweeps (C11-C14) — Optuna GridSampler

| Experiment | Parameter | Grid Values |
|---|---|---|
| **C11** EMA Decay | `training.ema_decay` | {0.990, 0.995, 0.9965, 0.998, 0.999} |
| **C12** Weight Decay | `training.weight_decay` | {0.04, 0.06, 0.095, 0.12, 0.15} |
| **C13** Recurrence Layers | `model.recurrence_layers` | {[3,4,5], [4,5], [2,3,4,5], [5,6,7], []} |
| **C14** Warmdown Fraction | `training.warmdown_frac` | {0.50, 0.60, 0.72, 0.80, 0.90} |

**Optuna config for C11-C14:**
- Sampler: GridSampler (exhaustive, all 5 values)
- No pruning (need full loss curves for comparison)
- Time per trial: 10 min
- Total: 20 runs × 10 min = 200 min

### 5.5 Quantization & Compression (C15-C17) — Post-Training

These share ONE trained baseline checkpoint. No retraining needed.

| Experiment | Variants |
|---|---|
| **C15** SDClip k-sweep | int6 k: {10, 11, 12, 12.85, 14, 16} × int8 embed k: {16, 18, 20, 22, 25} = 30 combos |
| **C16** Mixed Precision | A) all int6, B) int5 MLP + int6 attn, C) int6 MLP + int8 attn |
| **C17** Compressors | A) brotli, B) zstd-22, C) lzma-9, D) ANS/Huffman |

**Execution:** Train baseline once (20 min), save checkpoint, then run quantization/compression variants (seconds each).

### 5.6 Eval-Time Techniques (C18-C20) — Post-Training

Same shared baseline checkpoint.

| Experiment | Variants | Search method |
|---|---|---|
| **C18** TTT HP Sweep | epochs × LR × optimizer × LoRA rank = 120 combos | Optuna TPE, 30 trials, ~5 min each |
| **C19** SLOT vs LoRA | A) LoRA rank-8, B) SLOT 512-dim delta, C) both | Grid (3 runs) |
| **C20** Sliding Window | stride × window = 12 combos | Grid |

### 5.7 Vocab/Embedding Innovation (C21)

Requires new SP16384 tokenizer + re-tokenized FineWeb. 4 variants, each a full training run:

| Variant | Embedding | Params freed | Model change |
|---|---|---|---|
| **A** SP8192 baseline | Standard 8192×512 | None (reference) | None |
| **B** SP16384 + factorized | 16384×64 + 64×512 | ~6.3 MB | Option B1: 2x wider MLP, B2: 14L, B3: both |
| **C** SP16384 + multi-hash | 3×5461×512 sum | None (same budget) | None |
| **D** SP16384 + factorized multi-hash | 3×5461×64 + 64×512 | ~6.3 MB | Same as B options |

**Execution:** Train tokenizer → retokenize FineWeb → 4 training runs × 20 min = 80 min

---

## 6. HP Optimization

### 6.1 `golfcomp/experiments/optuna_search.py`

```python
class OptunaSearcher:
    """Wraps Optuna for HP search with experiment configs.
    
    Two modes:
    - Bayesian (TPESampler): for architecture experiments (C1-C5, C18)
    - Grid (GridSampler): for explicit sweeps (C11-C14)
    """
    def __init__(self, base_config: ExperimentConfig, search_space: dict,
                 mode: str = "bayesian", n_trials: int = 5):
        self.base_config = base_config
        self.search_space = search_space
        
        if mode == "bayesian":
            self.sampler = optuna.samplers.TPESampler(seed=42)
            self.pruner = optuna.pruners.MedianPruner(
                n_startup_trials=1, n_warmup_steps=100
            )
        elif mode == "grid":
            self.sampler = optuna.samplers.GridSampler(search_space)
            self.pruner = optuna.pruners.NopPruner()
        
        self.study = optuna.create_study(
            direction="minimize",
            sampler=self.sampler,
            pruner=self.pruner,
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """Single trial: build config from suggested HPs, train, return loss."""
        config = deepcopy(self.base_config)
        
        # Apply suggested HPs
        for param_name, spec in self.search_space.items():
            if spec["type"] == "float":
                value = trial.suggest_float(param_name, spec["low"], spec["high"],
                                           log=spec.get("log", False))
            elif spec["type"] == "int":
                value = trial.suggest_int(param_name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                value = trial.suggest_categorical(param_name, spec["choices"])
            
            # Set value on config (supports nested: "training.matrix_lr")
            self._set_nested(config, param_name, value)
        
        # Build model + train
        model = build_model(config.model)
        trainer = Trainer(model, config, seed=42)
        
        # Report intermediate values for pruning
        trainer.set_pruning_callback(
            lambda step, loss: trial.report(loss, step)
        )
        
        result = trainer.train()
        return result["final_loss"]
    
    def search(self) -> dict:
        """Run all trials. Returns best config + all trial results."""
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "all_trials": [
                {"params": t.params, "value": t.value, "state": t.state.name}
                for t in self.study.trials
            ],
        }
```

### 6.2 Search Spaces (defined in `golfcomp/experiments/search_spaces.py`)

```python
# Architecture experiment search spaces
C1_GLA_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
    "training.gate_lr":   {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
    "training.ema_decay": {"type": "float", "low": 0.990, "high": 0.999},
    "model.gla_expand_ratio": {"type": "categorical", "choices": [1, 2]},
}

C2_MAMBA_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
    "training.ssm_lr":    {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
    "model.mamba_d_state": {"type": "categorical", "choices": [16, 32, 64]},
}

C3_XLSTM_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
    "training.mlstm_lr":  {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
}

C4_RWKV_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.005, "high": 0.04, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
}

C5_MIXED_SPACE = {**C1_GLA_SPACE, **C2_MAMBA_SPACE}  # combined

# HP sweep grid spaces (for GridSampler)
C11_EMA_GRID = {"training.ema_decay": [0.990, 0.995, 0.9965, 0.998, 0.999]}
C12_WD_GRID = {"training.weight_decay": [0.04, 0.06, 0.095, 0.12, 0.15]}
C13_RECURRENCE_GRID = {"model.recurrence_layers": [[3,4,5], [4,5], [2,3,4,5], [5,6,7], []]}
C14_WARMDOWN_GRID = {"training.warmdown_frac": [0.50, 0.60, 0.72, 0.80, 0.90]}

# TTT search space (C18)
C18_TTT_SPACE = {
    "eval.ttt_epochs":    {"type": "int", "low": 1, "high": 8},
    "eval.ttt_lr":        {"type": "float", "low": 0.001, "high": 0.01, "log": True},
    "eval.ttt_optimizer": {"type": "categorical", "choices": ["sgd_momentum", "adamw_cosine"]},
    "eval.ttt_lora_rank": {"type": "categorical", "choices": [4, 8, 16]},
}
```

---

## 7. Execution Pipeline

### 7.1 `golfcomp/experiments/runner.py`

```python
class ExperimentRunner:
    """Orchestrates a single experiment: train → quantize → evaluate → log."""
    
    def __init__(self, config: ExperimentConfig, seed: int = 42):
        self.config = config
        self.seed = seed
    
    def run(self) -> dict:
        set_seed(self.seed)
        model = build_model(self.config.model)
        
        # Train
        trainer = Trainer(model, self.config, self.seed)
        train_result = trainer.train()
        
        # Quantize
        if self.config.quant.method != "none":
            quantizer = build_quantizer(self.config.quant)
            quant_result = quantizer.quantize_model(model)
        
        # Compress
        artifact = ArtifactPacker.pack(model, self.config)
        compress_result = Compressor.compress(artifact, self.config.compression.method)
        
        # Evaluate
        evaluator = BPBEvaluator(model, self.config.training.tokenizer_path, self.config.eval)
        eval_result = evaluator.evaluate(self.config.training.data_path + "/val/")
        
        # Log
        result = {
            **train_result,
            **eval_result,
            "artifact_bytes": len(compress_result),
            "seed": self.seed,
            "config": asdict(self.config),
        }
        self._save_result(result)
        return result
    
    def run_post_training(self, checkpoint_path: str) -> dict:
        """For C15-C20: load checkpoint, apply quant/eval variants."""
        model = load_checkpoint(checkpoint_path)
        ...
```

### 7.2 `scripts/run_all_cold.py` — Full Orchestration

```
EXECUTION ORDER (total ~10-12 hours on 5090):
═══════════════════════════════════════════════════════════════
Phase 0: Data Prep                                    [~30 min]
  ├── Download FineWeb SP8192 tokenized data
  ├── Train SP16384 tokenizer (for C21)
  └── Tokenize FineWeb with SP16384

Phase 1: Cold Baseline                                [~25 min]
  ├── Train baseline (20 min)
  ├── Save checkpoint (reused by C6-C20)
  └── Evaluate + log reference BPB

Phase 2: Architecture Comparisons (C1-C5)             [~5 hours]
  ├── C1 GLA Hybrid:    Optuna 5 trials × 10 min + best × 20 min
  ├── C2 Mamba-3 Hybrid: Optuna 5 trials × 10 min + best × 20 min
  ├── C3 xLSTM Hybrid:  Optuna 5 trials × 10 min + best × 20 min
  ├── C4 RWKV-7:        Optuna 5 trials × 10 min + best × 20 min
  └── C5 Mixed Hybrid:  Optuna 5 trials × 10 min + best × 20 min
  (Sequential: each needs GPU exclusively)

Phase 3: Technique Ablations (C6-C10)                 [~100 min]
  ├── C6  Basis Sharing:       20 min
  ├── C7  Relaxed Recursive:   20 min
  ├── C8  EngramLite:          20 min
  ├── C9  SwiGLU:              20 min
  └── C10 No Parallel Res:     20 min
  (Sequential)

Phase 4: HP Sweeps (C11-C14)                          [~200 min]
  ├── C11 EMA Decay:      5 × 10 min = 50 min
  ├── C12 Weight Decay:   5 × 10 min = 50 min
  ├── C13 Recurrence:     5 × 10 min = 50 min
  └── C14 Warmdown:       5 × 10 min = 50 min

Phase 5: Quantization & Compression (C15-C17)         [~5 min]
  (Uses Phase 1 checkpoint — no retraining)
  ├── C15 SDClip k-sweep:   30 quantize runs (seconds each)
  ├── C16 Mixed Precision:   3 quantize runs
  └── C17 Compressors:       4 compress runs

Phase 6: Eval-Time Techniques (C18-C20)               [~3 hours]
  (Uses Phase 1 checkpoint — no retraining)
  ├── C18 TTT Sweep: Optuna 30 trials × ~5 min each = 150 min
  ├── C19 SLOT vs LoRA: 3 eval runs × ~10 min = 30 min
  └── C20 Sliding Window: 12 eval configs × ~5 min = 60 min

Phase 7: Vocab/Embedding (C21)                        [~80 min]
  ├── C21-A SP8192 reference:              20 min (may reuse baseline)
  ├── C21-B SP16384 + factorized:          20 min
  ├── C21-C SP16384 + multi-hash:          20 min
  └── C21-D SP16384 + factorized+multi:    20 min

Phase 8: Analysis                                     [~5 min]
  ├── Generate comparison tables
  ├── Plot loss curves
  ├── Rank experiments by promotion criteria
  └── Output promotion recommendations for hot experiments
═══════════════════════════════════════════════════════════════
TOTAL: ~12 hours (vs 8 hours in cold-exp.md — delta is Optuna overhead)
```

### 7.3 `golfcomp/experiments/analysis.py`

```python
class ResultsAnalyzer:
    """Compares experiment results and generates promotion recommendations."""
    
    PROMOTION_CRITERIA = {
        "architecture": {"throughput_gain": 0.05, "loss_tolerance": 0.01},
        "ablation": {"bpb_improvement": 0.002},
        "hp_sweep": {"differs_from_frontier": True},
        "quantization": {"artifact_reduction_kb": 200, "bpb_improvement": 0.001},
        "eval_time": {"bpb_improvement": 0.001},
    }
    
    def load_all_results(self, results_dir: str) -> pd.DataFrame: ...
    
    def compare_to_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add delta_bpb, delta_throughput columns."""
        ...
    
    def rank_by_category(self, df: pd.DataFrame) -> dict:
        """Rank experiments within each category."""
        ...
    
    def recommend_promotions(self, df: pd.DataFrame) -> list[dict]:
        """Apply promotion criteria, return list of experiments to promote to hot."""
        ...
    
    def plot_loss_curves(self, df: pd.DataFrame, output_dir: str):
        """Generate loss-vs-tokens and loss-vs-wallclock plots per category."""
        ...
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Markdown report with tables, rankings, and promotion recommendations."""
        ...
```

---

## 8. Dependencies

```toml
# pyproject.toml
[project]
name = "golfcomp"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "flash-attn>=2.5",           # FA2
    "sentencepiece>=0.2",
    "optuna>=3.5",
    "scipy>=1.12",               # Power law fitting
    "pandas>=2.0",               # Results analysis
    "matplotlib>=3.8",           # Plotting
    "brotli>=1.1",
    "zstandard>=0.22",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
gla = ["fla>=0.1"]              # Flash Linear Attention (C1)
mamba = ["mamba-ssm>=2.0"]       # Mamba-3 (C2)
xlstm = ["xlstm>=0.1"]          # xLSTM (C3)
rwkv = ["rwkv>=0.1"]            # RWKV-7 (C4)
all = ["golfcomp[gla,mamba,xlstm,rwkv]"]
```

---

## 9. Verification

### Build & Smoke Test
```bash
pip install -e ".[all]"
python -c "from golfcomp.models import build_model; print('OK')"
```

### Per-Component Tests
```bash
# Model forward pass (no training)
python scripts/run_experiment.py --config configs/baseline.yaml --max-steps 10 --dry-run

# Optuna smoke (1 trial, 1 min)
python scripts/run_optuna.py --config configs/c01_gla.yaml --trials 1 --max-time 60

# Quantization pipeline
python scripts/run_experiment.py --config configs/c15_sdclip.yaml --checkpoint checkpoints/baseline/

# BPB evaluator
python scripts/run_experiment.py --config configs/baseline.yaml --eval-only --checkpoint checkpoints/baseline/
```

### Full Cold Suite
```bash
python scripts/run_all_cold.py --output results/ --phases all
# Or phase by phase:
python scripts/run_all_cold.py --output results/ --phases 1,2
```

---

## NOT in scope

- **Hot experiments (H1-H20):** These need 8xH100. Cold results determine which hot experiments to run.
- **DDP / multi-GPU:** Single 5090 only. DDP is hot-experiment territory.
- **FlashAttention 3:** FA2 only for cold (as specified).
- **Submission export:** Flattening to single train_gpt.py is post-cold work.
- **Custom Triton kernels:** Engineering-heavy, deferred to hot phase (H6).
- **N-gram eval cache (C17/legal check):** Not in cold-exp.md scope. Addressed in hot (H14).
- **Complementary training:** Not a cold experiment. Referenced in hot experiments.

## What already exists

- `experiments/cold-exp.md` — Full experiment definitions (source of truth for configs)
- `experiments/hot-exp.md` — Hot experiment definitions (post-cold)
- `experiments/experiment_plan.md` — Higher-level experiment plan with 20 experiments
- `research/techniques.md` — Comprehensive technique catalog with HPs and results
- `research/compatibility_matrix.md` — Technique interaction matrix (critical for avoiding conflicts)
- `research/research_architectures_and_efficiency.md` — Architecture research
- `baseline.md` — Current frontier reproduction commands and stack details
- **Parameter-golf repo `train_gpt.py`** — Reference implementation with Muon, QAT, data loading

---

## Implementation Phases (for build order)

```
Phase A: Core infrastructure    [CC: ~45 min]
  config.py, data.py, trainer.py, optimizers.py, schedulers.py, ema.py, seed.py, logging.py

Phase B: Model components       [CC: ~30 min]
  attention.py, embeddings.py, activations.py, position.py, residuals.py

Phase C: Baseline model         [CC: ~15 min]
  transformer.py (TransformerModel + TransformerBlock), base.py

Phase D: Recurrence variants    [CC: ~20 min]
  recurrence.py (DepthRecurrence, BasisSharing, RelaxedRecursive)

Phase E: Alt architectures      [CC: ~30 min]
  gla_hybrid.py, mamba_hybrid.py, xlstm_hybrid.py, rwkv.py, mixed_hybrid.py

Phase F: Quantization + compression  [CC: ~20 min]
  sdclip.py, gptq.py, mixed.py, compression.py, artifact.py

Phase G: Evaluation             [CC: ~20 min]
  evaluator.py, ttt.py, metrics.py

Phase H: Experiment runner      [CC: ~15 min]
  runner.py, optuna_search.py, search_spaces.py

Phase I: Configs + scripts      [CC: ~15 min]
  21 YAML configs, run_experiment.py, run_sweep.py, run_optuna.py, run_all_cold.py

Phase J: Analysis               [CC: ~10 min]
  analysis.py, analyze_results.py, train_tokenizer.py
```

Total estimated CC time: ~3.5 hours of implementation.

---

## 10. Comprehensive Results & Error Logging

### 10.1 Logging Infrastructure

Every experiment produces structured logs via `ExperimentLogger` (`golfcomp/utils/logging.py`):

**Output files per experiment** in `results/{experiment_name}/`:
- `config.yaml` — Frozen experiment config snapshot
- `events.jsonl` — Full structured event log (every lifecycle event)
- `metrics.csv` — Time-series: step, wall_time, loss, tokens_seen, tokens/sec, lr_scale, grad_norm, gpu_mem_gb, gpu_util, ema_decay, qat_active, recurrence_active
- `summary.json` — Final results: BPB, tokens/sec, artifact size, promotion status
- `errors.log` — Human-readable error log (empty if clean run)
- `optuna_trials.json` — Optuna trial history (for C1-C5, C18)
- `checkpoint/` — Model + optimizer state (for reuse by C15-C20)
- `loss_curve.png` — Auto-generated loss-vs-tokens plot

**Key classes:**
- `ExperimentLogger` — Writes events.jsonl + metrics.csv + summary.json
- `ExperimentErrorHandler` — Centralized error handling (recover or abort)
- `ResultsAnalyzer` — Post-run comparison tables, plots, promotion recommendations

### 10.2 Training Metrics (logged every 30s)

| Metric | Type | Purpose |
|--------|------|---------|
| `step` | int | Training step number |
| `wall_time_s` | float | Seconds since training start |
| `loss` | float | Cross-entropy loss (6 decimal places) |
| `tokens_seen` | int | Cumulative tokens processed |
| `tokens_per_sec` | float | Throughput (architecture comparison) |
| `lr_scale` | float | Current LR scale (warmup/warmdown) |
| `grad_norm` | float | Gradient norm before clipping |
| `gpu_mem_gb` | float | CUDA allocated memory |
| `gpu_util_pct` | float | GPU utilization percentage |
| `ema_decay` | float | Current EMA decay value |
| `qat_active` | bool | Whether QAT is active this step |
| `recurrence_active` | bool | Whether depth recurrence is active |

### 10.3 Per-Experiment-Category Logging

**Architecture (C1-C5) — additional metrics:**
- `backbone_forward_ms`: Time in GLA/Mamba/xLSTM layers specifically
- `attn_forward_ms`: Time in attention layers
- Library version logged at init
- C3 (xLSTM): Flagged with "SPEED_CAVEAT: PyTorch impl, Triton needed for speed comparison"

**Technique Ablations (C6-C10) — experiment-specific monitors:**
- **C6 BasisSharing:** `basis_condition_number` (every 100 steps), `basis_effective_rank`. ALERT if condition > 1000.
- **C7 RelaxedRecursive:** `lora_adapter_norms` per-block per-loop. ALERT if norms < 0.01 after 500 steps (adapters not learning).
- **C8 EngramLite:** `per_head_gate_activation` (verify all n-gram heads used).
- **C9 SwiGLU:** `activation_sparsity` (compare with LeakyReLU²'s 84-98%). `param_count_match` verify.
- **C10 NoParallelRes:** `per_layer_gradient_norm` (detect gradient flow issues).

**HP Sweeps (C11-C14):**
- Full loss curve per sweep value (40+ data points per 10-min run)
- Edge detection: ALERT if best value is at sweep range boundary
- Cross-sweep sensitivity analysis in `results/hp_sweeps/cross_analysis.json`

**Quantization (C15-C17):**
- Per-k-value: `{artifact_bytes, quant_error_mse, eval_bpb, clipping_fraction}`
- Per-component breakdown for C16: `{mlp_error, attn_error, embed_error}`
- Pareto frontier identification (BPB vs artifact size)

**Eval-Time (C18-C20):**
- Per-TTT-trial: `{pre_bpb, post_bpb, improvement, adapt_time_s}`
- Per-document BPB for C19 SLOT vs LoRA comparison
- Diminishing returns curve for C20 stride sweep

**Vocab (C21):**
- `bpb_per_byte` (tokenizer-agnostic metric)
- `freed_param_budget_mb` per variant
- Collision rates for multi-hash variants

### 10.4 Error Handling Strategy

| Error Type | Severity | Action | Recovery |
|-----------|----------|--------|----------|
| Loss spike > 2x | Warning | Log, continue | Transient, usually recovers |
| Loss spike > 10x | Fatal | Log, abort experiment | Move to next experiment |
| NaN in loss/grads/weights | Fatal | Log, save state, abort | Move to next experiment |
| CUDA OOM | Fatal | Log, abort | Suggest reducing micro_batch |
| Library import failure | Skip | Log, skip experiment | Continue with others |
| GPTQ Hessian non-PD | Recoverable | Regularize (1e-6 diag) | Silent fix, logged |
| GPTQ layer failure | Recoverable | Fall back to SDClip for that layer | Logged as warning |
| Compression failure | Recoverable | Fall back to brotli | Logged as warning |
| Artifact > 16MB | Warning | Log, continue | Still useful for relative comparison |
| Basis condition > 1000 | Warning | Log alert | Continue, flag results unreliable |
| Basis condition > 10000 | Fatal | Abort C6 | Results meaningless |
| LoRA adapters not learning | Warning | Log after 500 steps | Flag C7 results as suspect |
| All Optuna trials pruned | Recoverable | Fall back to baseline HPs | Logged, rerun with baseline |

### 10.5 Post-Run Analysis Output

`run_all_cold.py` produces after all experiments complete:

```
results/
├── run_manifest.json              # Full run metadata (times, statuses, hardware)
├── comparison_table.csv           # All experiments side-by-side
├── promotion_recommendations.md   # Which experiments to promote to hot
└── plots/
    ├── loss_vs_tokens_architectures.png
    ├── loss_vs_tokens_ablations.png
    ├── hp_sweep_{ema,wd,recurrence,warmdown}.png
    ├── quant_pareto.png
    ├── ttt_sweep.png
    └── vocab_comparison.png
```

`comparison_table.csv` columns: `experiment | category | bpb | delta_bpb | tokens_per_sec | artifact_mb | status | errors | promoted`

`run_manifest.json` includes: hardware info, per-experiment start/end times and status, Optuna trial counts, overall summary, and promotion list.

---

## 11. Engineering Review Findings

### First Review — Architecture
| # | Issue | Resolution | Confidence |
|---|-------|-----------|------------|
| 1 | Mamba-3 not in mamba-ssm library | Implement 3 deltas on Mamba-2 base (complex gates, trapezoidal disc., MIMO) | 4/10 (downgraded after 2nd review) |
| 2 | Basis Sharing + GPTQ unknown interaction | Reconstruct per-layer weights then quantize normally | 7/10 |
| 3 | xLSTM no pip library | PyTorch mLSTM from paper, log speed caveat | 8/10 |
| 4 | Compute budget 12hrs vs 8hrs | Accept 12hrs, run overnight | 10/10 |
| 5 | RWKV-7 library is research-grade repo | Adapt from RWKV-LM GitHub, wrap in our model interface | 7/10 |
| 6 | Factorized embed tied logits (C21) | Factorized: `hidden @ proj.T @ low_rank.T` works. Multi-hash: NEEDS FIX (see 2nd review) | 5/10 (downgraded) |

### Second Review — Critical Fixes Required

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 7 | **OrthoInit missing** — SmearGate requires `torch.nn.init.orthogonal_()` on model init. Without it, SmearGate is +0.003 BPB worse. Affects EVERY experiment. | CRITICAL | Add `init_weights()` to BaseModel calling `orthogonal_` on all linear layers where SmearGate is used. Must be called after model construction. |
| 8 | **Muon Newton-Schulz wrong for wide matrices** — Down-projections (1536→512) have `shape[0] < shape[1]`. The N-S iteration `1.5X - 0.5 X@X^T@X` is for tall matrices only. Wide needs `1.5X - 0.5 X^T@X@X` or transpose the gradient first. | CRITICAL | In `Muon.step()`: if grad is wide (cols > rows), transpose before N-S, transpose back after. Or: always make gradient tall by choosing the smaller dimension. |
| 9 | **Multi-hash tied logits are wrong** — `sum(hidden @ table.weight.T for table in tables)` doesn't reverse the hash. Forward: `embed(id) = sum(table_i[h_i(id)])`. Reverse: for each vocab id, compute `hidden . (table_1[h_1(id)] + table_2[h_2(id)] + table_3[h_3(id)])`. The current code computes `hidden @ table_1.T + hidden @ table_2.T + hidden @ table_3.T` which gives logits for hash buckets, not vocab ids. | CRITICAL | Implement explicit logit computation: loop over vocab or use scatter to reconstruct per-vocab embedding, then dot with hidden. Or: don't tie output embeddings for multi-hash (use separate output projection). |
| 10 | **BasisSharing reconstructs every forward pass** — `get_weight(layer_idx)` called every step per layer per matrix. O(num_layers × dim × rank) overhead per forward. Will dominate training time. | HIGH | Cache reconstructed weights. Only recompute when coefficients are updated (once per optimizer step, not per forward micro-batch). |
| 11 | **C7 virtual layer math error** — cold-exp.md says "6 shared × 2 loops = 14 virtual layers." But 6×2=12. | MEDIUM | Fix: either 7 shared × 2 = 14, or 6 shared × 2 + 2 unshared = 14. Ask user to clarify cold-exp.md. Implement as 7×2=14 for now. |
| 12 | **XSA first-sequence behavior** — `_prev_kv = None` means first sequence skips XSA. Must match competition reference. | MEDIUM | Initialize `_prev_kv` to learned or zero-filled KV of correct shape. Match whatever train_gpt.py does. |
| 13 | **Optuna "best trial" ambiguity** — MedianPruner may prune promising trials early. "Best" should mean best completed trial, not best at-pruning loss. | LOW | Specify: `study.best_trial` returns best completed. Add note: if all trials pruned, re-run top-3 by at-pruning loss without pruning. |
| 14 | **SWA intentionally omitted** — Not in plan but in research docs. | LOW | Intentional: SWA is complementary to EMA but adds complexity. Defer to hot experiments. Document this in NOT-in-scope. |
| 15 | **xlstm/rwkv deps are fictional** — No pip packages exist. | LOW | Fix pyproject.toml: remove `[project.optional-dependencies]` for xlstm and rwkv. These are custom implementations. Only FLA and mamba-ssm are real deps. |
| 16 | **Mamba-3 custom impl is 5-10x slower than Mamba-2 kernel** — Complex-valued selective scan can't bolt onto existing CUDA kernel. PyTorch fallback destroys throughput. | HIGH | Downgrade confidence on C2/C5. Log "THROUGHPUT_CAVEAT: Pure PyTorch Mamba-3, not kernel-optimized" same as xLSTM. Throughput comparisons invalid for C2/C5, only sample efficiency. |

### Code Quality Review
- **DRY:** Extract `HybridBaseModel` for C1-C5 (shared embed→smear→bigram→[backbone]→[attn]→output pattern)
- **Config inheritance:** YAML configs use `_base: baseline.yaml` with per-experiment overrides
- **OrthoInit:** Add to model init (CRITICAL, see #7 above)
- **Muon wide-matrix fix:** Required in optimizer (CRITICAL, see #8 above)

### Test Plan

```
SMOKE TESTS (run before any experiment)
════════════════════════════════════════════════════════════
[+] golfcomp/models/
    ├── [GAP] TransformerModel.forward — shape test, 1 step
    ├── [GAP] GLAHybridModel.forward — verify FLA integration
    ├── [GAP] MambaHybridModel.forward — verify Mamba-3 impl
    ├── [GAP] XLSTMHybridModel.forward — verify mLSTM impl
    ├── [GAP] RWKVModel.forward — verify RWKV-7 wrapper
    ├── [GAP] MixedHybridModel.forward — verify layer composition
    ├── [GAP] BasisSharing.get_weight — SVD reconstruction
    └── [GAP] RelaxedRecursive.forward — LoRA pass differentiation

[+] golfcomp/training/
    ├── [GAP] Muon.step — Newton-Schulz convergence (5 iterations)
    ├── [GAP] WarmdownScheduler — LR curve shape matches 1-sqrt
    ├── [GAP] EMAWrapper — decay math correctness
    └── [GAP] Trainer.train — 10 steps, loss decreases

[+] golfcomp/quantization/
    ├── [GAP] SDClipQuantizer — roundtrip: quantize → dequantize, MSE < threshold
    ├── [GAP] GPTQQuantizer — single layer quantize, verify Hessian shape
    └── [GAP] Compressor — roundtrip: compress → decompress == original

[+] golfcomp/evaluation/
    ├── [GAP] BPBEvaluator — sliding window produces reasonable BPB (0.5-2.0 range)
    ├── [GAP] LoRATTT — adapt on 1 doc, loss decreases
    └── [GAP] SLOTTTT — adapt on 1 doc, loss decreases

[+] golfcomp/experiments/
    ├── [GAP] OptunaSearcher — 1 trial completes, returns result
    └── [GAP] ExperimentRunner — full pipeline: train(10 steps) → quant → eval

COVERAGE: 0/19 paths tested
GAPS: 19 smoke tests needed
```

All tests are unit/integration tests using pytest. No E2E browser tests (this is a CLI/GPU workload).

### Performance Review
- **Memory:** ~20M param models at 512d + FA2 + seq1024 + micro_batch=32 = ~2-3GB peak. 32GB 5090 is fine.
- **Optuna overhead:** Model construction per trial is <1s. Negligible vs 10-min training.
- **GPTQ calibration:** Self-generated data, one forward pass. ~500MB activation memory. Fine.
- **No N+1 or scaling issues.** Single GPU, sequential experiments.

### Failure Modes
| Codepath | Failure mode | Test? | Error handling? | User-visible? |
|----------|-------------|-------|-----------------|---------------|
| FLA GLA import | Library not installed / API changed | Smoke test | ImportError → skip C1 | Clear error msg |
| Mamba-3 custom impl | Numerical instability in complex gates | Smoke test | NaN check in trainer | Loss = NaN logged |
| BasisSharing SVD | Degenerate bases (rank collapse) | Smoke test | Monitor basis condition number | Silent degradation |
| GPTQ Cholesky | Non-PD Hessian | No | 1e-6 diagonal regularization | Silent fix |
| Optuna pruning | All trials pruned (bad search space) | Logged | Fallback to baseline HPs | Warning in results |
| SP16384 tokenizer | Training fails on FineWeb sample | Smoke test | Retry with different coverage | Clear error |

**Critical gap:** BasisSharing rank collapse has no test AND could be silent. Adding a condition number monitor to the training loop for C6.

---

## 11. Worktree Parallelization Strategy

Sequential implementation, no parallelization opportunity. All experiments run on the same GPU sequentially. The build phases (A-J) are sequential dependencies. However, implementation of Phases B-E (model components + architectures) could be parallelized across worktrees since they're independent modules that don't share files:

| Lane | Modules | Depends on |
|------|---------|------------|
| Lane A | config.py, data.py, trainer.py, optimizers.py, schedulers.py, ema.py | — |
| Lane B | attention.py, embeddings.py, activations.py, position.py, residuals.py | — |
| Lane C | transformer.py, base.py, recurrence.py | Lane A + B |
| Lane D | gla_hybrid.py, mamba_hybrid.py, xlstm_hybrid.py, rwkv.py | Lane B + C |
| Lane E | sdclip.py, gptq.py, compression.py, evaluator.py, ttt.py | Lane A |
| Lane F | runner.py, optuna_search.py, configs, scripts | Lane C + D + E |

**Parallel lanes:** Launch A + B in parallel. Then C + E in parallel. Then D. Then F.

---

## Eng Review Completion Summary (2 passes)

- Step 0: Scope Challenge — scope accepted as-is (all 21 experiments)
- Architecture Review (pass 1): 6 issues found, all resolved
- Architecture Review (pass 2): 10 additional issues found, 3 CRITICAL
- Code Quality Review: 2 critical fixes (OrthoInit, Muon wide-matrix)
- Test Review: 19 smoke tests identified, 0 covered, all added to plan
- Performance Review: 1 issue (BasisSharing forward-pass overhead, fix: caching)
- NOT in scope: written (hot experiments, DDP, FA3, Triton kernels, submission export, SWA)
- What already exists: written (7 research/experiment docs, baseline commands)
- Failure modes: 1 critical gap flagged (BasisSharing rank collapse monitor)
- Parallelization: 6 lanes, 3 parallel opportunities
- Lake Score: 5/5 recommendations chose complete option

### Critical Fixes Before Implementation
1. Add OrthoInit to model construction (affects all experiments)
2. Fix Muon Newton-Schulz for wide gradient matrices (affects all experiments)
3. Fix multi-hash tied embedding logit computation (affects C21 variants C, D)
4. Cache BasisSharing reconstructed weights (affects C6)
5. Fix C7 virtual layer count (7×2=14, not 6×2)
6. Add throughput caveats for C2, C3, C5 (PyTorch-only, no kernel optimization)

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Outside Voice | Agent subagent | Independent 2nd opinion | 1 | ISSUES FOUND | 3 critical, 4 high, 3 medium |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | CLEAR (PLAN) | 16 total issues, 3 critical fixed |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

**VERDICT:** ENG REVIEW CLEARED (pass 2) — 16 issues identified across 2 reviews, all addressed. 3 critical fixes documented for implementation.
