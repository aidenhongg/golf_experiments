# Launch Guide: Cold Experiments on RunPod RTX 5090

## 1. Provision RunPod Instance

- **GPU**: RTX 5090 (32GB GDDR7)
- **Template**: `runpod/pytorch:2.7.0-py3.11-cuda12.8.1-devel-ubuntu22.04`
- **Disk**: 100GB (data + checkpoints + results)
- **Cloud type**: Secure Cloud (lower latency)

Connect via SSH or web terminal once running.

---

## 2. Clone & Install

```bash
# Upload or clone your project
cd /workspace
# (upload COLD_EXPR/ directory via scp, rsync, or git)

cd /workspace/COLD_EXPR

# Install PyTorch 2.7+ for Blackwell SM_120 (if not already in the template)
pip install 'torch>=2.7' --index-url https://download.pytorch.org/whl/cu128

# Install the package + core deps
pip install -e .

# Verify GPU is recognized
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import torch; print('SM:', torch.cuda.get_device_capability())"
# Expected: SM: (12, 0) for RTX 5090

# Verify attention backend (uses PyTorch SDPA on Blackwell, no flash-attn needed)
python -c "from golfcomp.config import ExperimentConfig; print('golfcomp OK')"
```

### Optional: Install C1 (GLA) dependencies
```bash
# fla uses Triton kernels (arch-portable, works on SM_120)
pip install flash-linear-attention
```

### Optional: Install C2 (Mamba) dependencies
```bash
# mamba-ssm must be built from source for SM_120 (pre-built wheels lack it)
export TORCH_CUDA_ARCH_LIST="12.0"
export MAMBA_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
# Verify:
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
```

### Note on flash-attn (FA2)
flash-attn 2.x **cannot compile for SM_120** (nvcc segfaults, see
[Dao-AILab/flash-attention#2361](https://github.com/Dao-AILab/flash-attention/issues/2361)).
The codebase automatically falls back to PyTorch's native `scaled_dot_product_attention`
which uses the same memory-efficient/flash algorithms via CUTLASS on Blackwell.

---

## 3. Download & Prepare Data

### What the pipeline needs

```
COLD_EXPR/
└── data/
    ├── fineweb_sp8192/          # REQUIRED for phases 0-6
    │   ├── train/
    │   │   ├── shard_000.bin    # uint16 token IDs, ~100MB each
    │   │   ├── shard_001.bin
    │   │   └── ...
    │   └── val/
    │       └── shard_000.bin    # validation split (same format)
    ├── fineweb_sp16384/         # OPTIONAL, only for C21 (phase 7)
    │   ├── train/*.bin
    │   └── val/*.bin
    └── tokenizers/
        ├── fineweb_8192_bpe.model   # REQUIRED, SentencePiece BPE model
        └── fineweb_16384_bpe.model  # OPTIONAL, only for C21
```

- **Format**: Binary files of packed `uint16` token IDs (2 bytes per token)
- **Train shards**: ~10B tokens total across shards, ~20GB on disk
- **Val data**: Used by BPBEvaluator for scoring (sliding window, per-byte BPB)
- **Tokenizer**: SentencePiece `.model` file, needed for byte-length introspection during BPB eval

### Option A: Automated download (recommended)

```bash
cd /workspace/COLD_EXPR
pip install brotli huggingface_hub

# Downloads parameter-golf repo, runs their data script, symlinks into place
python scripts/download_data.py --variant sp8192

# Verify everything is in place
python scripts/download_data.py --verify-only
```

Expected output:
```
=== Data Verification ===
  [OK] SP8192 data: 42 shards, 19847 MB
  [OK] SP8192 tokenizer: data/tokenizers/fineweb_8192_bpe.model
  [SKIP] SP16384 tokenizer not found (only needed for C21)
  [SKIP] SP16384 data not found (only needed for C21)

  Ready for phases 0-6.
```

### Option B: Manual download

```bash
# 1. Clone parameter-golf repo
git clone --depth=1 https://github.com/openai/parameter-golf /tmp/parameter-golf
cd /tmp/parameter-golf
pip install -r requirements.txt

# 2. Download tokenized FineWeb (SP8192)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest

# 3. Link into COLD_EXPR data dirs
cd /workspace/COLD_EXPR
mkdir -p data/fineweb_sp8192/train data/fineweb_sp8192/val data/tokenizers

# Training shards
ln -s /tmp/parameter-golf/data/datasets/fineweb10B_sp8192/*.bin data/fineweb_sp8192/train/
# OR copy if symlinks cause issues:
# cp /tmp/parameter-golf/data/datasets/fineweb10B_sp8192/*.bin data/fineweb_sp8192/train/

# Validation data (check parameter-golf repo for val split location)
# If no separate val split, copy a few shards:
# cp data/fineweb_sp8192/train/shard_000.bin data/fineweb_sp8192/val/

# Tokenizer
ln -s /tmp/parameter-golf/data/tokenizers/fineweb_8192_bpe.model data/tokenizers/
```

### Option C: Pre-cached data (RunPod network volume)

```bash
# If you have data on a RunPod network volume:
ln -s /runpod-volume/fineweb_sp8192 data/fineweb_sp8192
ln -s /runpod-volume/tokenizers data/tokenizers
```

### SP16384 data (for C21 only, optional)

```bash
# 1. Train the 16K tokenizer
#    Needs raw FineWeb text — extract from existing shards or download separately
python scripts/train_tokenizer.py \
  --input /path/to/fineweb_raw_sample.txt \
  --prefix data/tokenizers/fineweb_16384_bpe \
  --vocab-size 16384

# 2. Re-tokenize FineWeb with 16K vocab
#    Use parameter-golf's tokenization script with the new model:
python scripts/download_data.py --variant sp16384 --raw-text /path/to/fineweb_raw_sample.txt
```

### Verify data before running experiments

```bash
# Quick check: confirms shards exist, tokenizer loads, format is correct
python scripts/download_data.py --verify-only

# Full smoke test: loads data, builds model, runs 10 steps
python scripts/run_experiment.py --config configs/baseline.yaml --dry-run
```

### Data format details

| Field | Value |
|-------|-------|
| Encoding | `uint16` (2 bytes per token) |
| Shard size | ~100-500MB each |
| Total train tokens | ~10B (FineWeb subset) |
| Vocab sizes | 8192 (SP8192) or 16384 (SP16384) |
| Sequence length | 1024 tokens per training sample |
| Doc boundaries | Token ID 0 separates documents (for eval) |
| Train loading | Sequential shard read, shuffled shard order per epoch |
| Eval loading | All val shards, split on doc boundaries, sliding window |

---

## 4. Smoke Test (2 minutes)

```bash
cd /workspace/COLD_EXPR

# Dry run: 10 steps, verify everything connects
python scripts/run_experiment.py --config configs/baseline.yaml --dry-run

# Expected output:
#   Result: BPB=N/A, loss=<some number>
# If this works, your stack is ready.
```

---

## 5. Run All Cold Experiments

### Option A: Full overnight run (~12 hours)

```bash
# Run all 9 phases sequentially
nohup python scripts/run_all_cold.py --output results/ --phases all \
  > cold_run.log 2>&1 &

# Monitor progress
tail -f cold_run.log
```

### Option B: Phase by phase (recommended for debugging)

```bash
# Phase 0: Verify data
python scripts/run_all_cold.py --output results/ --phases 0

# Phase 1: Train cold baseline (20 min) — MUST complete before phases 5-7
python scripts/run_all_cold.py --output results/ --phases 1

# Verify baseline result
cat results/cold_baseline/summary.json

# Phase 2: Architecture search (5 hrs) — C1-C5, Optuna 5 trials + best rerun
python scripts/run_all_cold.py --output results/ --phases 2

# Phase 3: Technique ablations (100 min) — C6-C10, single runs
python scripts/run_all_cold.py --output results/ --phases 3

# Phase 4: HP sweeps (200 min) — C11-C14, grid search
python scripts/run_all_cold.py --output results/ --phases 4

# Phase 5: Quantization (5 min) — C15-C17, post-training on baseline checkpoint
python scripts/run_all_cold.py --output results/ --phases 5

# Phase 6: Eval-time tricks (3 hrs) — C18-C20, TTT/sliding window sweeps
python scripts/run_all_cold.py --output results/ --phases 6

# Phase 7: Vocab experiment (80 min) — C21, requires SP16384 tokenizer
# (skip if you don't have SP16384 data prepared)
python scripts/run_all_cold.py --output results/ --phases 7

# Phase 8: Generate analysis
python scripts/run_all_cold.py --output results/ --phases 8
# Or standalone:
python scripts/analyze_results.py --results results/ --output results/
```

---

## 6. Run Individual Experiments

```bash
# Single experiment
python scripts/run_experiment.py --config configs/c09_swiglu.yaml --seed 42

# Optuna HP search (e.g., GLA hybrid, 5 trials at 10 min each)
python scripts/run_optuna.py --config configs/c01_gla_hybrid.yaml --trials 5 --max-time 600

# Post-training eval (needs baseline checkpoint)
python scripts/run_experiment.py --config configs/c18_ttt_sweep.yaml \
  --eval-only --checkpoint results/cold_baseline/checkpoint.pt
```

---

## 7. Analyze Results

```bash
python scripts/analyze_results.py --results results/ --output results/

# Outputs:
#   results/comparison_table.csv         — all experiments side-by-side
#   results/promotion_recommendations.md — what to promote to hot (8xH100)
#   results/plots/                       — loss curves, pareto charts
```

---

## 8. RTX 5090 Tuning Notes

| Setting | Value | Why |
|---------|-------|-----|
| `cudnn.benchmark` | `True` (already set) | Auto-tune kernels for 5090 architecture |
| `cudnn.deterministic` | `False` (already set) | 10-30% throughput gain |
| `torch.compile` | On activations + RoPE | Fuses ops on 5090 tensor cores |
| Attention | PyTorch SDPA | FA2 can't compile for SM_120; SDPA uses CUTLASS flash backend |
| Micro-batch | 32K tokens | Fits in 32GB GDDR7 with SDPA |
| Grad accum | 4 steps | Effective batch = 128K tokens |

**If OOM**: Reduce `micro_batch_tokens` in the YAML config from 32768 to 16384 and increase `grad_accum_steps` from 4 to 8.

**If slow**: Verify GPU architecture is detected correctly. RTX 5090 is SM_120 (Blackwell consumer). Run:
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# Should print (12, 0) for RTX 5090
```

---

## 9. Quick Reference

| Command | Time | What |
|---------|------|------|
| `--dry-run` | 1 min | Verify stack works |
| `--phases 1` | 20 min | Baseline only |
| `--phases 1,3` | 2 hrs | Baseline + ablations |
| `--phases 1,2,3,4` | 8 hrs | All training experiments |
| `--phases all` | 12 hrs | Full cold suite |

---

## 10. Promotion Criteria

After all experiments complete, check `results/promotion_recommendations.md`:

| Category | Promote if... |
|----------|---------------|
| Architecture (C1-C5) | >= 5% faster tokens/sec AND <= 1% worse loss |
| Ablation (C6-C10) | >= 0.002 BPB improvement |
| HP sweep (C11-C14) | Optimal value differs from frontier |
| Quantization (C15-C17) | >= 200KB artifact reduction OR >= 0.001 BPB |
| Eval-time (C18-C20) | >= 0.001 BPB improvement |

Promoted experiments get escalated to hot runs on 8xH100.
