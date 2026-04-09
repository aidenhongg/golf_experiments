# Parameter Golf — Baseline (Current Frontier)

**Date:** 2026-04-08
**Source:** PR #1477 (1.0822 BPB, 3-seed mean) + PR #1471 (1.0866 no-TTT variant)

---

## The Stack (PR #1477 — Current Best)

```
val_bpb = 1.0822 (3-seed mean, std 0.0005) | ~15.99 MB | 8xH100 SXM, 600s
```

### Architecture
- 11 layers, 512 dimensions, 8 attention heads, 4 KV heads (GQA)
- Depth recurrence on layers 4-5 (14 virtual layers)
- Parallel residuals from layer 7+ (attention + MLP on separate lanes, learned merge scalar)
- Skip gates (learned gates on residual connections)
- XSA on all 11 layers
- LeakyReLU(0.5)² activation
- VE128 (Value Embedding, dim=128, layers 9-10)
- Tied embeddings, logit softcap=30.0

### Tokenizer
- SP8192 (8192-token SentencePiece BPE, FineWeb-trained)

### Training
- MuonEq-R optimizer (equalized Muon variant)
- QK-Gain 5.0
- Weight decay: 0.095
- Matrix LR: 0.022
- EMA decay: 0.9965
- Warmdown: 72%
- Late recurrence start: step 2000
- FlashAttention 3
- Batch: 786K tokens/step on 8xH100
- ~7000 steps in 600s

### Quantization
- SDClip: clip = k * std(row), k=12.85 for int6, k=20.0 for int8 embeds
- Full Hessian GPTQ + Cholesky + actorder
- GPTQ on embeddings
- Zero selective pruning

### Eval
- Sliding window (stride=64)
- Score-First TTT (3 epochs, backward-looking only)

### Compression
- Brotli

---

## No-TTT Variant (PR #1471 — 1.0866)

Same stack minus TTT. Uses 3-layer depth recurrence (layers 3,4,5) instead of (4,5).
This is the "training-only" baseline — useful for comparing training improvements
without TTT confounding the results.

---

## Reproduction Commands

### With TTT (1.0822)
```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
SEED=42 TTT_ENABLED=1 PARALLEL_START_LAYER=7 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Without TTT (1.0866)
```bash
SEED=42 VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
RECUR_START_STEP=2000 WARMDOWN_FRAC=0.72 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## What Makes This Stack Work

The frontier breaks three "hard conflicts" from the earlier compatibility matrix:

1. **EMA + Depth Recurrence** — Solved by: EMA 0.9965 (not 0.997), late recurrence start (step 2000), WD=0.095, only 3 layers recur
2. **TTT + Depth Recurrence** — Solved by: score-first TTT is eval-time only, no gradient coupling during training
3. **XSA + Depth Recurrence** — Works: XSA applies to all 11 virtual layers including recurred ones

Key hyperparameter shifts from earlier consensus:
- WD 0.095 (was 0.04) — 2.4x higher
- EMA 0.9965 (was 0.997) — closer to 1
- Warmdown 72% (was ~50%) — much more aggressive
- SP8192 (was SP4096) — 2x vocabulary
