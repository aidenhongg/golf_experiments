# Cold-Start Experiments (Single 5090, 32GB GDDR7)

Experiments that can be run NOW on a single RTX 5090 to get **relative rankings**
between techniques. Results won't match absolute BPB on 8xH100 (fewer steps, smaller
batch), but directional comparisons transfer: if technique A beats B on your 5090,
it will very likely beat B on 8xH100 too.

**Hardware:** RTX 5090 (Blackwell), 32GB GDDR7, FP8 tensor cores, FA2
**Expected steps:** ~2500-4000 in 10 min (single GPU), or run longer (20-30 min)
**Batch size:** ~64-128K tokens/step with gradient accumulation
**Metric:** Relative BPB improvement vs cold baseline, NOT absolute BPB

---

## Cold Baseline

Run the frontier stack adapted for single GPU:
```
11L/512d/GQA, SP8192, LeakyReLU(0.5)²
XSA-all, EMA 0.9965, WD=0.095
3-layer depth recurrence (3,4,5), late start (proportional to total steps)
Parallel Residuals (L7+), QK-Gain 5.0, Skip Gates
SDClip, Brotli
Gradient accumulation to ~128K effective batch
FA2 (not FA3), single GPU
```
Run for 20 minutes. Record loss curve every 30 seconds. This is your reference.

---

## Architecture Comparisons (C1-C5)

These answer: "Which backbone learns better per token?"
Compare loss vs tokens-seen (NOT wall-clock) to isolate sample efficiency from speed.

### C1: GLA Hybrid vs Transformer
```
8 GLA layers (FLA library) + 3 attention layers (final)
  512d, XSA on final 3 attention layers
  EMA 0.9965, SDClip, SP8192
  Muon for projections, AdamW for GLA gates
  Everything else same as cold baseline
```
- **Measure:** tokens/sec throughput + loss-vs-tokens curve
- **Compare against:** cold baseline (pure transformer)
- **Key question:** Is GLA faster AND/OR more sample-efficient?
- **Duration:** 20 min

### C2: Mamba-3 Hybrid vs Transformer
```
8 Mamba-3 layers + 3 attention layers (final)
  512d, XSA on final 3 attention layers
  EMA 0.9965, SP8192
  Muon for projections, AdamW for SSM params
  Int8 for Mamba layers during eval (SSM quant sensitivity)
```
- **Measure:** tokens/sec + loss-vs-tokens
- **Key question:** Does Mamba-3 at dim=512 overcome the prior negative result?
- **Duration:** 20 min

### C3: xLSTM/mLSTM Hybrid vs Transformer
```
8 mLSTM blocks + 3 attention layers (final)
  512d, XSA on final 3 attention layers
  EMA 0.9965, SP8192
  Muon for projections, AdamW for mLSTM params
```
- **Measure:** tokens/sec (verify the 3.5x claim at small scale) + loss-vs-tokens
- **Key question:** Does the speed advantage actually materialize at 512d on consumer GPU?
- **Duration:** 20 min

### C4: RWKV-7 Pure vs Transformer
```
11L RWKV-7 backbone, 512d
  SP8192, no attention (no XSA possible)
  No EMA (no XSA to anchor it)
  WD=0.095, SDClip
```
- **Measure:** tokens/sec + loss-vs-tokens
- **Key question:** Can RWKV-7 compete without XSA/EMA?
- **Duration:** 20 min

### C5: GLA Hybrid + Mamba-3 Bottom Layers
```
4 Mamba-3 layers (bottom) + 4 GLA layers (middle) + 3 attention (top)
  512d, XSA on top 3 attention layers
  Mixed optimizer: AdamW for SSM, Muon for projections, AdamW for GLA gates
```
- **Measure:** Does mixing two sub-quadratic architectures help or hurt?
- **Key question:** Is this better than pure GLA hybrid or pure Mamba-3 hybrid?
- **Duration:** 20 min

---

## Technique Ablations (C6-C10)

These answer: "Does technique X improve the loss curve?"
All modifications are against the cold baseline. Change ONE thing at a time.

### C6: Basis Sharing vs Hard Depth Recurrence
```
Cold baseline BUT replace 3-layer depth recurrence with:
  SVD basis sharing across all 11 layers
  Shared bases + per-layer coefficients
  No late recurrence start needed (it's soft sharing)
  Keep EMA 0.9965, XSA-all
```
- **Key question:** Does Basis Sharing converge? Is loss competitive with hard recurrence?
- **Why it matters:** If yes, it avoids the fragile late-start/high-WD/high-decay workarounds

### C7: Relaxed Recursive (LoRA) vs Hard Depth Recurrence
```
Cold baseline BUT replace 3-layer recurrence with:
  6 shared blocks x 2 loops + rank-8 LoRA per pass
  14 virtual layers with per-pass differentiation
  Keep EMA 0.9965, XSA-all
```
- **Key question:** Does LoRA relaxation give clean EMA compatibility without the late-start hack?

### C8: EngramLite vs BigramHash
```
Cold baseline BUT replace BigramHash(3072) with:
  EngramLite (multi-head n-gram hashing + context gating)
  Same param budget
```
- **Key question:** Does EngramLite's richer representation beat BigramHash on SP8192?

### C9: SwiGLU vs LeakyReLU(0.5)²
```
Cold baseline BUT swap activation:
  SwiGLU replacing LeakyReLU(0.5)²
  Adjust MLP hidden dim to match param count
```
- **Key question:** Is LeakyReLU² actually optimal or just well-tuned?
- **Note:** SwiGLU enables potential 2:4 sparsity exploration (unlike ReLU² which already has natural sparsity)

### C10: Parallel Residuals Ablation
```
Cold baseline BUT remove parallel residuals:
  Standard serial residuals throughout (no separate attention/MLP lanes)
```
- **Key question:** How much do parallel residuals contribute?
- **If large:** they're a core technique. **If small:** they're tuning.

---

## Hyperparameter Sweeps (C11-C14)

These answer: "Are the frontier hyperparameters actually optimal?"
Critical because HP optima may shift with architecture changes.

### C11: EMA Decay Sweep
```
Cold baseline with EMA decay in {0.990, 0.995, 0.9965, 0.998, 0.999}
```
- **Key question:** Is 0.9965 a sharp optimum or a broad plateau?
- **Duration:** 5 runs x 10 min each

### C12: Weight Decay Sweep
```
Cold baseline with WD in {0.04, 0.06, 0.095, 0.12, 0.15}
```
- **Key question:** Is 0.095 optimal? Does higher WD help quantization further?

### C13: Recurrence Layer Sweep
```
Cold baseline with depth recurrence on:
  A) layers 3,4,5 (current)
  B) layers 4,5 only (PR #1477 uses this)
  C) layers 2,3,4,5
  D) layers 5,6,7
  E) no recurrence (11 unique layers)
```
- **Key question:** Which layers benefit most from recurrence? Is deeper or shallower better?

### C14: Warmdown Sweep
```
Cold baseline with warmdown fraction in {0.50, 0.60, 0.72, 0.80, 0.90}
```
- **Key question:** Is 72% optimal or can we push further?
```
A) SP8192 with full FP16 embedding (current frontier)
     8192 × 512 = 8.4 MB embedding, ~7.6 MB left for model

B) SP16384 with ALBERT-style factorized embedding
     16384 × 64 low-rank (FP16) + 64 × 512 projection = ~2.1 MB embedding
     ~13.9 MB left for model — use freed budget for:
       Option B1: 2x wider MLP (1536 → 3072)
       Option B2: 3 extra layers (11L → 14L)
       Option B3: both (13L + wider MLP)

C) SP16384 with multi-hash compositional embedding
     3 tables of 5461 × 512, compose via sum: table1[h1(t)] + table2[h2(t)] + table3[h3(t)]
     ~8.4 MB embedding (same budget as SP8192), virtually zero collisions

D) SP16384 with factorized + multi-hash hybrid
     3 tables of 5461 × 64 + shared 64 × 512 projection
     ~2.1 MB embedding, zero collisions, maximum freed budget
```
- **Key question:** Does SP16384's better text compression + freed param budget beat SP8192's full embeddings?
- **What to measure:** BPB per byte (tokenizer-agnostic), artifact size, tokens/sec
- **Critical detail:** Tied output embeddings must use the same factorized/hashed scheme. Verify the output logit computation works correctly with factorized embeddings: `logits = hidden @ projection.T @ low_rank.T`
- **Duration:** 4 runs x 20 min each
- **Why this could be a breakthrough:** 6.3 MB of freed artifact budget is enormous — it's enough for a fundamentally larger model while getting better tokenization

---

## Quantization & Compression (C15-C17)

These answer: "Can we squeeze more quality from the 16MB budget?"
Can test post-training without full training runs.

### C15: SDClip k-Value Sweep
```
Train cold baseline once, then quantize with:
  k in {10.0, 11.0, 12.0, 12.85, 14.0, 16.0} for int6
  k in {16.0, 18.0, 20.0, 22.0, 25.0} for int8 embeds
```
- **Key question:** Are the current k values optimal?
- **Duration:** Train once (20 min), quantize 30 times (fast)

### C16: Int5 MLP + Int6 Attention vs Uniform Int6
```
Train cold baseline once, then quantize:
  A) All int6 (current)
  B) Int5 for MLP weights, int6 for attention
  C) Int6 for MLP, int8 for attention
```
- **Key question:** Can mixed precision per-component beat uniform int6?
- **Note:** Int5 MLP frees artifact bytes for bigger model

### C17: Brotli vs Zstd-22 vs LZMA-9
```
Train cold baseline once, compress with each:
  A) Brotli (current frontier)
  B) zstd level 22
  C) LZMA level 9
  D) ANS/Huffman entropy coding
```
- **Key question:** Which compressor gives smallest artifact for int6 weights?
- **Duration:** Train once, compress 4 times (seconds each)

---

## Eval-Time Techniques (C18-C20)

These answer: "Can we extract more BPB at eval without changing the model?"
Eval runs on single GPU just fine. No multi-GPU needed.

### C18: Score-First TTT Hyperparameter Sweep
```
Train cold baseline once, then eval with TTT:
  Epochs: {1, 2, 3, 5, 8}
  LR: {0.001, 0.002, 0.005, 0.01}
  Optimizer: {SGD+momentum, AdamW+cosine}
  LoRA rank: {4, 8, 16}
```
- **Key question:** What are the optimal TTT hyperparameters?
- **Duration:** Train once (20 min), eval many times (~5 min each)

### C19: SLOT (Output-Head TTT) vs LoRA TTT
```
Train cold baseline once, then eval with:
  A) LoRA TTT rank-8 (current)
  B) SLOT: single 512-dim delta vector at last hidden layer
  C) Both combined
```
- **Key question:** Is SLOT cheaper and equally effective?

### C20: Sliding Window Stride Sweep
```
Train cold baseline once, then eval with:
  Stride: {32, 64, 128, 256}
  Window: {1024, 2048, 4096}
```
- **Key question:** Is stride=64 + window=2048 actually optimal? Would stride=32 help at eval-time cost?
- **Note:** Smaller stride = more context per token = better BPB but slower eval

---

## Vocab / Embedding Innovation (C21)

### C21: SP16384 + Factorized Embedding vs SP8192
```
A) SP8192 with full FP16 embedding (current frontier)
     8192 × 512 = 8.4 MB embedding, ~7.6 MB left for model

B) SP16384 with ALBERT-style factorized embedding
     16384 × 64 low-rank (FP16) + 64 × 512 projection = ~2.1 MB embedding
     ~13.9 MB left for model — use freed budget for:
       Option B1: 2x wider MLP (1536 → 3072)
       Option B2: 3 extra layers (11L → 14L)
       Option B3: both (13L + wider MLP)

C) SP16384 with multi-hash compositional embedding
     3 tables of 5461 × 512, compose via sum: table1[h1(t)] + table2[h2(t)] + table3[h3(t)]
     ~8.4 MB embedding (same budget as SP8192), virtually zero collisions

D) SP16384 with factorized + multi-hash hybrid
     3 tables of 5461 × 64 + shared 64 × 512 projection
     ~2.1 MB embedding, zero collisions, maximum freed budget
```
- **Key question:** Does SP16384's better text compression + freed param budget beat SP8192's full embeddings?
- **What to measure:** BPB per byte (tokenizer-agnostic), artifact size, tokens/sec
- **Critical detail:** Tied output embeddings must use the same factorized/hashed scheme. Verify the output logit computation works correctly with factorized embeddings: `logits = hidden @ projection.T @ low_rank.T`
- **Duration:** 4 runs x 20 min each
- **Why this could be a breakthrough:** 6.3 MB of freed artifact budget is enormous — it's enough for a fundamentally larger model while getting better tokenization

---

## Execution Protocol

### For Architecture Comparisons (C1-C5):
1. Run each architecture for 20 minutes
2. Log loss every 30 seconds (40 data points)
3. Plot loss-vs-tokens-seen (sample efficiency)
4. Plot loss-vs-wall-clock (total efficiency)
5. Measure tokens/sec throughput
6. Fit power law: `L(T) = a * T^(-b) + L_inf`
7. Rank architectures by predicted 10-min loss on 8xH100 (extrapolated)

### For Technique Ablations (C6-C10):
1. Run cold baseline + one modification for 20 minutes each
2. Compare final loss (and loss curve shape) to cold baseline
3. Any technique that's ≥0.002 BPB better than baseline is worth promoting to hot

### For HP Sweeps (C11-C14):
1. Run full sweep (5 values each, 10 min per run)
2. Find optimal value and verify it's not at the edge of the sweep range
3. If edge: extend sweep

### For Quantization/Eval (C15-C20):
1. Train ONE model to completion
2. Run quantization/eval variants on the same model
3. Cheapest experiments — many variants from one training run

---

## Promotion Criteria (Cold -> Hot)

A cold experiment earns a hot-start slot if:
- Architecture comparison: ≥5% better tokens/sec AND ≤1% worse loss-per-token
- Technique ablation: ≥0.002 BPB improvement over cold baseline
- HP sweep: optimal value differs from frontier default
- Quantization: artifact size reduction ≥200KB OR ≥0.001 BPB improvement
- Eval trick: ≥0.001 BPB improvement

---

## Total Compute Budget

```
Architecture comparisons (C1-C5):   5 x 20 min =  100 min
Technique ablations (C6-C10):       5 x 20 min =  100 min
HP sweeps (C11-C14):               20 x 10 min =  200 min
Quantization (C15-C17):             1 x 20 min + quantize = 25 min
Eval techniques (C18-C20):          1 x 20 min + evals   = 60 min

Total: ~8 hours on single 5090
```
