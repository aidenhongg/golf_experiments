# Parameter Golf — Experiment Plan

**Date:** 2026-04-08
**Objective:** Systematically test the 20 most promising technique combinations to push below the current frontier BPB.

---

## Critical Updates from Latest Submissions (Apr 8, 2026)

Before designing experiments, the competition landscape has shifted dramatically since our earlier research. Key corrections:

### New SOTA (not yet merged)
| PR | Score | Stack |
|----|-------|-------|
| **#1477** | **1.0822** (3-seed) | SP8192 + Parallel Residuals + Score-First TTT + Depth Recurrence (4-5) + MuonEq-R + QK-Gain 5.0 + SDClip + GPTQ embeds + Skip Gates + Brotli |
| **#1471** | **1.0866** (3-seed) | SP8192 + SDClip + 3-Layer Depth Recurrence (3,4,5) + EMA 0.9965 + XSA-all + Parallel Residuals + QK-Gain 5.0 + Skip Gates + Full GPTQ + Brotli |
| **#1460** | **1.0827** (3-seed) | SP8192 + TTT + Eval-Time Hash Embedding |
| Merged SOTA | **1.1147** | 11L AR Self-Gen GPTQ + XSA-all + BigramHash(3072) |

### Compatibility Matrix Corrections

Our earlier matrix had several entries that are now **proven wrong** by frontier submissions:

| "Hard Conflict" | Actual Status | How It Was Solved |
|-----------------|---------------|-------------------|
| **EMA + Depth Recurrence** (was: `--`, 1.42 BPB catastrophe) | **Works at 1.0866** | EMA decay 0.9965 (not 0.997), late recurrence start (step 2000), WD=0.095 (not 0.04), only 3 layers recur (not all) |
| **TTT + Depth Recurrence** (was: `--`, breaks training) | **Works at 1.0822** | Score-first TTT is eval-time only. Training uses recurrence normally. TTT adapts after scoring, avoiding gradient coupling |
| **XSA + Depth Recurrence** (was: `?`) | **Works at 1.0866** | XSA-all + 3-layer recurrence. XSA applies to all 11 virtual layers |

### New Techniques Now Standard in Frontier
| Technique | Description | Source |
|-----------|-------------|--------|
| **SP8192** | 8192-token BPE vocabulary. Dramatically better than SP4096 | All top submissions |
| **Parallel Residuals** | From layer 7+, attention and MLP operate on separate residual lanes with learned merge scalar | PR #1334 |
| **SDClip** | Quantization clipping via clip = k·std(row). k=12.85 for int6, k=20.0 for int8 embeds. Replaces percentile search | PR #1394 |
| **MuonEq-R** | Equalized variant of Muon optimizer | PR #1334 |
| **Skip Gates** | Learned gates on residual skip connections | PR #1334 |
| **QK-Gain 5.0** | Per-head query scaling, up from 4.0 | Consensus |
| **Late Recurrence Start** | Depth recurrence begins at step 2000 (not from step 0) | PR #1445 |
| **EMA 0.9965** | Higher decay than prior 0.997 standard | PR #1421 |
| **WD=0.095** | Much higher weight decay than prior 0.04 | PR #1331 |
| **Warmdown 72%** | Much more aggressive warmdown | PR #1445 |
| **Brotli compression** | Replaces zstd-22 | Frontier standard |

---

## Experiment Design

### Baseline (Experiment 0)

**Merged SOTA Reproduction**
```
11L / 512d / GQA(8Q/4KV)
SP4096, XSA-all, BigramHash(3072)
AR Self-Gen GPTQ, Parallel Muon
EMA 0.997, LeakyReLU(0.5)²
```
- **Target:** 1.1147 BPB (reproduce merged record)
- **Purpose:** Validate our training infrastructure against the known baseline
- **Risk:** None (exact reproduction)

---

### Tier 1: Low-Risk, Incremental (Expected: 1.07-1.09 BPB)

These build on the proven frontier stack with minor additions.

#### Experiment 1: Frontier Reproduction (No-TTT)
```
Reproduce PR #1471 exactly:
  SP8192, 11L/512d/GQA, 3-layer depth recurrence (3,4,5)
  EMA 0.9965, WD=0.095, XSA-all, SDClip
  Parallel Residuals (L7+), QK-Gain 5.0, Skip Gates
  LeakyReLU(0.5)², VE128 (L9-10), Full GPTQ + Cholesky
  Brotli, warmdown 72%, recurrence start step 2000
```
- **Target:** 1.0866 BPB
- **Purpose:** Establish our own reproducible frontier before modifying
- **Risk:** Low (exact reproduction)

#### Experiment 2: Frontier + Score-First TTT
```
Experiment 1 stack + Score-First TTT (3 epochs, AdamW+cosine)
  Document masking, backward-looking only
```
- **Target:** 1.0822 BPB (reproduce PR #1477)
- **Purpose:** Validate TTT works on top of our frontier
- **Risk:** Low (proven combination)
- **Note:** TTT is eval-time only, doesn't conflict with training-time recurrence

#### Experiment 3: Frontier + FP8 Training
```
Experiment 1 stack + FP8 via NVIDIA Transformer Engine
  FP8 for forward/backward GEMMs, FP32 master weights
  Expect ~2x throughput = ~2x more training steps
```
- **Target:** 1.07-1.08 BPB
- **Purpose:** Test if pure speed (more steps) improves BPB on already-converged frontier
- **Risk:** Low-medium (FP8 on H100 is well-supported; stability tuning may be needed)
- **Key question:** Is the frontier still compute-bound or has it converged?

#### Experiment 4: Frontier + FP8 + TTT
```
Experiment 3 + Score-First TTT
```
- **Target:** 1.06-1.08 BPB
- **Purpose:** Best of both worlds: more training steps + eval-time adaptation
- **Risk:** Low (combines two individually-proven improvements)

#### Experiment 5: Frontier + Custom Triton Fusions
```
Experiment 1 stack + fused kernels:
  - Fuse SmearGate + BigramHash + embedding lookup (1 kernel, not 3)
  - Fuse LayerNorm + QKV projection
  - Fuse backward MLP (CUTLASS EVT)
  Expect 20-40% throughput gain
```
- **Target:** 1.08-1.09 BPB
- **Purpose:** Pure engineering speed gain. Orthogonal to all other experiments.
- **Risk:** Low (pure speed optimization, no quality risk)
- **Engineering cost:** High (custom Triton kernels)

---

### Tier 2: Medium-Risk, Novel Combinations (Expected: 1.06-1.09 BPB)

These introduce new techniques not yet validated on the frontier stack.

#### Experiment 6: Frontier + EngramLite
```
Experiment 1 stack with BigramHash replaced by EngramLite:
  Multi-head n-gram hashing with context-aware gating
  (Used in the 1.1086 unofficial best from March)
```
- **Target:** 1.08-1.09 BPB
- **Purpose:** Test if EngramLite's richer n-gram representation beats BigramHash on the new SP8192 stack
- **Risk:** Medium (EngramLite was validated on older stack, not SP8192)
- **Note:** May have diminishing returns with SP8192's larger vocabulary

#### Experiment 7: Frontier + TrigramHash
```
Experiment 1 stack + TrigramHash(8192) alongside BigramHash(3072)
  Dedicated trigram table, not shared with bigram
```
- **Target:** 1.08 BPB
- **Purpose:** TrigramHash was the "largest single gain after recurrence" on earlier stacks
- **Risk:** Medium (needs sufficient training steps; may be redundant with SP8192)

#### Experiment 8: Basis Sharing Replacing Hard Recurrence
```
SP8192, 11L/512d/GQA
  SVD basis sharing across all 11 layers (shared bases + per-layer coefficients)
  EMA 0.9965, XSA-all     [no late-start hack needed: basis sharing is soft]
  SDClip, Parallel Residuals, QK-Gain 5.0, Skip Gates
  Full GPTQ, Brotli
  WD=0.095, warmdown 72%
```
- **Target:** 1.08-1.09 BPB
- **Purpose:** Test if Basis Sharing achieves comparable parameter savings to hard recurrence WITHOUT needing the late-start hack, higher EMA decay, or limited recurrence layers
- **Risk:** Medium (Basis Sharing not yet tested in competition)
- **Key advantage:** If it works, it avoids all the fragile recurrence workarounds

#### Experiment 9: Relaxed Recursive (LoRA Adapters)
```
SP8192, 6 shared blocks x 2 loops + rank-8 LoRA adapters per pass
  14 virtual layers with per-pass differentiation
  EMA 0.9965, XSA-all, SDClip
  Parallel Residuals (L7+), QK-Gain 5.0, Skip Gates
  Full GPTQ, Brotli
```
- **Target:** 1.08-1.09 BPB
- **Purpose:** Compare LoRA-relaxed recurrence vs hard 3-layer recurrence
- **Risk:** Medium (untested in competition, but theoretically cleaner than hard recurrence)
- **Key question:** Does LoRA relaxation avoid quantization error amplification better than late-start?

#### Experiment 10: GLA Hybrid + SP8192
```
8 GLA layers + 3 attention layers (final 3)
  SP8192, 512d
  XSA on final 3 attention layers only
  EMA 0.9965 (XSA present ✓)
  GLA layers use FLA library Triton kernels
  SDClip: int6 for attention, int8 for GLA layers (quantization sensitivity)
  Parallel Residuals (L7+), QK-Gain 5.0 (attention layers only)
  Muon for projections, AdamW for GLA gates
  Brotli
```
- **Target:** 1.08-1.10 BPB
- **Purpose:** Test if GLA layers provide training speed advantage while keeping XSA on attention layers
- **Risk:** Medium-high (GLA at 512d at this scale untested; quantization of GLA layers unknown)
- **Key advantage:** GLA length generalization (train 1024, eval 4096 for free)

---

### Tier 3: Higher-Risk Architecture Experiments (Expected: 1.06-1.12 BPB)

These involve less-tested architecture changes with wider outcome ranges.

#### Experiment 11: GLA Hybrid + FP8 + TTT
```
Experiment 10 + FP8 Training + Score-First TTT
  GLA training speed + FP8 = significantly more steps
  TTT on attention layers for eval-time adaptation
```
- **Target:** 1.06-1.09 BPB
- **Purpose:** Maximum speed hybrid with eval-time optimization
- **Risk:** High (stacking three unproven changes on top of each other)

#### Experiment 12: Mamba-3 Hybrid + SP8192
```
8 Mamba-3 layers + 3 attention layers (final 3)
  SP8192, 512d
  XSA on final 3 attention layers
  EMA 0.9965
  Int8 for Mamba-3 layers (SSM quantization sensitivity)
  Int6 + SDClip for attention layers
  Mixed optimizer: Muon for projections, AdamW for SSM params
  Brotli
```
- **Target:** 1.08-1.12 BPB
- **Purpose:** Test Mamba-3's halved state size + complex-valued gates on parameter golf
- **Risk:** High (prior SSM result at dim=512 was negative; Mamba-3 may change this)
- **Key question:** Is Mamba-3's throughput advantage real at seq1024 on H100?

#### Experiment 13: Mamba-3 Hybrid + FP8
```
Experiment 12 + FP8 Training
  Mamba-3 throughput + FP8 = potentially 3-4x more steps than BF16 transformer
```
- **Target:** 1.07-1.10 BPB
- **Purpose:** Maximum throughput via architecture + precision
- **Risk:** High (compounds two uncertain factors)

#### Experiment 14: xLSTM Hybrid + SP8192 + FP8
```
8 mLSTM blocks + 3 attention layers (final 3)
  SP8192, 512d
  xLSTM's 3.5x training speed + FP8 = potentially 7x more steps
  XSA on attention layers, EMA 0.9965
  SDClip, Brotli
```
- **Target:** 1.07-1.10 BPB
- **Purpose:** Exploit xLSTM's speed claim for massive step-count increase
- **Risk:** Very high (xLSTM at 20M params completely untested; kernel availability uncertain)
- **Potential:** If 7x steps materializes, could be transformative

#### Experiment 15: Monarch MLP + SP8192 Frontier
```
Experiment 1 stack with dense MLP replaced by Monarch-factorized MLP
  Block-diagonal product structure: O(N log N) vs O(N^2)
  70% MLP compute reduction = significantly more training steps
  Custom Triton kernels for Monarch matmul
```
- **Target:** 1.08-1.10 BPB
- **Purpose:** Speed via structured matrices without changing architecture identity
- **Risk:** Medium-high (quality-compute tradeoff unknown at 512d; needs custom kernels)
- **Note:** Incompatible with standard GPTQ (needs custom quantization for Monarch structure)

---

### Tier 4: Moonshots (Expected: 1.05-1.15 BPB, wide range)

High risk, high potential reward. At least one should be attempted.

#### Experiment 16: CLKV + Deeper Model (13L)
```
SP8192, 13L/512d/GQA
  Cross-layer KV sharing across adjacent attention layers
  Freed param budget used for 2 extra layers
  XSA-all, EMA 0.9965, SDClip
  Parallel Residuals, QK-Gain 5.0, Skip Gates
  Brotli
```
- **Target:** 1.07-1.09 BPB
- **Purpose:** Test if depth > width at the frontier (13L vs 11L+recurrence)
- **Risk:** Medium-high (13L may be too slow to train in 10 minutes)

#### Experiment 17: Complementary Training + N-gram Eval Cache
```
Experiment 1 stack + Complementary Training (7-gram)
  Down-weight tokens predictable by n-gram statistics during training
  N-gram eval cache (orders 2-10, entropy-adaptive alpha)
  kNN-LM (k=32, RBF kernel)
```
- **Target:** 0.9-1.05 BPB (if legal)
- **Purpose:** Test the strongest single lever (n-gram cache) on the current frontier
- **Risk:** Very high (many n-gram approaches were rule-invalidated; must verify legality first)
- **BLOCKER:** Check rule compliance before investing any compute

#### Experiment 18: Ternary Transformer + SP8192
```
BitNet 1.58-bit, 14L/768d/GQA
  SP8192 tokenizer (large vocab viable with tiny weights)
  ~100M+ parameters in 16MB
  STE from step 0, NeoMuon or AdamW
  FP16 tied embeddings, Brotli
  XSA-all, Skip Gates
```
- **Target:** 1.05-1.12 BPB
- **Purpose:** Test if raw parameter count (100M+ ternary vs ~21M int6) overcomes per-param quality loss
- **Risk:** High (competition BitNet result was 1.1570; but that was smaller and older stack)

#### Experiment 19: Ternary Mamba-3
```
BitNet 1.58-bit + 14L Mamba-3 / 768d (no attention)
  ~100M+ parameters, linear-time training
  No XSA (no attention), no EMA
  Completely novel regime
```
- **Target:** 1.05-1.15 BPB (enormous uncertainty)
- **Purpose:** Maximum parameter count + fastest architecture. True moonshot.
- **Risk:** Extreme (neither ternary Mamba nor Mamba-3 at this scale has been tested)

#### Experiment 20: Kitchen Sink (Best-of-Everything Compatible Stack)
```
8 GLA layers + 3 attention layers (final)
  SP8192, 512d
  Basis Sharing (shared SVD bases across all layers)
  FP8 Training
  Custom Triton fusions
  XSA on final 3 attention layers
  EMA 0.9965
  EngramLite (replacing BigramHash)
  Parallel Residuals (L7+), QK-Gain 5.0, Skip Gates
  SDClip, Brotli
  Score-First TTT (3 epochs)
  Complementary Training (if legal)
```
- **Target:** 1.05-1.08 BPB
- **Purpose:** Stack every compatible improvement. Test the multiplicative stacking hypothesis.
- **Risk:** High (interaction effects between 10+ simultaneous changes are unpredictable)
- **Key insight from competition:** Gains are multiplicative. If each technique adds 1%, stacking 10 gives ~10%.

---

## Experiment Priority & Execution Order

### Phase 1: Baselines (Run first, validate infrastructure)
| Order | Exp | Name | Expected BPB | Compute |
|-------|-----|------|-------------|---------|
| 1 | 0 | Merged SOTA Reproduction | 1.1147 | 10 min |
| 2 | 1 | Frontier Reproduction (no TTT) | 1.0866 | 10 min |
| 3 | 2 | Frontier + TTT | 1.0822 | 10 min + eval |

### Phase 2: Low-Hanging Fruit (Run in parallel)
| Order | Exp | Name | Expected BPB | Risk |
|-------|-----|------|-------------|------|
| 4 | 3 | Frontier + FP8 | 1.07-1.08 | Low |
| 5 | 5 | Frontier + Triton Fusions | 1.08-1.09 | Low |
| 6 | 4 | Frontier + FP8 + TTT | 1.06-1.08 | Low |

### Phase 3: Novel Techniques (Run based on Phase 2 results)
| Order | Exp | Name | Expected BPB | Risk |
|-------|-----|------|-------------|------|
| 7 | 6 | + EngramLite | 1.08-1.09 | Med |
| 8 | 7 | + TrigramHash | 1.08 | Med |
| 9 | 8 | Basis Sharing | 1.08-1.09 | Med |
| 10 | 9 | Relaxed Recursive | 1.08-1.09 | Med |

### Phase 4: Architecture Experiments (Run if Phase 3 shows promise)
| Order | Exp | Name | Expected BPB | Risk |
|-------|-----|------|-------------|------|
| 11 | 10 | GLA Hybrid | 1.08-1.10 | Med-High |
| 12 | 11 | GLA Hybrid + FP8 + TTT | 1.06-1.09 | High |
| 13 | 12 | Mamba-3 Hybrid | 1.08-1.12 | High |
| 14 | 14 | xLSTM Hybrid + FP8 | 1.07-1.10 | Very High |

### Phase 5: Moonshots (Run if budget allows)
| Order | Exp | Name | Expected BPB | Risk |
|-------|-----|------|-------------|------|
| 15 | 16 | CLKV + 13L | 1.07-1.09 | Med-High |
| 16 | 15 | Monarch MLP | 1.08-1.10 | Med-High |
| 17 | 17 | N-gram Cache (if legal) | 0.9-1.05 | Very High |
| 18 | 18 | Ternary Transformer | 1.05-1.12 | High |
| 19 | 19 | Ternary Mamba-3 | 1.05-1.15 | Extreme |
| 20 | 20 | Kitchen Sink | 1.05-1.08 | High |

---

## Success Criteria

| Level | BPB | What it means |
|-------|-----|---------------|
| **Minimum viable** | < 1.0866 | Beat the no-TTT frontier |
| **Competitive** | < 1.0822 | Beat the current TTT frontier |
| **Strong** | < 1.07 | Significant improvement over frontier |
| **Breakthrough** | < 1.05 | Novel techniques clearly working |
| **Exceptional** | < 1.00 | Multiple novel techniques stacking multiplicatively |

All experiments must be validated with **3 seeds** and **std < 0.001 BPB** before claiming results.

---

## Resource Requirements

| Resource | Per Experiment | Total (20 experiments) |
|----------|---------------|----------------------|
| 8xH100 SXM time | 10 min train + ~10 min eval | ~7 hours |
| 3-seed validation | 30 min per experiment | ~10 hours |
| Engineering time | 2-8 hours per experiment | ~80-120 hours |
| Custom kernels (Exp 5, 10, 14, 15) | 8-16 hours each | ~40-60 hours |

**Total estimated compute:** ~20 hours of 8xH100 time (including 3-seed validation).
**Total estimated engineering:** ~120-180 hours.

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| FP8 training instability | Exp 3, 4, 11, 13, 14 produce garbage | Start with FP8 for GEMMs only, keep gradients in BF16. Use NVIDIA Transformer Engine's auto-scaling |
| GLA/Mamba-3 kernels not available for H100 | Exp 10-14 blocked | Verify FLA library + Mamba repo support H100 before starting. Fall back to custom Triton |
| Basis Sharing + GPTQ interaction unknown | Exp 8 may fail | Test GPTQ on basis-shared weights in isolation first. Quantize bases and coefficients separately |
| N-gram cache ruled illegal | Exp 17 wasted | Check rules BEFORE investing any compute |
| xLSTM at 20M params doesn't work | Exp 14 wasted | Run a quick 1-GPU sanity check (1 min) before full 8xH100 run |
| Ternary + Mamba-3 has no reference implementation | Exp 19 blocked | Build incrementally: first validate ternary transformer, then ternary Mamba-3 |

---

## Tracking & Comparison

All experiments logged with:
- 3-seed BPB (mean + std)
- Training steps completed
- Wall-clock time
- Artifact size (bytes)
- Technique stack (for attribution)

Results added to a shared comparison table after each experiment completes.
