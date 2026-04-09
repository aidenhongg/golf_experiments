# Hot-Start Experiments (8xH100 SXM, Full Competition Runs)

Experiments that require the actual competition hardware. Either because they need
multi-GPU DDP, FlashAttention v3, ≥7000 training steps, Hopper-specific features,
or because they need absolute BPB measurement for submission.

**Run these AFTER cold screening.** Only promote experiments whose cold results
showed clear directional improvement. Don't waste 8xH100 time on unvalidated ideas.

**Hardware:** 8x H100 80GB SXM, 600s training budget
**Batch:** 786K tokens/step
**Steps:** ~7000
**Metric:** Absolute val_bpb (3-seed mean, std < 0.001)

---

## Phase 1: Baselines & Frontier Reproduction (H1-H3)

Run first. Validate infrastructure. Establish your own reference numbers.

### H1: Merged SOTA Reproduction
```
11L/512d/GQA, SP4096
AR Self-Gen GPTQ + XSA-all + BigramHash(3072)
Parallel Muon, EMA 0.997, LeakyReLU(0.5)²
zstd-22
```
- **Target:** 1.1147 BPB
- **Purpose:** Sanity check. If you can't reproduce this, nothing else matters.
- **Seeds:** 42, 1337, 2024

### H2: No-TTT Frontier Reproduction (PR #1471)
```
SP8192, 11L/512d, 3-layer depth recurrence (3,4,5)
EMA 0.9965, WD=0.095, XSA-all, SDClip
Parallel Residuals (L7+), QK-Gain 5.0, Skip Gates
Full GPTQ + Cholesky + actorder, Brotli
Recurrence start step 2000, warmdown 72%
```
- **Target:** 1.0866 BPB
- **Purpose:** Your working frontier. All other hot experiments compare to this.
- **Seeds:** 42, 1337, 2024

### H3: TTT Frontier Reproduction (PR #1477)
```
H2 stack + Score-First TTT (3 epochs)
Depth recurrence on layers 4-5 (not 3,4,5)
MuonEq-R, GPTQ embeddings
```
- **Target:** 1.0822 BPB
- **Purpose:** Your ceiling reference. Everything else tries to beat this.
- **Seeds:** 42, 314, 999

---

## Phase 2: Speed Improvements (H4-H6)

These exploit 8xH100 Hopper features that don't exist on consumer GPUs.

### H4: Frontier + FP8 Training
```
H2 stack + FP8 via NVIDIA Transformer Engine
  FP8 for forward/backward GEMMs
  FP32 master weights, BF16 gradients
  Expect ~2x throughput = ~14000 steps in 600s
```
- **Target:** 1.07-1.08 BPB
- **Depends on:** Nothing (standalone test)
- **Key question:** Is the frontier still compute-bound? More steps should help.
- **Note:** 5090 has FP8 too, but throughput on 8xH100 is different. Need real numbers.

### H5: Frontier + FlashAttention 3 Optimizations
```
H2 stack with FA3-specific tuning:
  Verify FA3 is actually enabled (not falling back to FA2)
  Test FA3 with head_dim=64 vs 128
  Measure actual step time vs FA2
```
- **Target:** Marginal improvement (FA3 is likely already used in H2)
- **Purpose:** Ensure we're not leaving FA3 performance on the table

### H6: Frontier + Custom Triton Fusions (Hopper-optimized)
```
H2 stack + fused kernels targeting Hopper:
  Fuse SmearGate + BigramHash + embedding (1 kernel)
  Fuse LN + QKV projection
  CUTLASS EVT backward MLP fusion
  Warp-specialized attention kernels
```
- **Target:** 20-40% throughput gain = 1.08 BPB or better
- **Depends on:** Engineering effort (8-16 hours kernel development)
- **Key question:** How much throughput is left on the table from unfused ops?

---

## Phase 3: Promoted Cold Results (H7-H12)

These are cold experiments that passed screening and need full-budget validation.
**Only run if the cold experiment showed clear improvement.**

### H7: Best Architecture Hybrid (from C1-C5 winner)
```
Winner from cold architecture screening, fully optimized:
  Full SP8192 frontier stack
  FA3 (if architecture supports it)
  786K batch, 8xH100 DDP
  Full GPTQ + SDClip quantization
  3-seed validation
```
- **Target:** Beat H2 (1.0866)
- **Depends on:** C1-C5 results. If no hybrid beats transformer, skip this.
- **Promoted architecture:** [TBD after cold screening]

### H8: Best Architecture Hybrid + FP8 + TTT
```
H7 winner + FP8 Training + Score-First TTT
```
- **Target:** Beat H3 (1.0822)
- **Depends on:** H7 showing promise

### H9: Basis Sharing (if C6 showed convergence)
```
Full SP8192 frontier stack with SVD basis sharing
  replacing hard depth recurrence
  8xH100, full 7000 steps, 3-seed validation
```
- **Target:** Beat H2 (1.0866)
- **Depends on:** C6 showing competitive loss curve
- **Key advantage:** Cleaner than hard recurrence, no late-start fragility

### H10: Relaxed Recursive (if C7 showed promise)
```
Full SP8192 frontier stack with LoRA-relaxed recurrence
  8xH100, 3-seed validation
```
- **Target:** Beat H2 (1.0866)
- **Depends on:** C7 results

### H11: EngramLite on Frontier (if C8 showed improvement)
```
H2 stack with EngramLite replacing BigramHash
  8xH100, 3-seed validation
```
- **Target:** Beat H2 by ≥0.002 BPB
- **Depends on:** C8 results

### H12: Optimal HPs from Sweeps (if C11-C14 found improvements)
```
H2 stack with optimal HP values from cold sweeps:
  EMA decay: [best from C11]
  WD: [best from C12]
  Recurrence layers: [best from C13]
  Warmdown: [best from C14]
```
- **Target:** Beat H2 by ≥0.003 BPB (cumulative from HP tuning)
- **Depends on:** C11-C14 identifying at least one suboptimal frontier HP

---

## Phase 4: Multi-GPU Dependent (H13-H16)

These specifically need large batch or multi-GPU for the technique to work.

### H13: Frontier + TrigramHash
```
H2 stack + TrigramHash(8192) alongside BigramHash(3072)
  Dedicated trigram table
  Needs ≥7000 steps for sparse trigram patterns to converge
```
- **Target:** 1.08 BPB
- **Why hot-only:** TrigramHash confirmed to fail on single GPU due to insufficient steps
- **Note:** This was "largest single gain after recurrence" on earlier stacks

### H14: Frontier + Complementary Training + N-gram Eval
```
H2 stack + Complementary Training (7-gram) during training
  N-gram eval cache (orders 2-10, entropy-adaptive alpha)
  kNN-LM (k=32, RBF kernel)
```
- **Target:** 0.9-1.05 BPB (if legal)
- **BLOCKER:** Verify rule compliance BEFORE running. Many n-gram approaches invalidated.
- **Why hot-only:** N-gram cache quality depends on seeing enough eval tokens; large batch helps

### H15: Deeper Model (13L) + Cross-Layer KV Sharing
```
SP8192, 13L/512d/GQA
  Cross-layer KV sharing on adjacent attention layers
  Freed param budget used for 2 extra layers (vs 11L baseline)
  XSA-all, EMA 0.9965, SDClip
  Parallel Residuals, Skip Gates, Brotli
```
- **Target:** 1.07-1.09 BPB
- **Why hot-only:** 13L may need full DDP throughput to fit within 600s
- **Key question:** Is depth at 13L worth the speed cost?

### H16: Larger Batch Muon Convergence Test
```
H2 stack but test batch sizes:
  A) 786K (current)
  B) 1.5M (gradient accumulation)
  C) 393K (half batch, 2x steps)
```
- **Target:** Find optimal batch-step tradeoff
- **Why hot-only:** 786K+ batch requires 8xH100 memory
- **Key question:** Is 786K the right balance? More steps at smaller batch might win.

---

## Phase 5: Full Novel Stacks (H17-H20)

Complete new stacks that combine multiple innovations. Only run after Phase 2-4
have identified which individual techniques work.

### H17: Speed Stack (FP8 + Triton + optimal HPs + TTT)
```
H2 stack + every validated speed improvement:
  FP8 Training
  Custom Triton fusions
  Optimal HPs from C11-C14
  Score-First TTT
  Expected: ~2.5x more steps than H2
```
- **Target:** 1.06-1.08 BPB
- **Depends on:** H4 and H6 showing speed gains translate to BPB gains
- **Philosophy:** Identical architecture, maximum engineering optimization

### H18: Architecture + Speed Stack
```
Best hybrid (H7) + FP8 + Triton + TTT + optimal HPs
  The full combination of architecture change + speed
```
- **Target:** 1.05-1.08 BPB
- **Depends on:** H7 and H17 both showing gains
- **Risk:** High (many simultaneous changes)

### H19: Ternary Transformer (BitNet 1.58)
```
BitNet 1.58-bit, 14L/768d/GQA, SP8192
  ~100M+ parameters in 16MB
  Absmean + STE from step 0
  NeoMuon or AdamW
  XSA-all, Skip Gates
  FP16 tied embeddings, Brotli
```
- **Target:** 1.05-1.12 BPB
- **Why hot-only:** Ternary training is very sensitive to batch size; needs 786K batch for convergence
- **Depends on:** Cold ternary experiment (not in current cold list, could add) showing convergence
- **Note:** Completely different optimization regime. No int6, no GPTQ, no SDClip.

### H20: Kitchen Sink (Best of Everything)
```
Combine EVERY technique that individually passed validation:
  [Architecture from H7/H8]
  [Speed from H17: FP8 + Triton]
  [Weight sharing from H9/H10: Basis Sharing or Relaxed Recursive]
  [Embeddings from H11: EngramLite or BigramHash]
  [HPs from H12: optimal sweep values]
  [Eval from C18-C20: optimal TTT + sliding window]
  [Quant from C15-C17: optimal SDClip k + compressor]
  3-seed validation
```
- **Target:** Best possible BPB
- **Depends on:** Everything above. This is the final submission candidate.
- **Run last.** This is the culmination experiment.

---

## Execution Order & Dependencies

```
Phase 1 (Baselines):     H1 → H2 → H3                    [sequential, ~30 min]
Phase 2 (Speed):          H4, H5, H6                       [parallel, ~30 min]
Phase 3 (Cold winners):   H7-H12                           [parallel, ~60 min]
                          (only run those promoted from cold)
Phase 4 (Multi-GPU):      H13-H16                          [parallel, ~40 min]
Phase 5 (Full stacks):    H17 → H18 → H19 → H20           [sequential, ~2 hrs]
                          (each builds on prior results)

Total: ~4-5 hours of 8xH100 time (including 3-seed validation)
Cost at ~$25/hr for 8xH100: ~$100-125
```

---

## Decision Gates

After each phase, decide whether to continue:

| Gate | Question | If NO |
|------|----------|-------|
| After Phase 1 | Can we reproduce frontier? | Debug infrastructure, don't proceed |
| After Phase 2 | Does FP8/Triton give ≥10% more steps? | Skip speed-dependent experiments |
| After Phase 3 | Did any cold winner beat H2? | Focus on speed (H17) not architecture |
| After Phase 4 | Did TrigramHash/CLKV/N-gram work? | Focus on validated techniques only |
| After Phase 5 | Is H20 our best result? | Submit best from any phase |

---

## Submission Checklist

Before submitting any result:
- [ ] 3-seed mean BPB computed (std < 0.001)
- [ ] Artifact ≤ 16,000,000 bytes
- [ ] Training completed in ≤ 600s on 8xH100
- [ ] Eval completed in ≤ 600s
- [ ] Beats current merged SOTA (1.1147) by ≥ 0.005 nats at p < 0.01
- [ ] No network calls during training or eval
- [ ] Score-first TTT uses only backward-looking (already-graded) tokens
- [ ] N-gram cache (if used) normalizes over full vocabulary
- [ ] All code + weights fit in single artifact
- [ ] README with reproduction commands included
