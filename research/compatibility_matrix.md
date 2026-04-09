# Parameter Golf — Technique Compatibility Matrix

## Legend

| Symbol | Meaning |
|--------|---------|
| `++` | **Strong synergy** — significantly better together than either alone |
| `+` | **Mild synergy** — complementary, small bonus from pairing |
| `.` | **Neutral** — independent, no known interaction |
| `-` | **Mild conflict** — diminishing returns or slight interference |
| `--` | **Hard conflict** — actively hurts, avoid combining |
| `DEP` | **Dependency** — first technique requires the second to function |
| `?` | **Untested / unknown** |

---

## Core Interaction Matrix

Techniques are abbreviated. Read as: **row + column = interaction**.

```
                  XSA   EMA   SWA   SmGt  Orth  BGH   TGH   PRoP  LRe²  Muon  QAT6  GPTQ  TTT   DpRc  Ngra  CmpT  SlWn  FP16E UNet  EngL
XSA               ·    ++     +     .     .     .     .     .     .     .     .     +     --    ?     .     .     +     .     .     .
EMA              ++     ·     +     .     .     .     .     .     .     .     +     +     -     --    .     .     .     .     .     .
SWA               +     +     ·     .     .     .     .     .     .     .     +     .     .     -     .     .     .     .     .     .
SmGt              .     .     .     ·    DEP    +     +     .     .     .     .     .     .     .     .     .     .     .     .     .
Orth              .     .     .    DEP    ·     .     .     .     .     .     .     .     .     .     .     .     .     .     .     .
BGH               .     .     .     +     .     ·     +     .     .     .     .     .     .     .     .     +     .     .     .     -
TGH               .     .     .     +     .     +     ·     .     .     .     .     .     .     .     .     +     .     .     .     ?
PRoP              .     .     .     .     .     .     .     ·     .     .     .     .     .     .     .     .     +     .     .     .
LRe²              .     .     .     .     .     .     .     .     ·     .     .     .     .     .     .     .     .     .     .     .
Muon              .     .     .     .     .     .     .     .     .     ·     .     .     .     .     .     .     .     .     .     .
QAT6              .     +     +     .     .     .     .     .     .     .     ·     +     -     -     .     .     .     .     .     .
GPTQ              +     +     .     .     .     .     .     .     .     .     +     ·     -     -     .     .     .     .     .     .
TTT              --     -     .     .     .     .     .     .     .     .     -     -     ·     --    .     .     .     .     .     .
DpRc              ?    --     -     .     .     .     .     .     .     .     -     -     --    ·     .     .     .     .     .     .
Ngra              .     .     .     .     .     .     .     .     .     .     .     .     .     .     ·    ++     .     .     .     .
CmpT              .     .     .     .     .     +     +     .     .     .     .     .     .     .    ++     ·     .     .     .     .
SlWn              +     .     .     .     .     .     .     +     .     .     .     .     .     .     .     .     ·     .     .     .
FP16E             .     .     .     .     .     .     .     .     .     .     .     .     .     .     .     .     .     ·     .     .
UNet              .     .     .     .     .     .     .     .     .     .     .     .     .     -     .     .     .     .     ·     .
EngL              .     .     .     .     .     -     .     .     .     .     .     .     .     .     .     .     .     .     .     ·
```

### Abbreviation Key

| Abbrev | Technique |
|--------|-----------|
| XSA | Cross-Sequence Attention |
| EMA | Exponential Moving Average (decay=0.997) |
| SWA | Stochastic Weight Averaging |
| SmGt | SmearGate |
| Orth | OrthoInit (Orthogonal Initialization) |
| BGH | BigramHash |
| TGH | TrigramHash |
| PRoP | Partial RoPE |
| LRe² | LeakyReLU(0.5)² |
| Muon | Muon Optimizer |
| QAT6 | Late Int6 STE QAT |
| GPTQ | Full GPTQ / GPTQ-lite |
| TTT | Test-Time Training (LoRA) |
| DpRc | Depth Recurrence (weight sharing) |
| Ngra | N-gram Eval Cache |
| CmpT | Complementary Training |
| SlWn | Sliding Window Eval |
| FP16E | FP16 Tied Embeddings |
| UNet | U-Net Skip Connections |
| EngL | EngramLite |

---

## Hard Dependencies (DEP)

These are not optional pairings — the first technique **breaks** without the second.

| Technique | Requires | Why |
|-----------|----------|-----|
| SmearGate | OrthoInit | Without OrthoInit, SmearGate is +0.003 BPB *worse*. Every successful SmearGate submission uses OrthoInit |
| SWA | WD ≥ 0.04 | SWA shows zero effect below weight decay 0.04 |
| SWA | FP32 accumulation | bf16 SWA averaging is catastrophic |
| Late QAT | Warmdown schedule | QAT activation is keyed to LR scale threshold (< 0.10–0.15) |

---

## Hard Conflicts (--)

These combinations are **confirmed harmful** — avoid.

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| **TTT + XSA** | +0.016 BPB worse | XSA already captures cross-sequence context; TTT adaptation on top of that creates conflicting gradient signals |
| **TTT + Depth Recurrence** | Breaks training | TTT weight updates affect both loop iterations simultaneously, compounding gradients through recurrence in unexpected ways |
| **EMA + Depth Recurrence** | 1.42 BPB catastrophe | Short effective training per unique block means EMA shadow weights accumulate poorly-converged early states |
| **EMA without XSA** | Hurts performance | EMA smoothing only helps when XSA provides the cross-sequence signal to smooth over; without XSA the averaging just blurs useful specialization |
| **Block sharing > 2 cycles** | 900× quant error amplification | Quantization error compounds multiplicatively through each reuse of the same weights |
| **SGD-based TTT + Full GPTQ** | +0.030 BPB | SGD updates corrupt the carefully calibrated GPTQ weight arrangement; AdamW+cosine TTT works instead |

---

## Strong Synergies (++)

These combinations are **significantly better together** than the sum of their parts.

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| **EMA + XSA** | -0.003 BPB beyond individual gains | XSA extends the effective context; EMA smooths over training noise in that extended context. Each enables the other's strength |
| **N-gram cache + Complementary Training** | Designed to pair | Complementary training teaches the neural model to *ignore* what n-grams can predict. N-gram cache handles those at eval. Division of labor |
| **SmearGate + BigramHash** | Complementary bigram signal | SmearGate provides soft continuous blending of adjacent embeddings; BigramHash provides discrete lookup. Different mechanisms, additive gains |
| **Late QAT + GPTQ** | Stacked quant resilience | QAT trains the model to be robust to rounding; GPTQ then optimally places the rounding boundaries. QAT makes GPTQ's job easier |
| **EMA + SWA** | Different averaging horizons | EMA provides continuous exponential smoothing; SWA provides discrete checkpoint averaging. They capture different frequency bands of the loss landscape |

---

## Mild Synergies (+)

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| XSA + Sliding Window Eval | XSA borrows context across windows; sliding window ensures rich context within each window | Complementary context extension |
| XSA + GPTQ | XSA's zero-parameter design means GPTQ budget is fully spent on MLP/attention weights | No parameter competition |
| Partial RoPE + Sliding Window | Partial RoPE avoids dimension explosion at long context; sliding window exploits longer context | Enables longer effective eval |
| BigramHash + Complementary Training | BigramHash provides the bigram statistics that complementary training down-weights against | Coherent training signal |
| TrigramHash + Complementary Training | Same as above but for trigram patterns | Coherent training signal |
| EMA + Late QAT | EMA smoothing improves the weight distribution that QAT then learns to quantize | Cleaner quantization targets |
| SWA + Late QAT | SWA checkpoint averaging reduces weight outliers, making int6 quantization more uniform | Less clipping loss |
| EMA + GPTQ | Smoother weights → fewer outliers → less GPTQ reconstruction error | Quantization-friendly |

---

## Mild Conflicts (-)

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| TTT + EMA | Marginal negative | EMA-smoothed weights are already well-averaged; TTT perturbations fight the smoothing |
| TTT + Late QAT | TTT updates may push weights outside the int6 grid QAT trained for | Quantization grid mismatch |
| TTT + GPTQ | Non-SGD TTT partially OK (AdamW+cosine), but any TTT risks corrupting Hessian-calibrated weight placement | Hessian invalidation |
| Depth Recurrence + SWA | Averaging over checkpoints of recurrent models amplifies the same quantization-error compounding issue as EMA | Error accumulation |
| Depth Recurrence + QAT | Quantization error flows through the same weights multiple times during forward pass | Multiplicative error |
| Depth Recurrence + GPTQ | Same mechanism as QAT — Hessian calibration doesn't account for multi-pass error | Calibration mismatch |
| Depth Recurrence + UNet | Skip connections across encoder-decoder assume unique layers; recurrence complicates the skip topology | Architectural tension |
| BigramHash + EngramLite | Overlapping functionality — both provide n-gram-aware embeddings | Diminishing returns |

---

## Confirmed Neutral (.)

These are genuinely independent — combining them gives roughly the sum of individual gains.

| Combo | Why independent |
|-------|----------------|
| Muon + anything architectural | Optimizer is agnostic to model structure |
| LeakyReLU² + anything | Activation function choice is orthogonal to most other decisions |
| FP16 Embeddings + anything | Precision decision for one layer, doesn't interact with other techniques |
| zstd compression + anything | Post-training compression is fully decoupled |
| FlashAttention v3 + anything | Pure systems optimization, no quality interaction |
| Sliding Window Eval + training techniques | Eval-time only; doesn't interact with how the model was trained |
| N-gram cache + training techniques (except CmpT) | Eval-time only overlay |

---

## Context-Dependent Interactions

These interactions depend on **specific configurations or hardware**.

| Combo | Condition | Effect |
|-------|-----------|--------|
| TrigramHash + any | Needs ≥7000 training steps (8xH100). Fails on single GPU due to insufficient steps for sparse trigram patterns to converge | Hardware-dependent |
| Larger batch (786K) + any | Only beneficial when total training time allows enough tokens. With very short wallclock, smaller batch + more steps wins | Time-budget dependent |
| SP4096 tokenizer + deeper model | Larger vocab = larger embedding layer = fewer layers fit in 16MB. SP4096 trades depth for better token compression | Artifact-budget tradeoff |
| 8192 BPE + ternary quantization | Only viable because ternary weights are so small that large embedding fits. Doesn't work with int6 | Quantization-scheme dependent |
| Seq length curriculum + XSA | Curriculum starts with short attention spans; XSA cross-sequence benefit is minimal at short lengths. Gains appear later in training | Training-phase dependent |
| Long-context eval (4096) + NTK-RoPE | Works. But Full RoPE + long context = quality degradation. Must use NTK-aware or Partial RoPE | RoPE-variant dependent |

---

## Strategy Archetypes

Based on the compatibility constraints above, here are the **known-valid stacks** — combinations where all interactions are synergistic or neutral, with no conflicts.

### Archetype A: "Pure Neural" (no eval tricks)
```
11L / 512d / GQA(8Q/4KV)
+ LeakyReLU(0.5)²
+ XSA-4 or XSA-all
+ EMA (0.997)                  [requires XSA ✓]
+ SWA (WD≥0.04, FP32)
+ SmearGate + OrthoInit         [dependency satisfied ✓]
+ BigramHash (2048–3072)
+ Partial RoPE (16/64)
+ LN Scale
+ Muon + AdamW (mixed)
+ Late QAT int6 STE
+ GPTQ-lite post-training
+ FP16 tied embeddings
+ U-Net skips
+ zstd-22
+ Sliding window eval (stride=64)
```
**No conflicts.** This is the canonical SOTA stack. ~1.1147–1.1233 BPB.

### Archetype B: "Depth Recurrence" (parameter-efficient)
```
8 blocks × 2 loops (16 effective layers)
+ FiLM conditioning per loop
+ LeakyReLU(0.5)²
+ BigramHash (20480) + TrigramHash (8192)
+ SmearGate + OrthoInit
+ Muon + AdamW
+ Late QAT int6 STE
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
```
**Must avoid:** EMA, SWA, TTT, GPTQ (all conflict with recurrence).
**Must avoid:** >2 cycles, UNet skips (architectural tension).
~1.1570–1.1634 BPB.

### Archetype C: "TTT-Focused" (eval-time adaptation)
```
11L / 512d / GQA
+ LeakyReLU(0.5)²
+ Partial RoPE
+ LN Scale
+ SmearGate + OrthoInit
+ BigramHash
+ Muon + AdamW
+ Late QAT int6 STE
+ GPTQ-lite (not Full GPTQ)    [Full GPTQ conflicts with TTT]
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
+ LoRA TTT (rank-8, AdamW+cosine, score-first)
+ Document masking (BOS boundaries)
```
**Must avoid:** XSA (hard conflict with TTT), EMA (mild conflict), Full GPTQ (SGD TTT breaks it), depth recurrence.
~1.1190 BPB.

### Archetype D: "N-gram Hybrid" (neural + statistical)
```
11L / 512d / GQA
+ LeakyReLU(0.5)²
+ XSA-4
+ EMA (0.997)
+ SmearGate + OrthoInit
+ BigramHash
+ Partial RoPE
+ Muon + AdamW
+ Complementary Training (7-gram)   [pairs with n-gram cache ✓]
+ Late QAT int6 STE
+ GPTQ-lite
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
+ N-gram eval cache (orders 2–10, entropy-adaptive alpha)
+ kNN-LM (k=32, RBF kernel)
```
**Must avoid:** TTT (conflicts with XSA which is needed for EMA).
Note: Many n-gram cache approaches were **invalidated** by rule enforcement (must normalize over full vocabulary). Check current rules.
~0.4–0.9 BPB range (if eval cache is legal).

### Archetype E: "Ternary / BitNet" (extreme quantization)
```
10L / 768d / GQA
+ 8192 BPE tokenizer           [large vocab viable due to tiny weights]
+ BitNet 1.58-bit (absmean + STE from step 0)
+ FP8 QAT
+ YaRN positional encoding
+ NeoMuon optimizer
+ U-Net skips
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
```
**Different optimization regime entirely.** Most int6/GPTQ techniques don't apply.
~1.1570 BPB.

### Archetype F: "EngramLite + Turbo" (best unofficial)
```
11L / 512d / GQA
+ LeakyReLU(0.5)²
+ EngramLite (multi-head n-gram hashing + context gating)
+ Partial RoPE
+ Turbo-Muon optimizer
+ Mixed-precision GPTQ (int5 MLP / int6 attn or similar)
+ Late QAT
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
```
**Note:** EngramLite may have diminishing returns with BigramHash (overlapping functionality).
~1.1086 BPB (best known).

---

## Conflict Summary Graph

```
Hard conflicts (never combine):
  TTT ──✗── XSA
  TTT ──✗── Depth Recurrence
  EMA ──✗── Depth Recurrence
  SGD TTT ──✗── Full GPTQ
  Block sharing >2 ──✗── Any quantization

Hard dependencies (must co-deploy):
  SmearGate ──DEP── OrthoInit
  SWA ──DEP── WD≥0.04
  SWA ──DEP── FP32 accumulation
  EMA ──DEP── XSA (or EMA hurts)

This creates two mutually exclusive families:
  Family 1: XSA + EMA (no TTT, no depth recurrence)
  Family 2: TTT (no XSA, no EMA, no depth recurrence, no Full GPTQ)
  Family 3: Depth Recurrence (no EMA, no SWA, no TTT, no GPTQ, careful with QAT)
```

---

---

# EXTENDED MATRIX: Alternative Architectures & Efficiency Techniques

The following sections expand the compatibility analysis to include alternative architectures (Mamba-3, GLA, xLSTM, RWKV-7) and advanced efficiency techniques (Basis Sharing, Relaxed Recursive, Monarch matrices, FP8 training, Cross-Layer KV Sharing, BitNet) from the research in `research_architectures_and_efficiency.md`.

---

## Extended Abbreviation Key

| Abbrev | Technique | Category |
|--------|-----------|----------|
| Mb3 | Mamba-3 SSM layers | Architecture |
| GLA | Gated Linear Attention layers | Architecture |
| xLST | xLSTM / mLSTM blocks | Architecture |
| RW7 | RWKV-7 layers | Architecture |
| BsSh | Basis Sharing (cross-layer SVD) | Efficiency |
| RxRc | Relaxed Recursive (weight tying + LoRA) | Efficiency |
| Mnrc | Monarch / BLAST structured matrices | Efficiency |
| FP8T | FP8 Training (H100 Transformer Engine) | Efficiency |
| TrFu | Custom Triton Fused Kernels | Systems |
| CLKV | Cross-Layer KV Sharing | Efficiency |
| BtNt | BitNet / Ternary (1.58-bit) | Quantization |

---

## Extended Interaction Matrix: New Techniques vs Existing Stack

Read as: **row (new technique) + column (existing technique) = interaction**.

```
            XSA   EMA   SWA   SmGt  BGH   PRoP  LRe²  Muon  QAT6  GPTQ  TTT   DpRc  SlWn  FP16E UNet
Mb3          ctx   .     .     .     .     -     +     ?     -     -     ?     --    .     .     ?
GLA          ctx   .     .     .     .     ctx   +     ?     ?     -     ?     --    .     .     .
xLST         ctx   .     .     .     .     N/A   +     ?     ?     ?     ?     --    .     .     ?
RW7          --    .     .     -     .     N/A   ?     ?     ?     ?     ?     ?     .     .     .
BsSh         .     +     .     .     .     .     .     .     .     ?     ?     --    .     .     .
RxRc         ?     ?     ?     .     .     .     .     .     ?     ?     ?    repl   .     .     -
Mnrc         .     .     .     .     .     .     +     ?     ?     -     .     .     .     .     .
FP8T         .     .     .     .     .     .     .     .     +     +     .     .     .     .     .
TrFu         .     .     .     ++    ++    .     .     .     .     .     .     .     .     .     .
CLKV         +     .     .     .     .     .     .     .     -     -     .     .     .     .     .
BtNt         .     ?     .     .     .     .     .     -     --    --    .     .     .     +     .
```

## Extended Interaction Matrix: New Techniques vs Each Other

```
            Mb3   GLA   xLST  RW7   BsSh  RxRc  Mnrc  FP8T  TrFu  CLKV  BtNt
Mb3          ·     -     -     -     ?     ?     +     ++    +     N/A   ?
GLA          -     ·     -     -     ?     ?     +     ++    +     +     ?
xLST         -     -     ·     -     ?     ?     ?     ++    +     N/A   ?
RW7          -     -     -     ·     ?     ?     ?     +     +     N/A   ?
BsSh         ?     ?     ?     ?     ·     --    .     .     .     .     .
RxRc         ?     ?     ?     ?     --    ·     .     .     .     .     .
Mnrc         +     +     ?     ?     .     .     ·     ++    -     .     .
FP8T        ++    ++    ++     +     .     .     ++    ·     +     .     --
TrFu         +     +     +     +     .     .     -     +     ·     .     .
CLKV        N/A    +    N/A   N/A    .     .     .     .     .     ·     .
BtNt         ?     ?     ?     ?     .     .     .     --    .     .     ·
```

---

## New Hard Conflicts (--)

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| **Pure RWKV-7 + XSA** | Incompatible | XSA operates across attention sequence boundaries. RWKV-7 has no attention mechanism, so XSA has nothing to attach to. Only works in a hybrid where some layers are attention |
| **Mamba-3 + Depth Recurrence** | Same 900x quant error | Mamba-3 state transitions compound through reused blocks just like transformer attention does. Selective scan parameters are equally sensitive to accumulated quantization noise |
| **GLA + Depth Recurrence** | Same mechanism | GLA's gated recurrence compounds errors through shared weights |
| **xLSTM + Depth Recurrence** | Same mechanism | Matrix memory state update amplifies errors through reuse |
| **Basis Sharing + Relaxed Recursive** | Mutually exclusive | Both solve the same problem (reducing unique cross-layer parameters) via different mechanisms. Using both is redundant and the interactions are undefined |
| **Basis Sharing + Depth Recurrence** | Mutually exclusive | Basis Sharing IS the soft alternative to Depth Recurrence. They address the same parameter-saving goal |
| **BitNet + QAT6** | Incompatible regimes | BitNet trains ternary from step 0. Int6 QAT is a completely different quantization grid. Cannot mix |
| **BitNet + GPTQ** | Incompatible regimes | GPTQ calibrates for dense int6/int8 weight distributions. Ternary weights have a fundamentally different structure (only 3 values) |
| **BitNet + FP8 Training** | Incompatible regimes | FP8 training precision is meaningless when weights are ternary. BitNet uses STE from step 0 with its own precision regime |

---

## New Strong Synergies (++)

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| **FP8 Training + Mamba-3** | Compounding speed | Mamba-3 already has higher throughput than transformers. FP8 tensor cores on H100 give additional ~2x throughput on the GEMM portions. Combined: potentially 3-4x more training steps than BF16 transformer in 10 minutes |
| **FP8 Training + GLA** | Compounding speed | GLA's chunkwise algorithm is GEMM-heavy. FP8 accelerates all GEMMs. FLA library may need FP8 kernel variants |
| **FP8 Training + xLSTM** | Compounding speed | xLSTM already claims 3.5x training speed. FP8 could push this further. Potentially 5-7x more steps than BF16 transformer |
| **FP8 Training + Monarch/BLAST** | Triple compounding | Monarch reduces FLOP count by 70%. FP8 doubles throughput per FLOP. Combined: ~6x faster MLP computation. Massive step-count increase |
| **Triton Fusions + SmearGate** | Outsized impact at small scale | SmearGate + BigramHash + embedding lookup are 3 separate kernel launches for a small op each. Fusing into one kernel eliminates 2 launches. At 512d, kernel launch overhead dominates |
| **Triton Fusions + BigramHash** | Same mechanism | Part of the same fusion opportunity |
| **Monarch + Mamba-3** | Speed compounding | Monarch factorizes the linear projections inside Mamba-3 blocks. Mamba-3 already eliminated the causal conv. Leaner per-step compute |
| **Monarch + GLA** | Speed compounding | GLA's input/output projections can be Monarch-factorized for O(N log N) instead of O(N^2) matmuls |

---

## New Mild Synergies (+)

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| **Mamba-3 + LeakyReLU²** | Compatible | Mamba-3 interleaves MLP layers. LeakyReLU² applies to those MLPs. The sparsity benefit transfers |
| **GLA + LeakyReLU²** | Compatible | Same reason: GLA uses MLP sublayers |
| **xLSTM + LeakyReLU²** | Compatible | xLSTM residual blocks include MLPs |
| **FP8T + QAT6** | Smooth transition | FP8 training produces tightly concentrated weight distributions. This makes subsequent int6 QAT easier: fewer outliers to clip. Research confirms "QAT can improve performance of FP8 quantized models" |
| **FP8T + GPTQ** | Smoother calibration | "After FP8 training, activation distributions are tightly concentrated, allowing for direct per-tensor static PTQ." GPTQ Hessian calibration benefits from well-behaved weight distributions |
| **CLKV + XSA** | Orthogonal sharing | XSA shares context across sequences (horizontal). CLKV shares parameters across layers (vertical). Different dimensions, no interference |
| **CLKV + GLA** | Fewer unique params | In a hybrid with some GLA + some attention layers, sharing KV projections across the attention layers frees param budget for the GLA layers |
| **Basis Sharing + EMA** | Smooth shared bases | EMA averages weights. With basis sharing, the shared bases get smoothed while per-layer coefficients stay adaptive. Could make the bases better-conditioned for quantization |
| **Triton Fusions + FP8T** | Additive speed | Fused kernels reduce launch overhead. FP8 increases per-kernel throughput. Different bottlenecks, both help |
| **Triton Fusions + Mamba-3/GLA/xLSTM/RWKV-7** | Custom kernels needed | Each alt architecture benefits from fused forward/backward kernels. FLA library already provides these for GLA. Others need custom work |
| **FP8T + RWKV-7** | Speed gain | RWKV-7 linear ops benefit from FP8 tensor cores, though delta rule dynamics may need FP32 master state |

---

## New Mild Conflicts (-)

| Combo | Effect | Mechanism |
|-------|--------|-----------|
| **Mamba-3 + Partial RoPE** | Redundant/conflicting position encoding | Mamba-3 uses its own "RoPE trick" internally (complex-valued state = data-dependent rotary embedding). Applying Partial RoPE externally on Mamba layers is redundant. In a hybrid, PRoP only applies to attention layers anyway |
| **Mamba-3 + QAT6** | SSM quantization sensitivity | "Despite applying quantization only to the hidden state, it is far more vulnerable than anticipated." Selective scan parameters are fragile under quantization. QAT helps but can't fully fix SSM sensitivity |
| **Mamba-3 + GPTQ** | Hessian mismatch | GPTQ's Hessian-based calibration assumes standard linear layers. Mamba's selective scan, input-dependent A/B/C parameters, and state transitions need different calibration. Standard GPTQ may degrade quality |
| **GLA + GPTQ** | Post-QK sensitivity | Research shows "post-QK operations exhibit higher sensitivity to quantization in linear attention." GPTQ calibration may not account for GLA's gated dynamics. However, "gated attention exhibited significantly more stable results with quantization" overall |
| **RWKV-7 + SmearGate** | Redundant token mixing | RWKV-7 has its own token mixing mechanism (delta rule with vector-valued gating). SmearGate adds a second layer of adjacent-token blending that overlaps with what RWKV already does |
| **Mamba-3 + GLA/xLSTM/RWKV-7** | Redundant sub-quadratic layers | Mixing multiple alternative architectures in the same model adds complexity without clear benefit. Each solves the same problem (replace attention) differently. Pick one |
| **Monarch + GPTQ** | Calibration mismatch | GPTQ assumes dense weight matrices for Hessian reconstruction. Monarch's block-diagonal factored form breaks this assumption. Need custom GPTQ variant |
| **Monarch + Triton Fusions** | Fusion conflict | Monarch matrices have their own specialized GEMM routines. Generic Triton fusions (e.g., Liger-Kernel fused linear+CE) may not compose with Monarch's block-diagonal multiply |
| **BitNet + Muon** | Gradient mismatch | Muon performs Newton-Schulz orthogonalization on gradient matrices. Ternary weight STE gradients are fundamentally different (sparse, discrete). The orthogonalization may not converge or may produce poor updates |
| **CLKV + QAT6/GPTQ** | Shared quantization error | Shared KV projections mean quantization errors in those projections affect multiple layers simultaneously. Error doesn't average out across layers as with independent weights |
| **Relaxed Recursive + UNet** | Topological tension | Same issue as hard recurrence: skip connections assume unique encoder/decoder layers. LoRA relaxation helps differentiate passes but the skip topology is still confused by reuse |
| **Monarch + Muon (?)** | Unknown dynamics | Muon's Newton-Schulz orthogonalization is designed for dense matrices. On block-diagonal Monarch structures, the orthogonalization operates within blocks but may not capture cross-block interactions. Effect unknown |

---

## New Context-Dependent Interactions (ctx)

| Combo | Condition | Effect |
|-------|-----------|--------|
| **Mamba-3 + XSA** | Only in hybrid (some attention + some Mamba layers) | XSA applies to the attention layers only. Mamba layers don't interfere with XSA but also don't benefit from it. The 1:3 attention:Mamba ratio means only ~25% of layers use XSA. Diminished but still positive impact |
| **GLA + XSA** | Only in hybrid (some attention + some GLA layers) | Same as Mamba: XSA on the few attention layers only. GLA layers handle their own context via data-dependent gates. The attention layers provide XSA's cross-sequence benefit while GLA handles within-sequence efficiently |
| **xLSTM + XSA** | Only in hybrid | Same pattern. xLSTM layers have their own memory mechanism. Attention layers at the end provide XSA |
| **GLA + Partial RoPE** | Only on attention layers in hybrid | GLA doesn't use RoPE by default (uses its own position-dependent gating). Partial RoPE applies only to the attention layers in a hybrid config |
| **Relaxed Recursive + EMA** | Depends on LoRA rank | If LoRA rank is high enough to differentiate passes sufficiently, EMA may not cause the catastrophic 1.42 BPB failure seen with hard recurrence. Lower LoRA rank = more like hard recurrence = more risk. Needs ablation |
| **Relaxed Recursive + GPTQ** | Depends on LoRA rank | Higher LoRA rank = more differentiation between passes = less quantization error amplification. But still some amplification since the base weights are shared. Risk proportional to sharing degree |
| **Mamba-3 + Muon** | Depends on which parameters | Muon works on matrix-shaped parameters (linear projections). Mamba-3's input/output projections are standard linear layers (Muon OK). The SSM-specific parameters (A, B, C, state update) may need AdamW instead. Use mixed optimizer: Muon for projections, AdamW for SSM params |
| **GLA + Muon** | Same as Mamba-3 | GLA's input/output projections are linear (Muon OK). The gating parameters may need AdamW. Mixed optimizer approach |
| **xLSTM + Muon** | Matrix memory complicates things | mLSTM's matrix memory uses a covariance update rule. Standard Muon on the projection matrices should work. The memory-specific parameters need special handling |
| **BitNet + EMA** | Unknown but theoretically possible | EMA averages ternary weights in FP32 shadow. The resulting average is no longer ternary, but the shadow is only used for averaging, then requantized. Should work in theory but untested for ternary |

---

## New Hard Dependencies

| Technique | Requires | Why |
|-----------|----------|-----|
| **CLKV in hybrid** | At least 2 attention layers | Cross-layer KV sharing requires multiple attention layers to share between. Doesn't apply to Mamba/GLA/xLSTM layers |
| **Mamba-3 hybrid + XSA** | Final 2-4 layers must be attention | XSA needs attention layers to operate. Place attention at the end for maximum XSA benefit. Mamba for lower/faster layers |
| **GLA hybrid + XSA** | Same as Mamba-3 | Same reasoning |
| **FP8T + FP32 master weights** | FP32 weight copies | FP8 training uses FP8 for forward/backward compute but needs FP32 master weights for stability. Not optional |
| **Monarch + custom GEMM** | Triton/CUDA implementation | Standard PyTorch matmul doesn't know about Monarch block-diagonal structure. Need custom kernel to get the speed benefit |

---

## Updated Strategy Archetypes (with new techniques)

### Archetype G: "GLA Hybrid" (speed + XSA compatibility)
```
8 GLA layers + 3 attention layers (final)
  512d, GQA(8Q/4KV) on attention layers
+ LeakyReLU(0.5)² on all MLP sublayers
+ XSA on final 3 attention layers        [requires attention layers ✓]
+ EMA (0.997)                             [requires XSA ✓]
+ SWA (WD≥0.04, FP32)
+ SmearGate + OrthoInit
+ BigramHash (2048)
+ Partial RoPE on attention layers only
+ LN Scale
+ Muon for projections, AdamW for GLA gates  [mixed optimizer]
+ Late QAT int6 STE                       [attention layers only; GLA may need int8]
+ GPTQ-lite post-training                 [attention layers only]
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval (stride=64)
+ FP8 Training via Transformer Engine      [accelerates all GEMMs]
```
**Key tradeoff:** GLA layers train faster but may quantize worse than attention.
GLA provides length generalization (train 1024, eval 4096 for free).
~8 GLA layers replace 8 attention layers, potentially 30-40% faster step time.
**Must verify:** GLA int6 quantization quality. Fall back to int8 for GLA layers if needed.
**Estimated range:** Speculative, but faster training + XSA compatibility could approach 1.10-1.12 BPB.

### Archetype H: "Mamba-3 Hybrid" (max throughput + XSA)
```
8 Mamba-3 layers + 3 attention layers (final)
  512d, GQA on attention layers
+ LeakyReLU(0.5)² on MLP sublayers (Mamba-3 interleaves MLPs)
+ XSA on final 3 attention layers
+ EMA (0.997)
+ SmearGate + OrthoInit
+ BigramHash (2048)
+ Partial RoPE on attention layers only
+ Muon for linear projections, AdamW for SSM params  [critical: mixed optimizer]
+ Int8 for Mamba-3 layers (SSM is fragile under int6)
+ Int6 + GPTQ-lite for attention layers
+ FP16 tied embeddings
+ FP8 Training
+ zstd-22
+ Sliding window eval
```
**Key tradeoff:** Mamba-3 throughput is higher but SSM quantization is harder.
Need int8 (not int6) for Mamba layers, costing ~1MB more artifact space.
**Must verify:** Mamba-3 kernel availability/speed on H100 vs FlashAttention v3.
**Risk:** High. Prior SSM result at dim=512 was negative. Mamba-3 may change this but untested.

### Archetype I: "Basis Sharing Neural" (soft recurrence, no conflicts)
```
11L / 512d / GQA(8Q/4KV)
  Shared SVD bases across all 11 layers
  Unique per-layer coefficients (small)
+ LeakyReLU(0.5)²
+ XSA-4 or XSA-all
+ EMA (0.997)                             [compatible with basis sharing ✓]
+ SWA (WD≥0.04, FP32)
+ SmearGate + OrthoInit
+ BigramHash (2048-3072)
+ Partial RoPE (16/64)
+ LN Scale
+ Muon + AdamW (mixed)
+ Late QAT int6 STE
+ GPTQ-lite post-training
+ FP16 tied embeddings
+ U-Net skips
+ zstd-22
+ Sliding window eval (stride=64)
```
**This is Archetype A + Basis Sharing.** The key difference: Basis Sharing reduces
unique parameters substantially (shared bases + small per-layer coefficients), freeing
artifact budget for wider MLPs, more layers, or larger BigramHash tables.
**Avoids all recurrence conflicts** because each layer still has unique effective weights.
**Must verify:** GPTQ interaction with shared bases (quantize bases + coefficients separately).

### Archetype J: "FP8 Speed Demon" (max training steps)
```
11L / 512d / GQA(8Q/4KV)
+ FP8 Training via Transformer Engine     [~2x throughput]
+ Custom Triton fusions (SmearGate+BGH+embed, LN+QKV)  [additional 20-40% throughput]
+ LeakyReLU(0.5)²
+ XSA-4
+ EMA (0.997)
+ SmearGate + OrthoInit
+ BigramHash (2048)
+ Partial RoPE
+ Muon + AdamW
+ Late QAT int6 STE                       [FP8 -> int6 transition is smooth ✓]
+ GPTQ-lite post-training                 [FP8 weights have tighter distributions ✓]
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
```
**This is Archetype A + FP8 + Triton fusions.** Conservative architecture, aggressive
systems optimization. Goal: ~2.5x more training steps in 10 minutes than current SOTA.
More training steps = lower BPB even with identical architecture.
**Risk:** Low. FP8 is well-supported on H100. Triton fusions are pure engineering.
**Estimated range:** 1.10-1.12 BPB (more steps on the same architecture that already works).

### Archetype K: "xLSTM Speed Hybrid" (untested high-risk)
```
8 mLSTM blocks + 3 attention layers (final)
  512d, GQA on attention layers
+ XSA on final 3 attention layers
+ EMA (0.997)
+ SmearGate + OrthoInit
+ BigramHash
+ FP8 Training                            [compounds with xLSTM's 3.5x speed]
+ Muon for projections, AdamW for mLSTM memory params
+ Late QAT (attention layers int6, mLSTM layers int8)
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
```
**The moonshot.** If xLSTM's 3.5x speed claim holds AND FP8 adds another 2x,
you get potentially 7x more training steps than current SOTA in the same 10 minutes.
**Risk:** Very high. xLSTM at 20M params is completely untested. mLSTM quantization is unknown.
Kernel availability is uncertain. But the upside is enormous.

### Archetype L: "Ternary Mamba-3" (extreme compression)
```
14L Mamba-3 / 768d (no attention)
  BitNet 1.58-bit (absmean + STE from step 0)
  ~100M+ effective parameters in 16MB
+ 8192 BPE tokenizer
+ YaRN position encoding
+ NeoMuon or AdamW (Muon may conflict with ternary STE)
+ FP16 tied embeddings
+ zstd-22
+ Sliding window eval
```
**No XSA, no EMA, no GPTQ, no standard quantization.** Completely different paradigm.
Betting on raw parameter count (100M+ ternary vs 21M int6) overcoming per-param quality loss.
**Must avoid:** Muon (likely conflicts with ternary STE), int6 QAT, GPTQ, all attention-specific tricks.
**Risk:** Very high. Neither ternary Mamba nor Mamba-3 at this scale has been tested.
**Potential:** If ternary quality-per-param improves at 100M+ scale (research suggests it does), this could break through.

---

## Updated Conflict Summary Graph

```
Hard conflicts (never combine):
  TTT ──✗── XSA
  TTT ──✗── Depth Recurrence
  EMA ──✗── Depth Recurrence
  SGD TTT ──✗── Full GPTQ
  Block sharing >2 ──✗── Any quantization
  Pure RWKV-7 ──✗── XSA  (no attention to attach to)
  BitNet ──✗── QAT6/GPTQ  (incompatible quant regimes)
  BitNet ──✗── FP8 Training  (incompatible precision regimes)
  Basis Sharing ──✗── Relaxed Recursive  (mutually exclusive approaches)
  Basis Sharing ──✗── Depth Recurrence  (mutually exclusive approaches)
  All alt architectures ──✗── Depth Recurrence  (same error amplification)

Hard dependencies (must co-deploy):
  SmearGate ──DEP── OrthoInit
  SWA ──DEP── WD≥0.04
  SWA ──DEP── FP32 accumulation
  EMA ──DEP── XSA (or EMA hurts)
  FP8T ──DEP── FP32 master weights
  Monarch ──DEP── Custom GEMM kernel
  Mamba-3/GLA/xLSTM hybrid + XSA ──DEP── Final 2-4 layers must be attention
  CLKV ──DEP── ≥2 attention layers

This creates FIVE mutually exclusive families:

  Family 1: XSA + EMA (transformer)
    No TTT, no depth recurrence
    Can hybrid with GLA/Mamba-3/xLSTM if final layers are attention

  Family 2: TTT
    No XSA, no EMA, no depth recurrence, no Full GPTQ

  Family 3: Depth Recurrence (hard)
    No EMA, no SWA, no TTT, no GPTQ, no alt architectures

  Family 4: BitNet / Ternary
    No int6 QAT, no GPTQ, no FP8 training
    Can use with any backbone (transformer, Mamba-3, RWKV-7)

  Family 5: Basis Sharing (soft recurrence)
    Compatible with EMA, XSA, GPTQ (potentially)
    Alternative to both Family 3 and Family 1's parameter budget
    Cannot combine with Depth Recurrence or Relaxed Recursive
```

---

## Updated Practical Guidance

1. **Archetype A** (Pure Neural Transformer) remains the safest bet with most validated synergies. ~1.1147 BPB.

2. **Archetype J** (FP8 Speed Demon) is the lowest-risk improvement: same proven architecture, more training steps via FP8 + Triton fusions. Try this first.

3. **Archetype I** (Basis Sharing Neural) replaces hard recurrence with soft sharing. Gets parameter savings of recurrence without losing EMA/GPTQ compatibility. Medium risk, medium reward.

4. **Archetype G** (GLA Hybrid) is the most promising architecture change. GLA trains fast, generalizes to long context, and keeps attention for XSA on final layers. Medium-high risk.

5. **Archetype H** (Mamba-3 Hybrid) has higher throughput potential but SSM quantization is a real problem. Only if you can afford int8 on Mamba layers.

6. **Archetype K** (xLSTM Speed) is theoretically the fastest but completely untested at small scale. High risk, potentially enormous reward.

7. **Archetype L** (Ternary Mamba-3) is the moonshot. 100M+ params in 16MB. Only try this if everything else has been exhausted.

8. **Archetypes B-F** (from original matrix) remain valid. Recurrence (B) is now superseded by Basis Sharing (I) which avoids its conflicts.

9. **Never mix quantization families.** BitNet and int6 are incompatible. Pick one.

10. **In hybrid architectures, always put attention layers last.** XSA needs attention. EMA needs XSA. Placing attention at the end preserves these critical dependencies.
