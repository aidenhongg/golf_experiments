# Parameter Golf Research: Alternative Architectures & Efficiency Techniques

Research compiled for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) challenge.
**Constraint reminder:** 16MB artifact, 10 min training on 8xH100, lowest BPB on FineWeb wins.

---

## Part 1: Alternative Model Architectures (Not Standard Transformer+MLP)

The current SOTA stack is a standard transformer (11L, 512d, GQA) with MLP blocks. Below are architectures that could theoretically compete or be hybridized.

---

### 1.1 Mamba / Selective State Space Models (SSM)

**What it is:** Linear-time sequence model where SSM parameters are functions of the input (selective mechanism). No attention, no MLP blocks in the original design. Uses a hardware-aware parallel scan algorithm.

**Why it could work for Parameter Golf:**
- 5x higher throughput than transformers at inference (more training steps in 10 min)
- Linear scaling in sequence length (cheap long-context)
- At ~40M params, matches Transformer++ with 3-4x fewer parameters
- Mamba-3B matches Transformer-6B perplexity

**Why it might NOT work:**
- Competition already tested SSM hybrids: "showed -18% improvement at dim=192 but +2.7% worse at dim=512" (scale deception)
- Lags behind transformers on copying / in-context learning tasks
- FlashAttention v3 on H100 is extremely fast, narrowing the throughput advantage
- Selective scan kernels may not be as optimized as FlashAttention for short sequences

**Mamba-2 improvements:** Restructured computation as large matrix multiplications for better hardware utilization on tensor cores. State Space Duality (SSD) framework connects SSMs to attention.

**Mamba-3 (ICLR 2026):**
- Generalized trapezoidal discretization (2nd-order accurate)
- Complex-valued state updates via "RoPE trick" (data-dependent rotary embedding)
- MIMO formulation improving performance without increasing decode latency
- Removes the short causal convolution entirely
- At 1.5B: 57.6% avg accuracy, 2.2pp leap over standard Transformer
- Achieves comparable perplexity to Mamba-2 with **half the state size**

**Verdict for Parameter Golf:** Worth exploring, especially Mamba-3's halved state size. The prior negative result at dim=512 was with older Mamba. Mamba-3's improvements might change the picture. Main risk: kernel optimization lag behind FlashAttention on H100.

**References:**
- [Mamba paper](https://arxiv.org/abs/2312.00752)
- [Mamba-3 (ICLR 2026)](https://arxiv.org/abs/2603.15569)
- [Mamba-3 blog](https://www.together.ai/blog/mamba-3)

---

### 1.2 RWKV-7 "Goose"

**What it is:** RNN architecture combining benefits of recurrent and attention-based systems. Uses a generalized delta rule with vector-valued gating and in-context learning rates. Fully parallelizable training, linear-time inference, constant memory (no KV cache).

**Why it could work for Parameter Golf:**
- Linear time, constant space, fast training, infinite context length
- 2.9B model achieves new 3B SOTA on multilingual tasks
- Can be directly trained like a GPT transformer (parallelizable)
- Free sentence embedding (useful for eval-time tricks?)
- "Default config only requires 1 GPU with 10G VRAM" (extremely lean)

**Why it might NOT work:**
- Less battle-tested at tiny scale (<50M params)
- Custom CUDA kernels needed for competitive speed
- Community/tooling smaller than transformer ecosystem
- Unclear how well it quantizes to int6

**RWKV-X (2025):** Linear complexity hybrid that can dynamically switch between attention and SSM at different token lengths.

**Verdict for Parameter Golf:** Strong candidate if training speed advantage materializes on H100. The constant-memory property means you could potentially train with much longer sequences without memory issues, enabling better context utilization within the 10-minute budget. RWKV-7's delta rule formulation is mathematically richer than older RWKV versions.

**References:**
- [RWKV-7 paper](https://arxiv.org/abs/2503.14456)
- [RWKV GitHub](https://github.com/BlinkDL/RWKV-LM)
- [RWKV-X](https://arxiv.org/html/2504.21463v2)

---

### 1.3 Griffin / Hawk (Google DeepMind)

**What it is:** Hawk is a pure RNN with gated linear recurrences. Griffin is a hybrid mixing gated linear recurrences with local attention.

**Why it could work for Parameter Golf:**
- Hawk exceeds Mamba on downstream tasks
- Griffin matches Llama-2 performance trained on 6x fewer tokens (data efficiency matters in 10-min budget)
- Can extrapolate on sequences significantly longer than training
- Matches hardware efficiency of Transformers during training
- Lower inference latency + significantly higher throughput

**Why it might NOT work:**
- Primarily tested at 1B+ scale
- Local attention component in Griffin still has quadratic cost within window
- Less community tooling than Mamba or standard transformers

**Verdict for Parameter Golf:** Griffin's hybrid approach (local attention + gated linear recurrence) is interesting because you get the best of both worlds. The 6x data efficiency claim is particularly relevant given the fixed training budget. The local attention window could be tuned to balance speed vs quality.

**References:**
- [Griffin paper](https://arxiv.org/abs/2402.19427)

---

### 1.4 Hymba (NVIDIA, Hybrid Mamba-Transformer)

**What it is:** Parallel hybrid-head architecture integrating transformer attention and SSM heads within the same layer. Attention and SSM process input in parallel, then combine.

**Why it could work for Parameter Golf:**
- 1.5B model outperformed all sub-2B public models
- Surpassed Llama-3.2-3B with 1.32% higher accuracy
- 11.67x reduction in cache size, 3.49x higher throughput
- Learnable "meta tokens" prepended to every input reduce attention burden
- Cross-layer KV sharing + partial sliding window attention

**Why it might NOT work:**
- Designed for 1B+ scale, unclear benefit at 20M params
- Parallel attention+SSM adds architectural complexity
- Meta tokens consume sequence positions

**Key insight for Parameter Golf:** The cross-layer KV sharing technique from Hymba could be adapted even within a standard transformer to save parameters. The meta token concept (learnable prefixes storing critical info) could complement the existing SmearGate approach.

**References:**
- [Hymba paper](https://arxiv.org/html/2411.13676v1)
- [NVIDIA blog](https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/)

---

### 1.5 xLSTM (Extended LSTM)

**What it is:** Modern LSTM with exponential gating + two variants: sLSTM (scalar memory, new memory mixing) and mLSTM (matrix memory, fully parallelizable, covariance update rule). Stacked in residual blocks.

**Why it could work for Parameter Golf:**
- 3.5x faster training than baseline Transformer at same size
- Exponential gating with normalization/stabilization
- mLSTM is fully parallelizable (no sequential bottleneck)
- Performance favorable vs state-of-the-art Transformers and SSMs
- Good scaling properties

**Why it might NOT work:**
- Less tested at tiny scale
- Matrix memory in mLSTM could be parameter-hungry
- Quantization behavior unknown for exponential gating

**Verdict for Parameter Golf:** The 3.5x training speed claim is extremely attractive for a 10-minute budget. If that translates to 3.5x more training steps, the BPB impact could be significant. The mLSTM variant with matrix memory is the parallelizable one to target.

**References:**
- [xLSTM paper](https://arxiv.org/abs/2405.04517)

---

### 1.6 RetNet (Retentive Network)

**What it is:** Retention mechanism supporting parallel, recurrent, and chunkwise computation. Dual form of recurrence and attention.

**Why it could work for Parameter Golf:**
- O(1) inference cost per step
- Three computation modes enable flexible training/eval tradeoffs
- Competitive with Transformer on language modeling

**Why it might NOT work:**
- GLA (Gated Linear Attention) outperforms RetNet at same scale (GLA 1.3B: 17.22 perplexity vs RetNet 18.64 on WikiText)
- Data-independent scalar decay is a limitation (GLA fixes this with data-dependent gates)

**Verdict for Parameter Golf:** Consider GLA over RetNet. GLA's data-dependent gating is strictly more expressive, with better benchmarks and the flash-linear-attention library provides efficient Triton kernels.

**References:**
- [RetNet paper](https://arxiv.org/abs/2307.08621)
- [GLA paper](https://arxiv.org/abs/2312.06635)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

---

### 1.7 GLA (Gated Linear Attention)

**What it is:** Linear attention with data-dependent scalar gates. Hardware-efficient chunkwise algorithm.

**Why it could work for Parameter Golf:**
- Competitive with LLaMA-architecture Transformer at same scale
- Exceptional length generalization: trained on 2K, generalizes to 20K+
- Efficient Triton kernels in FLA library
- Better than RetNet and competitive with Mamba

**Why it might NOT work:**
- Relatively new, less hyperparameter tuning knowledge
- Unclear quantization properties
- May not beat optimized FlashAttention v3 at short (1024-2048) sequences

**Verdict for Parameter Golf:** Strong candidate, especially combined with the length generalization property. Train at 1024, eval at 4096 with essentially no quality loss. The FLA library provides ready-to-use Triton kernels.

**References:**
- [GLA paper](https://arxiv.org/abs/2312.06635)
- [FLA library](https://github.com/fla-org/flash-linear-attention)

---

### 1.8 Hyena (Long Convolutions)

**What it is:** Subquadratic operator using implicitly parameterized long convolutions and data-controlled gating. Drop-in replacement for attention.

**Why it could work for Parameter Golf:**
- At 335M params, matches Transformer perplexity with 20% fewer FLOPs
- 2x faster than optimized attention at seq 8K, 100x at seq 64K
- Sublinear parameter scaling in sequence length
- Unrestricted context length

**Why it might NOT work:**
- Speed advantage only kicks in at longer sequences (8K+)
- At 1024-2048 seq length (Parameter Golf range), advantage may be minimal
- Less tested at very small scale (<50M params)
- Implicit convolution parameterization adds complexity

**Verdict for Parameter Golf:** Unlikely to help at the 1024-2048 sequence lengths used in the competition. The speed advantage is for long sequences. Skip unless you're building a long-context-focused approach.

**References:**
- [Hyena paper](https://arxiv.org/abs/2302.10866)
- [Hazy Research blog](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)

---

### 1.9 BitNet / Ternary Architectures

**What it is:** Models natively trained with 1.58-bit (ternary: -1, 0, +1) weights from scratch using QAT.

**Why it could work for Parameter Golf:**
- Dramatically more parameters per byte: a 73.7M param model fits in 16MB
- At 2B params + 4T tokens: performance within 1-2 points of full-precision SOTA
- "Matches FP16 baselines starting from 3B size"
- 100K-48M parameter models show 1.58-bit QAT can get close to SOTA on SLMs
- Memory footprint reduction from ~2GB to 0.4GB at 2B scale

**Why it might NOT work:**
- Competition already tested: BitNet achieved 1.1570 BPB (worse than SOTA 1.1086)
- Different optimization regime (can't easily combine with int6 techniques)
- Training ternary from scratch requires specialized kernels
- Muon optimizer may not work well with ternary gradients

**Verdict for Parameter Golf:** Already explored in the competition with decent but not SOTA results. The parameter budget advantage (73.7M params in 16MB vs ~21M with int6) is real, but the quality-per-parameter is lower. Could improve with Mamba-3 or RWKV-7 backbone instead of transformer.

**References:**
- [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [BitNet b1.58 Reloaded (small networks)](https://arxiv.org/html/2407.09527v1)

---

### 1.10 Hybrid Approaches (Most Promising)

Based on the research, the most promising unexplored directions for Parameter Golf are **hybrid architectures** that mix mechanisms:

| Hybrid | Idea | Why it could win |
|--------|------|-----------------|
| **Transformer + Mamba-3 layers** | Replace some transformer layers with Mamba-3 blocks | Mamba-3 achieves same perplexity with half the state, potentially fewer params per layer. Use attention only where it matters most (final layers for XSA compatibility) |
| **GLA + attention hybrid** | Use GLA for most layers, attention for last 2-3 (for XSA) | GLA trains fast and generalizes to long context. Keep attention only where XSA provides its proven benefit |
| **xLSTM (mLSTM) + attention** | mLSTM for speed, attention for final layers | 3.5x training speed claim could yield dramatically more training steps |
| **Ternary Mamba-3** | BitNet 1.58 weights inside Mamba-3 architecture | Mamba-3's halved state size + ternary compression = massive parameter budget. Could fit 100M+ effective params |

---

## Part 2: Memory & Speed Efficiency Techniques

---

### 2.1 Advanced Weight Sharing

#### Basis Sharing (ICLR 2025)
- Decompose weight matrices across layers via SVD into shared basis vectors + unique per-layer coefficients
- Outperforms standard SVD compression, especially at large compression ratios
- **Application:** Share basis vectors across all 11 layers, unique coefficients per layer. Could dramatically reduce unique parameters while maintaining per-layer expressiveness

#### MASA (Matrix Atom Sharing in Attention, 2025)
- Learns a compact set of matrix "atoms" capturing shared patterns across layers
- Each layer reconstructs weights via layer-specific coefficients
- More flexible than ALBERT-style rigid weight tying
- **Application:** Instead of the current 2-cycle depth recurrence (which conflicts with EMA/GPTQ), use MASA-style soft sharing that avoids the quantization error amplification problem

#### Relaxed Recursive Transformers
- Weight tying across repeated blocks relaxed with per-block low-rank (LoRA) modules
- Avoids the 900x quantization error amplification of hard recurrence
- **Application:** Use 6 shared blocks with per-block LoRA adapters (rank 4-8). Gets the parameter savings of recurrence without the hard conflicts

#### Cross-Layer KV Sharing (from Hymba)
- Share key-value projections across adjacent layers
- Reduces KV parameter count substantially
- **Application:** Could free up parameter budget for wider MLPs or more layers

---

### 2.2 Low-Rank Training from Scratch

#### Monarch Matrices
- Parameterize weight matrices as products of two block-diagonal matrices
- O(N log N) matrix-vector products instead of O(N^2)
- Up to 70% reduction in computational complexity while recovering original accuracy
- **Application:** Replace dense MLP weight matrices with Monarch parameterization. Faster matmuls = more training steps in 10 minutes

#### Butterfly Factorizations
- Factor N x N matrix into O(log N) sparse matrices with O(N) nonzeros each
- O(N log N) operations for matrix-vector products
- Can be used in training from scratch with random initialization
- **Application:** For the 512x1536 MLP matrices, butterfly factorization could reduce compute while maintaining expressiveness

#### BLAST (Block-Level Adaptive Structured Matrices)
- Training from scratch with BLAST-format weights
- Up to 70% reduction in computational complexity
- "Effectively recovers original accuracy"
- **Application:** Most promising for the MLP blocks which dominate parameter count

---

### 2.3 Structured Pruning & Sparse Training

#### Lottery Ticket at Training Time
- Train with progressive pruning from the start
- Sparse-to-sparse training avoids ever materializing the full dense model
- **Application:** Start with a larger model, prune during training to fit 16MB. Could discover more efficient subnetworks than training a small dense model directly

#### 2:4 Structured Sparsity (Revisited)
- Competition found "large negative results" with 2:4 sparsity
- BUT: this was with ReLU^2 which already produces 84-98% unstructured sparsity
- With a different activation (SwiGLU, Mamba gates), 2:4 might work
- H100 tensor cores give 2x speedup for 2:4 patterns
- **Application:** Only worth revisiting if switching away from ReLU^2 activation

---

### 2.4 Fused Kernel Optimizations

#### Liger-Kernel (Already Partially Used)
- Fused RMSNorm: 7x speedup, 3x memory reduction
- Fused RoPE: 8x faster, 3x less memory
- Fused linear+CrossEntropy: 3x speedup
- Total: 20-43% throughput improvement
- **Application:** Already used in top submissions. Ensure ALL available fusions are applied

#### Custom Triton Kernels
- Operation fusion eliminates kernel launch overhead (critical for small models where launch overhead dominates)
- Up to 8x speed improvements and 50% memory reduction
- Key opportunity: fuse SmearGate + BigramHash + embedding lookup into a single kernel
- Key opportunity: fuse LayerNorm + attention QKV projection
- **Application:** For small models, kernel launch overhead is proportionally larger. Fusing more operations has outsized impact

#### CUTLASS EVT (Epilogue Visitor Tree)
- Already explored: fuses backward MLP computation into GEMM epilogue
- 3.7% step time reduction = ~500 extra training steps
- Hopper-only (H100 qualifies)
- **Application:** Already in use, but could be extended to more operations

---

### 2.5 Mixed Precision Training Strategies

#### FP8 Training (H100 Native)
- H100 has native FP8 tensor cores (2x throughput vs BF16)
- FP8 training reduces memory by 2x for activations
- Challenge: training stability requires careful scaling
- **Application:** Use FP8 for forward pass activations, BF16 for master weights and gradients. Could nearly double training throughput

#### Per-Tensor vs Per-Channel FP8 Scaling
- Per-tensor: simpler, some accuracy loss
- Per-channel: better accuracy, more overhead
- Block-wise scaling (like FP8 in Transformer Engine): best tradeoff
- **Application:** NVIDIA Transformer Engine handles this automatically on H100. Ensure it's enabled

---

### 2.6 Gradient Checkpointing

- Trades compute for memory: ~30% compute overhead for major memory savings
- Sweet spot: checkpoint every 2-4 layers for 40-60% memory savings with 10-20% compute overhead
- Can enable larger batch sizes, which improves GPU utilization
- **Application:** For 11 layers at 512d, memory likely isn't the bottleneck on H100 80GB. But if switching to a wider/deeper architecture, checkpointing enables it. Also useful if running FP8 training with larger batch sizes

---

### 2.7 Knowledge Distillation

- Competition found: "I/O overhead fatal in 600s budget" for distillation
- However, **offline distillation** (pre-compute teacher logits, store as training data) avoids runtime overhead
- TAID approach: gradually adapt teacher based on student's learning progress
- At 70M student / large teacher: meaningful quality transfer demonstrated
- **Application:** Pre-compute soft labels from a larger model on FineWeb training data. Store as part of the 16MB artifact (tricky with space budget). Or, use the model's own predictions from early training as a curriculum signal (self-distillation)

---

### 2.8 Data Efficiency Techniques

#### Curriculum Learning
- Sequence length curriculum already used (128 -> 2048)
- **Unexplored:** difficulty-based curriculum (easy documents first, hard later)
- **Unexplored:** domain-based curriculum (cluster FineWeb, train on each cluster sequentially)

#### Token Weighting / Complementary Training
- Already explored: down-weight tokens predictable by n-grams
- **Extension:** train a tiny byte-level n-gram model first (offline), use its confidences to weight all training tokens. Essentially pre-compute the complementary training signal

#### Data Mixing
- FineWeb contains diverse web data
- Some domains may be more compressible than others
- **Application:** Analyze FineWeb shard statistics, over-sample domains that improve BPB most per training step

---

### 2.9 Inference-Time Efficiency (for Eval)

#### Speculative Decoding Style
- Not directly applicable (eval is perplexity, not generation)
- But: could use a tiny draft model to identify "easy" tokens and skip full model computation

#### Batched Eval
- Process multiple sequences simultaneously
- Maximize GPU utilization during eval
- Already implicit in sliding window eval

#### KV Cache Compression During Eval
- For sliding window eval with stride=64, the KV cache is recomputed repeatedly
- Cross-sequence KV sharing (from Hymba) could reduce this overhead
- Quantize KV cache to int8 during eval to save memory

---

## Part 3: Synthesis — What's Actually Worth Trying

### Tier 1: High confidence, likely positive impact

| Technique | Why | Risk |
|-----------|-----|------|
| **Basis Sharing** across layers | Replaces hard weight tying (2-cycle recurrence) with soft sharing that's compatible with EMA/GPTQ | Medium: implementation complexity |
| **Custom Triton fusions** for SmearGate+BigramHash+embedding | Reduces kernel launch overhead at small model scale | Low: pure speed gain |
| **FP8 training** via Transformer Engine | Nearly 2x training throughput on H100 | Medium: stability tuning needed |
| **Relaxed Recursive Transformer** with LoRA adapters | Gets recurrence parameter savings without EMA/GPTQ conflicts | Medium: needs ablation |
| **GLA layers** replacing some attention layers | Fast, length-generalizing, compatible with FLA library | Medium: integration work |

### Tier 2: Moderate confidence, worth exploring

| Technique | Why | Risk |
|-----------|-----|------|
| **Mamba-3 hybrid** (Mamba-3 for lower layers, attention for top 3-4 + XSA) | Mamba-3 halved state size + complex gates, attention where XSA matters | High: two different codebases |
| **Monarch/BLAST matrices** for MLP | Reduce MLP compute by 70%, enabling more training steps | Medium: quality-compute tradeoff unknown at this scale |
| **xLSTM mLSTM blocks** | 3.5x training speed claim | High: untested at tiny scale |
| **Pre-computed soft labels** (offline distillation) | Better training signal without runtime overhead | Medium: space budget for labels |
| **MASA-style attention sharing** | Parameter savings with per-layer adaptation | Medium: novel technique |

### Tier 3: Speculative, high risk/reward

| Technique | Why | Risk |
|-----------|-----|------|
| **Full Mamba-3 (no attention)** | If training speed advantage is real, more steps = lower BPB | High: prior SSM results were negative |
| **RWKV-7 backbone** | Linear time + constant memory + parallelizable | High: tooling gap |
| **Ternary Mamba-3** (BitNet + Mamba-3) | 100M+ params in 16MB with a faster architecture | Very high: uncharted territory |
| **Butterfly-factorized everything** | O(N log N) all matrix ops | High: quality loss unknown |
| **2:4 sparsity with SwiGLU** (not ReLU^2) | Avoids the ReLU^2 conflict that killed prior attempts | High: may still not help |

---

## Sources

### Architecture Papers
- [Mamba](https://arxiv.org/abs/2312.00752) — Selective SSM, linear-time
- [Mamba-3 (ICLR 2026)](https://arxiv.org/abs/2603.15569) — Complex-valued, MIMO, halved state
- [Griffin/Hawk](https://arxiv.org/abs/2402.19427) — Gated linear recurrence + local attention
- [xLSTM](https://arxiv.org/abs/2405.04517) — Exponential gating, matrix memory
- [RetNet](https://arxiv.org/abs/2307.08621) — Retention mechanism, three computation modes
- [GLA](https://arxiv.org/abs/2312.06635) — Gated linear attention, data-dependent gates
- [Hyena](https://arxiv.org/abs/2302.10866) — Long convolutions, subquadratic
- [Hymba](https://arxiv.org/html/2411.13676v1) — Parallel hybrid attention+SSM heads
- [RWKV-7](https://arxiv.org/abs/2503.14456) — Generalized delta rule, vector gating
- [BitNet b1.58 Reloaded](https://arxiv.org/html/2407.09527v1) — Ternary at small scale

### Efficiency Techniques
- [Basis Sharing (ICLR 2025)](https://arxiv.org/abs/2410.03765) — Cross-layer SVD sharing
- [Monarch Matrices](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf) — Structured efficient matrices
- [BLAST](https://arxiv.org/pdf/2410.21262) — Block-level adaptive structured matrices
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — Fused Triton kernels for LLM training
- [Flash-Linear-Attention](https://github.com/fla-org/flash-linear-attention) — Efficient kernels for GLA, RetNet, Mamba2
- [TAID Distillation](https://sakana.ai/taid/) — Adaptive teacher-student knowledge transfer
- [NVIDIA Hymba blog](https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/)
- [SSM vs Transformer Tradeoffs](https://goombalab.github.io/blog/2025/tradeoffs/) — Albert Gu's analysis
- [Latent Space: 2024 Post-Transformers](https://www.latent.space/p/2024-post-transformers) — NeurIPS recap
