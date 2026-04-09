# OpenAI Parameter Golf — Comprehensive Technique Catalog

**Goal:** Train the best language model fitting in a 16MB artifact, trainable in ≤10 min on 8xH100s. Scored by bits-per-byte (BPB) on FineWeb validation. Current SOTA: **1.1086 BPB** (down from 1.2244 baseline).

---

## 1. Architecture (Macro)

| Technique | Description | Impact |
|---|---|---|
| **Deeper over wider** | 11 layers at 512d beats 9L/512d baseline; 12L viable only at seq1024 (too slow at seq2048) | Core design choice |
| **3× MLP expansion** | Hidden dim 1536 (3×512) instead of baseline 2× — better capacity per compressed byte | Standard in top submissions |
| **U-Net skip connections** | Encoder-decoder style shortcuts between corresponding layers (5 encoder, 6 decoder) with learned per-skip weights | Used in SOTA stack |
| **Depth recurrence / weight tying** | Reuse the same N blocks K times (e.g., 8 blocks × 2 loops = 16 effective layers). Massive parameter savings | Works up to 2 cycles only |
| **FiLM conditioning on loops** | Tiny scale/shift params per loop iteration (~3K params) so the model knows which pass it's on | Enables recurrence |
| **Hourglass FFN** | Stacked narrow-to-narrow sub-MLPs with residuals — deeper MLP at fewer params | Explored |
| **Tied embeddings** | Input/output embedding weight sharing | Universal |

**What failed:**
- **MoE** — optimal sparsity = 0 below 500M params
- **SSM hybrids** — looked good at dim=192 but worse at dim=512 (scale deception)
- **MLA (Multi-Head Latent Attention)** — quality present but 83ms/step, too slow
- **Block sharing beyond 2 cycles** — 900× quantization error amplification
- **AttnRes (learned softmax over depth)** — 54% throughput penalty

---

## 2. Attention Mechanisms

| Technique | Description | Impact |
|---|---|---|
| **Grouped Query Attention (GQA)** | 8 query heads, 4 KV heads — reduces parameters while maintaining expressiveness | Universal in top runs |
| **Cross-Sequence Attention (XSA)** | Attention across sequence boundaries, borrowing context from adjacent sequences. XSA-all (11 layers) best; XSA-4 (last 4 layers) is the speed/quality sweet spot. **Zero additional parameters** | Key differentiator in SOTA |
| **QK-Norm** | L2-normalize Q/K before dot product + learned per-head temperature | Stabilizes training |
| **QK-Gain** | Per-head learned scalar on queries (gain=4.0 optimal) | Minor gain |
| **LN Scale** | LayerNorm scaling in early layers: `1/√(layer_idx+1)` — addresses attention logit explosion | Used in top stacks |
| **FlashAttention v3** | Hardware-efficient attention implementation | Universal |
| **Logit softcap** | Softcap=30.0 on output logits | Stabilization |

---

## 3. Embedding & Input Innovations

| Technique | Description | Impact |
|---|---|---|
| **SmearGate** | ~512-param sigmoid gate blending current token embedding with previous token — injects bigram context. **Requires OrthoInit** (hurts without it) | ~-0.003 BPB |
| **BigramHash** | XOR-hash token pairs → learned embedding table (2048–20480 buckets, dim=128, projected to 512). Near-zero param cost | Standard in top runs |
| **TrigramHash** | Extension to 3 consecutive tokens. Dedicated table matters (can't share with bigram). Largest single gain after recurrence | -0.008 BPB |
| **EngramLite** | Multi-head n-gram hashing with context-aware gating (inspired by DeepSeek Engram) | Used in 1.1086 submission |
| **ValueEmbedding** | Shared value embedding (dim=128) injected at specific layers (e.g., 9-10) with per-layer learned scales | Used in SOTA |
| **FP16 tied embedding** | Keep embedding in FP16 even when rest is int6. Single highest-value precision decision (~1MB cost) | Universal in top runs |

---

## 4. Positional Encodings

| Technique | Description | Impact |
|---|---|---|
| **Partial RoPE** | Apply rotary embeddings to only 16/64 head dimensions — reduces dimension explosion | Standard in top runs |
| **NTK-aware RoPE** | Dynamic scaling for evaluating at longer sequences than training length | Enables seq2048→4096 eval |
| **YaRN** | Yet Another RoPE eNhancement — used in the ternary/FP8 submission for 2048 max position | Alternative to NTK |

**What failed:** Full RoPE causes quality degradation beyond training length; SelfExtend/4096 context was +0.48 BPB worse.

---

## 5. Activation Functions

| Technique | Description | Impact |
|---|---|---|
| **LeakyReLU(0.5)²** | `(leaky_relu(x, negative_slope=0.5))²` — produces 84-98% natural sparsity | Dominant in SOTA stack |
| **ReLU²** | Squared ReLU — high sparsity but no negative gradient flow | Baseline choice |
| **SwiGLU** | Gated linear unit variant replacing standard MLP | Used in some submissions |
| **Star-ReLU** | AI-discovered (GEPA) activation with per-layer scaling | Explored |

---

## 6. Quantization & Compression

### Quantization Methods

| Technique | Description | Impact |
|---|---|---|
| **Int6 STE QAT** | 64 levels [-32,31], straight-through estimator for gradient flow. Applied late (70–85% through training, or when LR scale < 0.15) | Core technique |
| **GPTQ-lite** | Per-row optimal clip percentile search (5 candidates: 0.999–1.0), select min MSE. Deterministic, zero training cost | -0.0006 BPB |
| **Full GPTQ** | Per-block Hessian-aware quantization. Must run within 600s training budget | Stronger than GPTQ-lite |
| **Self-generated calibration data** | Use model's own non-val outputs for GPTQ calibration | Legal, used in SOTA |
| **Mixed precision** | Int5 for MLP + int6 for attention, or int6 for MLP/attention + int8 for embeddings | Architecture-dependent |
| **Ternary / BitNet 1.58-bit** | Absmean quantization + STE from step 0. 73.7M params at 10L/768d. Achieved 1.1570 BPB | Different optimization regime |
| **FP8 QAT** | Used in the 74M ternary submission | Niche |
| **OptRot** | Rotation matrix redistributing weight outliers pre-quantization, fused into adjacent layers (zero artifact cost) | -0.001–0.003 BPB |
| **GuidedQuant** | Gradient-aware PTQ integrating end-loss gradients into layer-wise quantization | -0.002–0.005 BPB |
| **YAQA Adaptive Rounding** | Kronecker-factored Hessian guidance, ~30% less quant error than GPTQ | -0.001–0.003 BPB |

**What failed:**
- Int4 quantization — 0.065 BPB cost vs int6
- Mixed int4 attention / int8 MLP — +0.047 BPB
- Product Quantization — +292% BPB

### Compression

| Technique | Description | Impact |
|---|---|---|
| **zstd level 22** | Standard final compression. ~580KB better than LZMA-9 on int6 weights | Universal |
| **Byte-shuffle pre-filter** | Groups MSB/LSB for better compression ratio | Used |
| **Brotli-11** | Alternative: ~580KB better than LZMA-9 | Explored |
| **ANS/Huffman entropy coding** | Better than general-purpose zstd on quantized weights | Advanced |
| **Frequency-ordered tokenization** | Variable-length integer pre-compression encoding. Saves 200–500KB of artifact budget | Enables larger models |

---

## 7. Optimizers

| Technique | Description | Impact |
|---|---|---|
| **Muon** | SGD + Nesterov momentum post-processed by Newton-Schulz orthogonalization. ~35% faster convergence than AdamW. Settings: momentum 0.99 (warmup 0.92→0.99 over 1500 steps), lr=0.02, WD=0.04 | **Consensus best optimizer** |
| **Parallel Muon** | Batched Newton steps distributed across GPUs | Speed improvement |
| **NorMuon** | Per-neuron adaptive LR from accumulated second-order stats | Marginal gain |
| **Turbo-Muon** | AOL preconditioning + Polar Express coefficients | Marginal gain |
| **NeoMuon** | Used in ternary submission | Variant |
| **MUD** | Triangular whitening replacing Newton-Schulz | Slower in practice |
| **Mousse** | Curvature-aware Muon | 12% effectiveness |
| **Mixed optimizer strategy** | Muon for matrix params, AdamW for embeddings (lr=0.035) and scalars (lr=0.025) | Universal in top runs |

---

## 8. Learning Rate & Training Schedules

| Technique | Description | Impact |
|---|---|---|
| **Warmdown** | Linear cooldown from peak LR over 3000–3500 iterations (wallclock-based) | Standard |
| **1-sqrt cooldown** | `1-sqrt((t-T0)/(T+1-T0))` outperforms linear/cosine shapes | Better than linear |
| **Gradient clipping** | Norm ≤ 0.3 (tighter than typical 1.0) | Stability |
| **Batch size evolution** | 262K → 786K tokens/step — more total tokens matters more than gradient steps at frontier | Important at scale |
| **Momentum warmup** | 0.92 → 0.99 over 1500 steps | Standard with Muon |
| **Sequence length curriculum** | Start with short attention (128–384 tokens), grow to full 2048. Faster early steps = more total steps | Training efficiency |
| **Late QAT activation** | STE int6 fake-quantization enabled when LR scale drops below 0.10–0.15 | Consensus approach |

---

## 9. Weight Averaging

| Technique | Description | Impact |
|---|---|---|
| **EMA** | Decay=0.997 every step, applied before quantization. Best when paired with XSA (hurts without it) | -0.003 BPB with XSA |
| **SWA** | Stochastic weight averaging over 16+ checkpoints at intervals ("Tight SWA: every 50 steps when scale<0.2"). Requires WD≥0.04 and FP32 accumulation (bf16 catastrophic) | Complementary to EMA |
| **EMA failure mode** | Short training runs (~4K steps) cause shadow weights to accumulate poorly-converged early states → 1.42 BPB catastrophe | Watch out with recurrence |

---

## 10. Evaluation-Time Tricks

| Technique | Description | Impact |
|---|---|---|
| **Sliding window eval** | Overlapping windows (stride=64, window=2048) — each scored token gets 1984+ context tokens | **-0.034 BPB** vs non-overlapping |
| **N-gram eval cache** | Build hash table from already-scored tokens on-the-fly. Multi-order backoff (2–10 grams). Entropy-adaptive alpha: `0.05 + 0.55 * sigmoid(2*(H-4.0))`. Zero artifact cost | **Largest single lever: 0.4–0.9 BPB range** |
| **kNN-LM** | 512-dim hidden state ring buffer, k=32 nearest neighbors with RBF kernel, additive to n-gram cache | -0.007 BPB on top of n-gram |
| **Complementary training** | Down-weight tokens predictable by bigram/n-gram statistics during training: `w = 1 - alpha * p_ngram`. Model specializes on what n-grams can't predict | Pairs with eval n-gram |
| **Long-context eval** | Evaluate at 2048–4096 tokens after training at 1024 (with NTK-RoPE) | Free BPB improvement |
| **Logit chunking** | Process validation in 65K-token segments | Memory management |

**What was invalidated (rule violations):**
- Val-calibrated GPTQ (uses val data at eval time)
- Two-pass scoring (score → TTT → rescore)
- N-gram caches that don't normalize over full vocabulary
- Pre-eval TTT with token reordering

---

## 11. Test-Time Training (TTT)

| Technique | Description | Impact |
|---|---|---|
| **LoRA TTT** | Rank-8 LoRA adapters trained per-document during eval on already-scored tokens. "Score-first" single-pass. AdamW+cosine works; SGD hurts Full GPTQ models (+0.030 BPB) | 1.1190–1.1928 BPB |
| **SLOT** | Single learnable delta vector (512 dims) at last hidden layer, optimized per-batch. Lighter than LoRA, avoids GPTQ weight corruption | -0.002–0.006 BPB |
| **Per-layer LR groups** | 3× LR multiplier for layers with high quantization error | TTT refinement |
| **Document masking** | BOS-based boundary detection preventing cross-document leakage during TTT | Required for legal TTT |

**What failed:**
- Multi-epoch TTT → memorization gradient (0.95→0.78 BPB over 5 epochs — overfits)
- TTT + XSA actively hurts (+0.016 BPB)
- TTT with depth recurrence — updates affect both loop iterations simultaneously
- MC Dropout ensembling (K=16, dropout=0.30: +0.005 BPB)
- Full-model TTT on frontier models (neutral or negative)

---

## 12. Tokenization Strategies

| Technique | Description | Impact |
|---|---|---|
| **SP1024** | SentencePiece 1024-vocabulary baseline | Default |
| **SP4096** | Better compression ratio (0.306 tokens/byte) | Common upgrade |
| **8192 BPE** | Experimental larger vocab — tradeoff: more embedding params = fewer layers | Used in ternary submission |
| **TokenMonster (Scylla)** | Ungreedy multi-branch search (6 parallel branches). 37.5% fewer tokens vs BPE at same vocab. Custom 998-token vocab achieved 1.0806 BPB | Best tokenizer approach |
| **FineWeb-aligned training** | Train tokenizer on FineWeb specifically, not generic English | Better than generic |

---

## 13. Initialization & Regularization

| Technique | Description | Impact |
|---|---|---|
| **OrthoInit** | Orthogonal weight initialization. **Co-required with SmearGate** | Critical |
| **Overtone spectral init** | SVD-based power-law spectrum shaping | Explored |
| **muP-scaled output projections** | Maximal update parameterization for output layers | Used in top runs |
| **Decoupled weight decay** | WD=0.04 standard. Makes weights smaller → better for quantization and zstd compression | Universal |
| **LN Scale regularization** | `1/√(layer+1)` scaling factors in layer norms | Stability |
| **Focal loss** | `(1-p)^γ * (-log p)` — down-weight easy tokens, focus on hard | Explored, not dominant |
| **Token frequency weighted loss** | Differential weighting based on token occurrence frequency | Used in recurrent submission |

---

## 14. Systems / Kernel Optimizations

| Technique | Description | Impact |
|---|---|---|
| **Liger-Kernel** | Fused RMSNorm (6×), fused linear+CE (3×), fused residual+norm. 20–43% throughput gains | More training steps |
| **Backward MLP fusion (CUTLASS EVT)** | Fuses `(grad @ W_down) * act_grad` into GEMM epilogue. Hopper-only. ~3.7% step time reduction | +500 extra steps |
| **torch.compile + DDP** | Standard distributed training with compilation | Universal |
| **DistributedTokenLoader** | Custom data distribution across ranks | Infrastructure |

**What failed:** 2:4 structured sparsity — despite ReLU² producing 84-98% sparsity, enforcing NVIDIA 2:4 pattern gave large negative results.

---

## 15. Exotic / Novel Approaches

| Technique | Description | Result |
|---|---|---|
| **Context Tree Weighting (CTW)** | Bayesian-optimal weighting over all context tree models | Failed: +0.005 BPB, 46 min eval |
| **Fixed-Share Hedge** | Non-stationary expert tracking with switching between experts | -0.003–0.008 BPB expected |
| **PPMII-Style Escape Estimation** | Principled escape probabilities + information inheritance + full exclusions | -0.01–0.03 BPB expected |
| **Match Model** | Longest-match prediction from previously-scored data | -0.005–0.01 BPB expected |
| **Sparse/Skip-Gram Context Models** | Non-contiguous positions (tokens at -1, -3, -5) capturing patterns with variable gaps | -0.005–0.015 BPB expected |
| **Information-theoretic mixing** | Log-odds space mixing instead of linear interpolation | -0.002–0.005 BPB expected |
| **Dirichlet backoff** | Probabilistic n-gram backoff method | Used in some n-gram stacks |
| **GEPA architecture** | AI-discovered architecture using Star-ReLU | Explored |

---

## 16. The Winning Formula (SOTA Stack)

The top submissions combine **8–12 orthogonal techniques simultaneously** via multiplicative stacking:

**Official SOTA (#1019, 1.1147 BPB):**
> 11L AR Self-Gen GPTQ + XSA-all + BigramHash(3072) + Parallel Muon

**Best unofficial (#1089, 1.1086 BPB):**
> Turbo-Muon + EngramLite hash embeddings + mixed-precision GPTQ

**Typical top-tier stack:**
1. 11 layers, 512d, GQA (8Q/4KV)
2. LeakyReLU(0.5)² activation
3. XSA on last 4+ layers
4. SmearGate + BigramHash + OrthoInit
5. Partial RoPE (16/64 dims)
6. Muon optimizer (momentum 0.99, warmup from 0.92)
7. EMA (0.997) + Tight SWA
8. Late QAT int6 STE + GPTQ-lite post-training
9. FP16 tied embeddings, int6 everything else
10. zstd-22 compression
11. Sliding window eval (stride=64)
12. 1-sqrt warmdown schedule

---

## Key Insight

The competition demonstrates that **gains are multiplicative, not additive**. Better tokenizer × better quantization × better evaluation × better optimization compounds exponentially. No single technique dominates — the edge comes from combining the most orthogonal techniques without introducing conflicts (e.g., EMA needs XSA; SmearGate needs OrthoInit; TTT conflicts with depth recurrence and XSA).

---

## Sources

- [GitHub - openai/parameter-golf](https://github.com/openai/parameter-golf)
- [OpenAI Model Craft: Parameter Golf](https://openai.com/index/parameter-golf/)
- [DeepWiki - openai/parameter-golf](https://deepwiki.com/openai/parameter-golf)
- [Parameter Golf Field Guide](https://sameersegal.github.io/learn-parameter-golf/)
- [Live AI Commentary - Issue #140](https://github.com/openai/parameter-golf/issues/140)
- [PR #181 - 11L LeakyReLU² XSA4 PartialRoPE LNScale EMA ParallelMuon TTT](https://github.com/openai/parameter-golf/pull/181)
- [11L EMA GPTQ-lite Record README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md)
- [Substack: I Entered Parameter Golf With Zero ML Knowledge](https://namspdr.substack.com/p/i-entered-openais-parameter-golf)
- [PR #551 - Efficient 21K Parameter Model](https://github.com/openai/parameter-golf/pull/551)
- [PR #423 - Efficient Recurrent GPT](https://github.com/openai/parameter-golf/pull/423)
- [PR #342 - SmearGate + BigramHash](https://github.com/openai/parameter-golf/pull/342)
- [Runpod Blog - Parameter Golf](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod)
- [TokenMonster](https://github.com/alasdairforsythe/tokenmonster)
