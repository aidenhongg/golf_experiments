import torch.nn as nn
from golfcomp.config import ModelConfig
from golfcomp.models.base import BaseModel
from golfcomp.models.components.attention import GQAAttention
from golfcomp.models.components.embeddings import (
    TokenEmbedding, SmearGate, BigramHash, EngramLite, FactorizedEmbedding,
)
from golfcomp.models.components.activations import LeakyReLUSq, SwiGLU
import torch.nn.functional as F
from golfcomp.models.components.position import PartialRoPE
from golfcomp.models.components.residuals import ParallelResidual, SkipGate
from golfcomp.models.components.recurrence import (
    DepthRecurrence, BasisSharing, RelaxedRecursive,
)


class TransformerBlock(nn.Module):
    """LN -> Attn -> residual -> LN -> MLP -> residual. Parallel or serial."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.attn = GQAAttention(
            dim=config.model_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qk_gain=config.qk_gain,
            use_xsa=config.use_xsa,
            logit_softcap=config.logit_softcap,
            layer_idx=layer_idx,
            total_layers=config.num_layers,
        )
        if config.activation == "swiglu":
            self.mlp = SwiGLU(config.model_dim, hidden_dim=config.model_dim * config.mlp_mult)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.model_dim, config.model_dim * config.mlp_mult),
                LeakyReLUSq(),
                nn.Linear(config.model_dim * config.mlp_mult, config.model_dim),
            )
        self.use_parallel = (
            config.use_parallel_residuals and layer_idx >= config.parallel_start_layer
        )
        if self.use_parallel:
            self.parallel_res = ParallelResidual(config.model_dim)
            self.skip1 = self.skip2 = None
        else:
            self.skip1 = SkipGate(config.model_dim) if config.use_skip_gates else None
            self.skip2 = SkipGate(config.model_dim) if config.use_skip_gates else None

    def forward(self, x, rope_fn=None):
        if self.use_parallel:
            return self.parallel_res(x, self.attn(self.norm1(x), rope_fn=rope_fn),
                                     self.mlp(self.norm2(x)))
        h = self.attn(self.norm1(x), rope_fn=rope_fn)
        x = self.skip1(x, h) if self.skip1 else x + h
        h = self.mlp(self.norm2(x))
        x = self.skip2(x, h) if self.skip2 else x + h
        return x


class Transformer(BaseModel):
    """Full transformer. Configurable for baseline + all ablations via ModelConfig."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Embedding
        if config.embedding_type == "standard":
            self.embed = TokenEmbedding(config.vocab_size, config.model_dim)
        else:
            self.embed = FactorizedEmbedding(
                config.vocab_size, config.model_dim,
                low_rank=config.factorized_embed_dim,
                embed_type=config.embedding_type,
                num_tables=config.num_hash_tables,
            )
        self.smear_gate = SmearGate(config.model_dim) if config.use_smear_gate else None

        # N-gram features
        if config.use_engramlite:
            self.ngram_embed = EngramLite(
                config.vocab_size, model_dim=config.model_dim,
                orders=config.engramlite_orders,
                buckets_per_head=config.engramlite_buckets_per_head,
            )
        elif config.bigram_hash_buckets > 0:
            self.ngram_embed = BigramHash(
                config.bigram_hash_buckets, config.bigram_hash_dim, config.model_dim
            )
        else:
            self.ngram_embed = None

        self.rope = PartialRoPE(config.model_dim // config.num_heads, config.rope_partial_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])

        # Recurrence variants (mutually exclusive)
        self.depth_recurrence = None
        self.basis_sharing = None
        self.relaxed_recursive = None
        if config.recurrence_type == "depth":
            self.depth_recurrence = DepthRecurrence(
                config.recurrence_layers, config.model_dim
            )
        elif config.recurrence_type == "basis_sharing":
            self.basis_sharing = BasisSharing(
                config.num_layers, config.model_dim, config.basis_rank
            )
        elif config.recurrence_type == "relaxed_recursive":
            shared = nn.ModuleList(list(self.layers[:config.num_shared_blocks]))
            self.relaxed_recursive = RelaxedRecursive(
                shared, config.num_loops, config.lora_rank
            )
            # Keep only non-shared layers to avoid double-registering shared parameters
            self.non_shared_layers = nn.ModuleList(list(self.layers[config.num_shared_blocks:]))
            del self.layers

        # Output head
        self.norm = nn.LayerNorm(config.model_dim)
        if config.tie_embeddings and config.embedding_type == "standard":
            self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)
            self.output.weight = self.embed.weight  # weight tying
        elif config.tie_embeddings:
            self.output = None  # use embed.compute_logits
        else:
            self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)

        self.init_weights()

    def set_recurrence_active(self, active: bool):
        if self.depth_recurrence:
            self.depth_recurrence.active = active

    def forward(self, input_ids):
        x = self.embed(input_ids)
        if self.smear_gate:
            x = self.smear_gate(x)
        if self.ngram_embed is not None:
            x = x + self.ngram_embed(input_ids)

        rope_fn = self.rope

        if self.relaxed_recursive is not None:
            x = self.relaxed_recursive(x, rope_fn=rope_fn)
            # Run remaining non-shared layers
            for layer in self.non_shared_layers:
                x = layer(x, rope_fn=rope_fn)
        elif self.basis_sharing is not None:
            self.basis_sharing.invalidate_cache()
            self.basis_sharing._rebuild_cache()
            for i, layer in enumerate(self.layers):
                h = layer(x, rope_fn=rope_fn)
                w = self.basis_sharing.get_weight(i)
                x = h + F.linear(x, w)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, rope_fn=rope_fn)
                if self.depth_recurrence and self.depth_recurrence.should_repeat(i):
                    h = layer(x, rope_fn=rope_fn)
                    x = x + self.depth_recurrence.apply_film(h, i, loop=1)

        x = self.norm(x)
        if self.output is not None:
            return self.output(x)
        return self.embed.compute_logits(x)
