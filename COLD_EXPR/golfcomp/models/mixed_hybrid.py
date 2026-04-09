import torch.nn as nn
from golfcomp.config import ModelConfig
from golfcomp.models.base import BaseModel
from golfcomp.models.transformer import TransformerBlock
from golfcomp.models.components.embeddings import TokenEmbedding, SmearGate, BigramHash
from golfcomp.models.components.position import PartialRoPE
from golfcomp.models.mamba_hybrid import MambaBlock
from golfcomp.models.gla_hybrid import GLABlock


class MixedHybridModel(BaseModel):
    """C5: 4 Mamba-3 (bottom) + 4 GLA (middle) + 3 attention (top).
    THROUGHPUT_CAVEAT: Mamba layers are pure PyTorch.
    Mixed optimizer: AdamW for SSM, Muon for projections, AdamW for GLA gates."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.embed = TokenEmbedding(config.vocab_size, config.model_dim)
        self.smear_gate = SmearGate(config.model_dim) if config.use_smear_gate else None
        self.ngram_embed = BigramHash(config.bigram_hash_buckets, config.bigram_hash_dim, config.model_dim)
        self.rope = PartialRoPE(config.model_dim // config.num_heads, config.rope_partial_dim)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(config.model_dim, config.mamba_d_state) for _ in range(4)
        ])
        self.gla_layers = nn.ModuleList([
            GLABlock(config.model_dim, config.gla_expand_ratio) for _ in range(4)
        ])
        self.attn_layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=8 + i) for i in range(3)
        ])

        self.norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embed.weight
        self.init_weights()

    def forward(self, input_ids):
        x = self.embed(input_ids)
        if self.smear_gate:
            x = self.smear_gate(x)
        x = x + self.ngram_embed(input_ids)
        for layer in self.mamba_layers:
            x = layer(x)
        for layer in self.gla_layers:
            x = layer(x)
        for layer in self.attn_layers:
            x = layer(x, rope_fn=self.rope)
        return self.output(self.norm(x))
