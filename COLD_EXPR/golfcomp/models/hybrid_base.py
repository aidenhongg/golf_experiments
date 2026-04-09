import torch.nn as nn
from golfcomp.config import ModelConfig
from golfcomp.models.base import BaseModel
from golfcomp.models.transformer import TransformerBlock
from golfcomp.models.components.embeddings import TokenEmbedding, SmearGate, BigramHash
from golfcomp.models.components.position import PartialRoPE


class HybridBaseModel(BaseModel):
    """Base for C1-C5 hybrids. Shares: embed -> smear -> bigram -> [backbone] -> [attn top] -> norm -> output."""

    def __init__(self, config: ModelConfig, num_backbone_layers: int):
        super().__init__(config)
        self.embed = TokenEmbedding(config.vocab_size, config.model_dim)
        self.smear_gate = SmearGate(config.model_dim) if config.use_smear_gate else None
        self.ngram_embed = BigramHash(config.bigram_hash_buckets, config.bigram_hash_dim, config.model_dim)
        self.rope = PartialRoPE(config.model_dim // config.num_heads, config.rope_partial_dim)

        # Subclass fills self.backbone_layers
        self.backbone_layers = nn.ModuleList()

        # Top attention layers (remaining after backbone)
        num_attn = config.num_layers - num_backbone_layers
        self.attn_layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=num_backbone_layers + i)
            for i in range(num_attn)
        ])

        self.norm = nn.LayerNorm(config.model_dim)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        if self.smear_gate:
            x = self.smear_gate(x)
        x = x + self.ngram_embed(input_ids)
        for layer in self.backbone_layers:
            x = layer(x)
        for layer in self.attn_layers:
            x = layer(x, rope_fn=self.rope)
        return self.output(self.norm(x))
