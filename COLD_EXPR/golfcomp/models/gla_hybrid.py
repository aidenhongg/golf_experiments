import torch.nn as nn
from golfcomp.models.hybrid_base import HybridBaseModel
from golfcomp.models.components.activations import LeakyReLUSq


class GLABlock(nn.Module):
    """Wrapper: FLA GatedLinearAttention + LeakyReLU^2 MLP."""

    def __init__(self, dim, expand_ratio=1):
        super().__init__()
        from fla.layers import GatedLinearAttention
        self.norm1 = nn.LayerNorm(dim)
        # fla >= 0.3: API uses hidden_size + expand_k/expand_v (not d_model/expand_ratio)
        self.gla = GatedLinearAttention(hidden_size=dim, expand_k=expand_ratio, expand_v=expand_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 3), LeakyReLUSq(), nn.Linear(dim * 3, dim),
        )

    def forward(self, x):
        # fla >= 0.3: GatedLinearAttention.forward() returns (output, state)
        gla_out = self.gla(self.norm1(x))
        x = x + (gla_out[0] if isinstance(gla_out, tuple) else gla_out)
        x = x + self.mlp(self.norm2(x))
        return x


class GLAHybridModel(HybridBaseModel):
    """C1: 8 GLA (FLA library) + 3 attention. XSA on attn only. EMA 0.9965."""

    def __init__(self, config):
        super().__init__(config, num_backbone_layers=config.gla_num_layers)
        self.backbone_layers = nn.ModuleList([
            GLABlock(config.model_dim, config.gla_expand_ratio)
            for _ in range(config.gla_num_layers)
        ])
        self.init_weights()
