import torch
import torch.nn as nn
import torch.nn.functional as F
from golfcomp.models.hybrid_base import HybridBaseModel
from golfcomp.models.components.activations import LeakyReLUSq


class MLSTMBlock(nn.Module):
    """mLSTM with matrix memory + exponential gating (arxiv 2405.04517).
    THROUGHPUT_CAVEAT: Pure PyTorch, Triton kernels needed for speed comparison."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.i_gate = nn.Linear(dim, 1)
        self.f_gate = nn.Linear(dim, 1)
        self.o_gate = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 3), LeakyReLUSq(), nn.Linear(dim * 3, dim),
        )

    def forward(self, x):
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Exponential gates (log-space for stability)
        log_f = F.logsigmoid(self.f_gate(h))  # [B, S, 1]
        log_i = F.logsigmoid(self.i_gate(h))  # [B, S, 1]
        o = torch.sigmoid(self.o_gate(h))  # [B, S, D]

        # Cumulative forget gate in log-space
        log_f_cum = torch.cumsum(log_f, dim=1)  # [B, S, 1]

        # Attention-like parallel formulation:
        # score[t,s] = exp(log_i[s] + log_f_cum[t] - log_f_cum[s])
        log_scores = log_i.transpose(1, 2) + log_f_cum - log_f_cum.transpose(1, 2)  # [B, S, S]

        # Causal mask
        causal = torch.triu(torch.full((S, S), -1e9, device=x.device), diagonal=1)
        scores = torch.exp(log_scores + causal.unsqueeze(0))  # [B, S, S]

        # Normalize for stability
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

        # Gated linear attention output
        attn = scores @ v  # [B, S, D]
        x = x + o * attn
        x = x + self.mlp(self.norm2(x))
        return x


class XLSTMHybridModel(HybridBaseModel):
    """C3: 8 mLSTM + 3 attention. THROUGHPUT_CAVEAT: Pure PyTorch."""

    def __init__(self, config):
        super().__init__(config, num_backbone_layers=config.xlstm_num_layers)
        self.backbone_layers = nn.ModuleList([
            MLSTMBlock(config.model_dim) for _ in range(config.xlstm_num_layers)
        ])
        self.init_weights()
