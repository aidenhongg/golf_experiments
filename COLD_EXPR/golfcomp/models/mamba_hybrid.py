import torch
import torch.nn as nn
import torch.nn.functional as F
from golfcomp.models.hybrid_base import HybridBaseModel
from golfcomp.models.components.activations import LeakyReLUSq


class SimpleSSM(nn.Module):
    """Fallback SSM if mamba-ssm not available. Diagonal S4-style SSM."""

    def __init__(self, dim, d_state=32):
        super().__init__()
        self.d_state = d_state
        # Complex diagonal state matrix (log-space for stability)
        self.log_A = nn.Parameter(torch.randn(dim, d_state) * 0.01 - 4.0)
        self.B_proj = nn.Linear(dim, d_state)
        self.C_proj = nn.Linear(dim, d_state)
        self.D = nn.Parameter(torch.ones(dim))
        self.dt_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, S, D = x.shape
        dt = F.softplus(self.dt_proj(x))  # [B, S, D]
        A = -torch.exp(self.log_A)  # [D, N]
        b = self.B_proj(x)  # [B, S, N]
        c = self.C_proj(x)  # [B, S, N]

        # Discretize: A_bar = exp(A * dt), B_bar = dt * B
        # Sequential scan (fallback -- no parallel scan kernel)
        out = torch.zeros_like(x)
        h = torch.zeros(B, D, self.d_state, device=x.device)
        for t in range(S):
            dt_t = dt[:, t].unsqueeze(-1)  # [B, D, 1]
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)  # [B, D, N]
            B_bar = dt_t * b[:, t].unsqueeze(1)  # [B, D, N] (broadcast D)
            h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)  # [B, D, N]
            out[:, t] = (h * c[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]
        return out


class MambaBlock(nn.Module):
    """Mamba-3 block. THROUGHPUT_CAVEAT: Pure PyTorch, no kernel optimization.
    Uses mamba-ssm Mamba2 if available, else SimpleSSM fallback."""

    def __init__(self, dim, d_state=32):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
            self.ssm = Mamba2(d_model=dim, d_state=d_state)
        except ImportError:
            self.ssm = SimpleSSM(dim, d_state)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 3), LeakyReLUSq(), nn.Linear(dim * 3, dim),
        )

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MambaHybridModel(HybridBaseModel):
    """C2: 8 Mamba-3 + 3 attention. THROUGHPUT_CAVEAT: Pure PyTorch SSM fallback."""

    def __init__(self, config):
        super().__init__(config, num_backbone_layers=config.mamba_num_layers)
        self.backbone_layers = nn.ModuleList([
            MambaBlock(config.model_dim, config.mamba_d_state)
            for _ in range(config.mamba_num_layers)
        ])
        self.init_weights()
