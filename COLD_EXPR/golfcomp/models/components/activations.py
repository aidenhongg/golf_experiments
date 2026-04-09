import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyReLUSq(nn.Module):
    @torch.compile
    def forward(self, x):
        return F.leaky_relu(x, 0.5).square()


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=1024):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
