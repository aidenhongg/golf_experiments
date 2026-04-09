import torch
import torch.nn as nn


class ParallelResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.merge = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, attn_out, mlp_out):
        return x + self.merge * attn_out + (1 - self.merge) * mlp_out


class SkipGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, x, residual):
        return x + self.gate * residual
