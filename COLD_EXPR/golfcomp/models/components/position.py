import torch
import torch.nn as nn


class PartialRoPE(nn.Module):
    def __init__(self, head_dim, partial_dim=16, max_seq_len=8192):
        super().__init__()
        self.partial_dim = partial_dim
        assert partial_dim % 2 == 0
        freqs = 1.0 / (10000.0 ** (torch.arange(0, partial_dim, 2).float() / partial_dim))
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    @torch.compile
    def forward(self, q, k):
        # q, k: [B, S, H, D]
        S = q.shape[1]
        pd = self.partial_dim
        cos = self.cos_cache[:S]  # [S, pd//2]
        sin = self.sin_cache[:S]  # [S, pd//2]

        q_rope, q_pass = q[..., :pd], q[..., pd:]
        k_rope, k_pass = k[..., :pd], k[..., pd:]

        q_rope = self._apply_rope(q_rope, cos, sin)
        k_rope = self._apply_rope(k_rope, cos, sin)

        return torch.cat([q_rope, q_pass], dim=-1), torch.cat([k_rope, k_pass], dim=-1)

    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: [B, S, H, pd], cos/sin: [S, pd//2]
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = cos.view(1, -1, 1, half)
        sin = sin.view(1, -1, 1, half)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
