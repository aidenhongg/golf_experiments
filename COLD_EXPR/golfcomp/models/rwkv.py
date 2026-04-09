import torch
import torch.nn as nn
import torch.nn.functional as F
from golfcomp.config import ModelConfig
from golfcomp.models.base import BaseModel
from golfcomp.models.components.embeddings import TokenEmbedding, SmearGate, BigramHash


class RWKV7Block(nn.Module):
    """RWKV-7 block: time-mixing (WKV) + channel-mixing.
    No attention -> no XSA. No EMA (no attention to anchor it)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Time-mixing
        self.time_decay = nn.Parameter(torch.randn(dim) * 0.01 - 5)
        self.time_faaaa = nn.Parameter(torch.randn(dim) * 0.01)
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        # Channel-mixing
        self.cm_r_proj = nn.Linear(dim, dim, bias=False)
        self.cm_k_proj = nn.Linear(dim, dim * 3, bias=False)
        self.cm_v_proj = nn.Linear(dim * 3, dim, bias=False)
        self.cm_mix = nn.Parameter(torch.ones(1, 1, dim) * 0.5)

    def forward(self, x):
        B, S, D = x.shape
        h = self.norm1(x)

        # Time mixing with linear interpolation
        prev = F.pad(h[:, :-1], (0, 0, 1, 0))
        r = self.r_proj(h * self.time_mix_r + prev * (1 - self.time_mix_r))
        k = self.k_proj(h * self.time_mix_k + prev * (1 - self.time_mix_k))
        v = self.v_proj(h * self.time_mix_v + prev * (1 - self.time_mix_v))

        # WKV computation
        wkv = self._parallel_wkv(r, k, v)
        x = x + self.o_proj(wkv)

        # Channel mixing
        h = self.norm2(x)
        prev = F.pad(h[:, :-1], (0, 0, 1, 0))
        mixed = h * self.cm_mix + prev * (1 - self.cm_mix)
        r_cm = torch.sigmoid(self.cm_r_proj(mixed))
        k_cm = self.cm_k_proj(mixed).square().relu()
        x = x + r_cm * self.cm_v_proj(k_cm)
        return x

    def _parallel_wkv(self, r, k, v):
        """WKV via sequential scan with exponential decay."""
        B, S, D = r.shape
        w = torch.exp(-torch.exp(self.time_decay))  # [D]
        r_sig = torch.sigmoid(r)

        out = torch.zeros_like(r)
        # State: running weighted sum of k*v outer products (reduced to D via dot)
        state = torch.zeros(B, D, device=r.device)
        for t in range(S):
            # time-first bonus on first token
            bonus = torch.exp(self.time_faaaa) * k[:, t] * v[:, t] if t == 0 else torch.zeros_like(state)
            state = state * w.unsqueeze(0) + k[:, t] * v[:, t]
            out[:, t] = r_sig[:, t] * (state + bonus)
        return out


class RWKVModel(BaseModel):
    """C4: Pure RWKV-7 backbone. 11 layers, 512d.
    No attention -> no XSA -> no EMA."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.embed = TokenEmbedding(config.vocab_size, config.model_dim)
        self.smear_gate = SmearGate(config.model_dim) if config.use_smear_gate else None
        self.ngram_embed = BigramHash(config.bigram_hash_buckets, config.bigram_hash_dim, config.model_dim)
        self.layers = nn.ModuleList([RWKV7Block(config.model_dim) for _ in range(config.rwkv_num_layers)])
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
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))
