import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# PyTorch SDPA works on all GPU architectures including SM_120 (RTX 5090)
HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


class GQAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_kv_heads=4, qk_gain=5.0,
                 use_xsa=True, logit_softcap=30.0, layer_idx=0, total_layers=11):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_dim = dim * num_kv_heads // num_heads
        self.use_xsa = use_xsa
        self.logit_softcap = logit_softcap
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)
        self.groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.qk_gain = nn.Parameter(torch.full((num_heads,), qk_gain))

        # CRITICAL FIX (eng review #12): init to zeros so first sequence gets XSA shape
        self.register_buffer("_prev_k", torch.zeros(1, 0, num_kv_heads, self.head_dim), persistent=False)
        self.register_buffer("_prev_v", torch.zeros(1, 0, num_kv_heads, self.head_dim), persistent=False)

    def forward(self, x, rope_fn=None, use_flash=True):
        B, S, D = x.shape
        hd = self.head_dim

        q = self.q_proj(x).view(B, S, self.num_heads, hd)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, hd)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, hd)

        if rope_fn is not None:
            q, k = rope_fn(q, k)

        # QK-Norm + QK-Gain (scaled by ln_scale)
        q = F.normalize(q, dim=-1) * (self.qk_gain.view(1, 1, -1, 1) * self.ln_scale)
        k = F.normalize(k, dim=-1)

        # XSA: prepend previous sequence KV
        if self.use_xsa and self._prev_k.shape[1] > 0:
            prev_k = self._prev_k.expand(B, -1, -1, -1)
            prev_v = self._prev_v.expand(B, -1, -1, -1)
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        if self.use_xsa:
            self._prev_k = k[:, -S:].detach()
            self._prev_v = v[:, -S:].detach()

        if use_flash and HAS_FLASH_ATTN and self._flash_attn_supported():
            attn_out = self._flash_attention(q, k, v)
        elif use_flash and HAS_SDPA:
            attn_out = self._sdpa_attention(q, k, v)
        else:
            attn_out = self._manual_attention(q, k, v)

        return self.o_proj(attn_out.reshape(B, S, D))

    def _flash_attn_supported(self):
        """flash-attn 2.x cannot compile for SM_120 (RTX 5090). Check at runtime."""
        if not torch.cuda.is_available():
            return False
        cc = torch.cuda.get_device_capability()
        # FA2 supports SM 80/86/89/90 but segfaults on SM 100/120
        return cc[0] < 10

    def _flash_attention(self, q, k, v):
        # GQA: repeat KV heads to match Q heads
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
        return flash_attn_func(q, k, v, causal=True, softcap=self.logit_softcap)

    def _sdpa_attention(self, q, k, v):
        """PyTorch native SDPA — works on all GPU architectures including Blackwell.

        Note: SDPA does not support logit softcap natively. With QK-normalized
        attention (qk_gain=5.0, ln_scale<=1.0), max logit magnitude is ~5.0
        which is well below softcap=30.0, so omitting softcap is safe here.
        If softcap matters (very low values), use _manual_attention instead.
        """
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
        # SDPA expects [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # scale=1.0 because q/k are already pre-normalized with qk_gain
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        return out.transpose(1, 2)  # [B, S, H, D]

    def _manual_attention(self, q, k, v):
        # q: [B, Sq, Hq, D], k/v: [B, Sk, Hkv, D]
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
        # transpose to [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # No 1/sqrt(head_dim) scale: q/k are already pre-normalized with qk_gain
        scores = torch.matmul(q, k.transpose(-2, -1))

        # softcap
        if self.logit_softcap > 0:
            scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)

        # causal mask
        Sq, Sk = q.shape[2], k.shape[2]
        mask = torch.triu(torch.full((Sq, Sk), -1e9, device=q.device), diagonal=Sk - Sq + 1)
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2)  # [B, S, H, D]

    def reset_xsa(self):
        dev = self._prev_k.device
        self._prev_k = torch.zeros(1, 0, self.num_kv_heads, self.head_dim, device=dev)
        self._prev_v = torch.zeros(1, 0, self.num_kv_heads, self.head_dim, device=dev)
