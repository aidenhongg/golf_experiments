import torch
from .sdclip import SDClipQuantizer

# Optimal k values per bit width (std-deviation clip multipliers)
_K_TABLE = {4: 8.0, 5: 10.5, 6: 12.85, 7: 16.0, 8: 20.0}


class MixedPrecisionQuantizer:
    """Per-component bit allocation: different bits for MLP, attention, embeddings.
    Wraps SDClipQuantizer with per-component bit overrides.
    A) All int6 (baseline)  B) Int5 MLP + int6 attn  C) Int6 MLP + int8 attn"""

    def __init__(self, mlp_bits=6, attn_bits=6, embed_bits=8,
                 body_k=12.85, embed_k=20.0):
        self.mlp_bits = mlp_bits
        self.attn_bits = attn_bits
        self.embed_bits = embed_bits
        self.body_k = body_k
        self.embed_k = embed_k
        self._base = SDClipQuantizer(body_bits=6, embed_bits=embed_bits,
                                     body_k=body_k, embed_k=embed_k)

    @staticmethod
    def _k_for_bits(bits):
        return _K_TABLE.get(bits, 12.85)

    def _classify(self, name):
        """Returns (bits, k) for a parameter name."""
        if "embed" in name or "table" in name:
            return self.embed_bits, self.embed_k
        if any(s in name for s in ("mlp", "gate_proj", "up_proj", "down_proj")):
            return self.mlp_bits, self._k_for_bits(self.mlp_bits)
        if any(s in name for s in ("attn", "q_proj", "k_proj", "v_proj", "o_proj")):
            return self.attn_bits, self._k_for_bits(self.attn_bits)
        return 6, self.body_k

    def quantize_model(self, model):
        """Quantize with per-component bit allocation."""
        state = {}
        for name, param in model.named_parameters():
            if param.ndim < 2:
                state[name] = {"raw": param.data.half()}
                continue
            bits, k = self._classify(name)
            q, meta = self._base.quantize_tensor(param.data.float(), bits, k)
            state[name] = {"quantized": q, **meta}
        return state

    def dequantize_state(self, state):
        return self._base.dequantize_state(state)
