import torch


class SDClipQuantizer:
    """SDClip: clip = k * std(row). Per-row symmetric quantization.
    k=12.85 for int6 body weights, k=20.0 for int8 embeddings."""

    def __init__(self, body_bits=6, embed_bits=8, body_k=12.85, embed_k=20.0):
        self.body_bits = body_bits
        self.embed_bits = embed_bits
        self.body_k = body_k
        self.embed_k = embed_k

    def quantize_tensor(self, tensor, bits, k):
        """Per-row: compute std, clip at k*std, quantize to int levels.
        Returns (quantized_int8_tensor, {"scale": per_row_scale, "bits": bits})"""
        levels = 2 ** bits
        half = levels // 2
        row_std = tensor.std(dim=-1, keepdim=True).clamp(min=1e-8)
        clip_val = k * row_std
        clipped = tensor.clamp(-clip_val, clip_val)
        scale = clip_val / half
        quantized = (clipped / scale).round().clamp(-half, half - 1).to(torch.int8)
        return quantized, {"scale": scale.squeeze(-1), "bits": bits}

    def quantize_model(self, model):
        """Quantize all parameters. Returns dict of {name: {quantized, scale, bits}}."""
        state = {}
        for name, param in model.named_parameters():
            data = param.data.float()
            if data.ndim < 2:
                state[name] = {"raw": data.half()}
                continue
            is_embed = "embed" in name or "table" in name
            bits = self.embed_bits if is_embed else self.body_bits
            k = self.embed_k if is_embed else self.body_k
            q, meta = self.quantize_tensor(data, bits, k)
            state[name] = {"quantized": q, **meta}
        return state

    def dequantize_tensor(self, quantized, scale, bits):
        """Reconstruct float tensor from quantized + scale."""
        return quantized.float() * scale.unsqueeze(-1)

    def dequantize_state(self, state):
        """Reconstruct full state dict from quantized state."""
        result = {}
        for name, entry in state.items():
            if "raw" in entry:
                result[name] = entry["raw"].float()
            else:
                result[name] = self.dequantize_tensor(
                    entry["quantized"], entry["scale"], entry["bits"]
                )
        return result
