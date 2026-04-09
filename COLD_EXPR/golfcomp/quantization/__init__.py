from .sdclip import SDClipQuantizer
from .gptq import GPTQQuantizer
from .mixed import MixedPrecisionQuantizer
from .compression import Compressor


def build_quantizer(config):
    """Factory: returns quantizer based on config.method."""
    m = config.method if hasattr(config, "method") else config.quant.method
    cfg = config if hasattr(config, "method") else config.quant

    if m == "sdclip":
        return SDClipQuantizer(
            body_bits=cfg.body_bits, embed_bits=cfg.embed_bits,
            body_k=cfg.int6_k, embed_k=cfg.int8_embed_k,
        )
    elif m == "gptq":
        return GPTQQuantizer(
            bits=cfg.body_bits, actorder=cfg.gptq_actorder,
        )
    elif m == "mixed":
        return MixedPrecisionQuantizer(
            mlp_bits=cfg.mlp_bits, attn_bits=cfg.attn_bits,
            embed_bits=cfg.embed_bits,
            body_k=cfg.int6_k, embed_k=cfg.int8_embed_k,
        )
    else:
        raise ValueError(f"Unknown quantization method: {m}")


__all__ = [
    "build_quantizer", "SDClipQuantizer", "GPTQQuantizer",
    "MixedPrecisionQuantizer", "Compressor",
]
