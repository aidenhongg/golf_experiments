def build_model(config):
    arch = config.model.arch if hasattr(config, "model") else config.arch
    cfg = config.model if hasattr(config, "model") else config
    if arch == "transformer":
        from golfcomp.models.transformer import Transformer
        return Transformer(cfg)
    elif arch == "gla_hybrid":
        from golfcomp.models.gla_hybrid import GLAHybridModel
        return GLAHybridModel(cfg)
    elif arch == "mamba_hybrid":
        from golfcomp.models.mamba_hybrid import MambaHybridModel
        return MambaHybridModel(cfg)
    elif arch == "xlstm_hybrid":
        from golfcomp.models.xlstm_hybrid import XLSTMHybridModel
        return XLSTMHybridModel(cfg)
    elif arch == "rwkv":
        from golfcomp.models.rwkv import RWKVModel
        return RWKVModel(cfg)
    elif arch == "mixed_hybrid":
        from golfcomp.models.mixed_hybrid import MixedHybridModel
        return MixedHybridModel(cfg)
    else:
        raise ValueError(f"Unknown arch: {arch}")
