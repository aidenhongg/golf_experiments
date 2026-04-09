"""Search space definitions for all experiment campaigns."""

# Architecture experiments (C1-C5) -- Bayesian TPE
C1_GLA_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
    "training.gate_lr":   {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
    "training.ema_decay": {"type": "float", "low": 0.990, "high": 0.999},
    "model.gla_expand_ratio": {"type": "categorical", "choices": [1, 2]},
}

C2_MAMBA_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
    "training.gate_lr":   {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
    "model.mamba_d_state": {"type": "categorical", "choices": [16, 32, 64]},
}

C3_XLSTM_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.01, "high": 0.04, "log": True},
    "training.gate_lr":   {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
}

C4_RWKV_SPACE = {
    "training.matrix_lr": {"type": "float", "low": 0.005, "high": 0.04, "log": True},
    "training.weight_decay": {"type": "float", "low": 0.04, "high": 0.15},
}

C5_MIXED_SPACE = {**C1_GLA_SPACE, **C2_MAMBA_SPACE}

# HP sweeps (C11-C14) -- Grid
C11_EMA_GRID = {"training.ema_decay": [0.990, 0.995, 0.9965, 0.998, 0.999]}
C12_WD_GRID = {"training.weight_decay": [0.04, 0.06, 0.095, 0.12, 0.15]}
C13_RECURRENCE_GRID = {"model.recurrence_layers": [[3, 4, 5], [4, 5], [2, 3, 4, 5], [5, 6, 7], []]}
C14_WARMDOWN_GRID = {"training.warmdown_frac": [0.50, 0.60, 0.72, 0.80, 0.90]}

# TTT search (C18) -- Bayesian TPE
C18_TTT_SPACE = {
    "eval.ttt_epochs":    {"type": "int", "low": 1, "high": 8},
    "eval.ttt_lr":        {"type": "float", "low": 0.001, "high": 0.01, "log": True},
    "eval.ttt_optimizer": {"type": "categorical", "choices": ["sgd_momentum", "adamw_cosine"]},
    "eval.ttt_lora_rank": {"type": "categorical", "choices": [4, 8, 16]},
}

# Map experiment names -> (mode, n_trials, space)
SEARCH_SPACES = {
    "c01_gla_hybrid":      ("bayesian", 5, C1_GLA_SPACE),
    "c02_mamba_hybrid":    ("bayesian", 5, C2_MAMBA_SPACE),
    "c03_xlstm_hybrid":    ("bayesian", 5, C3_XLSTM_SPACE),
    "c04_rwkv":            ("bayesian", 5, C4_RWKV_SPACE),
    "c05_mixed_hybrid":    ("bayesian", 5, C5_MIXED_SPACE),
    "c11_ema_sweep":       ("grid", 5, C11_EMA_GRID),
    "c12_wd_sweep":        ("grid", 5, C12_WD_GRID),
    "c13_recurrence_sweep": ("grid", 5, C13_RECURRENCE_GRID),
    "c14_warmdown_sweep":  ("grid", 5, C14_WARMDOWN_GRID),
    "c18_ttt_sweep":       ("bayesian", 30, C18_TTT_SPACE),
}
