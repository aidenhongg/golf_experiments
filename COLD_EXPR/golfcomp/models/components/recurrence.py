import torch
import torch.nn as nn


class DepthRecurrence(nn.Module):
    """Hard depth recurrence: reuse layers with FiLM conditioning.
    Layers in recurrence_layers run twice; second pass gets FiLM (scale+shift)."""

    def __init__(self, recurrence_layers: list, dim: int, num_loops: int = 2):
        super().__init__()
        self.recurrence_layers = set(recurrence_layers)
        self.num_loops = num_loops
        self.active = False
        self.film_scale = nn.ParameterDict()
        self.film_shift = nn.ParameterDict()
        for idx in recurrence_layers:
            for loop in range(1, num_loops):
                key = f"l{idx}_loop{loop}"
                self.film_scale[key] = nn.Parameter(torch.ones(dim))
                self.film_shift[key] = nn.Parameter(torch.zeros(dim))

    def should_repeat(self, layer_idx: int) -> bool:
        return self.active and layer_idx in self.recurrence_layers

    def apply_film(self, h: torch.Tensor, layer_idx: int, loop: int) -> torch.Tensor:
        key = f"l{layer_idx}_loop{loop}"
        return h * self.film_scale[key] + self.film_shift[key]


class BasisSharing(nn.Module):
    """SVD basis sharing: W_l = U @ diag(s_l) @ V^T.
    U, V shared; s_l per-layer. Cache reconstructed weights."""

    def __init__(self, num_layers: int, dim: int, rank: int = 64):
        super().__init__()
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.ones(rank)) for _ in range(num_layers)
        ])
        self._cached_weights: list = [None] * num_layers
        self._cache_valid = False

    def invalidate_cache(self):
        self._cache_valid = False

    def _rebuild_cache(self):
        """Rebuild ALL layer weights atomically."""
        new_cache = []
        for coeff in self.coefficients:
            new_cache.append((self.U * coeff) @ self.V)
        self._cached_weights = new_cache
        self._cache_valid = True

    def get_weight(self, layer_idx: int) -> torch.Tensor:
        if not self._cache_valid:
            self._rebuild_cache()
        return self._cached_weights[layer_idx]


class LoRAAdapter(nn.Module):
    """Low-rank adapter: x @ A^T @ B^T."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def reset_parameters(self):
        nn.init.normal_(self.A, std=0.01)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T) @ self.B.T


class LoRASet(nn.Module):
    """LoRA adapters for dim-preserving Linear layers in a transformer block."""

    def __init__(self, block: nn.Module, rank: int = 8, dim: int = None):
        super().__init__()
        self.adapters = nn.ModuleDict()
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                safe_name = name.replace('.', '_')
                self.adapters[safe_name] = LoRAAdapter(
                    module.in_features, module.out_features, rank
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for adapter in self.adapters.values():
            out = out + adapter(x)
        return out


class RelaxedRecursive(nn.Module):
    """Weight tying + per-pass LoRA. shared_blocks x num_loops virtual layers."""

    def __init__(self, shared_blocks: nn.ModuleList, num_loops: int = 2, lora_rank: int = 8):
        super().__init__()
        self.shared_blocks = shared_blocks
        self.num_loops = num_loops
        self.lora_adapters = nn.ModuleDict()
        for block_idx in range(len(shared_blocks)):
            for loop in range(num_loops):
                key = f"b{block_idx}_l{loop}"
                self.lora_adapters[key] = LoRASet(shared_blocks[block_idx], rank=lora_rank)

    def forward(self, x: torch.Tensor, rope_fn=None) -> torch.Tensor:
        for loop in range(self.num_loops):
            for block_idx, block in enumerate(self.shared_blocks):
                key = f"b{block_idx}_l{loop}"
                h = block(x, rope_fn=rope_fn)
                x = h + self.lora_adapters[key](x)
        return x
