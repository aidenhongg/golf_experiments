import torch.nn as nn
from golfcomp.config import ModelConfig


class BaseModel(nn.Module):
    """Abstract base for all models.
    OrthoInit on all Linear weights where SmearGate is used -- without it, +0.003 BPB."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def set_recurrence_active(self, active: bool):
        pass

    def enable_qat(self, bits: int):
        pass

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset_xsa(self):
        for m in self.modules():
            if hasattr(m, 'reset_xsa') and m is not self:
                m.reset_xsa()
