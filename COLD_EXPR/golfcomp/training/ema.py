import torch


class EMAWrapper:
    def __init__(self, model, decay: float = 0.9965):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            self.shadow[n].mul_(d).add_(p.data, alpha=1 - d)

    def apply(self, model):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}
