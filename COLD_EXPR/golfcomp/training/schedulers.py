import math


class WarmdownScheduler:
    def __init__(self, optimizers, warmup_steps: int, total_steps: int, warmdown_frac: float, shape: str = "1_sqrt"):
        self.optimizers = [o for o in optimizers if o is not None]
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmdown_frac = warmdown_frac
        self.shape = shape
        self._base_lrs = []
        for opt in self.optimizers:
            self._base_lrs.append([g["lr"] for g in opt.param_groups])

    def get_lr_scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        warmdown_start = self.total_steps * (1 - self.warmdown_frac)
        if step < warmdown_start:
            return 1.0
        t = (step - warmdown_start) / max(self.total_steps - warmdown_start, 1)
        t = min(t, 1.0)
        if self.shape == "1_sqrt":
            return 1 - math.sqrt(t)
        elif self.shape == "cosine":
            return 0.5 * (1 + math.cos(math.pi * t))
        return 1 - t

    def step(self, step: int):
        scale = self.get_lr_scale(step)
        for opt, base_lrs in zip(self.optimizers, self._base_lrs):
            for g, blr in zip(opt.param_groups, base_lrs):
                g["lr"] = blr * scale
        return scale


class MomentumWarmup:
    def __init__(self, optimizer, start: float = 0.92, end: float = 0.99, warmup_steps: int = 1500):
        self.optimizer = optimizer
        self.start = start
        self.end = end
        self.warmup_steps = warmup_steps

    def step(self, current_step: int):
        if self.optimizer is None:
            return
        if current_step >= self.warmup_steps:
            mu = self.end
        else:
            t = current_step / max(self.warmup_steps, 1)
            mu = self.start + (self.end - self.start) * t
        for g in self.optimizer.param_groups:
            g["momentum"] = mu
