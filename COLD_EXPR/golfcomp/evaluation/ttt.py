"""Test-time training: LoRA, SLOT, and Combined adapters."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from golfcomp.models.components.recurrence import LoRAAdapter


class LoRATTT:
    """LoRA-based test-time training. Score-first, backward-looking.

    Adds rank-R LoRA adapters to attention projections.
    Trains per-document on already-scored tokens.
    AdamW + cosine LR (SGD hurts full GPTQ models).
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.adapters: dict[str, LoRAAdapter] = {}
        self._hooks: list = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(
                k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")
            ):
                adapter = LoRAAdapter(
                    module.in_features, module.out_features, config.ttt_lora_rank
                ).cuda()
                self.adapters[name] = adapter

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks that add LoRA output to matching layers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        name_to_module = dict(self.model.named_modules())
        for name, adapter in self.adapters.items():
            module = name_to_module[name]
            hook = module.register_forward_hook(
                lambda mod, inp, out, a=adapter: out + a(inp[0])
            )
            self._hooks.append(hook)

    def adapt(self, scored_tokens: torch.Tensor):
        """Train LoRA adapters on already-scored tokens."""
        params = [p for a in self.adapters.values() for p in a.parameters()]
        if not params:
            return

        if self.config.ttt_optimizer == "adamw_cosine":
            opt = torch.optim.AdamW(params, lr=self.config.ttt_lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(self.config.ttt_epochs, 1)
            )
        else:
            opt = torch.optim.SGD(params, lr=self.config.ttt_lr, momentum=0.9)
            sched = None

        self.model.train()
        for _ in range(self.config.ttt_epochs):
            logits = self.model(scored_tokens.unsqueeze(0))
            loss = F.cross_entropy(
                logits[0, :-1].reshape(-1, logits.size(-1)),
                scored_tokens[1:].reshape(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched:
                sched.step()
        self.model.eval()

    def reset(self):
        for adapter in self.adapters.values():
            adapter.reset_parameters()


class SLOTTTT:
    """Single Learnable Output Transform. Lighter than LoRA.

    Single delta vector at last hidden layer, optimized per-document.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.delta = nn.Parameter(torch.zeros(model.config.model_dim, device="cuda"))
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        """Hook into the final layer norm to add delta."""
        if self._hook:
            self._hook.remove()
        # Add delta to the output of the final norm layer
        self._hook = self.model.norm.register_forward_hook(
            lambda mod, inp, out: out + self.delta
        )

    def adapt(self, scored_tokens: torch.Tensor):
        opt = torch.optim.AdamW([self.delta], lr=self.config.ttt_lr)
        self.model.train()
        for _ in range(self.config.ttt_epochs):
            logits = self.model(scored_tokens.unsqueeze(0))
            loss = F.cross_entropy(
                logits[0, :-1].reshape(-1, logits.size(-1)),
                scored_tokens[1:].reshape(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.model.eval()

    def reset(self):
        self.delta.data.zero_()


class CombinedTTT:
    """LoRA + SLOT combined."""

    def __init__(self, model, config):
        self.lora = LoRATTT(model, config)
        self.slot = SLOTTTT(model, config)

    def adapt(self, tokens: torch.Tensor):
        self.lora.adapt(tokens)
        self.slot.adapt(tokens)

    def reset(self):
        self.lora.reset()
        self.slot.reset()
