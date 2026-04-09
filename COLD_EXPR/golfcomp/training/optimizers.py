import math
import torch
from torch.optim import Optimizer, AdamW


class Muon(Optimizer):
    def __init__(self, params, lr=0.022, momentum=0.99, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                orig_shape = grad.shape
                if grad.ndim < 2:
                    X = grad.unsqueeze(0)
                else:
                    X = grad.view(grad.shape[0], -1)
                transposed = X.shape[0] < X.shape[1]
                if transposed:
                    X = X.T
                X = X / (X.norm() + 1e-7)
                for _ in range(ns):
                    X = 1.5 * X - 0.5 * X @ (X.T @ X)
                if transposed:
                    X = X.T
                grad_orth = X.view(orig_shape)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                buf = state["momentum_buffer"]
                update = grad_orth + mu * buf
                buf.mul_(mu).add_(grad_orth)
                p.data.add_(update, alpha=-lr)
        return loss


class MuonEqR(Muon):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, ns = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                orig_shape = grad.shape
                if grad.ndim < 2:
                    X = grad.unsqueeze(0)
                else:
                    X = grad.view(grad.shape[0], -1)
                transposed = X.shape[0] < X.shape[1]
                if transposed:
                    X = X.T
                X = X / (X.norm() + 1e-7)
                for _ in range(ns):
                    X = 1.5 * X - 0.5 * X @ (X.T @ X)
                if transposed:
                    X = X.T
                scale = math.sqrt(X.shape[0] * X.shape[-1]) / (X.norm() + 1e-7)
                grad_orth = (X * scale).view(orig_shape)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                buf = state["momentum_buffer"]
                update = grad_orth + mu * buf
                buf.mul_(mu).add_(grad_orth)
                p.data.add_(update, alpha=-lr)
        return loss


def _classify_param(name, param):
    if "embed" in name or "table" in name:
        return "embed"
    if param.ndim >= 2 and param.numel() >= 256:
        return "matrix"
    return "scalar"


def build_optimizer(model, config):
    matrix_params, embed_params, scalar_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        cls = _classify_param(name, param)
        if cls == "embed":
            embed_params.append(param)
        elif cls == "matrix":
            matrix_params.append(param)
        else:
            scalar_params.append(param)
    OptimizerCls = MuonEqR if config.optimizer == "muoneq_r" else Muon
    if config.optimizer == "adamw":
        adamw = AdamW([
            {"params": matrix_params, "lr": config.matrix_lr},
            {"params": embed_params, "lr": config.embed_lr},
            {"params": scalar_params, "lr": config.scalar_lr},
        ], weight_decay=config.weight_decay)
        return None, adamw
    muon = OptimizerCls(matrix_params, lr=config.matrix_lr, momentum=0.99)
    adamw = AdamW([
        {"params": embed_params, "lr": config.embed_lr},
        {"params": scalar_params, "lr": config.scalar_lr},
    ], weight_decay=config.weight_decay)
    return muon, adamw


_GATE_SSM_KEYWORDS = {"ssm", "dt_proj", "A_log", "B_proj", "C_proj", "time_decay", "time_faaaa", "log_A", "mamba", "gla.gate"}


def _is_gate_ssm(name):
    return any(kw in name for kw in _GATE_SSM_KEYWORDS)


def build_hybrid_optimizer(model, config):
    proj_params, gate_params, scalar_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_gate_ssm(name):
            gate_params.append(param)
        elif param.ndim >= 2 and param.numel() >= 256:
            proj_params.append(param)
        else:
            scalar_params.append(param)
    OptimizerCls = MuonEqR if config.optimizer == "muoneq_r" else Muon
    muon = OptimizerCls(proj_params, lr=config.matrix_lr, momentum=0.99)
    adamw = AdamW([
        {"params": gate_params, "lr": config.gate_lr},
        {"params": scalar_params, "lr": config.scalar_lr},
    ], weight_decay=config.weight_decay)
    return muon, adamw
