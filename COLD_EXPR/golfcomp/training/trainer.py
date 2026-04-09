import time
import math
import torch
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..utils.seed import set_seed
from ..utils.logging import ExperimentLogger, ExperimentErrorHandler
from .data import FineWebDataset, TokenStream
from .optimizers import build_optimizer, build_hybrid_optimizer
from .schedulers import WarmdownScheduler, MomentumWarmup
from .ema import EMAWrapper


class Trainer:
    def __init__(self, model, config: ExperimentConfig, seed: int):
        set_seed(seed)
        self.config = config
        self.tc = config.training
        self.mc = config.model
        self.seed = seed
        self.model = model.cuda()
        is_hybrid = self.mc.arch != "transformer"
        if is_hybrid:
            self.muon, self.adamw = build_hybrid_optimizer(model, self.tc)
        else:
            self.muon, self.adamw = build_optimizer(model, self.tc)
        self.scheduler = WarmdownScheduler(
            [self.muon, self.adamw],
            self.tc.warmup_steps, self.tc.max_steps,
            self.tc.warmdown_frac, self.tc.cooldown_shape,
        )
        self.mom_warmup = MomentumWarmup(
            self.muon,
            *self.tc.momentum_warmup,
        )
        self.ema = EMAWrapper(model, self.tc.ema_decay) if self.tc.use_ema else None
        dataset = FineWebDataset(self.tc.data_path, self.mc.seq_len, seed)
        self.stream = TokenStream(dataset, self.tc.micro_batch_tokens, self.tc.grad_accum_steps, self.mc.seq_len)
        self.logger = ExperimentLogger(f"logs/{config.name}_{seed}")
        self.error_handler = ExperimentErrorHandler(self.logger)
        self._pruning_cb = None
        self._qat_active = False
        self._recurrence_active = False

    def set_pruning_callback(self, fn):
        self._pruning_cb = fn

    def train(self) -> dict:
        self.logger.log_event("train_start", config=self.config.name, seed=self.seed)
        start = time.time()
        step = 0
        tokens_seen = 0
        loss = float("inf")
        last_log = start
        data_iter = iter(self.stream)

        for micro_batches in data_iter:
            if step >= self.tc.max_steps:
                break
            if time.time() - start >= self.tc.max_time_seconds:
                self.logger.log_event("time_limit", step=step)
                break

            # Late recurrence activation
            if (self.mc.recurrence_type != "none"
                    and not self._recurrence_active
                    and step >= self.mc.recurrence_start_step):
                self._recurrence_active = True
                if hasattr(self.model, "set_recurrence_active"):
                    self.model.set_recurrence_active(True)
                self.logger.log_event("recurrence_activated", step=step)

            lr_scale = self.scheduler.step(step)
            self.mom_warmup.step(step)

            # Late QAT activation
            if (self.tc.use_qat and not self._qat_active
                    and lr_scale < self.tc.qat_start_lr_scale):
                self._qat_active = True
                if hasattr(self.model, "enable_qat"):
                    self.model.enable_qat(self.tc.qat_bits)
                self.logger.log_event("qat_activated", step=step, lr_scale=lr_scale)

            try:
                loss, grad_norm = self._train_step(micro_batches)
            except torch.cuda.OutOfMemoryError:
                self.error_handler.handle_oom(step)
                continue

            batch_tokens = sum(mb["input_ids"].numel() for mb in micro_batches)
            tokens_seen += batch_tokens
            step += 1

            err = self.error_handler.check_loss(loss, step)
            if err == "nan_divergence":
                self.logger.log_event("nan_divergence_abort", step=step)
                break

            if self.ema is not None:
                self.ema.update(self.model)

            now = time.time()
            if now - last_log >= self.tc.log_interval_seconds:
                elapsed = now - start
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                self.logger.log_metrics(
                    step=step, wall_time_s=round(elapsed, 1), loss=round(loss, 4),
                    tokens_seen=tokens_seen,
                    tokens_per_sec=round(tokens_seen / max(elapsed, 1)),
                    lr_scale=round(lr_scale, 4),
                    grad_norm=round(grad_norm, 4), gpu_mem_gb=round(gpu_mem, 2), gpu_util_pct=0,
                    ema_decay=self.tc.ema_decay if self.ema else 0,
                    qat_active=int(self._qat_active),
                    recurrence_active=int(self._recurrence_active),
                )
                last_log = now

            if self._pruning_cb is not None:
                self._pruning_cb(loss, step)

        # Apply EMA before returning
        if self.ema is not None:
            self.ema.apply(self.model)

        elapsed = time.time() - start
        summary = {
            "name": self.config.name, "seed": self.seed,
            "final_loss": loss if step > 0 else float("inf"),
            "steps": step, "tokens_seen": tokens_seen,
            "wall_time_s": round(elapsed, 1),
            "tokens_per_sec": round(tokens_seen / max(elapsed, 1)),
            "qat_active": self._qat_active,
            "recurrence_active": self._recurrence_active,
        }
        self.logger.set_summary(**summary)
        self.logger.save_summary()
        self.logger.log_event("train_end", **summary)
        return summary

    def _train_step(self, micro_batches) -> float:
        self.model.train()
        if self.muon is not None:
            self.muon.zero_grad()
        self.adamw.zero_grad()
        total_loss = 0.0

        for mb in micro_batches:
            ids = mb["input_ids"].cuda()
            labels = mb["labels"].cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / len(micro_batches)
            loss.backward()
            total_loss += loss.item()

        grad_norm = 0.0
        if self.tc.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip).item()

        if self.muon is not None:
            self.muon.step()
        self.adamw.step()
        return total_loss, grad_norm

    def save_checkpoint(self, path: str):
        state = {
            "model": self.model.state_dict(),
            "adamw": self.adamw.state_dict(),
        }
        if self.muon is not None:
            state["muon"] = self.muon.state_dict()
        if self.ema is not None:
            state["ema_shadow"] = self.ema.shadow
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location="cuda", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.adamw.load_state_dict(state["adamw"])
        if self.muon is not None and "muon" in state:
            self.muon.load_state_dict(state["muon"])
        if self.ema is not None and "ema_shadow" in state:
            self.ema.shadow = state["ema_shadow"]
