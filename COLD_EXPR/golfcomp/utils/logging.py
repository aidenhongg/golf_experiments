import csv
import json
import time
from pathlib import Path

METRICS_FIELDS = [
    "step", "wall_time_s", "loss", "tokens_seen", "tokens_per_sec",
    "lr_scale", "grad_norm", "gpu_mem_gb", "gpu_util_pct",
    "ema_decay", "qat_active", "recurrence_active",
]


class ExperimentLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self._events_path = self.log_dir / "events.jsonl"
        self._metrics_path = self.log_dir / "metrics.csv"
        self._summary_path = self.log_dir / "summary.json"
        self._summary = {}
        with open(self._metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_FIELDS)

    def log_event(self, event_type: str, **kwargs):
        record = {"t": time.time() - self.start_time, "type": event_type, **kwargs}
        with open(self._events_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_metrics(self, **kwargs):
        with open(self._metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([kwargs.get(k, "") for k in METRICS_FIELDS])

    def set_summary(self, **kwargs):
        self._summary.update(kwargs)

    def save_summary(self):
        with open(self._summary_path, "w") as f:
            json.dump(self._summary, f, indent=2)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class ExperimentErrorHandler:
    def __init__(self, logger: ExperimentLogger, nan_threshold: int = 3, spike_ratio: float = 5.0):
        self.logger = logger
        self.nan_threshold = nan_threshold
        self.spike_ratio = spike_ratio
        self.nan_count = 0
        self.prev_loss = None

    def check_loss(self, loss: float, step: int) -> str | None:
        import math
        if math.isnan(loss) or math.isinf(loss):
            self.nan_count += 1
            self.logger.log_event("nan_detected", step=step, count=self.nan_count)
            if self.nan_count >= self.nan_threshold:
                return "nan_divergence"
            return "nan_warning"
        self.nan_count = 0
        if self.prev_loss is not None and loss > self.prev_loss * self.spike_ratio:
            self.logger.log_event("loss_spike", step=step, loss=loss, prev=self.prev_loss)
            self.prev_loss = loss
            return "spike_warning"
        self.prev_loss = loss
        return None

    def handle_oom(self, step: int):
        self.logger.log_event("oom", step=step)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
