"""Loss tracking with power law extrapolation."""

import numpy as np


class LossTracker:
    """Logs loss every N seconds. Fits power law extrapolation."""

    def __init__(self, log_interval: float = 30.0):
        self.records: list[tuple[int, float, int, float]] = []  # (step, loss, tokens, wall_time)
        self.last_log_time = -999.0
        self.log_interval = log_interval

    def log(self, step: int, loss: float, tokens_seen: int, wall_time: float):
        if wall_time - self.last_log_time >= self.log_interval:
            self.records.append((step, loss, tokens_seen, wall_time))
            self.last_log_time = wall_time

    def fit_power_law(self) -> dict | None:
        """Fit L(T) = a * T^(-b) + L_inf. Returns {a, b, L_inf} or None."""
        if len(self.records) < 5:
            return None
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            return None

        tokens = np.array([r[2] for r in self.records], dtype=np.float64)
        losses = np.array([r[1] for r in self.records], dtype=np.float64)

        def power_law(T, a, b, L_inf):
            return a * np.power(T, -b) + L_inf

        try:
            popt, _ = curve_fit(
                power_law, tokens, losses, p0=[1.0, 0.5, 0.5], maxfev=5000
            )
            return {"a": float(popt[0]), "b": float(popt[1]), "L_inf": float(popt[2])}
        except (RuntimeError, ValueError):
            return None

    def summary(self) -> dict:
        if not self.records:
            return {"final_loss": float("inf"), "steps": 0, "tokens_seen": 0}
        return {
            "final_loss": self.records[-1][1],
            "steps": self.records[-1][0],
            "tokens_seen": self.records[-1][2],
            "wall_time": self.records[-1][3],
            "loss_curve": list(self.records),
            "power_law": self.fit_power_law(),
        }
