"""Optuna HP search: Bayesian (TPE) or Grid mode."""

from copy import deepcopy
import optuna

from ..config import ExperimentConfig, set_nested
from ..models import build_model
from ..training.trainer import Trainer


class OptunaSearcher:
    """Two modes: Bayesian (TPESampler) for C1-C5/C18, Grid (GridSampler) for C11-C14."""

    def __init__(self, base_config: ExperimentConfig, search_space: dict,
                 mode: str = "bayesian", n_trials: int = 5,
                 time_per_trial: int = 600):
        self.base_config = base_config
        self.search_space = search_space
        self.mode = mode
        self.n_trials = n_trials
        self.time_per_trial = time_per_trial

        if mode == "bayesian":
            sampler = optuna.samplers.TPESampler(seed=42)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=100)
        else:  # grid
            sampler = optuna.samplers.GridSampler(search_space)
            pruner = optuna.pruners.NopPruner()

        self.study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner,
        )

    def objective(self, trial: optuna.Trial) -> float:
        config = deepcopy(self.base_config)
        config.training.max_time_seconds = self.time_per_trial

        for param_name, spec in self.search_space.items():
            if self.mode == "grid":
                value = trial.suggest_categorical(param_name, spec)
            elif spec["type"] == "float":
                value = trial.suggest_float(param_name, spec["low"], spec["high"],
                                            log=spec.get("log", False))
            elif spec["type"] == "int":
                value = trial.suggest_int(param_name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                value = trial.suggest_categorical(param_name, spec["choices"])
            else:
                raise ValueError(f"Unknown spec type: {spec['type']}")
            set_nested(config, param_name, value)

        model = build_model(config)
        trainer = Trainer(model, config, seed=42)

        # Pruning callback -- Trainer calls cb(loss, step)
        def report_fn(loss, step):
            trial.report(loss, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trainer.set_pruning_callback(report_fn)

        result = trainer.train()
        return result["final_loss"]

    def search(self) -> dict:
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "all_trials": [
                {"params": t.params, "value": t.value, "state": t.state.name}
                for t in self.study.trials
            ],
        }
