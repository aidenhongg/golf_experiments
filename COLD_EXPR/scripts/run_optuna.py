#!/usr/bin/env python3
"""Run Optuna HP search. Usage: python scripts/run_optuna.py --config configs/c01_gla_hybrid.yaml --trials 5"""
import argparse
from golfcomp.config import load_config
from golfcomp.experiments.optuna_search import OptunaSearcher
from golfcomp.experiments.search_spaces import SEARCH_SPACES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--max-time", type=int, default=600)
    args = parser.parse_args()

    config = load_config(args.config)

    if config.name not in SEARCH_SPACES:
        print(f"No search space for {config.name}")
        return

    mode, default_trials, space = SEARCH_SPACES[config.name]
    n_trials = args.trials or default_trials

    searcher = OptunaSearcher(config, space, mode=mode, n_trials=n_trials, time_per_trial=args.max_time)
    result = searcher.search()

    print(f"Best: {result['best_value']:.6f}")
    print(f"Params: {result['best_params']}")


if __name__ == "__main__":
    main()
