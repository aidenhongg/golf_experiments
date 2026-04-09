#!/usr/bin/env python3
"""Run a single experiment. Usage: python scripts/run_experiment.py --config configs/baseline.yaml"""
import argparse
from golfcomp.config import load_config
from golfcomp.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.dry_run:
        config.training.max_steps = 10
        config.training.max_time_seconds = 60

    runner = ExperimentRunner(config, seed=args.seed)
    if args.eval_only and args.checkpoint:
        result = runner.run_post_training(args.checkpoint)
    else:
        result = runner.run()

    print(f"Result: BPB={result.get('bpb', 'N/A')}, loss={result.get('final_loss', 'N/A')}")


if __name__ == "__main__":
    main()
