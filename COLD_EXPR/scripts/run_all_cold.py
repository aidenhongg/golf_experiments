#!/usr/bin/env python3
"""Run all cold experiments in correct order.
Usage: python scripts/run_all_cold.py --output results/ --phases all
       python scripts/run_all_cold.py --phases 1,2
"""
import argparse
import json
import os
import time

from golfcomp.config import load_config, set_nested
from golfcomp.experiments.runner import ExperimentRunner
from golfcomp.experiments.optuna_search import OptunaSearcher
from golfcomp.experiments.search_spaces import SEARCH_SPACES

ARCH_CONFIGS = [f"configs/c{i:02d}_{n}.yaml" for i, n in [
    (1, "gla_hybrid"), (2, "mamba_hybrid"), (3, "xlstm_hybrid"), (4, "rwkv"), (5, "mixed_hybrid"),
]]
ABLATION_CONFIGS = [f"configs/c{i:02d}_{n}.yaml" for i, n in [
    (6, "basis_sharing"), (7, "relaxed_recursive"), (8, "engramlite"), (9, "swiglu"), (10, "no_parallel_res"),
]]
HP_CONFIGS = [f"configs/c{i:02d}_{n}.yaml" for i, n in [
    (11, "ema_sweep"), (12, "wd_sweep"), (13, "recurrence_sweep"), (14, "warmdown_sweep"),
]]
QUANT_CONFIGS = [f"configs/c{i:02d}_{n}.yaml" for i, n in [
    (15, "sdclip_sweep"), (16, "mixed_precision"), (17, "compressor_sweep"),
]]
EVAL_CONFIGS = [f"configs/c{i:02d}_{n}.yaml" for i, n in [
    (18, "ttt_sweep"), (19, "slot_vs_lora"), (20, "sliding_window"),
]]


def run_single(cfg_path, seed=42, checkpoint=None):
    config = load_config(cfg_path)
    runner = ExperimentRunner(config, seed=seed)
    if checkpoint:
        return runner.run_post_training(checkpoint)
    return runner.run()


def run_optuna_then_best(cfg_path, search_trials=5, search_time=600, best_time=1200):
    config = load_config(cfg_path)
    if config.name not in SEARCH_SPACES:
        print(f"  No search space for {config.name}, running single")
        return run_single(cfg_path)

    mode, default_trials, space = SEARCH_SPACES[config.name]
    n_trials = search_trials or default_trials

    # Optuna search phase
    searcher = OptunaSearcher(config, space, mode=mode, n_trials=n_trials, time_per_trial=search_time)
    search_result = searcher.search()
    print(f"  Search best: {search_result['best_value']:.6f} | {search_result['best_params']}")

    # Rerun best with full time budget
    best_config = load_config(cfg_path)
    for k, v in search_result["best_params"].items():
        set_nested(best_config, k, v)
    best_config.training.max_time_seconds = best_time

    runner = ExperimentRunner(best_config, seed=42)
    return {**runner.run(), "search": search_result}


def save_manifest(manifest, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/")
    parser.add_argument("--phases", default="all")
    args = parser.parse_args()

    phases = set(range(9)) if args.phases == "all" else {int(p) for p in args.phases.split(",")}
    manifest = {"start_time": time.time(), "experiments": {}, "phases": {}}
    baseline_ckpt = None

    # Phase 0: Data prep
    if 0 in phases:
        print("=== Phase 0: Data prep ===")
        for d in ["./data/fineweb_sp8192", "./data/tokenizers"]:
            assert os.path.isdir(d), f"Missing {d}"
        manifest["phases"]["0_data"] = "ok"
        print("  Data verified")

    # Phase 1: Baseline
    if 1 in phases:
        print("=== Phase 1: Baseline ===")
        result = run_single("configs/baseline.yaml")
        manifest["experiments"]["cold_baseline"] = result
        baseline_ckpt = f"results/cold_baseline/checkpoint.pt"
        manifest["phases"]["1_baseline"] = {"bpb": result.get("bpb"), "loss": result.get("final_loss")}
        print(f"  Baseline: BPB={result.get('bpb')}")

    if baseline_ckpt is None:
        baseline_ckpt = "results/cold_baseline/checkpoint.pt"

    # Phase 2: Architecture comparisons C1-C5
    if 2 in phases:
        print("=== Phase 2: Architecture (C1-C5) ===")
        for cfg_path in ARCH_CONFIGS:
            name = load_config(cfg_path).name
            print(f"  Running {name}...")
            result = run_optuna_then_best(cfg_path, search_trials=5, search_time=600, best_time=1200)
            manifest["experiments"][name] = result
        manifest["phases"]["2_arch"] = "done"

    # Phase 3: Technique ablations C6-C10
    if 3 in phases:
        print("=== Phase 3: Ablations (C6-C10) ===")
        for cfg_path in ABLATION_CONFIGS:
            name = load_config(cfg_path).name
            print(f"  Running {name}...")
            result = run_single(cfg_path)
            manifest["experiments"][name] = result
        manifest["phases"]["3_ablation"] = "done"

    # Phase 4: HP sweeps C11-C14
    if 4 in phases:
        print("=== Phase 4: HP Sweeps (C11-C14) ===")
        for cfg_path in HP_CONFIGS:
            config = load_config(cfg_path)
            print(f"  Running {config.name}...")
            if config.name in SEARCH_SPACES:
                mode, n_trials, space = SEARCH_SPACES[config.name]
                searcher = OptunaSearcher(config, space, mode=mode, n_trials=n_trials, time_per_trial=600)
                result = searcher.search()
                manifest["experiments"][config.name] = result
            else:
                manifest["experiments"][config.name] = run_single(cfg_path)
        manifest["phases"]["4_hp"] = "done"

    # Phase 5: Quantization C15-C17
    if 5 in phases:
        print("=== Phase 5: Quantization (C15-C17) ===")
        for cfg_path in QUANT_CONFIGS:
            name = load_config(cfg_path).name
            print(f"  Running {name}...")
            result = run_single(cfg_path, checkpoint=baseline_ckpt)
            manifest["experiments"][name] = result
        manifest["phases"]["5_quant"] = "done"

    # Phase 6: Eval-time C18-C20
    if 6 in phases:
        print("=== Phase 6: Eval-time (C18-C20) ===")
        for cfg_path in EVAL_CONFIGS:
            config = load_config(cfg_path)
            print(f"  Running {config.name}...")
            if config.name in SEARCH_SPACES:
                mode, n_trials, space = SEARCH_SPACES[config.name]
                searcher = OptunaSearcher(config, space, mode=mode, n_trials=n_trials, time_per_trial=600)
                result = searcher.search()
                manifest["experiments"][config.name] = result
            else:
                result = run_single(cfg_path, checkpoint=baseline_ckpt)
                manifest["experiments"][config.name] = result
        manifest["phases"]["6_eval"] = "done"

    # Phase 7: Vocab C21
    if 7 in phases:
        print("=== Phase 7: Vocab (C21) ===")
        result = run_single("configs/c21_sp16384.yaml")
        manifest["experiments"]["c21_sp16384"] = result
        manifest["phases"]["7_vocab"] = "done"

    # Phase 8: Analysis
    if 8 in phases:
        print("=== Phase 8: Analysis ===")
        baseline_bpb = manifest.get("experiments", {}).get("cold_baseline", {}).get("bpb")
        summary = []
        for name, res in manifest["experiments"].items():
            bpb = res.get("bpb") or res.get("best_value")
            delta = f"{bpb - baseline_bpb:+.4f}" if baseline_bpb and bpb else "N/A"
            summary.append({"name": name, "bpb": bpb, "delta": delta})
        manifest["phases"]["8_analysis"] = summary
        for s in sorted(summary, key=lambda x: x.get("bpb") or 999):
            print(f"  {s['name']:30s}  BPB={s['bpb']}  delta={s['delta']}")

    manifest["end_time"] = time.time()
    manifest["elapsed_s"] = manifest["end_time"] - manifest["start_time"]
    save_manifest(manifest, args.output)
    print(f"\nDone. Manifest saved to {args.output}/manifest.json")


if __name__ == "__main__":
    main()
