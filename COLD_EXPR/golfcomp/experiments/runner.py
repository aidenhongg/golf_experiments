"""Experiment runner: train -> quantize -> compress -> evaluate -> log."""

import json
import os
import torch

from ..config import ExperimentConfig
from ..models import build_model
from ..training.trainer import Trainer
from ..quantization import build_quantizer
from ..quantization.compression import Compressor
from ..evaluation.evaluator import BPBEvaluator
from ..utils.artifact import ArtifactPacker
from ..utils.seed import set_seed


class ExperimentRunner:
    """Orchestrates one experiment: train -> quantize -> compress -> evaluate -> log."""

    def __init__(self, config: ExperimentConfig, seed: int = 42):
        self.config = config
        self.seed = seed

    def run(self) -> dict:
        """Full pipeline."""
        set_seed(self.seed)
        model = build_model(self.config)

        # Train
        trainer = Trainer(model, self.config, self.seed)
        train_result = trainer.train()

        # Quantize
        if self.config.quant.method != "none":
            quantizer = build_quantizer(self.config.quant)
            quant_state = quantizer.quantize_model(model)
            # Dequantize back into model so eval sees quantized weights
            # GPTQ modifies in-place; only dequantize for SDClip/Mixed
            if hasattr(quantizer, 'dequantize_state'):
                deq = quantizer.dequantize_state(quant_state)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in deq:
                            param.copy_(deq[name])

        # Pack + Compress (pack quantized weights, not raw)
        artifact_bytes = ArtifactPacker.pack(model.state_dict())
        compressed = Compressor.compress(artifact_bytes, self.config.compression.method)

        # Evaluate
        evaluator = BPBEvaluator(model, self.config.training.tokenizer_path, self.config.eval)
        eval_result = evaluator.evaluate(self.config.training.data_path)

        result = {
            **train_result,
            **eval_result,
            "artifact_bytes": len(compressed),
            "artifact_mb": len(compressed) / (1024 * 1024),
            "seed": self.seed,
            "config_name": self.config.name,
        }
        self._save_result(result)
        return result

    def run_post_training(self, checkpoint_path: str) -> dict:
        """For C15-C20: load checkpoint, apply quant/eval variants only (no training)."""
        model = self._load_checkpoint(checkpoint_path)

        if self.config.quant.method != "none":
            quantizer = build_quantizer(self.config.quant)
            quant_state = quantizer.quantize_model(model)
            if hasattr(quantizer, 'dequantize_state'):
                deq = quantizer.dequantize_state(quant_state)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in deq:
                            param.copy_(deq[name])

        artifact_bytes = ArtifactPacker.pack(model.state_dict())
        compressed = Compressor.compress(artifact_bytes, self.config.compression.method)

        evaluator = BPBEvaluator(model, self.config.training.tokenizer_path, self.config.eval)
        eval_result = evaluator.evaluate(self.config.training.data_path)

        result = {
            **eval_result,
            "artifact_bytes": len(compressed),
            "artifact_mb": len(compressed) / (1024 * 1024),
            "seed": self.seed,
            "config_name": self.config.name,
            "checkpoint": checkpoint_path,
        }
        self._save_result(result)
        return result

    def _save_result(self, result: dict):
        out_dir = f"results/{self.config.name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    def _load_checkpoint(self, path: str):
        model = build_model(self.config)
        state = torch.load(path, map_location="cuda", weights_only=True)
        model.load_state_dict(state["model"])
        return model.cuda()
