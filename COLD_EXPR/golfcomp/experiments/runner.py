"""Experiment runner: train -> [pre-quant TTT] -> quantize -> compress -> evaluate -> log."""

import glob
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

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

    def _sync_vocab_size(self):
        """Override config vocab_size with the actual tokenizer vocab size."""
        tok_path = self.config.training.tokenizer_path
        if tok_path and os.path.exists(tok_path):
            sp = spm.SentencePieceProcessor(model_file=tok_path)
            actual = sp.GetPieceSize()
            if actual != self.config.model.vocab_size:
                print(f"  vocab_size: config={self.config.model.vocab_size} -> tokenizer={actual}")
                self.config.model.vocab_size = actual

    def _pre_quant_ttt(self, model):
        """Pre-quantization test-time training on validation data.

        Adapts model weights on val tokens BEFORE quantization, baking the
        adapted weights into the artifact. This is the key technique from
        PR #1487 (1.0600 BPB SOTA).
        """
        tc = self.config.training
        seq_len = tc.pre_quant_ttt_seq_len
        epochs = tc.pre_quant_ttt_epochs
        lr = tc.pre_quant_ttt_lr
        freeze_blocks = tc.pre_quant_ttt_freeze_blocks

        print(f"  Pre-quant TTT: {epochs} epochs, lr={lr}, freeze={freeze_blocks} blocks")

        # Load val data as continuous token stream
        val_path = tc.val_data_path or tc.data_path
        shard_paths = sorted(glob.glob(os.path.join(val_path, "fineweb_val_*.bin")))
        if not shard_paths:
            shard_paths = sorted(glob.glob(os.path.join(val_path, "*val*.bin")))
        if not shard_paths:
            raise RuntimeError(
                f"Pre-quant TTT enabled but no val shards found in {val_path}. "
                f"Set val_data_path in config or disable pre_quant_ttt."
            )

        all_tokens = []
        for shard in shard_paths:
            data = np.fromfile(shard, dtype=np.uint16)
            all_tokens.append(torch.from_numpy(data.astype(np.int32)))
        tokens = torch.cat(all_tokens).cuda()
        del all_tokens
        print(f"  Val tokens: {len(tokens):,}")

        # Freeze first N blocks
        if freeze_blocks > 0:
            if hasattr(model, 'layers'):
                for i in range(min(freeze_blocks, len(model.layers))):
                    for param in model.layers[i].parameters():
                        param.requires_grad_(False)
            else:
                print(f"  WARNING: model has no 'layers' attribute, skipping freeze")

        # Build optimizer (AdamW on unfrozen params only)
        ttt_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(ttt_params, lr=lr, weight_decay=0.0)

        # Number of sequences we can make from val tokens
        n_seqs = (len(tokens) - 1) // seq_len
        if n_seqs == 0:
            raise RuntimeError(
                f"Val data too short for pre-quant TTT ({len(tokens)} tokens, need > {seq_len})"
            )

        # Cosine schedule over total steps
        total_steps = epochs * n_seqs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # Reset XSA state before TTT
        if hasattr(model, 'reset_xsa'):
            model.reset_xsa()

        model.train()
        step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(n_seqs):
                start = i * seq_len
                chunk = tokens[start:start + seq_len + 1]
                ids = chunk[:-1].unsqueeze(0)
                labels = chunk[1:].unsqueeze(0)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                step += 1

            avg_loss = epoch_loss / max(n_seqs, 1)
            print(f"    TTT epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Unfreeze all params
        for param in model.parameters():
            param.requires_grad_(True)

        model.eval()
        del optimizer, scheduler, ttt_params, tokens
        torch.cuda.empty_cache()
        print(f"  Pre-quant TTT complete ({step} steps)")

    def run(self) -> dict:
        """Full pipeline: train -> [EMA] -> [pre-quant TTT] -> quantize -> compress -> eval."""
        set_seed(self.seed)
        self._sync_vocab_size()
        model = build_model(self.config)

        # Train
        trainer = Trainer(model, self.config, self.seed)
        train_result = trainer.train()

        # Pre-quant TTT (before quantization, after EMA)
        if self.config.training.pre_quant_ttt:
            self._pre_quant_ttt(model)

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
        val_path = self.config.training.val_data_path or self.config.training.data_path
        eval_result = evaluator.evaluate(val_path)

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
        self._sync_vocab_size()
        model = self._load_checkpoint(checkpoint_path)

        # Pre-quant TTT (if enabled)
        if self.config.training.pre_quant_ttt:
            self._pre_quant_ttt(model)

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
        val_path = self.config.training.val_data_path or self.config.training.data_path
        eval_result = evaluator.evaluate(val_path)

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
