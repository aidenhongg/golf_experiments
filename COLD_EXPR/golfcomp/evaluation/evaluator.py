"""BPB evaluator with sliding window and optional TTT."""

import math
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path

from golfcomp.config import EvalConfig
from golfcomp.evaluation.ttt import LoRATTT, SLOTTTT, CombinedTTT


class BPBEvaluator:
    """Competition-accurate BPB evaluation.

    Sliding window with configurable stride, SentencePiece byte introspection
    for per-byte BPB, document boundary handling, optional TTT integration.
    """

    def __init__(self, model, tokenizer_path: str, config: EvalConfig):
        self.model = model
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.config = config
        self.ttt = self._build_ttt() if config.use_ttt else None

        # Cache byte lengths per token id
        self._byte_len_cache: dict[int, int] = {}

    def evaluate(self, val_data_path: str) -> dict:
        """Returns {"bpb": float, "total_bytes": int, "total_bits": float}."""
        self.model.eval()
        total_bits = 0.0
        total_bytes = 0

        for doc_tokens, doc_byte_count in self._load_validation(val_data_path):
            if hasattr(self.model, 'reset_xsa'):
                self.model.reset_xsa()

            if self.ttt:
                self.ttt.reset()

            doc_tokens = doc_tokens.cuda()

            for window_start in range(0, max(1, len(doc_tokens) - self.config.window_size + 1), self.config.stride):
                window_end = min(window_start + self.config.window_size, len(doc_tokens))
                input_ids = doc_tokens[window_start:window_end]

                with torch.no_grad():
                    logits = self.model(input_ids.unsqueeze(0))

                # Score only the "new" stride portion (all for first window)
                seq_len = window_end - window_start
                if window_start == 0:
                    score_start = 0
                else:
                    score_start = self.config.window_size - self.config.stride
                    score_start = min(score_start, seq_len - 1)

                log_probs = F.log_softmax(logits[0, score_start:seq_len - 1], dim=-1)
                targets = doc_tokens[window_start + score_start + 1:window_end]

                if len(targets) == 0:
                    continue

                assert len(log_probs) == len(targets), (
                    f"log_probs/targets length mismatch: {len(log_probs)} vs {len(targets)}"
                )

                token_nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

                # Vectorized per-token byte counting
                byte_counts = torch.tensor(
                    [self._token_byte_length(t.item()) for t in targets],
                    device=token_nll.device, dtype=token_nll.dtype,
                )
                total_bits += (token_nll.sum() / math.log(2)).item()
                total_bytes += int(byte_counts.sum().item())

            # TTT: adapt AFTER scoring (score-first protocol)
            if self.ttt:
                self.ttt.adapt(doc_tokens)

        bpb = total_bits / max(total_bytes, 1)
        return {"bpb": bpb, "total_bytes": total_bytes, "total_bits": total_bits}

    def _token_byte_length(self, token_id: int) -> int:
        if token_id not in self._byte_len_cache:
            piece = self.sp.id_to_piece(token_id)
            self._byte_len_cache[token_id] = len(
                piece.replace("\u2581", " ").encode("utf-8")
            )
        return self._byte_len_cache[token_id]

    def _load_validation(self, path: str):
        """Yield (token_tensor, byte_count) per document.

        Reads uint16 .bin files. Documents delimited by token 0 (if present),
        otherwise the whole shard is one document.
        """
        shard_paths = sorted(Path(path).glob("*.bin"))
        for shard in shard_paths:
            data = np.fromfile(str(shard), dtype=np.uint16)
            tokens = torch.from_numpy(data.astype(np.int64))

            # Split on token 0 as document boundary
            zero_idxs = (tokens == 0).nonzero(as_tuple=True)[0]
            if len(zero_idxs) > 0:
                starts = [0] + (zero_idxs + 1).tolist()
                ends = zero_idxs.tolist() + [len(tokens)]
                for s, e in zip(starts, ends):
                    doc = tokens[s:e]
                    if len(doc) < 2:
                        continue
                    yield doc, 0
            else:
                yield tokens, 0

    def _build_ttt(self):
        t = self.config.ttt_type
        if t == "lora":
            return LoRATTT(self.model, self.config)
        elif t == "slot":
            return SLOTTTT(self.model, self.config)
        elif t == "both":
            return CombinedTTT(self.model, self.config)
        raise ValueError(f"Unknown ttt_type: {t}")
