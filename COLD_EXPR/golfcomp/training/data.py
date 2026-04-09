import os
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset


class FineWebDataset(IterableDataset):
    def __init__(self, data_path: str, seq_len: int, seed: int):
        self.data_path = data_path
        self.seq_len = seq_len
        self.seed = seed
        self.shards = sorted(glob.glob(os.path.join(data_path, "*.bin")))
        assert self.shards, f"No .bin shards found in {data_path}"

    def __iter__(self):
        self._epoch = getattr(self, '_epoch', 0) + 1
        rng = np.random.RandomState(self.seed + self._epoch)
        shard_order = list(range(len(self.shards)))
        rng.shuffle(shard_order)
        buf = np.array([], dtype=np.uint16)
        for idx in shard_order:
            data = np.fromfile(self.shards[idx], dtype=np.uint16)
            buf = np.concatenate([buf, data])
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf = buf[self.seq_len + 1:]
                ids = torch.from_numpy(chunk.astype(np.int64))
                yield {"input_ids": ids[:-1], "labels": ids[1:]}


class TokenStream:
    def __init__(self, dataset: FineWebDataset, micro_batch_tokens: int, grad_accum_steps: int, seq_len: int):
        self.dataset = dataset
        self.micro_batch_size = micro_batch_tokens // seq_len
        self.grad_accum_steps = grad_accum_steps
        self.seq_len = seq_len

    def __iter__(self):
        it = iter(self.dataset)
        while True:
            accum = []
            for _ in range(self.grad_accum_steps):
                batch_ids, batch_labels = [], []
                for _ in range(self.micro_batch_size):
                    try:
                        sample = next(it)
                    except StopIteration:
                        it = iter(self.dataset)
                        sample = next(it)
                    batch_ids.append(sample["input_ids"])
                    batch_labels.append(sample["labels"])
                accum.append({
                    "input_ids": torch.stack(batch_ids),
                    "labels": torch.stack(batch_labels),
                })
            yield accum
