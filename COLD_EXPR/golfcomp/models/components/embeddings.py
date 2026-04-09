import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)

    def forward(self, ids):
        if ids.max() >= self.vocab_size or ids.min() < 0:
            bad = ids[(ids < 0) | (ids >= self.vocab_size)]
            raise ValueError(
                f"Token IDs out of range: got {bad[:10].tolist()} "
                f"(min={ids.min().item()}, max={ids.max().item()}, "
                f"vocab_size={self.vocab_size})"
            )
        return F.embedding(ids, self.weight)


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        g = torch.sigmoid(self.gate(x))
        return g * x + (1 - g) * prev


class BigramHash(nn.Module):
    def __init__(self, buckets=3072, hash_dim=128, model_dim=512):
        super().__init__()
        self.buckets = buckets
        self.table = nn.Embedding(buckets, hash_dim)
        self.proj = nn.Linear(hash_dim, model_dim)

    def forward(self, input_ids):
        prev = F.pad(input_ids[:, :-1], (1, 0))
        hashed = ((input_ids * 8191 + prev) % self.buckets)
        return self.proj(self.table(hashed))


class EngramLite(nn.Module):
    def __init__(self, vocab_size, num_heads=3, hash_dim=128, model_dim=512,
                 orders=(2, 3, 4), buckets_per_head=2048):
        super().__init__()
        self.orders = orders
        self.num_heads = num_heads
        self.buckets_per_head = buckets_per_head
        self.tables = nn.ModuleList([nn.Embedding(buckets_per_head, hash_dim) for _ in range(num_heads)])
        self.context_gate = nn.Linear(model_dim, num_heads)
        self.out_proj = nn.Linear(hash_dim * num_heads, model_dim)

    def forward(self, input_ids, hidden_state=None):
        B, S = input_ids.shape
        parts = []
        for i, (order, table) in enumerate(zip(self.orders, self.tables)):
            h = input_ids
            for j in range(1, order):
                shifted = F.pad(input_ids[:, :-j], (j, 0))
                h = h ^ shifted
            h = h % self.buckets_per_head
            parts.append(table(h))

        cat = torch.cat(parts, dim=-1)  # [B, S, hash_dim * num_heads]

        if hidden_state is not None:
            gates = torch.sigmoid(self.context_gate(hidden_state))  # [B, S, num_heads]
            hd = parts[0].shape[-1]
            for i in range(self.num_heads):
                cat[:, :, i*hd:(i+1)*hd] = cat[:, :, i*hd:(i+1)*hd] * gates[:, :, i:i+1]

        return self.out_proj(cat)


_HASH_PRIMES = [2654435761, 2246822519, 40503, 14348907, 73856093]
_HASH_OFFSETS = [0, 7, 13, 31, 61]


class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, low_rank=64,
                 embed_type="factorized", num_tables=3):
        super().__init__()
        self.embed_type = embed_type
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.low_rank = low_rank

        if embed_type == "factorized":
            self.low_rank_embed = nn.Embedding(vocab_size, low_rank)
            self.projection = nn.Linear(low_rank, model_dim, bias=False)
        elif embed_type == "multi_hash":
            table_size = (vocab_size + num_tables - 1) // num_tables
            self.table_size = table_size
            self.num_tables = num_tables
            self.tables = nn.ModuleList([nn.Embedding(table_size, model_dim) for _ in range(num_tables)])
            self._register_hashes(num_tables, table_size)
        elif embed_type == "factorized_multi_hash":
            table_size = (vocab_size + num_tables - 1) // num_tables
            self.table_size = table_size
            self.num_tables = num_tables
            self.tables = nn.ModuleList([nn.Embedding(table_size, low_rank) for _ in range(num_tables)])
            self.projection = nn.Linear(low_rank, model_dim, bias=False)
            self._register_hashes(num_tables, table_size)

    def _register_hashes(self, num_tables, table_size):
        self.hash_fns = []
        for i in range(num_tables):
            p = _HASH_PRIMES[i % len(_HASH_PRIMES)]
            o = _HASH_OFFSETS[i % len(_HASH_OFFSETS)]
            ts = table_size
            self.hash_fns.append(lambda ids, _p=p, _o=o, _ts=ts: ((ids * _p + _o) % _ts).long())

    @property
    def weight(self):
        if self.embed_type == "factorized":
            return self.low_rank_embed.weight @ self.projection.weight.T
        else:
            return self._get_all_embeddings()

    def _get_all_embeddings(self):
        all_ids = torch.arange(self.vocab_size, device=self.tables[0].weight.device)
        return self.forward(all_ids.unsqueeze(0)).squeeze(0)

    def forward(self, input_ids):
        if self.embed_type == "factorized":
            return self.projection(self.low_rank_embed(input_ids))

        dev = input_ids.device
        shape = input_ids.shape
        embed_dim = self.tables[0].embedding_dim
        out = torch.zeros(*shape, embed_dim, device=dev)
        for table, h in zip(self.tables, self.hash_fns):
            out = out + table(h(input_ids))
        if hasattr(self, "projection"):
            out = self.projection(out)
        return out

    def compute_logits(self, hidden):
        # CRITICAL FIX (eng review #9): per-vocab logits, not per-bucket
        if self.embed_type == "factorized":
            # projection.weight: [model_dim, low_rank]
            proj = hidden @ self.projection.weight  # [B, S, low_rank]
            return proj @ self.low_rank_embed.weight.T  # [B, S, vocab]
        else:
            # multi_hash / factorized_multi_hash: reconstruct full vocab embeddings
            all_ids = torch.arange(self.vocab_size, device=hidden.device)
            all_embeds = self.forward(all_ids.unsqueeze(0)).squeeze(0)  # [vocab, dim]
            return hidden @ all_embeds.T
