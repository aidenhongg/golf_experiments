import torch
import torch.nn as nn


class GPTQQuantizer:
    """Full GPTQ: Hessian-guided greedy quantization with Cholesky + activation ordering.
    Uses self-generated calibration data. Processes one layer at a time.
    CRITICAL: Hessian may not be PD. Add 1e-6 diagonal regularization."""

    def __init__(self, bits=6, actorder=True, use_cholesky=True, nsamples=128):
        self.bits = bits
        self.actorder = actorder
        self.use_cholesky = use_cholesky
        self.nsamples = nsamples

    def quantize_model(self, model, calib_loader):
        """Quantize all Linear layers using GPTQ with hook-based Hessian collection."""
        state = {}
        layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}

        for name, layer in layers.items():
            inputs = []
            handle = layer.register_forward_hook(
                lambda m, inp, out, store=inputs: store.append(inp[0].detach())
            )
            # Collect calibration inputs
            with torch.no_grad():
                count = 0
                for batch in calib_loader:
                    if count >= self.nsamples:
                        break
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    model(batch.to(next(model.parameters()).device))
                    count += batch.shape[0]
            handle.remove()

            if not inputs:
                state[name] = {"raw": layer.weight.data.half()}
                continue

            H = self._collect_hessian(inputs)
            Q = self._quantize_layer(layer.weight.data, H)
            state[name] = {"quantized": Q, "bits": self.bits}

            # Apply quantized weight back to model
            layer.weight.data.copy_(Q)

        return state

    def _quantize_layer(self, weight, H):
        """Core GPTQ algorithm on a single weight matrix."""
        W = weight.clone().float()
        n_rows, n_cols = W.shape

        # Actorder: sort columns by Hessian diagonal (most important first)
        perm = None
        if self.actorder:
            perm = torch.argsort(H.diag(), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        # Regularize and decompose
        damp = 1e-6 * torch.eye(n_cols, device=H.device)
        H = H + damp
        if self.use_cholesky:
            try:
                L = torch.linalg.cholesky(H)
                Hinv = torch.cholesky_inverse(L)
            except torch.linalg.LinAlgError:
                Hinv = torch.linalg.inv(H)
        else:
            Hinv = torch.linalg.inv(H)

        # Greedy column-by-column quantization
        Q = torch.zeros_like(W)
        levels = 2 ** self.bits
        half = levels // 2
        scale = W.abs().max(dim=0).values / half
        scale = scale.clamp(min=1e-8)

        for col in range(n_cols):
            w = W[:, col]
            s = scale[col]

            # Quantize column
            q = (w / s).round().clamp(-half, half - 1) * s
            Q[:, col] = q

            # Error compensation to remaining columns
            if col < n_cols - 1:
                err = w - q
                d = Hinv[col, col].clamp(min=1e-10)
                W[:, col + 1:] -= (err / d).unsqueeze(1) * Hinv[col, col + 1:].unsqueeze(0)

        # Undo actorder permutation
        if perm is not None:
            inv_perm = torch.argsort(perm)
            Q = Q[:, inv_perm]

        return Q

    def dequantize_state(self, state):
        """GPTQ modifies model in-place. Return float weights as-is."""
        result = {}
        for name, entry in state.items():
            if "raw" in entry:
                result[name] = entry["raw"].float()
            else:
                result[name] = entry["quantized"].float()
        return result

    def _collect_hessian(self, inputs):
        """H = (1/n) * sum(X_i^T X_i) where X_i are layer inputs."""
        H = None
        for inp in inputs:
            x = inp.float().reshape(-1, inp.shape[-1])
            h = x.T @ x
            H = h if H is None else H + h
        return H / len(inputs)
