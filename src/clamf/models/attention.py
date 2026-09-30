"""Attention modules: multi-head attention, CAM and LAAM (paper Sec. 2.2.2, Eq. 2-5, Fig. 4).

Shapes follow ``(B, L, d_model)``; attention weights are ``(B, n_heads, L_q, L_k)``. Masks are
additive float biases on the logits before the softmax (``-inf`` blocks, ``0`` keeps); the boolean
mask of :func:`lag_aware_binary_mask` means ``True`` = visible and is only for visualization (D-001).

Every attention runs through ``F.scaled_dot_product_attention`` unless its weights are requested,
in which case the explicit ``softmax(QK^T / sqrt(d_k) + bias) V`` is used (D-004).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

AttentionOutput = tuple[torch.Tensor, torch.Tensor | None]


class MultiHeadAttention(nn.Module):
    """Multi-head attention (Vaswani et al., 2017; paper Fig. 4a), optionally causal (CAM).

    With ``causal=True`` query ``i`` only attends to keys ``j <= i`` (CAM, Sec. 2.2.2); the mask is
    top-left aligned, as ``is_causal`` in PyTorch. Dropout is applied to the attention weights.
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float, causal: bool, fused: bool
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.causal = causal
        self.fused = fused
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> AttentionOutput:
        """``(B, L_q, d)``, ``(B, L_k, d)``, ``(B, L_k, d)`` -> ``(B, L_q, d)`` and the weights.

        ``bias`` is an additive float mask broadcastable to ``(B, n_heads, L_q, L_k)``. The weights
        (before dropout) are returned only when ``need_weights`` is true, otherwise ``None``.
        """
        q, k, v = (
            self._split(w(x)) for w, x in ((self.w_q, query), (self.w_k, key), (self.w_v, value))
        )
        if self.causal and bias is not None:
            bias = bias + causal_bias(q.shape[-2], k.shape[-2], q.device)
        causal_only = self.causal and bias is None
        p = self.dropout if self.training else 0.0
        weights = None
        if self.fused and not need_weights:
            mask = None if bias is None else bias.to(q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=p, is_causal=causal_only
            )
        else:
            logits = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
            if causal_only:
                logits = logits + causal_bias(q.shape[-2], k.shape[-2], q.device)
            if bias is not None:
                logits = logits + bias
            weights = logits.softmax(dim=-1)
            out = F.dropout(weights, p=p).to(v.dtype) @ v
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, self.n_heads * self.d_head)
        return self.w_o(out), weights if need_weights else None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, d)`` -> ``(B, n_heads, L, d_head)``."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)


class LagPredictor(nn.Module):
    """Causal lag-aware network, Eq. (2)-(4) and Fig. 4(b): one ``tau_i >= 0`` per position (D-009).

    ``Z_i = Concat(Q_i, K~_i)`` with the unprojected decoder state ``Q`` and encoder output ``K``;
    ``tau_i = Softplus(W2 LayerNorm(ReLU(W1 Z_i + b1 + Z_i)) + b2)``, shared across heads.
    """

    def __init__(self, d_model: int, eps: float, tau_init_bias: float) -> None:
        super().__init__()
        self.eps = eps
        self.fc1 = nn.Linear(2 * d_model, 2 * d_model)
        self.norm = nn.LayerNorm(2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, 1)
        nn.init.constant_(self.fc2.bias, tau_init_bias)  # tau ~ 0 at init (D-001, D-009)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """``(B, L, d)``, ``(B, L, d)`` -> ``tau`` ``(B, L)``, in time steps. Query and key must be
        aligned in time (same length, D-009)."""
        if query.shape[1] != key.shape[1]:
            raise ValueError(
                f"lag-aware attention needs aligned sequences, got {query.shape[1]} and {key.shape[1]}"
            )
        z = torch.cat([query, aggregate_keys(key, self.eps)], dim=-1)  # Eq. (3)
        hidden = self.norm(F.relu(self.fc1(z) + z))
        return F.softplus(self.fc2(hidden)).squeeze(-1)  # Eq. (4)


class LagAwareAttention(nn.Module):
    """LAAM: cross-attention with the soft lag-aware mask of Eq. (5) (Sec. 2.2.2; D-001, D-009).

    The decoder state (query) attends to the encoder output (key/value) with the additive bias of
    :func:`lag_aware_bias`, computed from the ``tau`` predicted by :class:`LagPredictor`.
    ``temperature`` is a plain attribute so it can be annealed during training (D-001).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        temperature: float,
        margin: float,
        eps: float,
        tau_init_bias: float,
        fused: bool,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.lag = LagPredictor(d_model, eps, tau_init_bias)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal=False, fused=fused)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """``(B, L, d)``, ``(B, L, d)`` -> output ``(B, L, d)``, weights or ``None``, ``tau (B, L)``."""
        tau = self.lag(query, memory)
        bias = lag_aware_bias(tau, memory.shape[1], self.temperature, self.margin)
        out, weights = self.attn(query, memory, memory, bias=bias, need_weights=need_weights)
        return out, weights, tau


def aggregate_keys(key: torch.Tensor, eps: float) -> torch.Tensor:
    """Content aggregation, Eq. (2): ``K~_i = sum_{j<=i} K_j / (P_i + eps)`` with ``P_i = i + 1``.

    That is the cumulative mean of the keys (D-009). ``(B, L, d)`` -> ``(B, L, d)``.
    """
    positions = torch.arange(1, key.shape[1] + 1, device=key.device, dtype=key.dtype)
    return key.cumsum(dim=1) / (positions[:, None] + eps)


def lag_aware_bias(
    tau: torch.Tensor, n_keys: int, temperature: float, margin: float
) -> torch.Tensor:
    """Soft lag-aware mask (D-001): ``bias_ij = logsigmoid((i + tau_i + margin - j) / T)``.

    ``tau (B, L_q)`` -> float32 bias ``(B, 1, L_q, n_keys)``, shared by all heads. As ``T -> 0`` it
    recovers the binary mask of Eq. (5).
    """
    tau = tau.float()
    i = torch.arange(tau.shape[1], device=tau.device, dtype=torch.float32)
    j = torch.arange(n_keys, device=tau.device, dtype=torch.float32)
    edge = i[:, None] + tau[..., None] + margin - j  # (B, L_q, n_keys)
    return F.logsigmoid(edge / temperature).unsqueeze(1)


def lag_aware_binary_mask(tau: torch.Tensor, n_keys: int) -> torch.Tensor:
    """Binary mask of Eq. (5), for visualization only (D-001): ``True`` where ``j <= i + tau_i``.

    ``tau (B, L_q)`` -> bool ``(B, L_q, n_keys)``.
    """
    i = torch.arange(tau.shape[1], device=tau.device)
    j = torch.arange(n_keys, device=tau.device)
    return j <= i[:, None] + tau[..., None]


def causal_bias(n_queries: int, n_keys: int, device: torch.device) -> torch.Tensor:
    """Causal mask of CAM (Sec. 2.2.2): ``0`` where ``j <= i``, ``-inf`` otherwise; ``(L_q, L_k)``."""
    blocked = torch.ones(n_queries, n_keys, dtype=torch.bool, device=device).triu(diagonal=1)
    return torch.zeros(n_queries, n_keys, device=device).masked_fill(blocked, float("-inf"))
