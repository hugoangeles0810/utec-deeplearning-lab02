"""Encoder and decoder blocks of CLAMF-Former (paper Sec. 2.2.1-2.2.2, Fig. 3; D-017).

Post-norm blocks as in Fig. 3 and Vaswani et al. (2017): every sub-layer is followed by
``Add & Norm``, i.e. ``LayerNorm(x + Dropout(Sublayer(x)))``, and the stacks have no final
LayerNorm. The encoder self-attention is causal (CAM) only with ``use_causal_encoder``; the decoder
self-attention is always causal, as in the classical Transformer; the cross-attention is the LAAM
with ``use_lag_aware_cross_attn`` and a standard unmasked cross-attention otherwise.

Shapes follow ``(B, L, d_model)``; attention weights are ``(B, n_heads, L_q, L_k)`` and ``tau`` is
``(B, L)``. Encoder and decoder must have the same length when the LAAM is used (D-009).
"""

from typing import NamedTuple

import torch
from torch import nn

from clamf.config import ModelConfig
from clamf.models.attention import LagAwareAttention, MultiHeadAttention


class DecoderAttention(NamedTuple):
    """Attention outputs of one decoder layer: weights are ``None`` unless requested; ``tau`` is
    ``None`` without the LAAM."""

    self_attn: torch.Tensor | None
    cross_attn: torch.Tensor | None
    tau: torch.Tensor | None


class FeedForward(nn.Module):
    """Position-wise feed-forward network: ``Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, d_model)`` -> ``(B, L, d_model)``."""
        return self.net(x)


class EncoderLayer(nn.Module):
    """Self-attention (CAM if ``causal``) and feed-forward, each with ``Add & Norm`` (Fig. 3)."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, causal: bool, fused: bool
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=causal, fused=fused)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(B, L, d)`` -> ``(B, L, d)`` and the self-attention weights or ``None``."""
        attn, weights = self.self_attn(x, x, x, need_weights=need_weights)
        x = self.norm1(x + self.dropout(attn))
        return self.norm2(x + self.dropout(self.ff(x))), weights


class DecoderLayer(nn.Module):
    """Causal self-attention, cross-attention (LAAM or standard) and feed-forward (Fig. 3).

    Each sub-layer is followed by ``Add & Norm``; the LAAM query is the output of the first one.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        lag_aware: bool,
        lag_temperature: float,
        lag_margin: float,
        lag_eps: float,
        lag_tau_init_bias: float,
        fused: bool,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal=True, fused=fused)
        self.cross_attn: LagAwareAttention | MultiHeadAttention
        if lag_aware:
            self.cross_attn = LagAwareAttention(
                d_model,
                n_heads,
                dropout,
                temperature=lag_temperature,
                margin=lag_margin,
                eps=lag_eps,
                tau_init_bias=lag_tau_init_bias,
                fused=fused,
            )
        else:
            self.cross_attn = MultiHeadAttention(
                d_model, n_heads, dropout, causal=False, fused=fused
            )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, DecoderAttention]:
        """``(B, L, d)`` decoder state, ``(B, L_enc, d)`` encoder output -> ``(B, L, d)``, attention."""
        attn, self_weights = self.self_attn(x, x, x, need_weights=need_weights)
        x = self.norm1(x + self.dropout(attn))
        tau = None
        if isinstance(self.cross_attn, LagAwareAttention):
            cross, cross_weights, tau = self.cross_attn(x, memory, need_weights=need_weights)
        else:
            cross, cross_weights = self.cross_attn(x, memory, memory, need_weights=need_weights)
        x = self.norm2(x + self.dropout(cross))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x, DecoderAttention(self_weights, cross_weights, tau)


class Encoder(nn.Module):
    """Stack of :class:`EncoderLayer` with no final LayerNorm (Fig. 3)."""

    def __init__(self, layers: list[EncoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """``(B, L, d)`` -> ``(B, L, d)`` and the self-attention weights of every layer."""
        weights = []
        for layer in self.layers:
            x, w = layer(x, need_weights=need_weights)
            weights.append(w)
        return x, weights


class Decoder(nn.Module):
    """Stack of :class:`DecoderLayer`; every layer attends to the last encoder output (Fig. 3)."""

    def __init__(self, layers: list[DecoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, list[DecoderAttention]]:
        """``(B, L, d)``, ``(B, L_enc, d)`` -> ``(B, L, d)`` and the attention of every layer."""
        attention = []
        for layer in self.layers:
            x, att = layer(x, memory, need_weights=need_weights)
            attention.append(att)
        return x, attention


def build_encoder(cfg: ModelConfig) -> Encoder:
    """``encoder_layers`` blocks; CAM when ``use_causal_encoder`` (Sec. 2.2.2, Table 6)."""
    return Encoder(
        [
            EncoderLayer(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_ff,
                cfg.dropout,
                causal=cfg.use_causal_encoder,
                fused=cfg.fused_attention,
            )
            for _ in range(cfg.encoder_layers)
        ]
    )


def build_decoder(cfg: ModelConfig) -> Decoder:
    """``decoder_layers`` blocks; LAAM when ``use_lag_aware_cross_attn``, one ``tau`` net per layer
    (D-009)."""
    return Decoder(
        [
            DecoderLayer(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_ff,
                cfg.dropout,
                lag_aware=cfg.use_lag_aware_cross_attn,
                lag_temperature=cfg.lag_temperature,
                lag_margin=cfg.lag_margin,
                lag_eps=cfg.lag_eps,
                lag_tau_init_bias=cfg.lag_tau_init_bias,
                fused=cfg.fused_attention,
            )
            for _ in range(cfg.decoder_layers)
        ]
    )
