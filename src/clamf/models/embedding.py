"""Input embedding: MSFM or Linear, plus positional encoding (paper Sec. 2.2.1, Fig. 3; D-010, D-012).

``InputEmbedding(x) = Dropout(Projection(x) + PE)``, where ``Projection`` is the MSFM
(``use_msfm: true``) or ``Linear(d_in -> d_model)`` (``use_msfm: false``, D-010), and ``PE`` is the
fixed sinusoidal encoding of Vaswani et al. (2017), added without ``sqrt(d_model)`` scaling (D-012).
Shapes follow ``(B, T, features)``.
"""

import math

import torch
from torch import nn

from clamf.config import ModelConfig
from clamf.models.msfm import MSFM


def sinusoidal_encoding(length: int, d_model: int) -> torch.Tensor:
    """Vaswani et al. (2017): ``PE[t, 2i] = sin(t / 10000^(2i/d))``, ``PE[t, 2i+1] = cos(...)``.

    Returns float32 ``(length, d_model)``; an odd ``d_model`` drops the last cosine column.
    """
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: d_model // 2])
    return pe


class PositionalEncoding(nn.Module):
    """Adds the fixed sinusoidal encoding (D-012); the table is a non-persistent buffer.

    Encoder and decoder build the same table, so hour ``t`` gets the same code in both (D-009).
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.register_buffer("pe", sinusoidal_encoding(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` -> ``(B, T, d_model)``, with ``T <= max_len``."""
        if x.shape[1] > self.pe.shape[0]:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_len {self.pe.shape[0]}")
        return x + self.pe[: x.shape[1]].to(x.dtype)


class InputEmbedding(nn.Module):
    """``Dropout(Projection(x) + PE)`` for the encoder or the decoder input (Fig. 3; D-010, D-012)."""

    def __init__(
        self, projection: MSFM | nn.Linear, d_model: int, max_len: int, dropout: float
    ) -> None:
        super().__init__()
        self.projection = projection
        self.positional = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None]:
        """``(B, T, d_in)`` -> ``(B, T, d_model)`` and the MSFM fusion weights.

        The weights (see :class:`~clamf.models.msfm.MSFM`) are ``None`` unless ``need_weights`` is
        true and the projection is the MSFM.
        """
        if isinstance(self.projection, MSFM):
            h, weights = self.projection(x, need_weights=need_weights)
        else:
            h, weights = self.projection(x), None
        return self.dropout(self.positional(h)), weights


def build_input_embedding(cfg: ModelConfig, d_in: int, seq_len: int) -> InputEmbedding:
    """Embedding for ``d_in`` input channels (11 meteorology, 1 discharge) over ``seq_len`` steps."""
    if cfg.positional_encoding != "sinusoidal":
        raise ValueError(f"unknown positional encoding {cfg.positional_encoding!r}")
    projection: MSFM | nn.Linear
    if cfg.use_msfm:
        projection = MSFM(
            d_in,
            cfg.d_fusion,
            cfg.d_model,
            cfg.msfm_scales,
            cfg.msfm_conv_kernel,
            cfg.n_heads,
            cfg.dropout,
            cfg.fused_attention,
        )
    else:
        projection = nn.Linear(d_in, cfg.d_model)
    return InputEmbedding(projection, cfg.d_model, seq_len, cfg.dropout)
