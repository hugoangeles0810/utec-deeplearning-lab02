"""CLAMF-Former: the full encoder-decoder model (paper Sec. 2.2.1, Fig. 3; D-011, D-017).

``enc_x -> InputEmbedding (MSFM or Linear, + PE) -> Encoder (CAM)`` gives the memory;
``dec_x -> InputEmbedding (MSFM or Linear, + PE) -> Decoder (causal self-attn + LAAM)`` attends to
it, and a ``Linear(d_model -> 1)`` on the last ``H`` decoder positions gives the forecast (D-011).
Every component follows the flags of :class:`~clamf.config.ModelConfig`, so ablations and the
vanilla baseline are only configs.

Shapes (``L = history_hours + horizon_hours``, ``H = horizon_hours``):

- ``enc_x`` ``(B, L, n_meteo)``: meteorology history followed by the horizon meteorology (D-006).
- ``dec_x`` ``(B, L, 1)``: normalized discharge history followed by ``H`` zeros (D-007).
- ``pred``  ``(B, H)``: normalized discharge forecast.
"""

from typing import NamedTuple

import torch
from torch import nn

from clamf.config import Config, ModelConfig
from clamf.models.embedding import build_input_embedding
from clamf.models.layers import DecoderAttention, build_decoder, build_encoder


class ModelAttention(NamedTuple):
    """Attention of every block; weights are ``None`` unless requested, ``tau`` is always set with
    the LAAM (D-017).

    - ``encoder``: self-attention weights per encoder layer, ``(B, n_heads, L, L)``.
    - ``decoder``: :class:`~clamf.models.layers.DecoderAttention` per decoder layer.
    - ``msfm_encoder`` / ``msfm_decoder``: fusion weights per coarse scale ``k``,
      ``(B, n_heads, L, L / k)``; ``None`` without the MSFM or without ``need_weights``.
    """

    encoder: list[torch.Tensor | None]
    decoder: list[DecoderAttention]
    msfm_encoder: dict[int, torch.Tensor] | None
    msfm_decoder: dict[int, torch.Tensor] | None


class ModelOutput(NamedTuple):
    """Forecast ``(B, H)`` and the attention of every block."""

    pred: torch.Tensor
    attention: ModelAttention

    @property
    def tau(self) -> torch.Tensor | None:
        """Predicted lags of every decoder layer, ``(decoder_layers, B, L)``; ``None`` without
        the LAAM (D-001: logged to MLflow)."""
        taus = [att.tau for att in self.attention.decoder]
        if any(t is None for t in taus):
            return None
        return torch.stack(taus)


class CLAMFFormer(nn.Module):
    """CLAMF-Former (Fig. 3) with MSFM, CAM and LAAM switchable by config.

    With ``use_msfm``, ``use_causal_encoder`` and ``use_lag_aware_cross_attn`` all false it is the
    vanilla encoder-decoder Transformer (Linear embedding, non-causal encoder, causal decoder,
    unmasked cross-attention; D-010, D-017).

    The last ``H`` positions of ``dec_x`` are replaced by zeros inside the model, so the forecast
    never depends on whatever the caller put in the horizon (D-007: the target never enters).
    """

    def __init__(
        self, cfg: ModelConfig, n_meteo: int, history_hours: int, horizon_hours: int
    ) -> None:
        super().__init__()
        if n_meteo <= 0 or history_hours <= 0 or horizon_hours <= 0:
            raise ValueError("n_meteo, history_hours and horizon_hours must be > 0")
        seq_len = history_hours + horizon_hours
        if cfg.use_msfm and any(seq_len % k for k in cfg.msfm_scales):
            raise ValueError(f"every msfm scale must divide history + horizon = {seq_len}")
        self.n_meteo = n_meteo
        self.history_hours = history_hours
        self.horizon_hours = horizon_hours
        self.enc_embedding = build_input_embedding(cfg, n_meteo, seq_len)
        self.dec_embedding = build_input_embedding(cfg, 1, seq_len)
        self.encoder = build_encoder(cfg)
        self.decoder = build_decoder(cfg)
        self.head = nn.Linear(cfg.d_model, 1)

    @property
    def seq_len(self) -> int:
        return self.history_hours + self.horizon_hours

    def forward(
        self, enc_x: torch.Tensor, dec_x: torch.Tensor, need_weights: bool = False
    ) -> ModelOutput:
        """``(B, L, n_meteo)``, ``(B, L, 1)`` -> ``(B, H)`` forecast and the attention."""
        self._check_inputs(enc_x, dec_x)
        horizon = self.horizon_hours
        dec_x = torch.cat([dec_x[:, :-horizon], dec_x.new_zeros(len(dec_x), horizon, 1)], dim=1)
        enc, msfm_enc = self.enc_embedding(enc_x, need_weights=need_weights)
        memory, enc_weights = self.encoder(enc, need_weights=need_weights)
        dec, msfm_dec = self.dec_embedding(dec_x, need_weights=need_weights)
        out, dec_attention = self.decoder(dec, memory, need_weights=need_weights)
        pred = self.head(out[:, -horizon:]).squeeze(-1)  # D-011
        return ModelOutput(pred, ModelAttention(enc_weights, dec_attention, msfm_enc, msfm_dec))

    def _check_inputs(self, enc_x: torch.Tensor, dec_x: torch.Tensor) -> None:
        expected = {
            "enc_x": (enc_x, (self.seq_len, self.n_meteo)),
            "dec_x": (dec_x, (self.seq_len, 1)),
        }
        for name, (x, shape) in expected.items():
            if x.dim() != 3 or tuple(x.shape[1:]) != shape:
                raise ValueError(
                    f"{name} must be (B, {shape[0]}, {shape[1]}), got {tuple(x.shape)}"
                )
        if len(enc_x) != len(dec_x):
            raise ValueError(f"batch sizes differ: {len(enc_x)} vs {len(dec_x)}")


def build_model(cfg: Config, n_meteo: int) -> CLAMFFormer:
    """CLAMF-Former for ``n_meteo`` meteorological channels (11 in the dataset, D-006)."""
    return CLAMFFormer(cfg.model, n_meteo, cfg.data.history_hours, cfg.data.horizon_hours)
