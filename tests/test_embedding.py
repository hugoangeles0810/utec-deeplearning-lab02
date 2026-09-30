import math

import pytest
import torch
from torch import nn

from clamf.config import ModelConfig
from clamf.models.embedding import (
    InputEmbedding,
    PositionalEncoding,
    build_input_embedding,
    sinusoidal_encoding,
)
from clamf.models.msfm import MSFM

B, T, D_IN, D_MODEL = 2, 24, 3, 8


def tiny_config(**overrides: object) -> ModelConfig:
    base = {
        "d_model": D_MODEL,
        "d_fusion": 8,
        "n_heads": 2,
        "dropout": 0.0,
        "msfm_scales": (1, 4, 12),
    }
    return ModelConfig(**{**base, **overrides})


def make_embedding(use_msfm: bool, d_in: int = D_IN, **overrides: object) -> InputEmbedding:
    torch.manual_seed(0)
    return build_input_embedding(tiny_config(use_msfm=use_msfm, **overrides), d_in, T).eval()


def inputs(seed: int = 1, d_in: int = D_IN, length: int = T) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, length, d_in)


# --- Sinusoidal encoding (D-012) ----------------------------------------------------------------


@pytest.mark.parametrize("d_model", [6, 5])  # an odd d_model drops the last cosine column
def test_sinusoidal_matches_vaswani_formula(d_model: int) -> None:
    pe = sinusoidal_encoding(5, d_model)
    assert pe.shape == (5, d_model)
    assert pe.dtype == torch.float32
    for t in range(5):
        for col in range(d_model):
            angle = t / 10000 ** (2 * (col // 2) / d_model)
            expected = math.sin(angle) if col % 2 == 0 else math.cos(angle)
            assert pe[t, col].item() == pytest.approx(expected, abs=1e-6)


def test_sinusoidal_first_position_and_range() -> None:
    pe = sinusoidal_encoding(384, 64)
    assert pe[0].tolist() == [0.0, 1.0] * 32
    assert pe.abs().max().item() <= 1.0
    assert len(torch.unique(pe, dim=0)) == 384  # every hour gets a distinct code


def test_positional_encoding_adds_without_scaling() -> None:
    x = inputs(d_in=D_MODEL)
    out = PositionalEncoding(D_MODEL, max_len=T)(x)
    torch.testing.assert_close(out - x, sinusoidal_encoding(T, D_MODEL).expand(B, -1, -1))


def test_positional_encoding_accepts_shorter_sequences_and_rejects_longer() -> None:
    pe = PositionalEncoding(D_MODEL, max_len=T)
    assert pe(inputs(d_in=D_MODEL, length=T - 4)).shape == (B, T - 4, D_MODEL)
    with pytest.raises(ValueError, match="max_len"):
        pe(inputs(d_in=D_MODEL, length=T + 1))


def test_positional_encoding_is_fixed_and_not_checkpointed() -> None:
    pe = PositionalEncoding(D_MODEL, max_len=T)
    assert list(pe.parameters()) == []
    assert "pe" not in pe.state_dict()


def test_positional_encoding_keeps_input_dtype() -> None:
    x = inputs(d_in=D_MODEL).to(torch.bfloat16)
    assert PositionalEncoding(D_MODEL, max_len=T)(x).dtype == torch.bfloat16


# --- Input embedding (D-010, D-012) -------------------------------------------------------------


@pytest.mark.parametrize("d_in", [11, 1])  # encoder meteorology and decoder discharge
def test_build_with_msfm(d_in: int) -> None:
    emb = make_embedding(use_msfm=True, d_in=d_in)
    assert isinstance(emb.projection, MSFM)
    out, weights = emb(inputs(d_in=d_in), need_weights=True)
    assert out.shape == (B, T, D_MODEL)
    assert set(weights) == {4, 12}
    assert emb(inputs(d_in=d_in))[1] is None


@pytest.mark.parametrize("d_in", [11, 1])
def test_build_without_msfm_uses_linear(d_in: int) -> None:
    emb = make_embedding(use_msfm=False, d_in=d_in)
    assert isinstance(emb.projection, nn.Linear)
    assert (emb.projection.in_features, emb.projection.out_features) == (d_in, D_MODEL)
    out, weights = emb(inputs(d_in=d_in), need_weights=True)
    assert out.shape == (B, T, D_MODEL)
    assert weights is None


@pytest.mark.parametrize("use_msfm", [False, True])
def test_embedding_is_projection_plus_encoding(use_msfm: bool) -> None:
    emb, x = make_embedding(use_msfm), inputs()
    projected = emb.projection(x)[0] if use_msfm else emb.projection(x)
    torch.testing.assert_close(emb(x)[0], projected + sinusoidal_encoding(T, D_MODEL))


def test_encoder_and_decoder_share_the_same_encoding() -> None:
    enc, dec = make_embedding(True, d_in=11), make_embedding(True, d_in=1)
    torch.testing.assert_close(enc.positional.pe, dec.positional.pe, atol=0, rtol=0)


def test_embedding_without_msfm_is_pointwise_in_time() -> None:
    # keeps the model causal when use_msfm is false (D-005): step t only changes output t
    emb, x, t = make_embedding(use_msfm=False), inputs(), 10
    perturbed = x.clone()
    perturbed[:, t] += 1.0
    changed = (emb(x)[0] - emb(perturbed)[0]).abs().amax(dim=(0, 2)) > 0
    assert changed.nonzero().flatten().tolist() == [t]


def test_dropout_only_in_training() -> None:
    emb, x = make_embedding(use_msfm=False, dropout=0.5), inputs()
    torch.testing.assert_close(emb(x)[0], emb(x)[0])
    emb.train()
    assert not torch.allclose(emb(x)[0], emb(x)[0])


def test_build_uses_config_dimensions() -> None:
    torch.manual_seed(0)
    emb = build_input_embedding(ModelConfig(), d_in=11, seq_len=384)
    assert isinstance(emb.projection, MSFM)
    assert emb.projection.scales == (1, 24, 96)
    assert emb.positional.pe.shape == (384, 64)
    assert emb.dropout.p == 0.1
    assert emb(torch.randn(1, 384, 11))[0].shape == (1, 384, 64)
