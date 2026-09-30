import pytest
import torch

from clamf.config import ModelConfig
from clamf.models.attention import LagAwareAttention, MultiHeadAttention
from clamf.models.layers import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    build_decoder,
    build_encoder,
)

B, L, D, HEADS, D_FF = 2, 12, 16, 4, 32


def tiny_config(**overrides: object) -> ModelConfig:
    base = {
        "d_model": D,
        "d_fusion": D,
        "n_heads": HEADS,
        "d_ff": D_FF,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.0,
    }
    return ModelConfig(**{**base, **overrides})


def make_encoder(**overrides: object) -> Encoder:
    torch.manual_seed(0)
    return build_encoder(tiny_config(**overrides)).eval()


def make_decoder(**overrides: object) -> Decoder:
    torch.manual_seed(0)
    return build_decoder(tiny_config(**overrides)).eval()


def inputs(seed: int = 1, length: int = L) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, length, D)


def perturb_after(x: torch.Tensor, i: int) -> torch.Tensor:
    perturbed = x.clone()
    perturbed[:, i + 1 :] += torch.randn_like(perturbed[:, i + 1 :])
    return perturbed


# --- Feed-forward and single layers -------------------------------------------------------------


def test_feed_forward_is_position_wise() -> None:
    torch.manual_seed(0)
    ff, x, t = FeedForward(D, D_FF, 0.0), inputs(), 4
    perturbed = x.clone()
    perturbed[:, t] += 1.0
    changed = (ff(x) - ff(perturbed)).abs().amax(dim=(0, 2)) > 0
    assert changed.nonzero().flatten().tolist() == [t]


def test_encoder_layer_is_post_norm() -> None:
    torch.manual_seed(0)
    layer, x = EncoderLayer(D, HEADS, D_FF, 0.0, causal=True, fused=False).eval(), inputs()
    h = layer.norm1(x + layer.self_attn(x, x, x)[0])
    expected = layer.norm2(h + layer.ff(h))
    out, weights = layer(x, need_weights=True)
    torch.testing.assert_close(out, expected)
    assert weights.shape == (B, HEADS, L, L)
    assert layer(x)[1] is None


@pytest.mark.parametrize("lag_aware", [False, True])
def test_decoder_layer_is_post_norm(lag_aware: bool) -> None:
    torch.manual_seed(0)
    layer = DecoderLayer(D, HEADS, D_FF, 0.0, lag_aware, 1.0, 0.5, 1e-6, -4.0, fused=False).eval()
    x, memory = inputs(1), inputs(2)
    h = layer.norm1(x + layer.self_attn(x, x, x)[0])
    if lag_aware:
        cross = layer.cross_attn(h, memory)[0]
    else:
        cross = layer.cross_attn(h, memory, memory)[0]
    h = layer.norm2(h + cross)
    expected = layer.norm3(h + layer.ff(h))
    torch.testing.assert_close(layer(x, memory)[0], expected)


# --- Builders -----------------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_build_encoder_follows_config(causal: bool) -> None:
    encoder = make_encoder(use_causal_encoder=causal, encoder_layers=3)
    assert len(encoder.layers) == 3
    assert all(layer.self_attn.causal is causal for layer in encoder.layers)


@pytest.mark.parametrize("lag_aware", [False, True])
@pytest.mark.parametrize("causal_encoder", [False, True])
def test_build_decoder_follows_config(lag_aware: bool, causal_encoder: bool) -> None:
    decoder = make_decoder(
        use_lag_aware_cross_attn=lag_aware, use_causal_encoder=causal_encoder, decoder_layers=3
    )
    assert len(decoder.layers) == 3
    for layer in decoder.layers:
        assert layer.self_attn.causal  # always causal, regardless of use_causal_encoder (D-017)
        expected = LagAwareAttention if lag_aware else MultiHeadAttention
        assert type(layer.cross_attn) is expected
        if not lag_aware:
            assert not layer.cross_attn.causal


def test_each_decoder_layer_has_its_own_tau_network() -> None:
    decoder = make_decoder(use_lag_aware_cross_attn=True)
    nets = [layer.cross_attn.lag for layer in decoder.layers]
    assert nets[0] is not nets[1]
    assert nets[0].fc1.weight.data_ptr() != nets[1].fc1.weight.data_ptr()


def test_builders_use_paper_dimensions() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig()
    encoder, decoder = build_encoder(cfg), build_decoder(cfg)
    assert len(encoder.layers) == len(decoder.layers) == 4
    ff = encoder.layers[0].ff.net
    assert (ff[0].in_features, ff[0].out_features) == (64, 256)
    assert decoder.layers[0].dropout.p == 0.1
    memory = encoder(torch.randn(1, 384, 64))[0]
    out, attention = decoder(torch.randn(1, 384, 64), memory)
    assert out.shape == (1, 384, 64)
    assert attention[0].tau.shape == (1, 384)


# --- Stacks: shapes and weights -----------------------------------------------------------------


def test_encoder_shapes_and_weights() -> None:
    encoder, x = make_encoder(), inputs()
    out, weights = encoder(x, need_weights=True)
    assert out.shape == (B, L, D)
    assert len(weights) == 2
    assert all(w.shape == (B, HEADS, L, L) for w in weights)
    assert encoder(x)[1] == [None, None]


@pytest.mark.parametrize("lag_aware", [False, True])
def test_decoder_shapes_and_weights(lag_aware: bool) -> None:
    decoder, x, memory = make_decoder(use_lag_aware_cross_attn=lag_aware), inputs(1), inputs(2)
    out, attention = decoder(x, memory, need_weights=True)
    assert out.shape == (B, L, D)
    assert len(attention) == 2
    for att in attention:
        assert att.self_attn.shape == att.cross_attn.shape == (B, HEADS, L, L)
        assert (att.tau.shape == (B, L)) if lag_aware else att.tau is None
    _, attention = decoder(x, memory)
    assert all(att.self_attn is None and att.cross_attn is None for att in attention)
    assert all((att.tau is not None) == lag_aware for att in attention)  # tau for MLflow logging


def test_standard_cross_attention_accepts_a_different_memory_length() -> None:
    decoder = make_decoder(use_lag_aware_cross_attn=False)
    out, attention = decoder(inputs(1), inputs(2, length=L + 4), need_weights=True)
    assert out.shape == (B, L, D)
    assert attention[0].cross_attn.shape == (B, HEADS, L, L + 4)


@pytest.mark.parametrize("lag_aware", [False, True])
def test_fused_matches_explicit(lag_aware: bool) -> None:
    x, memory = inputs(1), inputs(2)
    fused = make_encoder(fused_attention=True), make_decoder(use_lag_aware_cross_attn=lag_aware)
    explicit = (
        make_encoder(fused_attention=False),
        make_decoder(use_lag_aware_cross_attn=lag_aware, fused_attention=False),
    )
    outputs = []
    for encoder, decoder in (fused, explicit):
        mem = encoder(memory)[0]
        outputs.append(decoder(x, mem)[0])
    torch.testing.assert_close(outputs[0], outputs[1])
    # requesting weights forces the explicit path even when fused
    encoder, decoder = fused
    mem = encoder(memory, need_weights=True)[0]
    torch.testing.assert_close(decoder(x, mem, need_weights=True)[0], outputs[0])


# --- Causality (Sec. 2.2.2) ---------------------------------------------------------------------


def test_causal_encoder_ignores_future_inputs() -> None:
    encoder, x, i = make_encoder(use_causal_encoder=True), inputs(), 5
    out, out_perturbed = encoder(x)[0], encoder(perturb_after(x, i))[0]
    torch.testing.assert_close(out[:, : i + 1], out_perturbed[:, : i + 1])
    assert not torch.allclose(out[:, i + 1 :], out_perturbed[:, i + 1 :])


def test_non_causal_encoder_sees_the_future() -> None:
    encoder, x, i = make_encoder(use_causal_encoder=False), inputs(), 5
    assert not torch.allclose(
        encoder(x)[0][:, : i + 1], encoder(perturb_after(x, i))[0][:, : i + 1]
    )


@pytest.mark.parametrize("lag_aware", [False, True])
def test_decoder_ignores_future_decoder_inputs(lag_aware: bool) -> None:
    decoder, x, memory, i = (
        make_decoder(use_lag_aware_cross_attn=lag_aware),
        inputs(1),
        inputs(2),
        5,
    )
    out, out_perturbed = decoder(x, memory)[0], decoder(perturb_after(x, i), memory)[0]
    torch.testing.assert_close(out[:, : i + 1], out_perturbed[:, : i + 1])


def test_laam_decoder_ignores_future_memory_with_sharp_mask() -> None:
    # tau ~ 0 at init and T -> 0: position i only sees encoder steps j <= i (D-001)
    decoder = make_decoder(use_lag_aware_cross_attn=True, lag_temperature=1e-3)
    x, memory, i = inputs(1), inputs(2), 5
    out, out_perturbed = decoder(x, memory)[0], decoder(x, perturb_after(memory, i))[0]
    torch.testing.assert_close(out[:, : i + 1], out_perturbed[:, : i + 1])


def test_standard_cross_attention_sees_future_memory() -> None:
    decoder, x, memory, i = make_decoder(use_lag_aware_cross_attn=False), inputs(1), inputs(2), 5
    out, out_perturbed = decoder(x, memory)[0], decoder(x, perturb_after(memory, i))[0]
    assert not torch.allclose(out[:, : i + 1], out_perturbed[:, : i + 1])


def test_causal_encoder_and_laam_decoder_are_causal_end_to_end() -> None:
    # CLAAM without MSFM: perturbing both inputs after i leaves the decoder output up to i unchanged
    encoder = make_encoder(use_causal_encoder=True)
    decoder = make_decoder(use_lag_aware_cross_attn=True, lag_temperature=1e-3)
    x, meteo, i = inputs(1), inputs(2), 5

    def run(dec_in: torch.Tensor, enc_in: torch.Tensor) -> torch.Tensor:
        return decoder(dec_in, encoder(enc_in)[0])[0]

    out, out_perturbed = run(x, meteo), run(perturb_after(x, i), perturb_after(meteo, i))
    torch.testing.assert_close(out[:, : i + 1], out_perturbed[:, : i + 1])


# --- Training behaviour -------------------------------------------------------------------------


def test_dropout_only_in_training() -> None:
    decoder, x, memory = make_decoder(dropout=0.5), inputs(1), inputs(2)
    torch.testing.assert_close(decoder(x, memory)[0], decoder(x, memory)[0])
    decoder.train()
    assert not torch.allclose(decoder(x, memory)[0], decoder(x, memory)[0])


def test_gradients_reach_every_parameter() -> None:
    encoder, decoder = make_encoder(), make_decoder(use_lag_aware_cross_attn=True)
    encoder.train()
    decoder.train()
    out, _ = decoder(inputs(1), encoder(inputs(2))[0])
    out.square().mean().backward()
    for module in (encoder, decoder):
        for name, param in module.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
    tau_grad = decoder.layers[0].cross_attn.lag.fc2.weight.grad
    assert tau_grad.abs().sum() > 0


def test_runs_under_bf16_autocast() -> None:
    encoder, decoder = make_encoder(), make_decoder(use_lag_aware_cross_attn=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, attention = decoder(inputs(1), encoder(inputs(2))[0])
    assert out.shape == (B, L, D)
    assert torch.isfinite(out.float()).all()
    assert attention[0].tau is not None
