import dataclasses
import itertools

import pytest
import torch
from torch import nn

from clamf.config import Config, ModelConfig
from clamf.data.dataset import RainfallRunoffDataset
from clamf.data.prepare import prepare
from clamf.losses import get_loss
from clamf.models.attention import LagAwareAttention, MultiHeadAttention
from clamf.models.clamf_former import CLAMFFormer, build_model
from clamf.models.msfm import MSFM
from tests.synthetic import HIST, HOR, RawData

B, HISTORY, HORIZON, N_METEO, D, HEADS = 2, 24, 6, 3, 16, 4
L = HISTORY + HORIZON
SCALES = (1, 5, 15)
FLAGS = ("use_msfm", "use_causal_encoder", "use_lag_aware_cross_attn")
ALL_FLAGS = [
    dict(zip(FLAGS, values, strict=True)) for values in itertools.product([False, True], repeat=3)
]


def tiny_config(**overrides: object) -> ModelConfig:
    base = {
        "d_model": D,
        "d_fusion": D,
        "n_heads": HEADS,
        "d_ff": 32,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.0,
        "msfm_scales": SCALES,
    }
    return ModelConfig(**{**base, **overrides})


def make_model(**overrides: object) -> CLAMFFormer:
    torch.manual_seed(0)
    return CLAMFFormer(tiny_config(**overrides), N_METEO, HISTORY, HORIZON).eval()


def inputs(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    enc_x = torch.randn(B, L, N_METEO)
    dec_x = torch.cat([torch.randn(B, HISTORY, 1), torch.zeros(B, HORIZON, 1)], dim=1)
    return enc_x, dec_x


def perturb_after(x: torch.Tensor, i: int) -> torch.Tensor:
    perturbed = x.clone()
    perturbed[:, i + 1 :] += torch.randn_like(perturbed[:, i + 1 :])
    return perturbed


def flag_id(flags: dict[str, bool]) -> str:
    return "-".join(name.removeprefix("use_") for name, on in flags.items() if on) or "vanilla"


# --- Assembly and shapes (Fig. 3) ---------------------------------------------------------------


@pytest.mark.parametrize("flags", ALL_FLAGS, ids=flag_id)
def test_components_follow_flags(flags: dict[str, bool]) -> None:
    model = make_model(**flags)
    for embedding in (model.enc_embedding, model.dec_embedding):
        expected = MSFM if flags["use_msfm"] else nn.Linear
        assert type(embedding.projection) is expected
    assert all(
        layer.self_attn.causal is flags["use_causal_encoder"] for layer in model.encoder.layers
    )
    cross = LagAwareAttention if flags["use_lag_aware_cross_attn"] else MultiHeadAttention
    assert all(type(layer.cross_attn) is cross for layer in model.decoder.layers)
    assert (model.head.in_features, model.head.out_features) == (D, 1)


@pytest.mark.parametrize("flags", ALL_FLAGS, ids=flag_id)
def test_output_shapes_and_attention(flags: dict[str, bool]) -> None:
    model, (enc_x, dec_x) = make_model(**flags), inputs()
    out = model(enc_x, dec_x, need_weights=True)
    assert out.pred.shape == (B, HORIZON)
    att = out.attention
    assert [w.shape for w in att.encoder] == [(B, HEADS, L, L)] * 2
    for layer in att.decoder:
        assert layer.self_attn.shape == layer.cross_attn.shape == (B, HEADS, L, L)
    lag_aware = flags["use_lag_aware_cross_attn"]
    assert (out.tau.shape == (2, B, L)) if lag_aware else out.tau is None
    for weights in (att.msfm_encoder, att.msfm_decoder):
        if flags["use_msfm"]:
            assert {k: w.shape for k, w in weights.items()} == {
                k: (B, HEADS, L, L // k) for k in SCALES[1:]
            }
        else:
            assert weights is None


@pytest.mark.parametrize("flags", ALL_FLAGS, ids=flag_id)
def test_weights_only_when_requested_but_tau_always(flags: dict[str, bool]) -> None:
    out = make_model(**flags)(*inputs())
    att = out.attention
    assert att.encoder == [None, None]
    assert att.msfm_encoder is None and att.msfm_decoder is None
    assert all(layer.self_attn is None and layer.cross_attn is None for layer in att.decoder)
    assert (out.tau is not None) == flags["use_lag_aware_cross_attn"]  # for MLflow (D-001)


def test_forecast_is_linear_head_on_last_horizon_positions() -> None:
    model, (enc_x, dec_x) = make_model(), inputs()
    memory = model.encoder(model.enc_embedding(enc_x)[0])[0]
    out = model.decoder(model.dec_embedding(dec_x)[0], memory)[0]
    expected = model.head(out[:, -HORIZON:]).squeeze(-1)
    torch.testing.assert_close(model(enc_x, dec_x).pred, expected)


def test_fused_matches_explicit() -> None:
    enc_x, dec_x = inputs()
    fused = make_model(fused_attention=True)(enc_x, dec_x).pred
    explicit = make_model(fused_attention=False)(enc_x, dec_x).pred
    torch.testing.assert_close(fused, explicit)
    # requesting the weights forces the explicit path even when fused (D-004)
    torch.testing.assert_close(make_model()(enc_x, dec_x, need_weights=True).pred, fused)


# --- No target leakage (D-007) ------------------------------------------------------------------


@pytest.mark.parametrize("flags", ALL_FLAGS, ids=flag_id)
def test_decoder_horizon_values_are_ignored(flags: dict[str, bool]) -> None:
    # even a non-causal MSFM cannot see a target written into the zero horizon of dec_x
    model, (enc_x, dec_x) = make_model(**flags), inputs()
    leaked = dec_x.clone()
    leaked[:, HISTORY:] = torch.randn(B, HORIZON, 1)
    torch.testing.assert_close(model(enc_x, dec_x).pred, model(enc_x, leaked).pred)


def test_changing_the_real_future_does_not_change_the_forecast(raw: RawData, cfg: Config) -> None:
    cfg = dataclasses.replace(cfg, model=tiny_config())
    torch.manual_seed(0)
    model = build_model(cfg, n_meteo=11).eval()

    def forecast() -> torch.Tensor:
        batch = RainfallRunoffDataset(prepare(cfg, force=True), "train", HIST, HOR).load_all("cpu")
        return model(batch["enc_x"], batch["dec_x"]).pred

    before = forecast()
    raw.y = raw.y * 3 + 1
    raw.write()
    torch.testing.assert_close(forecast(), before)


# --- Causality (Sec. 2.2.2; D-005) --------------------------------------------------------------


def test_claam_forecast_ignores_future_meteorology() -> None:
    # CAM + sharp LAAM, no MSFM: horizon step h only sees meteorology up to hour HISTORY + h
    model = make_model(use_msfm=False, lag_temperature=1e-3)
    (enc_x, dec_x), i = inputs(), HISTORY + 2
    pred = model(enc_x, dec_x).pred
    perturbed = model(perturb_after(enc_x, i), dec_x).pred
    torch.testing.assert_close(pred[:, :3], perturbed[:, :3])
    assert not torch.allclose(pred[:, 3:], perturbed[:, 3:])


@pytest.mark.parametrize(
    "flags",
    [
        {"use_msfm": False, "use_causal_encoder": False, "use_lag_aware_cross_attn": False},
        {"use_msfm": True, "use_causal_encoder": True, "use_lag_aware_cross_attn": True},
    ],
    ids=["vanilla", "msfm-is-not-causal"],
)
def test_forecast_sees_future_meteorology(flags: dict[str, bool]) -> None:
    model = make_model(**flags, lag_temperature=1e-3)
    (enc_x, dec_x), i = inputs(), HISTORY + 2
    pred = model(enc_x, dec_x).pred
    assert not torch.allclose(pred[:, :3], model(perturb_after(enc_x, i), dec_x).pred[:, :3])


# --- Training behaviour -------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [ALL_FLAGS[0], ALL_FLAGS[-1]], ids=flag_id)
def test_gradients_reach_every_parameter(flags: dict[str, bool]) -> None:
    model = make_model(**flags).train()
    enc_x, dec_x = inputs()
    get_loss("freqmae")(model(enc_x, dec_x).pred, torch.randn(B, HORIZON)).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_a_few_adam_steps_reduce_the_loss() -> None:
    model = make_model(dropout=0.1).train()
    enc_x, dec_x = inputs()
    target = torch.randn(B, HORIZON)
    loss_fn = get_loss("freqmae")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model(enc_x, dec_x).pred, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_runs_under_bf16_autocast() -> None:
    model, (enc_x, dec_x) = make_model(), inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(enc_x, dec_x)
    assert out.pred.shape == (B, HORIZON)
    assert torch.isfinite(out.pred.float()).all()
    assert out.tau is not None


# --- Construction and validation ----------------------------------------------------------------


def test_build_model_uses_paper_setup() -> None:
    torch.manual_seed(0)
    model = build_model(Config(), n_meteo=11).eval()
    assert (model.history_hours, model.horizon_hours) == (336, 48)
    assert len(model.encoder.layers) == len(model.decoder.layers) == 4
    out = model(torch.randn(1, 384, 11), torch.zeros(1, 384, 1))
    assert out.pred.shape == (1, 48)
    assert out.tau.shape == (4, 1, 384)


def test_rejects_scales_that_do_not_divide_the_sequence() -> None:
    with pytest.raises(ValueError, match="divide"):
        CLAMFFormer(tiny_config(msfm_scales=(1, 7)), N_METEO, HISTORY, HORIZON)
    CLAMFFormer(tiny_config(msfm_scales=(1, 7), use_msfm=False), N_METEO, HISTORY, HORIZON)


@pytest.mark.parametrize(
    ("enc_shape", "dec_shape", "match"),
    [
        ((B, L, N_METEO + 1), (B, L, 1), "enc_x"),
        ((B, L - 1, N_METEO), (B, L, 1), "enc_x"),
        ((B, L, N_METEO), (B, L, 2), "dec_x"),
        ((B, L, N_METEO), (B + 1, L, 1), "batch"),
    ],
)
def test_rejects_bad_input_shapes(
    enc_shape: tuple[int, ...], dec_shape: tuple[int, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        make_model()(torch.randn(enc_shape), torch.randn(dec_shape))
