import math

import pytest
import torch
import torch.nn.functional as F

from clamf.models.attention import (
    LagAwareAttention,
    LagPredictor,
    MultiHeadAttention,
    aggregate_keys,
    causal_bias,
    lag_aware_bias,
    lag_aware_binary_mask,
)

B, L, D, HEADS = 2, 12, 16, 4


def make_mha(causal: bool, fused: bool = True, dropout: float = 0.0) -> MultiHeadAttention:
    torch.manual_seed(0)
    return MultiHeadAttention(D, HEADS, dropout, causal=causal, fused=fused).eval()


def make_laam(fused: bool = True, temperature: float = 1.0) -> LagAwareAttention:
    torch.manual_seed(0)
    return LagAwareAttention(
        D,
        HEADS,
        0.0,
        temperature=temperature,
        margin=0.5,
        eps=1e-6,
        tau_init_bias=-4.0,
        fused=fused,
    ).eval()


def inputs(seed: int = 1, length: int = L) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, length, D)


# --- MultiHeadAttention -------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_mha_shapes_and_weights(causal: bool) -> None:
    x = inputs()
    out, weights = make_mha(causal)(x, x, x, need_weights=True)
    assert out.shape == (B, L, D)
    assert weights.shape == (B, HEADS, L, L)
    torch.testing.assert_close(weights.sum(-1), torch.ones(B, HEADS, L))
    assert make_mha(causal)(x, x, x)[1] is None


def test_cross_attention_with_different_lengths() -> None:
    q, kv = inputs(1, length=5), inputs(2, length=9)
    out, weights = make_mha(causal=False)(q, kv, kv, need_weights=True)
    assert out.shape == (B, 5, D)
    assert weights.shape == (B, HEADS, 5, 9)


def test_rejects_d_model_not_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(10, 4, 0.0, causal=False, fused=True)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_matches_explicit(causal: bool, with_bias: bool) -> None:
    x = inputs()
    bias = torch.randn(B, 1, L, L) if with_bias else None
    fused = make_mha(causal, fused=True)
    explicit = make_mha(causal, fused=False)
    torch.testing.assert_close(fused(x, x, x, bias=bias)[0], explicit(x, x, x, bias=bias)[0])
    # requesting weights forces the explicit path even when fused
    torch.testing.assert_close(fused(x, x, x, bias=bias)[0], fused(x, x, x, bias, True)[0])


def test_matches_manual_single_head() -> None:
    torch.manual_seed(0)
    mha = MultiHeadAttention(4, 1, 0.0, causal=True, fused=False).eval()
    x = torch.randn(1, 3, 4)
    q, k, v = mha.w_q(x), mha.w_k(x), mha.w_v(x)
    logits = q @ k.transpose(-2, -1) / 2.0 + causal_bias(3, 3, x.device)
    expected = mha.w_o(logits.softmax(-1) @ v)
    torch.testing.assert_close(mha(x, x, x)[0], expected)


def test_causal_weights_are_lower_triangular() -> None:
    x = inputs()
    _, weights = make_mha(causal=True)(x, x, x, need_weights=True)
    assert torch.all(weights.triu(diagonal=1) == 0)


@pytest.mark.parametrize("fused", [False, True])
def test_cam_ignores_future_inputs(fused: bool) -> None:
    mha, x, i = make_mha(causal=True, fused=fused), inputs(), 5
    perturbed = x.clone()
    perturbed[:, i + 1 :] += torch.randn_like(perturbed[:, i + 1 :])
    out, out_perturbed = mha(x, x, x)[0], mha(perturbed, perturbed, perturbed)[0]
    torch.testing.assert_close(out[:, : i + 1], out_perturbed[:, : i + 1])
    assert not torch.allclose(out[:, i + 1 :], out_perturbed[:, i + 1 :])


def test_non_causal_attention_sees_the_future() -> None:
    mha, x, i = make_mha(causal=False), inputs(), 5
    perturbed = x.clone()
    perturbed[:, i + 1 :] += 1.0
    assert not torch.allclose(
        mha(x, x, x)[0][:, : i + 1], mha(perturbed, perturbed, perturbed)[0][:, : i + 1]
    )


def test_dropout_only_in_training() -> None:
    x = inputs()
    mha = make_mha(causal=False, dropout=0.5)
    torch.testing.assert_close(mha(x, x, x)[0], mha(x, x, x)[0])
    mha.train()
    assert not torch.allclose(mha(x, x, x)[0], mha(x, x, x)[0])


def test_causal_bias() -> None:
    bias = causal_bias(3, 3, torch.device("cpu"))
    inf = float("-inf")
    assert bias.tolist() == [[0.0, inf, inf], [0.0, 0.0, inf], [0.0, 0.0, 0.0]]


# --- Lag predictor (Eq. 2-4) --------------------------------------------------------------------


def test_aggregate_keys_is_cumulative_mean() -> None:
    key = torch.tensor([[[2.0], [4.0], [6.0]]])
    expected = torch.tensor([[[2.0 / (1 + 1e-6)], [6.0 / (2 + 1e-6)], [12.0 / (3 + 1e-6)]]])
    torch.testing.assert_close(aggregate_keys(key, eps=1e-6), expected)


def test_lag_predictor_matches_eq_4() -> None:
    torch.manual_seed(0)
    net = LagPredictor(D, eps=1e-6, tau_init_bias=-4.0)
    q, k = inputs(1), inputs(2)
    z = torch.cat([q, aggregate_keys(k, 1e-6)], dim=-1)
    expected = F.softplus(net.fc2(net.norm(F.relu(net.fc1(z) + z)))).squeeze(-1)
    tau = net(q, k)
    assert tau.shape == (B, L)
    torch.testing.assert_close(tau, expected)


def test_tau_is_positive_and_near_zero_at_init() -> None:
    torch.manual_seed(0)
    tau = LagPredictor(D, eps=1e-6, tau_init_bias=-4.0)(inputs(1), inputs(2))
    assert torch.all(tau > 0)
    assert tau.median().item() < 0.05  # Softplus(-4) ~ 0.018 (D-009)
    assert tau.max().item() < 0.5


def test_tau_only_depends_on_past_keys() -> None:
    torch.manual_seed(0)
    net, q, k, i = LagPredictor(D, eps=1e-6, tau_init_bias=-4.0), inputs(1), inputs(2), 5
    perturbed = k.clone()
    perturbed[:, i + 1 :] += 1.0
    torch.testing.assert_close(net(q, k)[:, : i + 1], net(q, perturbed)[:, : i + 1])


def test_lag_predictor_rejects_misaligned_sequences() -> None:
    net = LagPredictor(D, eps=1e-6, tau_init_bias=-4.0)
    with pytest.raises(ValueError, match="aligned"):
        net(inputs(1, length=5), inputs(2, length=6))


# --- Soft lag-aware mask (Eq. 5, D-001) ---------------------------------------------------------


def test_lag_aware_bias_values() -> None:
    tau = torch.tensor([[0.0, 1.5]])
    bias = lag_aware_bias(tau, n_keys=3, temperature=2.0, margin=0.5)
    assert bias.shape == (1, 1, 2, 3)
    for i in range(2):
        for j in range(3):
            edge = (i + tau[0, i].item() + 0.5 - j) / 2.0
            assert bias[0, 0, i, j].item() == pytest.approx(math.log(1 / (1 + math.exp(-edge))))


def test_lag_aware_bias_approaches_binary_mask_as_temperature_vanishes() -> None:
    tau = torch.tensor([[1.0, 0.0, 2.0, 1.0, 0.0]])
    bias = lag_aware_bias(tau, n_keys=5, temperature=1e-3, margin=0.5)[:, 0]
    visible = lag_aware_binary_mask(tau, n_keys=5)
    assert torch.all(bias[visible] > -1e-6)
    assert torch.all(bias[~visible] < -100)


def test_binary_mask_matches_paper_example() -> None:
    # docs/paper.md §4: tau = (1, 0, 2, 1, 0) with T = 5
    mask = lag_aware_binary_mask(torch.tensor([[1.0, 0.0, 2.0, 1.0, 0.0]]), n_keys=5)
    expected = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    assert mask[0].int().tolist() == expected


# --- LAAM ---------------------------------------------------------------------------------------


def test_laam_shapes() -> None:
    q, memory = inputs(1), inputs(2)
    out, weights, tau = make_laam()(q, memory, need_weights=True)
    assert out.shape == (B, L, D)
    assert weights.shape == (B, HEADS, L, L)
    assert tau.shape == (B, L)
    assert make_laam()(q, memory)[1] is None


def test_laam_fused_matches_explicit_with_tau_gradients() -> None:
    q, memory = inputs(1), inputs(2)
    grads = []
    for fused in (True, False):
        laam = make_laam(fused=fused).train()
        out, _, _ = laam(q, memory)
        out.square().mean().backward()
        grad = laam.lag.fc2.weight.grad
        assert grad is not None
        grads.append((out.detach(), grad.clone()))
    torch.testing.assert_close(grads[0][0], grads[1][0])
    torch.testing.assert_close(grads[0][1], grads[1][1])


def test_tau_receives_finite_nonzero_gradient() -> None:
    laam = make_laam().train()
    out, _, _ = laam(inputs(1), inputs(2))
    out.square().mean().backward()
    grad = laam.lag.fc2.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_laam_weights_follow_the_soft_mask() -> None:
    # with a tiny temperature and tau ~ 0 the LAAM behaves as causal cross-attention (D-001)
    _, weights, tau = make_laam(temperature=1e-3)(inputs(1), inputs(2), need_weights=True)
    assert torch.all(tau < 0.5)
    assert weights.triu(diagonal=1).max().item() < 1e-6


def test_laam_query_ignores_future_memory_beyond_its_lag() -> None:
    laam, q, memory, i = make_laam(temperature=1e-3), inputs(1), inputs(2), 5
    perturbed = memory.clone()
    perturbed[:, i + 1 :] += 1.0
    torch.testing.assert_close(laam(q, memory)[0][:, : i + 1], laam(q, perturbed)[0][:, : i + 1])
