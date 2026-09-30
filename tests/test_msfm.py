import pytest
import torch
import torch.nn.functional as F

from clamf.models.msfm import MSFM, ScaleExtraction

B, T, D_IN, D_FUSION, D_MODEL, HEADS = 2, 24, 3, 8, 12, 2
SCALES = (1, 4, 12)


def make_msfm(
    d_in: int = D_IN, fused: bool = True, dropout: float = 0.0, scales: tuple[int, ...] = SCALES
) -> MSFM:
    torch.manual_seed(0)
    return MSFM(
        d_in, D_FUSION, D_MODEL, scales, conv_kernel=3, n_heads=HEADS, dropout=dropout, fused=fused
    ).eval()


def inputs(seed: int = 1, d_in: int = D_IN, length: int = T) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, length, d_in)


# --- Scale extraction (Eq. 6-9) -----------------------------------------------------------------


@pytest.mark.parametrize("scale", SCALES)
def test_scale_extraction_shapes(scale: int) -> None:
    branch = ScaleExtraction(D_IN, D_FUSION, scale, conv_kernel=3)
    assert branch(inputs()).shape == (B, T // scale, D_FUSION)


def test_scale_one_is_conv_and_gelu_only() -> None:
    torch.manual_seed(0)
    branch, x = ScaleExtraction(D_IN, D_FUSION, 1, conv_kernel=3), inputs()
    expected = F.gelu(branch.conv(x.transpose(1, 2))).transpose(1, 2)
    torch.testing.assert_close(branch(x), expected)


def test_coarse_scale_is_block_max_of_the_hourly_features() -> None:
    torch.manual_seed(0)
    branch, x, k = ScaleExtraction(D_IN, D_FUSION, 4, conv_kernel=3), inputs(), 4
    hourly = F.gelu(branch.conv(x.transpose(1, 2))).transpose(1, 2)
    expected = hourly.view(B, T // k, k, D_FUSION).amax(dim=2)  # stride = k, no overlap (D-005)
    torch.testing.assert_close(branch(x), expected)


def test_conv_is_local_with_symmetric_zero_padding() -> None:
    # kernel 3, padding 1 (D-010): step t only reaches t - 1..t + 1, and nothing wraps around
    torch.manual_seed(0)
    branch, x, t = ScaleExtraction(D_IN, D_FUSION, 1, conv_kernel=3), inputs(), 10
    perturbed = x.clone()
    perturbed[:, t] += 1.0
    changed = (branch(x) - branch(perturbed)).abs().amax(dim=(0, 2)) > 0
    assert changed.nonzero().flatten().tolist() == [t - 1, t, t + 1]

    last = x.clone()
    last[:, -1] += 1.0
    torch.testing.assert_close(branch(x)[:, 0], branch(last)[:, 0])


def test_scale_extraction_rejects_indivisible_length() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ScaleExtraction(D_IN, D_FUSION, 5, conv_kernel=3)(inputs())


@pytest.mark.parametrize(("scale", "kernel"), [(0, 3), (2, 2), (2, 0)])
def test_scale_extraction_rejects_bad_arguments(scale: int, kernel: int) -> None:
    with pytest.raises(ValueError):
        ScaleExtraction(D_IN, D_FUSION, scale, conv_kernel=kernel)


# --- MSFM (Eq. 10-12) ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_in", [11, 1])  # encoder meteorology and decoder discharge (D-010)
def test_msfm_shapes_and_weights(d_in: int) -> None:
    msfm = make_msfm(d_in)
    out, weights = msfm(inputs(d_in=d_in), need_weights=True)
    assert out.shape == (B, T, D_MODEL)
    assert set(weights) == {4, 12}
    for k, w in weights.items():
        assert w.shape == (B, HEADS, T, T // k)
        torch.testing.assert_close(w.sum(-1), torch.ones(B, HEADS, T))
    assert msfm(inputs(d_in=d_in))[1] is None


def test_msfm_matches_eq_12() -> None:
    msfm, x = make_msfm(), inputs()
    fine = msfm.branches[0](x)
    fused = [
        attn(fine, branch(x), branch(x))[0]
        for branch, attn in zip(msfm.branches[1:], msfm.fusions, strict=True)
    ]
    expected = msfm.out(torch.cat([fine, *fused], dim=-1))
    torch.testing.assert_close(msfm(x)[0], expected)


def test_msfm_uses_one_linear_over_all_scales() -> None:
    msfm = make_msfm(scales=(1, 6))
    assert msfm.out.in_features == 2 * D_FUSION
    assert len(msfm.fusions) == 1
    assert msfm(inputs())[0].shape == (B, T, D_MODEL)


def test_msfm_shares_no_weights() -> None:
    msfm = make_msfm()
    params = [p for m in (*msfm.branches, *msfm.fusions) for p in m.parameters()]
    assert len({p.data_ptr() for p in params}) == len(params)


def test_fused_matches_explicit() -> None:
    x = inputs()
    fused, explicit = make_msfm(fused=True), make_msfm(fused=False)
    torch.testing.assert_close(fused(x)[0], explicit(x)[0])
    torch.testing.assert_close(fused(x)[0], fused(x, need_weights=True)[0])


def test_msfm_is_not_causal() -> None:
    # D-005: the literal MSFM lets position i see t > i (conv, pooling and unmasked fusion)
    msfm, x, i = make_msfm(), inputs(), 5
    perturbed = x.clone()
    perturbed[:, i + 2 :] += 1.0  # beyond the conv receptive field of positions <= i
    assert not torch.allclose(msfm(x)[0][:, : i + 1], msfm(perturbed)[0][:, : i + 1])


def test_msfm_rejects_scales_without_the_fine_branch() -> None:
    with pytest.raises(ValueError, match="start at 1"):
        make_msfm(scales=(4, 12))


def test_dropout_only_in_training() -> None:
    msfm, x = make_msfm(dropout=0.5), inputs()
    torch.testing.assert_close(msfm(x)[0], msfm(x)[0])
    msfm.train()
    assert not torch.allclose(msfm(x)[0], msfm(x)[0])


def test_gradients_reach_every_parameter() -> None:
    msfm = make_msfm().train()
    msfm(inputs())[0].square().mean().backward()
    for name, p in msfm.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
