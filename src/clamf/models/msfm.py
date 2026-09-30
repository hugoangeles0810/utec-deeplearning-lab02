"""Multi-Scale Fusion Module, MSFM (paper Sec. 2.2.3, Eq. 6-12, Fig. 5; D-005, D-010).

Literal and non-causal (D-005): a symmetric-padding Conv1D, a MaxPool with stride = ``k`` and an
unmasked fusion attention, so position ``i`` sees inputs at ``t > i``. Shapes follow
``(B, T, features)``; every scale ``k`` must divide ``T``.
"""

import torch
import torch.nn.functional as F
from torch import nn

from clamf.models.attention import MultiHeadAttention


class ScaleExtraction(nn.Module):
    """One branch of Eq. (6)-(9): ``MaxPool_k(GELU(Conv1D_{d -> d_fusion}(X)))``.

    The Conv1D uses an odd kernel with symmetric zero padding (keeps ``T``, D-010); the MaxPool has
    kernel = stride = ``k`` and is the identity for ``k = 1`` (D-005).
    """

    def __init__(self, d_in: int, d_fusion: int, scale: int, conv_kernel: int) -> None:
        super().__init__()
        if conv_kernel <= 0 or conv_kernel % 2 == 0:
            raise ValueError(f"conv_kernel must be a positive odd integer, got {conv_kernel}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.scale = scale
        self.conv = nn.Conv1d(d_in, d_fusion, conv_kernel, padding=conv_kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_in)`` -> ``(B, T / k, d_fusion)``."""
        if x.shape[1] % self.scale:
            raise ValueError(f"sequence length {x.shape[1]} not divisible by scale {self.scale}")
        h = F.gelu(self.conv(x.transpose(1, 2)))
        if self.scale > 1:
            h = F.max_pool1d(h, kernel_size=self.scale)
        return h.transpose(1, 2)


class MSFM(nn.Module):
    """Multi-scale decomposition, Eq. (6)-(9), and fusion, Eq. (10)-(12).

    ``scales[0]`` must be 1: that branch is ``F_daily`` (hourly here, D-002) and queries every
    coarser branch with its own multi-head attention (no mask, residual or LayerNorm; D-010).
    ``Output = Linear(Concat[F_1, F^_k2, F^_k3, ...])`` maps ``len(scales) * d_fusion -> d_model``.
    No weights are shared between branches, fusion attentions or MSFM instances (D-010).
    """

    def __init__(
        self,
        d_in: int,
        d_fusion: int,
        d_model: int,
        scales: tuple[int, ...],
        conv_kernel: int,
        n_heads: int,
        dropout: float,
        fused: bool,
    ) -> None:
        super().__init__()
        if not scales or scales[0] != 1:
            raise ValueError(f"scales must start at 1, got {scales}")
        self.scales = tuple(scales)
        self.branches = nn.ModuleList(
            ScaleExtraction(d_in, d_fusion, k, conv_kernel) for k in self.scales
        )
        self.fusions = nn.ModuleList(
            MultiHeadAttention(d_fusion, n_heads, dropout, causal=False, fused=fused)
            for _ in self.scales[1:]
        )
        self.out = nn.Linear(len(self.scales) * d_fusion, d_model)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None]:
        """``(B, T, d_in)`` -> ``(B, T, d_model)`` and, if requested, the fusion weights.

        The weights map each coarse scale ``k`` to ``(B, n_heads, T, T / k)``; ``None`` otherwise.
        """
        fine, *coarse = (branch(x) for branch in self.branches)  # Eq. (6)-(9)
        fused_scales = [fine]
        weights: dict[int, torch.Tensor] = {}
        for k, feats, attn in zip(self.scales[1:], coarse, self.fusions, strict=True):
            out, w = attn(fine, feats, feats, need_weights=need_weights)  # Eq. (10)-(11)
            fused_scales.append(out)
            if w is not None:
                weights[k] = w
        return self.out(torch.cat(fused_scales, dim=-1)), weights if need_weights else None  # (12)
