"""Training losses: FreqMAE (paper Sec. 2.2.4, Eq. 13-15; D-008) plus MSE and MAE.

All losses take ``pred`` and ``target`` of shape ``(B, H)`` (the forecast horizon only, in the
per-basin normalized space of D-007) and return a scalar. They are computed in float32 even under
bf16 autocast (D-004).
"""

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

LossName = Literal["freqmae", "mse", "mae"]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def freqmae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """FreqMAE, Eq. (15): ``sum_k |Yhat_k - Y_k|`` per series, averaged over the batch.

    ``Y_k`` is the unnormalized full DFT of Eq. (14) along the last dim, with ``N = H`` (no
    zero-padding). By linearity, ``Yhat_k - Y_k`` is the DFT of the error ``pred - target``.
    Shapes: ``(B, H)``, ``(B, H)`` -> scalar.
    """
    spectrum = torch.fft.fft(pred - target, dim=-1)
    return spectrum.abs().sum(dim=-1).mean()


_LOSSES: dict[str, LossFn] = {"freqmae": freqmae, "mse": F.mse_loss, "mae": F.l1_loss}


def get_loss(name: LossName) -> LossFn:
    """Loss selected by ``train.loss``; it casts inputs to float32 and checks their shapes."""
    if name not in _LOSSES:
        raise ValueError(f"unknown loss {name!r}; expected one of {sorted(_LOSSES)}")
    fn = _LOSSES[name]

    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")
        return fn(pred.float(), target.float())

    return loss
