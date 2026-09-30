"""Early stopping on the validation loss (paper Sec. 3.2; D-004, D-012)."""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from clamf.config import TrainConfig


@dataclass
class EarlyStopping:
    """Tracks the best val loss and the epochs without improvement.

    An epoch improves when its loss is strictly lower than ``best_loss - min_delta`` (D-012); a NaN
    loss never improves. Training stops after ``patience`` epochs in a row without improvement.
    """

    patience: int
    min_delta: float = 0.0
    best_loss: float = math.inf
    best_epoch: int = -1  # -1 until the first improvement
    bad_epochs: int = 0

    @classmethod
    def from_config(cls, cfg: TrainConfig) -> "EarlyStopping":
        return cls(cfg.early_stopping_patience, cfg.early_stopping_min_delta)

    def step(self, loss: float, epoch: int) -> bool:
        """Record the val loss of ``epoch``; return whether it is the new best."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.bad_epochs = loss, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
