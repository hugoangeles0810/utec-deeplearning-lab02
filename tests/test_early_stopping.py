import math

from clamf.config import TrainConfig
from clamf.utils.early_stopping import EarlyStopping


def test_stops_after_patience_epochs_without_improvement() -> None:
    stopper = EarlyStopping(patience=2)
    assert [stopper.step(loss, e) for e, loss in enumerate([3.0, 2.0, 2.5, 2.0])] == [
        True,
        True,
        False,
        False,  # equal to the best is not an improvement (strict, D-012)
    ]
    assert (stopper.best_loss, stopper.best_epoch, stopper.bad_epochs) == (2.0, 1, 2)
    assert stopper.should_stop


def test_improvement_resets_patience() -> None:
    stopper = EarlyStopping(patience=2)
    for epoch, loss in enumerate([3.0, 4.0, 1.0]):
        stopper.step(loss, epoch)
    assert (stopper.best_epoch, stopper.bad_epochs) == (2, 0)
    assert not stopper.should_stop


def test_min_delta_requires_a_larger_drop() -> None:
    stopper = EarlyStopping(patience=5, min_delta=0.5)
    assert stopper.step(2.0, 0)
    assert not stopper.step(1.6, 1)
    assert stopper.step(1.4, 2)


def test_nan_loss_never_improves() -> None:
    stopper = EarlyStopping(patience=1)
    assert not stopper.step(math.nan, 0)
    assert stopper.best_epoch == -1
    assert stopper.should_stop


def test_state_round_trip_and_config() -> None:
    stopper = EarlyStopping.from_config(
        TrainConfig(early_stopping_patience=7, early_stopping_min_delta=0.1)
    )
    stopper.step(1.0, 0)
    stopper.step(2.0, 1)
    restored = EarlyStopping(patience=1)
    restored.load_state_dict(stopper.state_dict())
    assert restored == stopper
    assert (restored.patience, restored.min_delta) == (7, 0.1)
