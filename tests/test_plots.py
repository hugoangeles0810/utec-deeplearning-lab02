from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from clamf import plots
from clamf.metrics import METRICS
from clamf.results import GRID_RUNS


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    plt.switch_backend("Agg")
    plots.use_style()
    yield
    plt.close("all")


def test_every_grid_run_has_its_own_color() -> None:
    assert set(plots.RUN_COLORS) == set(GRID_RUNS)
    assert len(set(plots.RUN_COLORS.values())) == len(GRID_RUNS)


def test_figures_draw_and_save(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    names = ["clamf", "vanilla", "dev"]  # dev is outside the grid: drawn in gray
    basins = {n: pd.DataFrame(rng.normal(size=(20, len(METRICS))), columns=METRICS) for n in names}
    leads = {
        n: pd.DataFrame({"nse_median": rng.random(48)}, index=pd.RangeIndex(1, 49, name="lead"))
        for n in names
    }
    history = pd.DataFrame(rng.random((5, 3)), columns=names)
    best = {n: 2 for n in names}
    preds = {
        n: pd.DataFrame(
            {
                "row_id": np.repeat([10, 11], 48),
                "lead": np.tile(np.arange(1, 49), 2),
                "obs": rng.random(96),
                "pred": rng.random(96),
            }
        )
        for n in ("clamf", "vanilla")
    }

    figures = {
        "boxplots": plots.metric_boxplots(basins),
        "lead": plots.lead_curves(leads, "nse_median", reference="vanilla"),
        "training": plots.training_curves(history, history + 1, best, ylim=(0, 3)),
        "tau": plots.tau_curves(history, best),
        "hydrographs": plots.hydrographs(preds, [10, 11], ["a", "b"]),
    }
    assert len(figures["boxplots"].axes) == len(METRICS)
    assert len([ax for ax in figures["training"].axes if ax.get_visible()]) == len(names)
    for name, fig in figures.items():
        path = plots.save(fig, name, tmp_path / "figures")
        assert path == tmp_path / "figures" / f"{name}.png"
        assert path.stat().st_size > 0
