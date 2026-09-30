"""Result figures of the experiment grid (AGENTS.md §11), drawn from :mod:`clamf.results` tables.

Each run keeps the same color in every figure (identity follows the run, not its rank), and every
figure with more than one run has a legend. ``save`` writes them to ``reports/figures/``.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from clamf.results import GRID_RUNS

# Categorical slots in fixed order (validated for color-vision deficiencies as adjacent series):
# the main comparison (CLAMF-Former vs baseline) takes the first two.
RUN_COLORS = {
    "clamf": "#2a78d6",
    "vanilla": "#eb6834",
    "msfm": "#1baf7a",
    "claam": "#eda100",
    "laam": "#e87ba4",
    "cam": "#008300",
}
OBSERVED_COLOR = "#0b0b0b"
MUTED = "#898781"
_STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "font.family": "sans-serif",
    "figure.dpi": 110,
}
METRIC_LABELS = {
    "nse": "NSE (↑)",
    "kge": "KGE (↑)",
    "rmse": "RMSE, mm/h (↓)",
    "tpe": "TPE-2 % (↓)",
    "bias": "BIAS (→ 0)",
}
LEAD_LABELS = {
    "nse_median": "NSE mediano",
    "nse_mean": "NSE medio",
    "rmse_median": "RMSE mediano (mm/h)",
    "rmse_mean": "RMSE medio (mm/h)",
}


def use_style() -> None:
    """Apply the figure style (light surface, hairline grid, 2 px lines) to matplotlib."""
    plt.rcParams.update(_STYLE)


def save(fig: Figure, name: str, out_dir: str | Path = "reports/figures") -> Path:
    """Write ``fig`` as ``<out_dir>/<name>.png`` (200 dpi) and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def metric_boxplots(
    basins: Mapping[str, pd.DataFrame],
    metrics: Sequence[str] = ("nse", "kge", "rmse", "tpe", "bias"),
) -> Figure:
    """One box per run for each metric, over the basins where it is defined (D-003).

    Outliers are not drawn (a few near-dry basins reach NSE in the thousands negative), so the
    boxes stay readable; the per-basin tables keep every value.
    """
    names = _ordered(basins)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.6))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        data = [basins[n][metric].dropna().to_numpy() for n in names]
        boxes = ax.boxplot(
            data,
            tick_labels=names,
            showfliers=False,
            widths=0.6,
            patch_artist=True,
            medianprops={"color": OBSERVED_COLOR, "linewidth": 1.5},
            whiskerprops={"color": MUTED, "linewidth": 1},
            capprops={"color": MUTED, "linewidth": 1},
        )
        for patch, name in zip(boxes["boxes"], names):
            patch.set_facecolor(_wash(_color(name), 0.35))
            patch.set_edgecolor(_color(name))
            patch.set_linewidth(1.5)
        if metric == "bias":
            ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def lead_curves(
    leads: Mapping[str, pd.DataFrame], column: str = "nse_median", reference: str = "vanilla"
) -> Figure:
    """``column`` per lead hour for every run (left) and its difference to ``reference`` (right)."""
    names = _ordered(leads)
    fig, (ax_abs, ax_diff) = plt.subplots(1, 2, figsize=(12, 4))
    for name in names:
        values = leads[name][column]
        ax_abs.plot(values.index, values, color=_color(name), label=name)
        if name != reference:
            diff = values - leads[reference][column]
            ax_diff.plot(diff.index, diff, color=_color(name), label=name)
    ax_diff.axhline(0, color=_color(reference), linewidth=1.5, label=reference)
    label = LEAD_LABELS.get(column, column)
    ax_abs.set(
        title=f"{label} por hora de anticipación", xlabel="hora de anticipación", ylabel=label
    )
    ax_diff.set(
        title=f"diferencia con {reference}", xlabel="hora de anticipación", ylabel=f"Δ {label}"
    )
    for ax in (ax_abs, ax_diff):
        ax.set_xlim(1, max(v.index.max() for v in leads.values()))
    ax_abs.legend(ncols=2)
    ax_diff.legend(ncols=2)
    fig.tight_layout()
    return fig


def training_curves(
    train: pd.DataFrame, val: pd.DataFrame, best_epoch: Mapping[str, int], ylim: tuple | None = None
) -> Figure:
    """Small multiples, one per run: train and val loss per epoch and the best epoch."""
    names = _ordered(val)
    ncols = 3
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True, squeeze=False
    )
    for ax, name in zip(axes.flat, names):
        t, v = train[name].dropna(), val[name].dropna()
        ax.plot(t.index, t, color=MUTED, linewidth=1.5, label="train")
        ax.plot(v.index, v, color=_color(name), label="val")
        best = best_epoch[name]
        ax.plot(
            best,
            v[best],
            "o",
            color=_color(name),
            markersize=8,
            markeredgecolor="#fcfcfb",
            markeredgewidth=2,
            label=f"mejor epoch ({best})",
        )
        ax.set_title(name)
        ax.set_xlabel("epoch")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper right")
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("FreqMAE (normalizado)")
    fig.tight_layout()
    return fig


def tau_curves(tau: pd.DataFrame, best_epoch: Mapping[str, int]) -> Figure:
    """Mean LAAM lag ``τ`` (hours) per epoch for the runs that have it, with the best epoch."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name in _ordered(tau):
        values = tau[name].dropna()
        ax.plot(values.index, values, color=_color(name), label=name)
        best = best_epoch[name]
        ax.plot(
            best,
            values[best],
            "o",
            color=_color(name),
            markersize=8,
            markeredgecolor="#fcfcfb",
            markeredgewidth=2,
        )
    ax.set(title="τ medio en val por epoch (● = mejor epoch)", xlabel="epoch", ylabel="τ (horas)")
    ax.legend()
    fig.tight_layout()
    return fig


def hydrographs(
    predictions: Mapping[str, pd.DataFrame], row_ids: Sequence[int], titles: Sequence[str]
) -> Figure:
    """Observed and forecast discharge over the horizon of each window in ``row_ids``.

    ``predictions``: long tables of :meth:`clamf.results.Grid.predictions`, one per run.
    """
    names = _ordered(predictions)
    ncols = min(len(row_ids), 3)
    nrows = -(-len(row_ids) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    indexed = {n: p.set_index("row_id") for n, p in predictions.items()}
    for ax, row_id, title in zip(axes.flat, row_ids, titles):
        first = indexed[names[0]].loc[row_id]
        ax.plot(first["lead"], first["obs"], color=OBSERVED_COLOR, linewidth=2.5, label="observado")
        for name in names:
            window = indexed[name].loc[row_id]
            ax.plot(window["lead"], window["pred"], color=_color(name), label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("hora de anticipación")
    for ax in axes.flat[len(row_ids) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("caudal (mm/h)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def _color(name: str) -> str:
    """The run's color; runs outside the grid are drawn in the muted gray."""
    return RUN_COLORS.get(name, MUTED)


def _ordered(tables: Mapping[str, object] | pd.DataFrame) -> list[str]:
    """Run names in the grid order of :data:`~clamf.results.GRID_RUNS`, others last."""
    names = list(tables.columns if isinstance(tables, pd.DataFrame) else tables)
    return sorted(names, key=lambda n: GRID_RUNS.index(n) if n in GRID_RUNS else len(GRID_RUNS))


def _wash(color: str, alpha: float) -> tuple[float, float, float]:
    """``color`` blended over the light surface, so boxes stay opaque over the grid."""
    rgb = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)]) / 255
    surface = np.array([0xFC, 0xFC, 0xFB]) / 255
    return tuple(alpha * rgb + (1 - alpha) * surface)
