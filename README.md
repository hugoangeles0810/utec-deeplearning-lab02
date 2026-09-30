# CLAMF-Former · Laboratorio 2 de Deep Learning (UTEC)

Implementación en PyTorch de **CLAMF-Former** (Yuan & Yan, *Journal of Hydrology*, 2026), un Transformer
para predecir caudales que combina atención causal y *lag-aware* (CLAAM), fusión multi-escala (MSFM) y
la pérdida FreqMAE. Se entrena y evalúa sobre el dataset Rainfall-Runoff del curso: con 336 h de
historia se predicen las 48 h siguientes. Además del modelo completo, se entrenan las ablaciones del
paper y un Transformer vanilla como baseline.

## Inicio rápido

```bash
uv sync
uv run python -m clamf.data.download
uv run python -m clamf.data.prepare --config configs/base.yaml
uv run python -m clamf.train --config configs/experiments/dev.yaml   # corrida corta de desarrollo
uv run pytest
```

Todos los comandos (evaluación, grilla, MLflow, lint) están en [`AGENTS.md` §4](AGENTS.md#4-entorno-y-comandos).

## Documentación

| Documento | Contenido |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Guía del proyecto: alcance, estructura, comandos, reglas de implementación y flujo de trabajo |
| [`docs/paper.md`](docs/paper.md) | Resumen técnico del paper ([PDF](docs/paper.pdf)) |
| [`docs/data_preparation.md`](docs/data_preparation.md) | Dataset, splits, normalización y cache |
| [`docs/experiments.md`](docs/experiments.md) | Grilla de experimentos y su mapeo a las tablas del paper |
| [`docs/runpod.md`](docs/runpod.md) | Entrenar la grilla en RunPod y bajar los resultados |
| [`docs/decisions.md`](docs/decisions.md) | Registro de decisiones de implementación |
