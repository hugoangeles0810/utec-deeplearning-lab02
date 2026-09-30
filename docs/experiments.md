# Experimentos

La grilla de entrenamiento del laboratorio: CLAMF-Former, las ablaciones de las Tablas 5 y 6 del paper y
el baseline Transformer vanilla. La decisión de agrupar las variantes así está en
[D-018](decisions.md#d-018--grilla-de-experimentos-sin-modelos-repetidos).

## La grilla

Cada YAML de [`configs/experiments/`](../configs/experiments/) es una **combinación distinta** de los tres
flags del modelo. Cambia solo los flags y `logging`; datos, hiperparámetros, pérdida (FreqMAE) y
presupuesto (200 epochs, paciencia 20, batch 256, seed 2025) salen de [`base.yaml`](../configs/base.yaml)
y son los mismos para todos (D-004).

| YAML (`run_name`) | MSFM | CAM | LAAM | Qué mide | Filas del paper |
|---|---|---|---|---|---|
| `vanilla` | ✗ | ✗ | ✗ | Baseline: Transformer vanilla | CLAMF-1 (Tabla 5), CLAAM-1 (Tabla 6) |
| `cam` | ✗ | ✓ | ✗ | Solo atención causal en el encoder | CLAAM-2 (Tabla 6) |
| `laam` | ✗ | ✗ | ✓ | Solo la cross-attention lag-aware | CLAAM-3 (Tabla 6) |
| `claam` | ✗ | ✓ | ✓ | CLAAM completo, sin MSFM | CLAMF-3 (Tabla 5), CLAAM (Tabla 6) |
| `msfm` | ✓ | ✗ | ✗ | Solo MSFM | CLAMF-2 (Tabla 5) |
| `clamf` | ✓ | ✓ | ✓ | Modelo completo | CLAMF (Tabla 5) |

- **MSFM** = `model.use_msfm`, **CAM** = `model.use_causal_encoder`, **LAAM** =
  `model.use_lag_aware_cross_attn`. Sin MSFM, la entrada pasa por la proyección lineal de D-010; sin
  CAM, la self-attention del encoder no es causal (la del decoder siempre lo es, D-017); sin LAAM, la
  cross-attention es la estándar sin máscara.
- No se entrenan "MSFM + CAM" ni "MSFM + LAAM": el paper no las evalúa y ninguna tabla las usa.
- `dev.yaml` no es parte de la grilla: el modelo completo en el experimento `dev`, para corridas
  cortas de desarrollo en la Mac. Hace 3 epochs de solo 50 batches de train cada una
  (`train.max_batches_per_epoch`, ~5 % de train; val se recorre completo), así prueba el pipeline de
  punta a punta (MLflow, checkpoints, `--resume`, `clamf.evaluate`) en pocos minutos. En la grilla
  `max_batches_per_epoch` es `0` (epoch completa) y el test de la grilla verifica que no cambie.

## Cómo se arman las tablas

Las tablas se generan a partir de los runs de MLflow (AGENTS.md §6): cada fila se lee del run con el
`run_name` que indica esta tabla. Un mismo run aparece en varias tablas:

| Tabla | Filas (en orden) |
|---|---|
| Tabla 5 · MSFM × CLAAM | `vanilla` (CLAMF-1), `msfm` (CLAMF-2), `claam` (CLAMF-3), `clamf` (CLAMF) |
| Tabla 6 · CAM × LAAM, sin MSFM | `vanilla` (CLAAM-1), `cam` (CLAAM-2), `laam` (CLAAM-3), `claam` (CLAAM) |
| CLAMF-Former vs baseline | `clamf`, `vanilla` |

Así las filas repetidas del paper (CLAMF-1 = CLAAM-1 y CLAMF-3 = CLAAM) muestran siempre los mismos
números.

## Cómo se lanzan

Los 6 runs van al experimento de MLflow `clamf-grid` y cada uno se llama como su YAML.

La grilla completa se lanza en RunPod con `clamf.grid`, que entrena, reanuda y evalúa en val cada
run según haga falta. Los pasos del pod y cómo bajar los resultados a la Mac están en
[`runpod.md`](runpod.md).

```bash
uv run python -m clamf.grid                                            # los 6 runs, en orden
uv run python -m clamf.grid --configs configs/experiments/dev.yaml     # solo algunos YAML
uv run python -m clamf.train --config configs/experiments/clamf.yaml
uv run python -m clamf.train --config configs/experiments/clamf.yaml --resume <mlflow run id>
uv run python -m clamf.train --config configs/experiments/dev.yaml     # prueba corta
uv run python -m clamf.evaluate --config configs/experiments/clamf.yaml --run-id <mlflow run id>
```

- `clamf.evaluate` evalúa el mejor checkpoint del run en val (provisional, D-003) y registra las
  métricas, las métricas por hora de anticipación y las predicciones **en el mismo run**, con el
  prefijo `val/`. Con `--split test` hace lo mismo sobre test cuando tenga `y_aux` (D-013).
  Detalle de lo que registra en D-003.

- Un run caído se reanuda con `--resume` y el **mismo YAML**, sin lanzar uno nuevo (D-004).
- Si una variante se vuelve a entrenar desde cero (por ejemplo, tras corregir un bug), el run anterior
  se borra o se renombra en MLflow para que quede **un solo run por `run_name`** en `clamf-grid`.
- La grilla se lanza en el RTX 4090 sin esperar a D-013 (equipo, 2026-09-30) y se reporta sobre val
  de forma provisional. Si D-013 termina en la opción b, se re-entrena toda la grilla.
- En la Mac, los resultados bajados del pod se ven con
  `uv run mlflow ui --backend-store-uri sqlite:///results/runpod/mlflow.db`, y las tablas se generan
  desde ese store.
