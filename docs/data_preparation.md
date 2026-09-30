# Preparación de datos

Cómo se preparan los datos del dataset Rainfall-Runoff para entrenar y evaluar CLAMF-Former. El EDA
está en [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) y las decisiones, en
[`decisions.md`](decisions.md).

## 1. Datos de partida

| Archivo | Clave | Shape | Contenido |
|---|---|---|---|
| `train.h5` | `X` | `(N, 336, 12)` | 336 h de historia: 11 canales meteorológicos + caudal (canal 11) |
| | `y` | `(N, 48)` | Caudal de las 48 h siguientes (target) |
| | `y_aux` | `(N, 48, 11)` | Meteorología de las 48 h siguientes |
| | `split` | `(N,)` | 0 = train (254 000), 1 = val (18 142) |
| | `basin_id` | `(N,)` | Cuenca (508, todas presentes en los tres splits) |
| `test.h5` | `X`, `basin_id` | `(27 983, 336, 12)` | Sin `y` ni `y_aux` |
| `test_targets.csv` | `Id, q_01..q_48` | `(27 983, 49)` | Target de test, alineado por fila con `test.h5` |

Los datos vienen **limpios**: no hay NaN, infinitos ni centinelas en ningún split.

## 2. Qué entra al modelo

| Tensor | Shape | Construcción |
|---|---|---|
| `enc_x` | `(B, 384, 11)` | Meteorología: `X[..., :11]` (336 h) seguida de `y_aux` (48 h) |
| `dec_x` | `(B, 384, 1)` | Caudal: `X[..., 11]` (336 h) seguido de 48 ceros |
| `target` | `(B, 48)` | `y`; solo se usa en la pérdida |
| `basin_id` | `(B,)` | Para normalizar y des-normalizar el caudal |

- El encoder recibe `y_aux`, como en el paper (D-006, decisión del equipo). Es la única excepción a
  `metadata.json`.
- Covariables conocidas en el horizonte: los 11 canales meteorológicos. El caudal del horizonte no se
  conoce (ceros). `basin_id` no es entrada del modelo y no hay atributos estáticos ni variables de
  calendario (D-006). Todos los modelos reciben las mismas entradas.
- `y` **nunca** entra al modelo.
- La predicción son las 48 últimas posiciones del decoder.

## 3. Splits

- Se usan los splits dados: train (`split == 0`), val (`split == 1`) y test (`test.h5`). Las ventanas
  de val y test no comparten ninguna hora con las de train.
- Las ventanas de train se usan tal cual, sin re-muestrear (D-014). Todas las de una cuenca empiezan
  a la misma hora del día, mientras que las de val/test empiezan a cualquier hora; se acepta ese
  desfase.
- **Test está en espera** (D-013): falta el `y_aux` de test y se le pidió al profesor. No hay fecha
  límite: se espera su respuesta. Mientras tanto:
  - no se mira `test_targets.csv`;
  - early stopping y selección de modelo se hacen con val, pero val no se reporta como resultado
    final;
  - las ablaciones y el baseline no se lanzan hasta tener respuesta;
  - si el profesor dice que no, se pasa a entrenar con dropout de la meteorología futura (opción b).

## 4. Normalización (D-007)

Todas las estadísticas se calculan **solo con train** (`split == 0`).

| Qué | Cómo | Por qué |
|---|---|---|
| Meteorología (11 canales) | z-score **global por canal** | Las escalas son muy distintas (`pressure` ~10⁵, `specific_humidity` ~10⁻²). Global porque `pressure` codifica la altitud de la cuenca y conviene conservarla |
| Caudal (`dec_x` y `target`) | z-score **por cuenca**: `(q − μ_b) / σ_b` | El caudal medio cambia ~70× entre cuencas; así cada cuenca pesa parecido en la pérdida, como en el NSE por cuenca. Conserva los ceros (~3 %) y la escala lineal |

- `σ_b` lleva un mínimo (`discharge_std_floor`, 0.001 mm/h) para no dividir por un valor cercano a
  cero en cuencas casi secas (D-015).
- Los 48 ceros del horizonte del decoder se ponen en espacio normalizado, así que equivalen a la media
  de la cuenca.
- Las predicciones se **des-normalizan** antes de calcular las métricas, que se reportan en mm/h.
- Los scalers se guardan como JSON y se registran como artifact en MLflow.
- `log(q + ε)` para el caudal queda como experimento opcional.

## 5. Valores faltantes

No hay, pero al cargar se verifica que todo sea finito y que claves, shapes y conteos coincidan con
`metadata.json`. Si algo falla, error explícito: nunca se imputa en silencio.

## 6. Pipeline

Código en `src/clamf/data/`, parámetros en la sección `data:` de la config (`configs/base.yaml`).

1. **Preparación (una vez)**: `uv run python -m clamf.data.prepare --config configs/base.yaml`
   (~1 min, ~5 GB). Hace esto:
   - valida los `.h5` y `test_targets.csv` (§5, `io.py`);
   - ajusta los scalers con train (§4, `scalers.py`);
   - escribe en `data/processed/` los arrays ya normalizados por split, junto con `scalers.json` y
     `manifest.json`;
   - comprueba que val y test no compartan horas con train.

   Si cambian los raw o la normalización, el cache queda obsoleto y hay que correrlo con `--force`
   (D-015).
2. **Datasets** (`dataset.py`): `RainfallRunoffDataset` abre el cache con memmap y devuelve los
   tensores de §2, más `row_id`. Test solo está disponible si trae `y_aux`; si no, da
   `MissingFutureMeteoError` (D-013).
3. **Loaders**: `build_dataloaders(cfg, device)` arma train (con shuffle sembrado, seed 2025), val y,
   si hay `y_aux`, test.
   - Con `preload_to_device: true` (default, D-004), cada split se carga **una vez** en el dispositivo
     (~5 s y 4.7 GB para train y val) y `DeviceLoader` arma cada batch con un solo indexado.
     Train se baraja con una permutación nueva por epoch, sacada de un generador con seed;
     guardar su estado alcanza para reanudar con el mismo orden.
   - Con `preload_to_device: false`, un `DataLoader` lee los memmaps y los batches quedan en CPU.
4. **Des-normalización**: `load_scalers(...).denormalize_q(pred, basin_id)` antes de las métricas.

| Clave (`data:`) | Default | Qué controla |
|---|---|---|
| `raw_dir`, `processed_dir` | `data/raw`, `data/processed` | Rutas de entrada y del cache |
| `history_hours` | 336 | Últimas horas de historia que entran al modelo (≤ 336) |
| `horizon_hours` | 48 | Horizonte de predicción (≤ 48) |
| `normalization.meteo` / `.discharge` | `global_zscore` / `basin_zscore` | Esquema de §4 |
| `normalization.discharge_std_floor` | 0.001 | Mínimo de `σ_b` en mm/h |
| `batch_size`, `eval_batch_size` | 256, 512 | Tamaño de batch de train y de val/test |
| `num_workers` | 0 | Workers del `DataLoader` (solo sin precarga) |
| `preload_to_device` | `true` | Cargar cada split en el dispositivo y armar los batches sin `DataLoader` (D-004) |
