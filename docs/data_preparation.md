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
  - val no se reporta como resultado final;
  - las ablaciones y el baseline no se lanzan hasta tener respuesta;
  - si el profesor dice que no, se pasa a entrenar con dropout de la meteorología futura (opción b).

## 4. Normalización (D-007)

Todas las estadísticas se calculan **solo con train** (`split == 0`).

| Qué | Cómo | Por qué |
|---|---|---|
| Meteorología (11 canales) | z-score **global por canal** | Las escalas son muy distintas (`pressure` ~10⁵, `specific_humidity` ~10⁻²). Global porque `pressure` codifica la altitud de la cuenca y conviene conservarla |
| Caudal (`dec_x` y `target`) | z-score **por cuenca**: `(q − μ_b) / σ_b` | El caudal medio cambia ~70× entre cuencas; así cada cuenca pesa parecido en la pérdida, como en el NSE por cuenca. Conserva los ceros (~3 %) y la escala lineal |

- `σ_b` lleva un mínimo `ε` para no dividir por un valor cercano a cero en cuencas casi secas.
- Los 48 ceros del horizonte del decoder se ponen en espacio normalizado, así que equivalen a la media
  de la cuenca.
- Las predicciones se **des-normalizan** antes de calcular las métricas, que se reportan en mm/h.
- Los scalers se guardan como JSON y se registran como artifact en MLflow.
- `log(q + ε)` para el caudal queda como experimento opcional.

## 5. Valores faltantes

No hay, pero al cargar se verifica que todo sea finito y que claves, shapes y conteos coincidan con
`metadata.json`. Si algo falla, error explícito: nunca se imputa en silencio.

## 6. Pipeline

1. **Carga y validación** de los `.h5` y `test_targets.csv` (§5).
2. **Ajuste de los scalers** con train (§4).
3. **Datasets** de train, val y test que devuelven los tensores de §2. El `y_aux` de test es opcional,
   para cuando llegue.
4. **DataLoader** sembrado (seed 2025).
5. **Des-normalización** de las predicciones antes de las métricas.

Parámetros en la config (`data:`): `history_hours`, `horizon_hours`, `use_future_meteo`,
`future_meteo_dropout` y la normalización elegida.
