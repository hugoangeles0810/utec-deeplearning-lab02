# AGENTS.md — Laboratorio 2 · Deep Learning (UTEC)

Guía para agentes de IA (Claude Code, Codex) y para el equipo. Léela completa antes de escribir código.

## 1. Contexto

El objetivo del laboratorio es **implementar en Python + PyTorch el modelo CLAMF-Former** descrito en
`docs/paper.pdf`, y entrenarlo/evaluarlo sobre **un dataset distinto al del paper**, que entrega el profesor.

- Paper: *Enhancing runoff prediction with causal lag-aware attention and multi-scale fusion in transformer
  models* — W. Yuan & H. Yan, Journal of Hydrology 664 (2026) 134369.
- Tarea del paper: predecir caudal (runoff) de los **próximos 7 días** a partir de **96 días** de historia
  de caudal + forzantes meteorológicos (CAMELS, 241 cuencas, 5 variables Daymet).
- **Estado del dataset: todavía no lo tenemos.** El pipeline de datos debe ser genérico (serie(s) temporal(es)
  multivariada(s): covariables + variable objetivo) y se adaptará cuando llegue. No asumas columnas, frecuencia
  ni número de cuencas/estaciones; déjalo parametrizado en la config.

## 2. Resumen técnico del paper (referencia rápida)

Arquitectura encoder–decoder Transformer con 3 aportes:

### 2.1 CLAAM — Causal Lag-Aware Attention Mechanism (Sec. 2.2.2)
- **CAM (causal attention)**: máscara causal (triangular) en la self-attention del **encoder y del decoder**
  (en el Transformer clásico solo el decoder es causal).
- **LAAM (lag-aware attention)**: reemplaza la cross-attention del decoder (query = caudal, key/value =
  meteorología). Una red predice un desfase `τ_i ≥ 0` por posición y la máscara permite ver `j ≤ i + τ_i`.
  - Agregación de contenido (Eq. 2): `K̃_i = (Σ_{j≤i} K_j) / (P_i + ε)`.
  - `Z_i = Concat(Q_i, K̃_i)` (Eq. 3).
  - `τ_i = Softplus(W2 · LayerNorm(ReLU(W1 Z_i + b1) + Z_i) + b2)` (Eq. 4, paréntesis ambiguos en el PDF).
  - Máscara (Eq. 5): `Mask_ij = 1 si j ≤ i + τ_i, 0 si no`.

### 2.2 MSFM — Multi-Scale Fusion Module (Sec. 2.2.3)
Se aplica a la entrada del encoder (meteorología) **y** del decoder (caudal), antes del positional encoding.
- 3 ramas: `Conv1D(d → d_fusion) + GELU + MaxPool(k)` con `k = 1` (diaria), `7` (semanal), `30` (mensual).
- Alineación: `F̂_weekly = CrossAttn(F_daily, F_weekly, F_weekly)`, idem mensual.
- Salida: `Linear(Concat[F_daily, F̂_weekly, F̂_monthly])`.

### 2.3 FreqMAE (Sec. 2.2.4)
`Loss = Σ_k |Ŷ_k − Y_k|`, donde `Y_k`, `Ŷ_k` son la DFT (`torch.fft`) de observado y predicho.

### 2.4 Setup experimental del paper (defaults)
| Parámetro | Valor |
|---|---|
| Longitud encoder / decoder | 103 / 103 (96 historia + 7 a predecir) |
| Entrada decoder | 96 días de caudal conocido + 7 días rellenos con ceros |
| Salida | 7 días |
| `d_model` / `d_fusion` | 64 / 64 |
| Capas encoder y decoder | 4 |
| Heads | 4 |
| `d_ff` (position-wise) | 256 |
| Dropout | 0.1 |
| Optimizador / LR | Adam / 1e-3 |
| Pérdida | FreqMAE |
| Epochs | 200 pre-entrenamiento + 50 fine-tuning |
| Early stopping | 20 epochs sin mejora en validación |
| Seed | 2025 |
| Split (CAMELS) | train 1980-10-01→1995-09-30 · val 1995-10-01→2000-09-30 · test 2000-10-01→2010-09-30 |

Métricas (Sec. 3.2): **NSE, KGE, RMSE, TPE-2 %, BIAS**, reportadas como **mediana y media** sobre
cuencas/series. Fórmulas en Eqs. 16–20.

## 3. Alcance del laboratorio

Dentro del alcance:
1. **Modelo completo CLAMF-Former** (CLAAM + MSFM + FreqMAE), entrenado y evaluado con las 5 métricas.
2. **Ablaciones** (Tablas 5 y 6 del paper):
   - `CLAMF-1`: sin MSFM, sin CLAAM · `CLAMF-2`: con MSFM, sin CLAAM · `CLAMF-3`: sin MSFM, con CLAAM · `CLAMF`: completo.
   - `CLAAM-1`: sin CAM, sin LAAM · `CLAAM-2`: solo CAM · `CLAAM-3`: solo LAAM · `CLAAM`: ambos (todas sin MSFM).
3. **Baseline: Transformer vanilla** (encoder–decoder estándar: self-attention del encoder no causal,
   cross-attention estándar), entrenado con el mismo pipeline, datos y presupuesto que CLAMF-Former.

Fuera del alcance salvo que el equipo lo pida: comparación de pérdidas (MAE/MSE/sjNSE), baselines
LSTM-MSV-S2S / RR-Former / DTSW-transformer.

Diseña los módulos para que **cada componente se active/desactive por config** (flags `use_msfm`,
`use_causal_encoder`, `use_lag_aware_cross_attn`, `loss`), de modo que ablaciones y baseline sean solo
archivos YAML distintos, no código duplicado. Los módulos de atención deben poder **devolver los pesos de
atención** (opcionalmente) para visualizarlos como en las Figs. 2 y 9.

## 4. Estructura del repositorio (objetivo)

```
.
├── AGENTS.md / CLAUDE.md
├── pyproject.toml, uv.lock
├── configs/
│   ├── base.yaml                  # defaults = setup del paper
│   └── experiments/               # un YAML por experimento/ablación/baseline
├── src/clamf/
│   ├── config.py                  # dataclasses tipadas + carga/merge de YAML
│   ├── data/                      # carga, preprocesamiento, ventanas, Dataset/DataLoader
│   ├── models/
│   │   ├── attention.py           # atención causal y lag-aware
│   │   ├── msfm.py                # Multi-Scale Fusion Module
│   │   ├── clamf_former.py        # modelo completo (componentes configurables)
│   │   └── vanilla_transformer.py # baseline
│   ├── losses.py                  # FreqMAE (+ MSE/MAE)
│   ├── metrics.py                 # NSE, KGE, RMSE, TPE-2%, BIAS
│   ├── train.py                   # entrypoint de entrenamiento
│   ├── evaluate.py                # entrypoint de evaluación
│   └── utils/                     # device, seeds, logging MLflow
├── tests/                         # pytest
├── notebooks/                     # EDA, experimentos exploratorios, demos
├── data/                          # raw/ y processed/ (NO se versiona)
├── reports/figures/               # figuras para presentación y video
└── docs/
    ├── paper.pdf
    └── decisions.md               # registro de decisiones de implementación
```

Reglas:
- **Todo el código reutilizable vive en `src/clamf/`.** El entrenamiento se lanza siempre por CLI
  (`python -m clamf.train ...`), **nunca desde un notebook**.
- Los notebooks solo importan de `clamf` para EDA, análisis de resultados, visualizaciones y demos.
  No definir modelos ni lógica de entrenamiento dentro de notebooks.
- No versionar `data/`, `mlruns/`, `mlflow.db`, checkpoints ni outputs pesados (añadir a `.gitignore`).

## 5. Entorno y comandos

Gestor: **uv** con `pyproject.toml` (Python 3.11, como el paper).

```bash
uv sync                                                        # instalar dependencias
uv run pytest                                                  # tests
uv run ruff check . && uv run ruff format .                    # lint + formato
uv run python -m clamf.train --config configs/experiments/clamf.yaml
uv run python -m clamf.evaluate --config configs/experiments/clamf.yaml --run-id <mlflow_run_id>
uv run mlflow ui                                               # ver experimentos
```

- Agrega dependencias con `uv add <paquete>` (o `uv add --dev` para herramientas); no edites el lockfile a mano.
- Mantén las dependencias mínimas: `torch`, `numpy`, `pandas`, `pyyaml`, `mlflow`, `matplotlib`;
  dev: `pytest`, `ruff`, `jupyter`.

### Dispositivos (Mac MPS y GPU NVIDIA)
- Selección de dispositivo centralizada en `clamf/utils/device.py`: `cuda` → `mps` → `cpu`,
  sobreescribible por config/CLI (`device: auto|cuda|mps|cpu`).
- MPS no soporta `float64`: usa `float32` en todo el pipeline de tensores.
- Si alguna operación (p. ej. FFT compleja) no está soportada en MPS, usar `PYTORCH_ENABLE_MPS_FALLBACK=1`
  y documentarlo; no escribir ramas de código específicas por dispositivo salvo que sea imprescindible.
- Los tests deben correr en CPU y ser rápidos (modelos diminutos, pocos pasos).

## 6. Configuración de experimentos

- YAML en `configs/`, cargado a **dataclasses tipadas** en `clamf/config.py` (secciones sugeridas:
  `data`, `model`, `train`, `logging`). Un experimento = un YAML que sobreescribe `base.yaml`.
- `base.yaml` reproduce los defaults del paper (tabla 2.4). **Ningún hiperparámetro hardcodeado** en el código.
- Validar la config al cargarla (claves desconocidas → error).

## 7. Tracking con MLflow

- Tracking local (`mlruns/` o `sqlite:///mlflow.db`), configurable por variable de entorno `MLFLOW_TRACKING_URI`.
- Un *experiment* de MLflow por estudio (p. ej. `clamf-main`, `ablation-clamf`, `ablation-claam`, `baseline`);
  un *run* por entrenamiento.
- Registrar siempre: config completa aplanada como params, seed, device, git commit; loss train/val por epoch;
  métricas finales de test (mediana y media de cada métrica); artifacts: YAML usado, mejor checkpoint,
  predicciones de test (CSV/Parquet) y figuras.
- Las tablas y figuras de resultados se generan **a partir de los runs de MLflow**, no copiando números a mano.

## 8. Reglas de implementación

### Fidelidad al paper
- El setup del paper es el **default**. Cualquier desviación (ventana, horizonte, hiperparámetros,
  preprocesamiento) debida al nuevo dataset se registra en `docs/decisions.md` con su justificación.
- Cita la sección/ecuación del paper en el docstring de cada módulo que la implemente (p. ej. `# Eq. (4)`).

### Ambigüedades del paper
Política: **elegir la interpretación más razonable, implementarla y documentarla** en `docs/decisions.md`
(qué dice el paper, qué se eligió, por qué, alternativas). Ambigüedades conocidas:
- `P_i` en Eq. 2 (¿índice de posición `i+1` → media acumulada, o el positional encoding?).
- Ubicación de la conexión residual y paréntesis en Eq. 4; si `τ` se predice por head o compartido.
- FreqMAE: `fft` vs `rfft`, suma vs media, sobre qué tramo se aplica (los 7 días predichos o toda la
  secuencia), valor de `N` (Eq. 14 permite `N ≥` longitud).
- MSFM: padding/stride de Conv1D y MaxPool, y si la cross-attention de alineación es causal. Ojo: un Conv1D
  con padding simétrico o un pooling que mira hacia adelante **filtra información futura** y contradice
  la causalidad del CLAAM; preferir padding causal (izquierdo) salvo decisión documentada.
- Cálculo de métricas con horizonte de 7 días (¿por día de anticipación, promedio del horizonte?).
- Pre-entrenamiento + fine-tuning: el paper no detalla sobre qué datos se hace cada fase; depende de si el
  dataset tiene múltiples series.

**⚠️ Decisión PENDIENTE — máscara lag-aware (Eq. 5):** la máscara binaria `j ≤ i + τ_i` no es diferenciable,
por lo que la red que predice `τ` no recibiría gradiente. El equipo **aún no ha decidido** la solución
(soft mask diferenciable, straight-through, u otra). Los agentes **no deben elegir una por su cuenta**:
expongan la estrategia como un parámetro de la config/interfaz y pregunten al equipo antes de implementarla.
Ver `docs/decisions.md`.

### Datos y fugas de información
- Split **cronológico** train/val/test; nunca aleatorio. Sin solapamiento temporal entre splits para la variable objetivo.
- Normalización con estadísticas **solo del train** (por serie/cuenca si hay varias); guardar los scalers
  como artifact. Las métricas se calculan en **unidades originales** (des-normalizar antes).
- Los 7 días futuros de la entrada del decoder van **rellenos con ceros**; el target nunca puede entrar al modelo.
- Documentar qué covariables se asumen conocidas en el horizonte de predicción (el paper usa la meteorología
  de los 7 días futuros como entrada del encoder).
- Manejo explícito de valores faltantes (p. ej. centinelas tipo `-999`): nunca dejar que entren como números.

### Reproducibilidad
- Seed global (`2025` por defecto) para `random`, `numpy` y `torch`; `DataLoader` con `generator`/`worker_init_fn`.
- Cada run debe poder reproducirse solo con su YAML y el commit.

### Tensores
- Convención de shapes: `(batch, seq_len, features)` (`batch_first=True`). Documentar shapes en docstrings.
- Las máscaras siguen la convención de PyTorch: se aplican como `-inf` (o sesgo aditivo) en los logits
  antes del softmax. Documentar si una máscara booleana significa "permitido" o "bloqueado".

## 9. Tests (pytest)

Obligatorios para las piezas críticas (`tests/`):
- **Shapes** de cada módulo (atenciones, MSFM, modelo completo, baseline).
- **Causalidad**: perturbar la entrada en `t > i` no cambia la salida del encoder/decoder en posiciones `≤ i`
  cuando CAM está activo (y sí puede cambiarla en el baseline vanilla).
- **Sin fuga del target**: cambiar los valores reales de los 7 días futuros no altera la predicción.
- **FreqMAE**: 0 para predicción perfecta, no negativa, gradiente finito; comparar con un cálculo manual pequeño.
- **Métricas**: NSE = 1 y BIAS = 0 para predicción perfecta; NSE = 0 al predecir la media; casos simples
  calculados a mano para KGE, RMSE y TPE-2 %.
- **Split/normalización**: sin solapamiento temporal y scalers ajustados solo con train.
- **Smoke test**: forward + backward + unos pocos pasos de entrenamiento en CPU con un modelo diminuto.

Todo PR debe pasar `uv run pytest` y `uv run ruff check .`.

## 10. Estilo de código

- **Idioma**: código, identificadores, docstrings y comentarios en **inglés**; documentación del proyecto
  (`AGENTS.md`, `README`, `docs/`, presentación) en **español**.
- `ruff` para lint y formato (line-length 100). Type hints en todas las funciones/métodos públicos.
- Docstrings breves con shapes de entrada/salida y la ecuación del paper que implementan.
- Módulos pequeños y enfocados; preferir composición (`nn.Module` reutilizables) a herencia profunda.
- Sin código muerto ni experimentos comentados; lo exploratorio va en `notebooks/`.

## 11. Flujo de trabajo en equipo (git)

- Trabajo en equipo con **ramas + Pull Requests** hacia `main`; no hacer commits directos a `main`.
- Ramas: `feat/<tema>`, `fix/<tema>`, `exp/<experimento>`, `docs/<tema>`.
- Commits con **Conventional Commits** (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `exp:`),
  mensajes en inglés, cambios pequeños y enfocados.
- Cada PR describe qué cambia, cómo se probó y, si aplica, el run de MLflow asociado y las decisiones
  añadidas a `docs/decisions.md`.
- Los agentes no hacen push, merge ni force-push sin pedírselo explícitamente al equipo.

## 12. Entregables

- **Presentación** y **video** explicando el paper, la implementación, las adaptaciones al dataset y los
  resultados (CLAMF-Former vs Transformer vanilla + ablaciones).
- Material de apoyo que el código debe poder generar de forma reproducible en `reports/figures/`:
  tablas de métricas (mediana/media), boxplots por métrica, curvas de predicción vs observado,
  mapas de atención (causal vs vanilla, patrón diagonal del LAAM) y curvas de entrenamiento.

## 13. Qué hacer ante dudas

1. Revisa el paper (`docs/paper.pdf`) y `docs/decisions.md`.
2. Si es una ambigüedad menor de implementación: decide, implementa y documenta.
3. Si afecta el alcance, el protocolo experimental, el dataset o una decisión marcada como **PENDIENTE**:
   detente y pregunta al equipo.
