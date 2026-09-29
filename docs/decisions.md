# Registro de decisiones de implementación

Cada decisión que se aparta del paper o resuelve una ambigüedad se registra aquí. El detalle técnico de
cada ambigüedad está en [`paper.md`](paper.md).

## Prioridades

| Prioridad | Significado | Quién decide |
|---|---|---|
| **P0** | Bloqueante: frena el trabajo o cambia el protocolo experimental | El equipo |
| **P1** | Riesgo de fuga de información: si se decide mal, los resultados no son válidos | El equipo (el agente puede proponer) |
| **P2** | Arquitectura y pérdida: detalles del modelo central | El agente al implementar, documentándolo aquí |
| **P3** | Defaults menores con un valor estándar razonable | El agente al implementar, documentándolo aquí |

## Decisiones por prioridad

| ID | Prioridad | Tema | Depende de | Estado |
|---|---|---|---|---|
| [D-002](#d-002--dataset-del-profesor) | P0 | Dataset del profesor | — | **DECIDIDA** |
| [D-001](#d-001--diferenciabilidad-de-la-máscara-lag-aware-eq-5) | P0 | Diferenciabilidad de la máscara lag-aware τ | — | **DECIDIDA** |
| [D-003](#d-003--evaluación-del-horizonte-de-predicción) | P0 | Evaluación del horizonte de predicción | — | **DECIDIDA** |
| [D-004](#d-004--estrategia-de-pre-entrenamiento-y-fine-tuning) | P0 | Pre-entrenamiento y fine-tuning | D-002 | **DECIDIDA** |
| [D-013](#d-013--meteorología-futura-en-test) | P0 | Meteorología futura en test | D-006 | PENDIENTE (en espera del profesor, sin fecha límite) |
| [D-005](#d-005--causalidad-del-msfm) | P1 | Causalidad del MSFM | — | **DECIDIDA** |
| [D-006](#d-006--covariables-conocidas-en-el-horizonte) | P1 | Covariables conocidas en el horizonte | D-002 | PENDIENTE |
| [D-007](#d-007--normalización-y-valores-faltantes) | P1 | Normalización y valores faltantes | D-002 | **DECIDIDA** |
| [D-014](#d-014--hora-de-inicio-de-las-ventanas-y-re-muestreo-de-train) | P1 | Hora de inicio de las ventanas y re-muestreo de train | — | **DECIDIDA** |
| [D-008](#d-008--detalles-de-freqmae) | P2 | Detalles de FreqMAE | D-003 | **DECIDIDA** |
| [D-009](#d-009--red-que-predice-τ) | P2 | Red que predice τ | — | PENDIENTE |
| [D-010](#d-010--estructura-del-msfm) | P2 | Estructura del MSFM | D-005 | PENDIENTE |
| [D-011](#d-011--posiciones-de-salida-de-la-predicción) | P2 | Posiciones de salida de la predicción | D-002 | **DECIDIDA** |
| [D-015](#d-015--detalles-de-los-scalers-y-del-cache-de-datos) | P2 | Detalles de los scalers y del cache de datos | D-007 | **DECIDIDA** |
| [D-012](#d-012--positional-encoding-batch-size-y-scheduler) | P3 | Positional encoding, batch size y scheduler | — | PENDIENTE |
| [D-016](#d-016--detalles-de-cálculo-de-las-métricas) | P3 | Detalles de cálculo de las métricas | D-003 | **DECIDIDA** |

Otros pendientes (no técnicos): registrar la **fecha de entrega** de la presentación y el video.

## Plantilla

```
## D-XXX · <título>
- Estado: PENDIENTE | DECIDIDA
- Prioridad: P0 | P1 | P2 | P3
- Paper: <sección/ecuación y qué dice>
- Pregunta abierta: <qué hay que definir>        (solo mientras esté PENDIENTE)
- Decisión: <qué se implementó>
- Justificación: <por qué>
- Alternativas consideradas: <...>
- Fecha / autor: <YYYY-MM-DD / nombre>
```

---

## P0 · Bloqueantes

## D-002 · Dataset del profesor
- Estado: **DECIDIDA**
- Prioridad: P0
- Paper: Sec. 3.1 y Tabla 1 (CAMELS, 241 cuencas, 5 forzantes Daymet, datos diarios).
- Datos recibidos (2026-09-26): folder de Drive "Rainfall-Runoff" del profesor. Se descarga a `data/raw/`
  con `uv run python -m clamf.data.download`. Según `metadata.json`:
  - Archivos: `train.h5` (5.1 GB; train + validación), `test.h5` (454 MB), `test_targets.csv`
    (`Id,q_01..q_48`) y `metadata.json`. El folder trae además `leer_datos.py`, un lector de referencia
    del profesor que no se descarga.
  - Claves de los `.h5` (se leen con `h5py`): `train.h5` tiene `X` `(N, 336, 12)`, `y` `(N, 48)`,
    `y_aux` `(N, 48, 11)`, `split` `(N,)` y `basin_id` `(N,)`; `test.h5` tiene solo `X` y `basin_id`.
    El `Id` de cada muestra es su índice de fila (base 0) dentro del archivo.
  - Frecuencia **horaria**. Muestras ya cortadas en ventanas: `X` con **336 h de historia**, `y` con
    **48 h a predecir**. Ejes `(sample, time, channel)`, `float32`.
  - 12 canales: 11 meteorológicos (`convective_fraction`, `longwave_radiation`, `potential_energy`,
    `potential_evaporation`, `pressure`, `shortwave_radiation`, `specific_humidity`, `temperature`,
    `total_precipitation`, `wind_u`, `wind_v`) y el target `specific_discharge` (canal 11, mm/h).
  - `split` en `train.h5`: 0 = train (254 000 muestras), 1 = validación (18 142). Test: 27 983 muestras.
  - `basin_id` anónimo y consistente entre splits.
  - `y_aux` (meteorología de las 48 h futuras) está marcada como
    `future_supervision_only_not_inference_inputs`: **no puede ser entrada del modelo**.
  - Sin normalización ni imputación aplicadas.
- Preguntas abiertas (las decide el equipo):
  - cómo adaptar la ventana 96→7 diaria del paper y las escalas 7/30 del MSFM a 336→48 horario;
  - si el split train/validación que viene dado es cronológico, y si se usa tal cual;
  - el rol de `y_aux` (ver D-006) y la normalización por `basin_id` con posibles NaNs (ver D-007).
- Hallazgos del análisis (2026-09-27):
  - val y test no comparten horas con train ni entre sí; no se puede verificar que sean posteriores a
    train (no hay fechas), solo que son disjuntos;
  - las ventanas de train se solapan y permiten reconstruir ~11.5 años horarios por cuenca;
  - varios canales meteorológicos son trihorarios interpolados; el caudal es horario.
- Nota (2026-09-28): el caudal es **específico** (`specific_discharge`, normalizado por el área de la
  cuenca). Las unidades `mm/h` se infieren del nombre de la variable (`metadata.json`: sin unidades
  declaradas) y el área no viene en los datos. Son coherentes con la magnitud: caudal medio mediano en
  val ≈ 0.048 mm/h ≈ 1.2 mm/día, del mismo orden que las cuencas del paper (28.6 % bajo 1 mm/día).
- Decisión (propuesta del 2026-09-27, aprobada por el equipo el 2026-09-28):
  - se mantiene la resolución **horaria**; no se agrega a diario;
  - encoder y decoder de **384 pasos** (336 h de historia + 48 h de horizonte), equivalente a los
    103 días (96 + 7) del paper;
  - la predicción son las **48 últimas posiciones** del decoder (D-011);
  - escalas del MSFM **`k = 1, 24, 96`** (hora, día, 4 días) en lugar de `1, 7, 30` (día, semana, mes);
  - el split viene dado y se usa tal cual: train (`split == 0`) para entrenar, val (`split == 1`) para
    early stopping y selección de modelo (D-013); `y_aux` y la normalización se deciden en D-006 y D-007.
- Justificación: agregar a diario dejaría solo 14 + 2 pasos por ventana y perdería la dinámica horaria
  del caudal. Con 384 pasos, `k = 24` y `k = 96` dividen la longitud exacta (16 y 4 pasos gruesos),
  así que el MaxPool con stride = k (D-005) no necesita padding. Un `k` mensual (720 h) no cabe en la
  ventana; `k = 96` da la escala más gruesa que todavía deja varios pasos para la fusión.
- Alternativas consideradas: agregar a resolución diaria y usar la ventana del paper; escalas
  `k = 1, 24, 168` (semana, que no divide 384 y deja ~2 pasos gruesos).
- Fecha / autor: 2026-09-28 / equipo.
- Impacto: condiciona D-004, D-006 y D-007.
- Registrada: 2026-09-25.

## D-001 · Diferenciabilidad de la máscara lag-aware (Eq. 5)
- Estado: **DECIDIDA**
- Prioridad: P0
- Paper: Sec. 2.2.2, Eq. (4)–(5). `τ_i = Softplus(MLP(Z_i))` y `Mask_ij = 1 si j ≤ i + τ_i, 0 si no`.
- Problema: una máscara binaria no es diferenciable respecto a `τ`, así que la red que predice el lag
  no recibiría gradiente y no aprendería. Implementada literalmente, `τ` sería una función aleatoria de la
  entrada fijada por la inicialización.
- Decisión: **máscara suave diferenciable**, implementada como sesgo aditivo sobre los logits de la
  cross-attention, antes del softmax:

  $$
  \text{bias}_{ij} = \log \sigma\!\left(\frac{i + \tau_i + \delta - j}{T}\right),
  \qquad
  \operatorname{Attn} = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \text{bias}\right) V
  $$

  Detalles de implementación:
  - Calcular el sesgo con `torch.nn.functional.logsigmoid` (numéricamente estable).
  - **Margen `δ = 0.5`**: desplaza el borde medio paso para que los días `j ≤ i` nunca se penalicen, aunque
    `τ_i = 0` (sin margen, el día actual recibiría `log(0.5) ≈ −0.69`).
  - **Temperatura `T` en la config** (default `1.0`, en pasos de tiempo), con opción de reducirla durante el
    entrenamiento (annealing) para acercarse a la máscara binaria del paper. Con `T → 0` se recupera la Eq. 5.
  - **Inicialización de `τ` cercana a 0**: sesgo negativo en la última capa antes del Softplus, para que el
    modelo arranque como atención causal y amplíe la ventana solo si le sirve.
  - **Monitoreo de `τ`**: registrar en MLflow su distribución (media y percentiles). El paper no acota `τ`;
    si crece hasta cubrir toda la secuencia, el LAAM termina siendo una cross-attention estándar. No es una
    fuga (la meteorología de toda la ventana es una entrada conocida), pero se pierde el sesgo inductivo del
    mecanismo y hay que tenerlo en cuenta al leer las ablaciones.
  - **Entrenamiento y evaluación usan la misma máscara suave.** La máscara binaria (`j ≤ i + τ_i`) se usa
    solo para inspección y visualización (p. ej. ventanas como en la Fig. 9), no para calcular métricas.
- Justificación: es la única opción con la que `τ` recibe un gradiente bien definido; conserva la semántica
  del paper como caso límite (`T → 0`); se implementa como un sesgo aditivo compatible con la atención
  estándar de PyTorch.
- Alternativas consideradas:
  - Máscara binaria literal: descartada, `τ` no aprende.
  - Máscara binaria en el forward con gradiente straight-through: descartada como opción principal. Con la
    máscara aplicada como `-inf` en los logits el gradiente no está definido, obliga a enmascarar después
    del softmax y renormalizar, y el gradiente es sesgado e inestable. Queda como experimento opcional si
    sobra tiempo.
  - Ambas configurables: descartada por el coste de implementación frente a lo que aporta al laboratorio.
- Fecha / autor: 2026-09-25 / equipo.

## D-003 · Evaluación del horizonte de predicción
- Estado: **DECIDIDA**
- Prioridad: P0
- Paper: Sec. 3.2 y Eq. 16–20. Las métricas se definen sobre una serie, pero el modelo predice 7 días por
  ventana y el paper no dice cómo se agregan. Tampoco dice nada sobre métricas indefinidas o cuencas
  excluidas: solo que NSE y KGE van de −∞ a 1. Los resultados son mediana y media sobre las 241 cuencas.
- Pregunta abierta: ¿se evalúa cada día de anticipación por separado (lead 1…7), solo el primer día, o el
  promedio del horizonte?
- Propuesta (2026-09-27): el horizonte es de 48 h y las ventanas de test no se
  solapan. Métrica principal por cuenca sobre todos los pares (ventana, hora de anticipación), con
  mediana y media sobre las 508 cuencas; como secundaria, NSE y RMSE por hora de anticipación (1…48).
- Decisión:
  - **Métrica principal (pooled):** por cuenca, las 5 métricas (NSE, KGE, RMSE, TPE-2 %, BIAS) se calculan
    sobre todos los pares (ventana, hora de anticipación 1…48) juntos, en unidades originales (mm/h).
    Se reporta la mediana y la media sobre las cuencas.
  - **Secundaria:** NSE y RMSE por hora de anticipación (1…48), con la misma agregación por cuenca.
  - **TPE-2 %:** sobre el 2 % de horas con mayor caudal observado de cada cuenca, dentro de la serie
    pooled.
  - **Casos degenerados:** si el denominador observado de una métrica en una cuenca es menor que un
    umbral (`min_obs_std` y `min_obs_mean`, por defecto `1e-3` mm/h, configurables), esa métrica queda
    en `NaN` para esa cuenca y no entra en la mediana ni en la media:
    - `σ_obs < min_obs_std` → NSE y KGE en `NaN` (denominador de NSE y de la λ del KGE);
    - `μ_obs < min_obs_mean` → BIAS, TPE-2 % y KGE en `NaN` (denominador de BIAS, TPE-2 % y γ del KGE);
    - RMSE no tiene denominador y se calcula siempre;
    - si la predicción es constante (`σ_pred = 0`), se toma ρ = 0 en el KGE en vez de excluir la cuenca.
  - La exclusión depende **solo de lo observado**, así que las cuencas excluidas son las mismas para
    todos los modelos. Se registra en MLflow cuántas cuencas se excluyen por métrica.
  - Además se registra la versión **literal** (las 508 cuencas, sin excluir) como referencia; las
    tablas principales usan la versión con exclusión.
  - **Split reportado:** val (`split == 1`) de forma **provisional** hasta tener el `y_aux` de test
    (D-013); en cuanto llegue, las tablas se regeneran sobre test a partir de los mismos runs.
- Justificación:
  - pooled aplica las Eq. 16–20 tal como están escritas (una serie por cuenca). Las ventanas de val
    son tramos de 48 h dispersos, sin solapamiento (solo 418 de 17 634 pares consecutivos son
    contiguos), así que no hay horas duplicadas;
  - el paper evalúa 10 años diarios por cuenca y nunca tiene denominadores diminutos. En val hay ~38
    ventanas por cuenca (mínimo 11), y en tramos secos la serie puede ser casi plana: 11 cuencas tienen
    `σ_obs < 1e-3` mm/h y 15 tienen `μ_obs < 1e-3` mm/h (15 en total). Con `σ_obs = 2·10⁻⁵` mm/h, un
    error de 0.001 mm/h da NSE ≈ −2 500 y mueve la media de las 508 cuencas en ~5 puntos;
  - el caudal es específico (normalizado por área, D-002), así que un umbral absoluto en mm/h trata
    igual a cuencas de distinto tamaño; `1e-3` mm/h ≈ 0.024 mm/día, ~40× por debajo del umbral de
    "caudal bajo" del paper (1 mm/día);
  - reportar sobre val sesga los resultados hacia arriba porque val también se usa para early stopping
    y selección de modelo; el sesgo es parecido entre modelos, así que las comparaciones siguen siendo
    razonables, pero no son una estimación limpia del desempeño.
- Alternativas consideradas: métrica por lead como principal (48 números por métrica); solo el lead 1;
  promedio de métricas por lead; no excluir cuencas (la media queda dominada por cuencas secas); NSE
  acotado `1/(2 − NSE)` (se aleja del paper).
- Impacto: define cómo se leen todas las tablas de resultados y la comparación con el baseline.
- Fecha / autor: 2026-09-28 / equipo.
- Registrada: 2026-09-25.

## D-004 · Estrategia de pre-entrenamiento y fine-tuning
- Estado: **DECIDIDA**
- Prioridad: P0
- Paper: Sec. 3.2 y Tabla 3 (200 epochs de pre-entrenamiento + 50 de fine-tuning; modelos por región y
  conjunto; early stopping de 20 epochs sin mejora en validación).
- Pregunta abierta: ¿con qué datos se hace cada fase (todas las series → cada serie o región)? ¿Se omite el
  fine-tuning si el dataset tiene una sola serie?
- Decisión:
  - **un modelo global** entrenado con las 508 cuencas, **sin fine-tuning** (ni por cuenca ni por grupo);
  - tope de **200 epochs** (`train.max_epochs`), con **early stopping de paciencia 20** sobre la pérdida
    de val (`train.early_stopping_patience`), como el paper; se evalúa val al final de cada epoch y se
    guarda el mejor checkpoint;
  - un epoch recorre las 254 000 ventanas de train tal como vienen (D-014);
  - el **mismo presupuesto** (tope, paciencia, batch, seed) para CLAMF, las ablaciones y el baseline;
  - infraestructura: la grilla final se entrena en **RunPod, Community Cloud, 1 × RTX 4090**
    (~$0.34/h); la MacBook (M5 Pro, MPS) queda para desarrollo, tests y corridas cortas.
- Mediciones (2026-09-28): proxy con los tamaños del paper (`d_model = 64`, 4 + 4 capas, 4 heads,
  `d_ff = 256`, ~600 k parámetros), 384 pasos, MSFM `k = 1, 24, 96`, máscara lag-aware suave y
  FreqMAE. No es el modelo real; el tiempo final puede variar ±50 %.

  | Dispositivo | Batch | bf16 | Atención fusionada (SDPA) | Tiempo por epoch | Memoria |
  |---|---|---|---|---|---|
  | M5 Pro (MPS) | 128–512 | no | no / sí | ~17 min | — |
  | RTX 4090 | 256 | no | no | 4.9 min | 14.5 GB |
  | RTX 4090 | 512 | no | no | sin memoria | > 23 GB |
  | RTX 4090 | 256 | no | sí | 2.2 min | 6.2 GB |
  | RTX 4090 | 256 | sí | no | 4.2 min | 16.8 GB |
  | **RTX 4090** | **256** | **sí** | **sí** | **0.9 min** | **4.2 GB** |
  | RTX 4090 | 512–1024 | sí | sí | 1.0 min | 8.4–16.7 GB |

  Con 0.9 min por epoch, una corrida que llega al tope cuesta ~3 h; las ~7 corridas de la grilla
  (CLAMF, ablaciones y baseline) suman como máximo ~21 h, unos **$7**, y menos con early stopping.
  En la MacBook la misma grilla tomaría ~2.5 semanas.
- **Ajustes obligatorios en el código** (sin ellos el 4090 es solo ~3.5× más rápido que la MacBook y el
  presupuesto no alcanza):
  1. **Atención fusionada.** Las atenciones (CAM, LAAM, fusión del MSFM y baseline) usan
     `torch.nn.functional.scaled_dot_product_attention`: `is_causal=True` para la máscara causal y
     `attn_mask` como sesgo aditivo float para la máscara lag-aware suave (D-001). El cálculo explícito
     (`softmax(QKᵀ/√d + sesgo)`) se usa **solo cuando se piden los pesos de atención** (figuras). Flag
     `model.fused_attention` (default `true`). Test: ambos caminos dan la misma salida (tolerancia
     float32).
  2. **Precisión mixta bf16 en CUDA.** Forward con `torch.autocast("cuda", dtype=torch.bfloat16)`.
     La FFT de FreqMAE, la pérdida y las métricas se calculan en **float32** (convertir antes de la
     FFT). Sin `GradScaler` (bf16 no lo necesita). Flag `train.amp: bf16 | none` (default `bf16`; en
     MPS y CPU se ignora y se registra en MLflow que no se usó).
  3. **Datos precargados en el dispositivo.** A 0.9 min por epoch se leen ~4.6 GB por epoch y un
     `DataLoader` con `num_workers: 0` sería el cuello de botella. El cache de train (4.4 GB) cabe en
     los 24 GB del 4090 junto al modelo: se carga una vez como tensores en el dispositivo y los batches
     se arman con índices barajados por el generador con seed. Flag `data.preload_to_device`
     (default `true`).
  4. **Batch de 256** (`data.batch_size`, ya en `base.yaml`). Con fp32 y batch 512 no cabe en 24 GB.
  5. **Checkpoints reanudables.** Guardar modelo, optimizador, epoch, mejor pérdida de val, contador
     de paciencia y estado del RNG al final de cada epoch, y poder reanudar desde ahí (el pod se puede
     caer o hay que apagarlo).
  6. **Registro en MLflow:** GPU y `amp` usados, tiempo por epoch, epoch del mejor checkpoint y
     epochs corridos.
- Justificación:
  - el paper entrena por región y ajusta, pero aquí las 508 cuencas comparten un solo split y no hay
    regiones: un modelo global aprovecha todas las ventanas y es el setup habitual para muchas
    cuencas;
  - el fine-tuning por cuenca no es caro en cómputo (~45 min por modelo en el 4090), pero habría que
    hacerlo en cada ablación y en el baseline para compararlos con justicia, deja 508 checkpoints por
    modelo y cada cuenca solo tiene ~500 ventanas de train, con riesgo de sobreajuste;
  - con los ajustes de arriba, el tope de 200 epochs y la paciencia del paper entran de sobra en el
    presupuesto de ~$20.
- Alternativas consideradas: presupuesto completo del paper (200 + 50) con fine-tuning global o por
  cuenca; tope de 40 epochs con paciencia 5–8 (~$1.5 en total, pero más lejos del paper); entrenar solo
  en la MacBook (inviable para la grilla).
- Fecha / autor: 2026-09-28 / equipo.
- Depende de: D-002.
- Registrada: 2026-09-25.

## D-013 · Meteorología futura en test
- Estado: **PENDIENTE — en espera de respuesta del profesor**
- Prioridad: P0
- Paper: Sec. 3.2. El encoder recibe la meteorología de toda la ventana, incluido el horizonte.
- Problema: el equipo decidió usar `y_aux` como entrada del encoder (D-006), pero `test.h5` solo trae
  `X` y `basin_id`. Las ventanas de test no se solapan entre sí, así que esa meteorología tampoco se
  puede reconstruir. Tal como está, el modelo no se puede evaluar en test.
- Opciones:
  - a. pedir al profesor el `y_aux` de test (recomendada como primer paso);
  - b. entrenar con dropout de la meteorología futura y un canal indicador; evaluar test sin ella y val
    con y sin ella (recomendada si no llega `y_aux` de test);
  - c. evaluar el modelo con `y_aux` en val y sacar una validación nueva de tramos de train;
  - d. no usar `y_aux` como entrada.
- Plan del equipo (2026-09-27): **opción a**. Se pide al profesor el `y_aux` de test y las predicciones
  en test quedan **en espera** hasta recibirlo. Mientras tanto:
  - el desarrollo sigue con train/val, sin mirar `test_targets.csv`;
  - early stopping y selección de modelo se hacen con **val** (`split == 1`), pero val no se reporta
    como resultado final (equipo, 2026-09-28; reemplaza la idea de separar tramos de train, que
    chocaba con D-014);
  - actualización (equipo, 2026-09-28): mientras no llegue test, las tablas se reportan sobre **val de
    forma provisional** (D-003), sabiendo que salen optimistas por usar val en la selección de modelo;
  - el pipeline acepta `y_aux` de test como opcional: si `test.h5` lo trae, `clamf.data.prepare` lo
    cachea y el split de test queda disponible; sin él, el Dataset de test lanza un error explícito.
    La opción b **no se implementa** hasta que el profesor responda que no;
  - solo corridas cortas de desarrollo. La grilla de ablaciones y el baseline se lanzan cuando haya
    respuesta, porque la opción b obliga a re-entrenar todo.
- Si el profesor entrega el `y_aux` de test, antes de usarlo se verifica:
  - que tenga shape `(27983, 48, 11)` y esté alineado por `Id` con `test.h5`;
  - que continúe a `X` sin salto, como pasa con `y_aux` en train/val.
- Alternativa: si el profesor dice que no, se pasa a la **opción b**. **No hay fecha límite**: se
  espera su respuesta (equipo, 2026-09-27).
- Registrada: 2026-09-27.

---

## P1 · Fugas de información

## D-005 · Causalidad del MSFM
- Estado: **DECIDIDA**
- Prioridad: P1
- Paper: Sec. 2.2.3, Eq. 6–12 y Fig. 5. El texto no especifica padding, kernel del Conv1D, stride ni
  máscara. La Fig. 5 dibuja `F_weekly` y `F_monthly` más cortas que `F_daily` (MaxPool sin solapamiento,
  stride = k) y una "Multi-Head Attention" de fusión sin máscara. La causalidad (Sec. 2.2.2) solo se
  impone en las atenciones del CLAAM, no en el MSFM.
- Pregunta abierta: un Conv1D con padding simétrico, un MaxPool de ventana 7/30 o una cross-attention de
  fusión sin máscara dejan que el día `i` vea días futuros, lo que contradice la causalidad del CLAAM.
- Decisión: **MSFM literal, no causal (opción C)**:
  - Conv1D con padding simétrico (`same`);
  - MaxPool con stride = k (sin solapamiento), de modo que las escalas gruesas se acortan a `T / k`;
  - cross-attention de fusión (Q = escala fina, K = V = escala gruesa) **sin máscara**.
- Justificación: es la lectura más fiel del paper (texto + Fig. 5) y, con toda probabilidad, lo que
  implementaron los autores. **No hay fuga del target**: las posiciones del horizonte del decoder
  entran como ceros (D-007), así que el MSFM solo mezcla información ya disponible en la entrada.
- Consecuencias:
  - con `use_msfm: true` el modelo **no es causal internamente**: la posición `i` ve información de
    `t > i` a través del MSFM, aunque el CAM esté activo;
  - el test de causalidad (`AGENTS.md` §8) se aplica con `use_msfm: false` y, con MSFM activo, a los
    bloques de atención que están después del MSFM; el test de **fuga del target** se aplica siempre,
    también con MSFM;
  - al interpretar la ablación MSFM/CLAAM (Tabla 5) hay que tener en cuenta que parte de la ganancia
    del MSFM puede venir de ver hacia adelante dentro de la ventana de entrada.
- Alternativas consideradas: (A) MSFM causal sin reducir longitud (Conv1D con padding izquierdo,
  MaxPool deslizante causal, cross-attention con máscara triangular); (B) fiel a la Fig. 5 con padding
  izquierdo y máscara por bloques completados (`fin del bloque ≤ i`) más una key nula para las primeras
  posiciones. Descartadas por alejarse de lo que describe el paper.
- Fecha / autor: 2026-09-28 / equipo.
- Registrada: 2026-09-25.

## D-006 · Covariables conocidas en el horizonte
- Estado: **PENDIENTE**
- Prioridad: P1
- Paper: Sec. 3.2. El encoder recibe la meteorología de los 103 días, incluidos los 7 días a predecir.
- Pregunta abierta: ¿nuestro dataset tiene covariables disponibles para el horizonte de predicción
  (observadas o pronosticadas)? Si no las tiene, hay que redefinir la entrada del encoder.
- Nota (2026-09-26): el dataset **no** permite usar la meteorología futura como entrada (`y_aux` es solo
  para supervisión), así que la entrada del encoder del paper (103 días incluyendo el horizonte) tiene que
  redefinirse.
- Decisión del equipo (2026-09-27): **`y_aux` entra al encoder en train/val** (384 h: `X[..., :11]`
  seguido de `y_aux`) para el LAAM del paper; es la única excepción a `metadata.json`. `y` se usa solo
  como target. Queda pendiente qué hacer en test, donde falta `y_aux` (ver D-013).
- Depende de: D-002.
- Registrada: 2026-09-25.

## D-007 · Normalización y valores faltantes
- Estado: **DECIDIDA**
- Prioridad: P1
- Paper: no especifica normalización ni tratamiento de faltantes.
- Pregunta abierta: normalización por serie o global, si se aplica log-transform al target y cómo se
  tratan los valores faltantes o centinelas (p. ej. `-999`). En cualquier caso, las estadísticas se ajustan
  solo con train (`AGENTS.md` §7).
- Nota (2026-09-26): los datos vienen sin normalizar ni imputar; hay que revisar NaNs en el EDA.
  `basin_id` permite normalizar por cuenca.
- Nota (2026-09-27): el EDA no encontró NaN, infinitos ni centinelas en ningún split.
- Decisión (detalle en [`data_preparation.md`](data_preparation.md) §4):
  - estadísticas ajustadas solo con train (`split == 0`);
  - meteorología: z-score global por canal;
  - caudal (entrada del decoder y target): z-score por cuenca, `(q − μ_b) / σ_b`, con un mínimo `ε`
    en `σ_b`;
  - los 48 ceros del horizonte del decoder van en espacio normalizado (equivalen a la media de la
    cuenca);
  - las predicciones se des-normalizan antes de las métricas; los scalers se guardan como artifact;
  - faltantes: no hay; al cargar se valida que todo sea finito y, si no, se lanza un error.
- Justificación: las escalas meteorológicas son muy distintas entre canales, y `pressure` codifica la
  altitud, así que normalizarla por cuenca borraría esa información. El caudal medio cambia ~70×
  entre cuencas; el z-score por cuenca iguala su peso en la pérdida, como el NSE por cuenca, y
  conserva los ceros y la escala lineal.
- Alternativas consideradas: `q / σ_b`; `log(q + ε)` con z-score global (queda como experimento
  opcional); normalización global del caudal (descartada: aplasta las cuencas secas).
- Fecha / autor: 2026-09-27 / equipo.
- Depende de: D-002.
- Registrada: 2026-09-25.

## D-014 · Hora de inicio de las ventanas y re-muestreo de train
- Estado: **DECIDIDA**
- Prioridad: P1
- Paper: no aplica (el paper usa datos diarios).
- Problema: todas las ventanas de train de una cuenca empiezan a la misma hora del día; las de val y
  test, a cualquier hora (se detecta con el ciclo diario de la radiación de onda corta). En train, la posición `t` equivale
  a una hora del día fija, y en evaluación no: es un desfase de distribución.
- Propuesta: reconstruir las series de train por cuenca a partir de las ventanas
  solapadas (0 conflictos, ~11.5 años por cuenca) y muestrear ventanas con inicio aleatorio dentro de
  cada tramo continuo. Solo se usan horas de train, que no comparten horas con val/test.
- Decisión: **no se re-muestrea**. Se usan las ventanas de train dadas tal cual y se acepta el
  desfase de hora de inicio.
- Justificación: mantiene el pipeline simple y usa los datos tal como los entrega el profesor.
- Alternativas consideradas: la propuesta de re-muestreo de arriba (descartada por el equipo).
- Fecha / autor: 2026-09-27 / equipo.

---

## P2 · Arquitectura y pérdida

## D-008 · Detalles de FreqMAE
- Estado: **DECIDIDA**
- Prioridad: P2
- Paper: Sec. 2.2.4, Eq. 13–15. Ver `paper.md` §6. La Eq. 15 suma el módulo de la diferencia de las
  DFT sobre `k = 0…N−1`; no dice sobre qué tramo, con qué `N`, qué normalización ni cómo se reduce
  sobre el batch.
- Decisión (`src/clamf/losses.py`):
  - **tramo:** solo las 48 h del horizonte (la salida del modelo, D-011); la historia no entra en la
    pérdida;
  - **`torch.fft.fft` completa**, no `rfft`: es la Eq. 15 literal. Para señales reales el espectro es
    simétrico, así que equivale a `rfft` con las frecuencias intermedias contadas dos veces;
  - **`N = H = 48`**, sin zero-padding, y normalización `"backward"` (DFT sin escalar, como la Eq. 14);
  - **reducción:** suma sobre las frecuencias (Eq. 15) y **media sobre el batch** (cada ventana es una
    serie);
  - **escala:** caudal **normalizado por cuenca** (D-007), el mismo espacio en que el modelo predice.
    Las métricas se calculan después en mm/h;
  - se calcula como `|DFT(ŷ − y)|`, que por linealidad es igual a `|DFT(ŷ) − DFT(y)|`;
  - la pérdida (FreqMAE, MSE o MAE) se calcula en **float32** aunque el forward use bf16 (D-004);
  - la FFT corre nativa en MPS (probado con torch 2.14), sin `PYTORCH_ENABLE_MPS_FALLBACK`.
- Justificación:
  - el horizonte es lo único que se evalúa (D-003) y la historia del decoder es entrada, no predicción;
  - `fft` completa y sin normalizar sigue la ecuación tal cual; Adam es casi invariante a la escala de
    la pérdida, así que el factor ~`N` no afecta al entrenamiento, y el early stopping solo compara
    valores dentro del mismo run;
  - zero-padding solo interpola el espectro, no añade información;
  - en el espacio normalizado por cuenca cada cuenca pesa parecido, igual que en el NSE por cuenca
    (misma razón que D-007). En mm/h dominarían las cuencas más húmedas.
- Alternativas consideradas: `rfft` (cambia el peso relativo de la componente continua y de Nyquist);
  `norm="ortho"` o media sobre las frecuencias (misma pérdida a escala); pérdida en mm/h; incluir la
  historia de la salida del decoder.
- Fecha / autor: 2026-09-29 / agente.
- Registrada: 2026-09-25.

## D-009 · Red que predice τ
- Estado: **PENDIENTE**
- Prioridad: P2
- Paper: Sec. 2.2.2, Eq. 2–4. Ver `paper.md` §4.4.
- Pregunta abierta:
  - qué es `P_i` en la Eq. 2 (lo más plausible: `i + 1`, es decir, media acumulada);
  - ubicación de la conexión residual en la Eq. 4;
  - `τ` por head o compartido;
  - qué `K` se agrega (proyectada por head o salida del encoder sin proyectar).
- Resuelto por D-001: `τ` es continuo y no se redondea (se usa directamente en la máscara suave), y su
  inicialización arranca cerca de 0.
- Registrada: 2026-09-25.

## D-010 · Estructura del MSFM
- Estado: **PENDIENTE**
- Prioridad: P2
- Paper: Sec. 2.2.3, Eq. 6–12. Ver `paper.md` §5.
- Pregunta abierta:
  - kernel y stride del Conv1D y del MaxPool;
  - qué hacer con `T = 103`, que no es divisible por 7 ni por 30;
  - si las tres ramas, y el MSFM del encoder y el del decoder, comparten pesos;
  - número de heads y si hay residual/LayerNorm en la cross-attention de fusión;
  - dimensión de salida del Linear final;
  - entrada de una sola variable en el decoder (`d = 1`).
- Nota (2026-09-27): con datos horarios (384 pasos), las escalas 7/30 días del paper se adaptan a
  `k = 1, 24, 96` (**decidido en D-002**, 2026-09-28).
- Nota (2026-09-28): D-005 ya fija padding simétrico en el Conv1D, MaxPool con stride = k y fusión sin
  máscara. Con `T = 384` y `k = 24, 96` la longitud es divisible, así que el MaxPool no necesita padding.
- Depende de: D-005.
- Registrada: 2026-09-25.

## D-011 · Posiciones de salida de la predicción
- Estado: **DECIDIDA**
- Prioridad: P2
- Paper: Sec. 2.2.1 y 3.2. La salida es de 7 días, pero no dice qué posiciones del decoder se usan.
- Pregunta abierta: lo natural son las 7 últimas posiciones del decoder, proyectadas a 1 dimensión.
- Decisión: las **48 últimas posiciones** del decoder (las del horizonte, que entran en ceros),
  proyectadas a 1 dimensión con una capa lineal (decidido en D-002).
- Fecha / autor: 2026-09-28 / equipo.
- Depende de: D-002.
- Registrada: 2026-09-25.

## D-015 · Detalles de los scalers y del cache de datos
- Estado: **DECIDIDA**
- Prioridad: P2
- Paper: no especifica normalización (ver D-007).
- Decisión:
  - las estadísticas se calculan sobre la historia `X` de las ventanas de train (`split == 0`), sin
    `y`/`y_aux`. Como las ventanas se solapan, una hora cuenta una vez por cada ventana que la
    contiene; el efecto es un peso casi uniforme dentro de cada tramo y no vale la pena deduplicar;
  - `discharge_std_floor = 0.001` mm/h. Con los datos reales, `σ_b` va de 0.0009 a 0.55 mm/h
    (mediana 0.106), así que el mínimo solo toca a una cuenca y apenas;
  - `clamf.data.prepare` escribe un cache `.npy` ya normalizado en `data/processed/` (~5 GB, ~1 min)
    que los Datasets abren con memmap. `manifest.json` guarda tamaño y mtime de los archivos raw y
    la normalización; si no coinciden con la config, el Dataset falla y pide re-generar con
    `--force`;
  - `prepare` también comprueba, con un hash de cada fila horaria, que val y test no compartan
    horas con train (0 compartidas con los datos reales).
- Justificación: normalizar una vez evita repetir el trabajo en cada run, y el memmap mantiene baja
  la RAM en el Mac y en la máquina NVIDIA. Con 0 workers el DataLoader entrega ~320 batches/s de 256
  ventanas en el Mac, así que la carga no es el cuello de botella.
- Alternativas consideradas: cargar todo en RAM (~6 GB por proceso); leer el `.h5` por muestra
  (lento con acceso aleatorio); guardar el cache sin normalizar y normalizar en `__getitem__`.
- Fecha / autor: 2026-09-28 / agente.

---

## P3 · Defaults menores

## D-012 · Positional encoding, batch size y scheduler
- Estado: **PENDIENTE**
- Prioridad: P3
- Paper: no especifica ninguno de los tres.
- Pregunta abierta: positional encoding sinusoidal o aprendido, tamaño de batch y si se usa un scheduler
  para el learning rate.
- Registrada: 2026-09-25.

## D-016 · Detalles de cálculo de las métricas
- Estado: **DECIDIDA**
- Prioridad: P3
- Paper: Eq. 16–20. No dice la varianza muestral o poblacional, cuántas horas forman el 2 % de picos,
  ni cómo se desempatan.
- Decisión (`src/clamf/metrics.py`):
  - desviaciones estándar **poblacionales** (`ddof = 0`) en el KGE y en el umbral `min_obs_std`;
  - TPE-2 %: `max(1, round(0.02 · n))` horas con el mayor caudal observado de la cuenca (serie pooled,
    D-003); los empates se resuelven por orden estable. Su denominador es la suma del caudal observado
    en esas horas; si es 0, la métrica queda en `NaN`;
  - un denominador exactamente 0 siempre da `NaN`, también en la versión literal (umbrales en 0);
  - KGE: predicción constante = rango 0 exacto (`ptp`), así se evita el ruido numérico de
    `std ≈ 1e-17`; ρ se recorta a `[−1, 1]`;
  - métricas por hora de anticipación: el umbral del NSE usa la `σ_obs` de la serie de esa hora;
  - mediana y media sobre las cuencas ignoran los `NaN` y se registra cuántos hay por métrica
    (`<métrica>_n_excluded`);
  - todo el cálculo se hace en `float64` con numpy (fuera del dispositivo, así que no aplica la
    restricción de MPS).
- Justificación: son los defaults habituales en hidrología. Redondear evita que 2 % de 100 horas dé 3
  por error de coma flotante, y el mínimo de 1 hora mantiene la métrica definida en series cortas.
- Alternativas consideradas: `ddof = 1`; `ceil` o `floor` para el número de horas pico.
- Fecha / autor: 2026-09-29 / agente.
- Registrada: 2026-09-29.
