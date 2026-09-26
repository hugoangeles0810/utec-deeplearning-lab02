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
| [D-002](#d-002--dataset-del-profesor) | P0 | Dataset del profesor | — | PENDIENTE |
| [D-001](#d-001--diferenciabilidad-de-la-máscara-lag-aware-eq-5) | P0 | Diferenciabilidad de la máscara lag-aware τ | — | **DECIDIDA** |
| [D-003](#d-003--evaluación-del-horizonte-de-7-días) | P0 | Evaluación del horizonte de 7 días | — | PENDIENTE |
| [D-004](#d-004--estrategia-de-pre-entrenamiento-y-fine-tuning) | P0 | Pre-entrenamiento y fine-tuning | D-002 | PENDIENTE |
| [D-005](#d-005--causalidad-del-msfm) | P1 | Causalidad del MSFM | — | PENDIENTE |
| [D-006](#d-006--covariables-conocidas-en-el-horizonte) | P1 | Covariables conocidas en el horizonte | D-002 | PENDIENTE |
| [D-007](#d-007--normalización-y-valores-faltantes) | P1 | Normalización y valores faltantes | D-002 | PENDIENTE |
| [D-008](#d-008--detalles-de-freqmae) | P2 | Detalles de FreqMAE | D-003 | PENDIENTE |
| [D-009](#d-009--red-que-predice-τ) | P2 | Red que predice τ | — | PENDIENTE |
| [D-010](#d-010--estructura-del-msfm) | P2 | Estructura del MSFM | D-005 | PENDIENTE |
| [D-011](#d-011--posiciones-de-salida-de-la-predicción) | P2 | Posiciones de salida de la predicción | — | PENDIENTE |
| [D-012](#d-012--positional-encoding-batch-size-y-scheduler) | P3 | Positional encoding, batch size y scheduler | — | PENDIENTE |

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
- Estado: **PENDIENTE**
- Prioridad: P0
- Paper: Sec. 3.1 y Tabla 1 (CAMELS, 241 cuencas, 5 forzantes Daymet, datos diarios).
- Pregunta abierta: todavía no tenemos el dataset. Hay que conocer las variables, la frecuencia, cuántas
  series hay, cuál es la variable objetivo y el periodo cubierto, y decidir si la ventana 96→7 y las escalas
  7/30 del MSFM siguen teniendo sentido.
- Impacto: bloquea todo el pipeline de datos y condiciona D-004, D-006 y D-007.
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

## D-003 · Evaluación del horizonte de 7 días
- Estado: **PENDIENTE**
- Prioridad: P0
- Paper: Sec. 3.2 y Eq. 16–20. Las métricas se definen sobre una serie, pero el modelo predice 7 días por
  ventana y el paper no dice cómo se agregan.
- Pregunta abierta: ¿se evalúa cada día de anticipación por separado (lead 1…7), solo el primer día, o el
  promedio del horizonte?
- Impacto: define cómo se leen todas las tablas de resultados y la comparación con el baseline.
- Registrada: 2026-09-25.

## D-004 · Estrategia de pre-entrenamiento y fine-tuning
- Estado: **PENDIENTE**
- Prioridad: P0
- Paper: Sec. 3.2 y Tabla 3 (200 epochs de pre-entrenamiento + 50 de fine-tuning; modelos por región y
  conjunto).
- Pregunta abierta: ¿con qué datos se hace cada fase (todas las series → cada serie o región)? ¿Se omite el
  fine-tuning si el dataset tiene una sola serie?
- Depende de: D-002.
- Registrada: 2026-09-25.

---

## P1 · Fugas de información

## D-005 · Causalidad del MSFM
- Estado: **PENDIENTE**
- Prioridad: P1
- Paper: Sec. 2.2.3, Eq. 6–12. No especifica padding, stride ni máscara.
- Pregunta abierta: un Conv1D con padding simétrico, un MaxPool de ventana 7/30 o una cross-attention de
  fusión sin máscara dejan que el día `i` vea días futuros, lo que contradice la causalidad del CLAAM.
- Recomendación actual (`AGENTS.md`): padding causal (a la izquierda) y alinear/enmascarar cada ventana
  agregada para que solo incluya días `≤ i`.
- Impacto: si no se resuelve, las ablaciones con y sin MSFM no son comparables.
- Registrada: 2026-09-25.

## D-006 · Covariables conocidas en el horizonte
- Estado: **PENDIENTE**
- Prioridad: P1
- Paper: Sec. 3.2. El encoder recibe la meteorología de los 103 días, incluidos los 7 días a predecir.
- Pregunta abierta: ¿nuestro dataset tiene covariables disponibles para el horizonte de predicción
  (observadas o pronosticadas)? Si no las tiene, hay que redefinir la entrada del encoder.
- Depende de: D-002.
- Registrada: 2026-09-25.

## D-007 · Normalización y valores faltantes
- Estado: **PENDIENTE**
- Prioridad: P1
- Paper: no especifica normalización ni tratamiento de faltantes.
- Pregunta abierta: normalización por serie o global, si se aplica log-transform al target y cómo se
  tratan los valores faltantes o centinelas (p. ej. `-999`). En cualquier caso, las estadísticas se ajustan
  solo con train (`AGENTS.md` §7).
- Depende de: D-002.
- Registrada: 2026-09-25.

---

## P2 · Arquitectura y pérdida

## D-008 · Detalles de FreqMAE
- Estado: **PENDIENTE**
- Prioridad: P2
- Paper: Sec. 2.2.4, Eq. 13–15. Ver `paper.md` §6.
- Pregunta abierta:
  - tramo sobre el que se calcula (solo los 7 días predichos o toda la secuencia);
  - `fft` o `rfft`;
  - valor de `N` y normalización de la FFT (`norm`);
  - reducción sobre el batch y las series;
  - escala de los datos (normalizada o unidades originales).
- Depende de: D-003 (para que la pérdida sea coherente con la evaluación).
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
- Depende de: D-005.
- Registrada: 2026-09-25.

## D-011 · Posiciones de salida de la predicción
- Estado: **PENDIENTE**
- Prioridad: P2
- Paper: Sec. 2.2.1 y 3.2. La salida es de 7 días, pero no dice qué posiciones del decoder se usan.
- Pregunta abierta: lo natural son las 7 últimas posiciones del decoder, proyectadas a 1 dimensión.
- Registrada: 2026-09-25.

---

## P3 · Defaults menores

## D-012 · Positional encoding, batch size y scheduler
- Estado: **PENDIENTE**
- Prioridad: P3
- Paper: no especifica ninguno de los tres.
- Pregunta abierta: positional encoding sinusoidal o aprendido, tamaño de batch y si se usa un scheduler
  para el learning rate.
- Registrada: 2026-09-25.
