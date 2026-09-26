# Registro de decisiones de implementación

Cada decisión que se aparta del paper o resuelve una ambigüedad se registra aquí.

Plantilla:

```
## D-XXX · <título>
- Estado: PENDIENTE | DECIDIDA
- Paper: <sección/ecuación y qué dice>
- Decisión: <qué se implementó>
- Justificación: <por qué>
- Alternativas consideradas: <...>
- Fecha / autor: <YYYY-MM-DD / nombre>
```

---

## D-001 · Diferenciabilidad de la máscara lag-aware (Eq. 5)
- Estado: **PENDIENTE**
- Paper: Sec. 2.2.2, Eq. (4)–(5). `τ_i = Softplus(MLP(Z_i))` y `Mask_ij = 1 si j ≤ i + τ_i, 0 si no`.
- Problema: una máscara binaria no es diferenciable respecto a `τ`, así que la red que predice el lag
  no recibiría gradiente y no aprendería.
- Alternativas consideradas:
  - Soft mask diferenciable: sesgo aditivo `log(sigmoid((i + τ_i − j) / T))` en los logits de atención.
  - Máscara binaria en el forward con gradiente straight-through en el backward.
  - Ambas configurables y comparadas como experimento adicional.
- Decisión: por definir por el equipo.
