# Ejecución en RunPod

Cómo entrenar la grilla ([`experiments.md`](experiments.md)) en un RTX 4090 de RunPod y traer los
resultados a la Mac. Las decisiones detrás de este flujo están en D-004 y D-013
([`decisions.md`](decisions.md)).

## Resumen

```
Mac                                  Pod (RTX 4090, volumen /workspace)
───                                  ──────────────────────────────────
merge a main  ─────────── git ─────▶ setup.sh      uv, entorno, datos, cache, tests
                                     run_grid.sh   clamf.grid → snapshot → stop pod
pull.sh  ◀──────────── rsync/ssh ─── mlflow.snapshot.db, mlruns/, checkpoints/, logs/
results/runpod/  (store reubicado)
```

Todo lo del pod vive en el volumen `/workspace`, el único disco que sobrevive a un stop: código, uv,
Python, caché de uv, datos, `mlflow.db`, `mlruns/`, `checkpoints/` y `logs/` (`scripts/runpod/env.sh`).

## 1. Antes de crear el pod (Mac)

- El pod clona `main` desde GitHub: lo que se vaya a correr tiene que estar mergeado y pusheado.
- Agregar la llave SSH pública en RunPod → *Settings* → *SSH Public Keys*.

## 2. Crear el pod

| Opción | Valor |
|---|---|
| Cloud | Community Cloud (~$0.34/h) o, si no hay 4090 con CUDA ≥ 13, Secure Cloud (~$0.74/h); siempre **on-demand** (un spot se puede interrumpir) |
| GPU | 1 × RTX 4090 |
| Filtro *CUDA version* | **13.0**. `torch 2.14` del lockfile trae wheels de CUDA 13.0 y necesita un driver NVIDIA ≥ 580 |
| Template | PyTorch oficial de RunPod. Solo aporta el sistema: uv instala su propio Python y torch |
| Volume disk | 50 GB en `/workspace` (datos ~10 GB, entorno ~8 GB, caché de uv) |
| Container disk | 20 GB |
| SSH | activar *SSH over exposed TCP* (hace falta para `rsync`) |

La conexión aparece en *Connect* → *SSH over exposed TCP*: `ssh root@<ip> -p <puerto>`.

## 3. Preparar el pod

```bash
git clone https://github.com/hugoangeles0810/utec-deeplearning-lab02 /workspace/utec-deeplearning-lab02
bash /workspace/utec-deeplearning-lab02/scripts/runpod/setup.sh          # o setup.sh <rama|commit>
```

`setup.sh` se puede correr varias veces; lo que ya está hecho se salta. Hace lo siguiente:

1. instala uv (en `/workspace/.local/bin`), `tmux` y `rsync`;
2. deja el repo en la rama o commit pedido (`main` por defecto);
3. corre `uv sync --frozen`;
4. verifica que torch vea la GPU. Si falla, el pod no tiene un driver para CUDA 13: hay que crear
   otro con el filtro de CUDA;
5. baja el dataset de Google Drive (`clamf.data.download`) y arma el cache (`clamf.data.prepare`);
6. corre los tests.

Si Google Drive limita la descarga, se suben los datos desde la Mac y se vuelve a correr `setup.sh`:

```bash
rsync -az -e "ssh -p <puerto>" data/raw/ root@<ip>:/workspace/utec-deeplearning-lab02/data/raw/
```

## 4. Smoke test (en el pod)

```bash
cd /workspace/utec-deeplearning-lab02
STOP_POD=0 bash scripts/runpod/run_grid.sh configs/experiments/dev.yaml
```

Entrena `dev.yaml` (3 epochs cortas) en el experimento `dev`, lo evalúa en val y deja el snapshot.
Revisar en el log:

- la línea `run <id> on cuda (amp: bf16)`;
- que no aparezcan errores y que termine con `clamf.grid exit code 0`.

Luego, desde la Mac, probar la descarga (sección 6) y abrir MLflow UI.

## 5. Lanzar la grilla (en el pod)

```bash
cd /workspace/utec-deeplearning-lab02
tmux new -d -s grid 'bash scripts/runpod/run_grid.sh'
```

`run_grid.sh` corre `python -m clamf.grid`. Entrena los 6 runs en orden (`clamf`, `vanilla`, `claam`,
`msfm`, `cam`, `laam`) en el experimento `clamf-grid` y evalúa cada uno en val. Al terminar hace un
snapshot del store y **detiene el pod** (`runpodctl stop pod`), haya salido bien o mal. Con `STOP_POD=0`
el pod queda encendido. Si un run falla, `clamf.grid` registra el error en el log y sigue con el
siguiente; al final imprime una tabla con el estado de cada run y sale con código distinto de 0.

Seguimiento:

```bash
tmux attach -t grid          # salir sin cortar: Ctrl-b d
tail -f logs/grid-*.log
nvidia-smi
```

Referencia (D-004): `clamf` tarda ~60 s por epoch y la grilla completa, ~4 h.

**Si el pod se reinicia o el proceso muere:** volver a correr `setup.sh` y relanzar `run_grid.sh`.
`clamf.grid` salta los runs terminados, reanuda el que quedó a medias desde su `last.pt` (mismo run
de MLflow) y evalúa lo que falte.

## 6. Bajar los resultados (Mac)

```bash
bash scripts/runpod/pull.sh root@<ip> <puerto>
# con una llave distinta de la default: SSH_OPTS="-i ~/.ssh/runpod" bash scripts/runpod/pull.sh ...
uv run mlflow ui --backend-store-uri sqlite:///results/runpod/mlflow.db
```

`mlflow ui` no lee `MLFLOW_TRACKING_URI`: sin `--backend-store-uri` abre el `mlflow.db` local de
desarrollo (experimentos `Default` y `dev`). Los scripts de Python (`clamf.evaluate`, notebooks) sí
usan `MLFLOW_TRACKING_URI`.

`pull.sh` hace tres cosas:

1. pide al pod un snapshot consistente del store (`clamf.utils.mlflow_store snapshot`);
2. copia con `rsync` el snapshot, `mlruns/`, `checkpoints/` y `logs/` a `results/runpod/`;
3. reescribe en `results/runpod/mlflow.db` las rutas de artifacts de `/workspace/...` a
   `results/runpod/...` (`clamf.utils.mlflow_store relocate`).

Se puede correr durante la grilla, como respaldo, y cuantas veces haga falta: cada vez reemplaza
`results/runpod/` completo. Por eso lo que se registre en ese store desde la Mac (por ejemplo,
`clamf.evaluate --split test`) se hace **después** del último `pull.sh`, apuntando a ese store:

```bash
MLFLOW_TRACKING_URI=sqlite:///results/runpod/mlflow.db \
  uv run python -m clamf.evaluate --config configs/experiments/clamf.yaml --run-id <run id> --split test
```

`clamf.evaluate` no encuentra `best.pt` en `checkpoints/<run id>/` de la Mac y lo baja de los artifacts
del run (ya reubicados). Usa el cache local de `data/processed/`, que tiene que incluir el `y_aux` de
test (D-013).

Comprobar que los datos del pod y de la Mac son los mismos: el `scalers.json` de cualquier run debe
ser idéntico al de la Mac.

```bash
diff results/runpod/mlruns/<exp>/<run>/artifacts/data/scalers.json data/processed/scalers.json
```

## 7. Al terminar

1. El pod se detiene solo. Se enciende desde la consola. `pull.sh` no necesita GPU: si la GPU de ese
   host está ocupada y la consola ofrece arrancarlo sin GPU, alcanza.
2. `bash scripts/runpod/pull.sh ...` y revisar en MLflow UI que estén los 6 runs con `val/…`.
3. **Terminar el pod** (*Terminate*) para dejar de pagar el volumen.

## Costos aproximados

- GPU: en la corrida del 2026-09-30 (Secure Cloud, $0.74/h) la grilla tomó ~4 h y el pod estuvo
  encendido ~4.6 h (~$3.40 con setup y smoke test; D-004). El peor caso, si ningún run para antes de
  las 200 epochs, es ~18 h.
- Volumen: se cobra también con el pod detenido; por eso se termina el pod apenas se bajan los
  resultados.

## Archivos

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `scripts/runpod/env.sh` | pod | rutas en `/workspace`, uv y `MLFLOW_TRACKING_URI` |
| `scripts/runpod/setup.sh` | pod | prepara el entorno, los datos y corre los tests |
| `scripts/runpod/run_grid.sh` | pod | `clamf.grid` + snapshot + stop del pod |
| `scripts/runpod/pull.sh` | Mac | copia y reubica el store en `results/runpod/` |
| `src/clamf/grid.py` | ambos | entrena, reanuda y evalúa la grilla de forma idempotente |
| `src/clamf/utils/mlflow_store.py` | ambos | `snapshot` y `relocate` del store sqlite |
