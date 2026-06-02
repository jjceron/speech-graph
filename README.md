# actual_project

Pipeline para analizar transcripciones procesadas con grafos de palabras, preservando las etiquetas clínicas/anotaciones en `[[...]]` como tokens del discurso.

## Estructura principal

```text
src/
  __init__.py
  analysis/          # correlaciones, perfiles por grupo, estabilidad entre ventanas
  config/            # rutas y parámetros por defecto
  features/          # etiquetas [[...]] y léxicos emocionales opcionales
  graphs/            # métricas de speech graphs, ventanas y baselines aleatorios
  io/                # lectura de transcripciones y metadatos xlsx
  pipeline/          # scripts orquestadores
  preprocessing/     # tokenización en español preservando [[...]]
  visualizations/    # reservado para figuras posteriores
```

En la raíz de `src/` solo quedan carpetas y `src/__init__.py`.

## Datos esperados

- Transcripciones: `data/processed/Transcripciones/*.txt`
- Metadatos/scores: `data/processed/df_dataset.xlsx`
- Código de unión: `code` en métricas contra `Cod` en el xlsx.

Las etiquetas `[[EE]]`, `[[PS]]`, `[[IF]]`, `[[DI StartTime=... EndTime=...]]`, etc. se preservan como tokens y también se cuantifican como variables `label_count_*` y `label_ratio_*`.

## Comandos básicos

Extraer métricas con ventana de 30 palabras:

```bash
python -m src.pipeline.extract_graph_metrics \
  --transcripts-dir data/processed/Transcripciones \
  --output-csv outputs/graph_metrics_w30.csv \
  --window-size 30 \
  --step 1
```

Unir con metadatos:

```bash
python -m src.pipeline.merge_metadata \
  --metrics-csv outputs/graph_metrics_w30.csv \
  --metadata-xlsx data/processed/df_dataset.xlsx \
  --output-csv outputs/graph_metrics_w30_with_meta.csv
```

Correr y comparar los tres esquemas de ventana 10/20/30:

```bash
python -m src.pipeline.run_window_schemes \
  --transcripts-dir data/processed/Transcripciones \
  --metadata-xlsx data/processed/df_dataset.xlsx \
  --output-dir outputs/window_schemes \
  --window-sizes 10,20,30 \
  --step 1
```

Esto genera:

```text
outputs/window_schemes/graph_metrics_w10_s1.csv
outputs/window_schemes/graph_metrics_w20_s1.csv
outputs/window_schemes/graph_metrics_w30_s1.csv
outputs/window_schemes/graph_metrics_all_windows.csv
outputs/window_schemes/graph_metrics_all_windows_with_meta.csv
outputs/window_schemes/analysis/correlations_by_window.csv
outputs/window_schemes/analysis/profile_by_group_and_window.csv
outputs/window_schemes/analysis/window_metric_stability.csv
```

## Baselines aleatorios / z-scores

Por defecto `--random-times 0` para que el pipeline sea rápido. Para una corrida estilo Mota con grafos aleatorios:

```bash
python -m src.pipeline.run_window_schemes --random-times 1000
```

En el corpus completo esto puede tardar bastante, especialmente con paso 1 y tres tamaños de ventana.

## Smoke test

Para probar estructura y parsing sin correr todo el corpus:

```bash
python -m src.pipeline.run_window_schemes --max-files 3 --output-dir outputs/smoke_test
```
