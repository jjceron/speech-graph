# Informe del test `test_w30_s20`

## 1. Comando ejecutado

El test analizado corresponde al siguiente comando del pipeline:

```bash
py -m src.pipeline.mota_test --random-times 0 --window-sizes 30 --max-files 20 --output-dir outputs/test_w30_s20
```

Salida observada en consola:

```text
[window=30] extracting metrics -> outputs\test_w30_s20\graph_metrics_w30_s1.csv
  transcript 1/20: CDMS-10-4A-JURAN
  transcript 20/20: CDMS-11-6B-AVROTO
Done. Main output: outputs\test_w30_s20\graph_metrics_all_windows_with_meta.csv
```

Para visualizar los resultados se usó el run `test_w30_s20`, ventana 30 y nivel `file`:

```bash
py -m src.visualization --runs test_w30_s20 --window-size 30 --level file --min-n 10 --min-abs-r 0.15
```

## 2. Qué significa este test

Este análisis procesó 20 transcripciones usando ventanas móviles de 30 palabras. El análisis se hizo a nivel de archivo completo (`level = file`), por lo que cada fila representa una transcripción/sujeto, no segmentos internos.

El parámetro `--random-times 0` indica que no se generó baseline aleatorio. Por tanto, este test analiza las métricas observadas de los grafos, pero todavía no calcula z-scores ni compara los grafos reales contra grafos aleatorizados.

Este resultado debe entenderse como una prueba exploratoria ampliada. Sirve para revisar si el pipeline funciona y para detectar señales candidatas, pero no debe usarse como evidencia final.

## 3. Estado técnico del run

El run terminó correctamente. La consola indicó que se procesaron 20 transcripciones y que se generó el archivo principal:

```text
outputs/test_w30_s20/graph_metrics_all_windows_with_meta.csv
```

El merge con metadata parece operativo, porque las figuras cruzan métricas NLP con variables del Excel: Barratt, tareas cognitivas, coherencia narrativa, comprensión lectora, fluidez verbal y variables educativas.

El archivo `metadata_without_transcript.csv` contiene sujetos del Excel que no entraron en este test. Esto es esperable porque se corrió con `--max-files 20`; por tanto, la mayor parte del Excel queda fuera de este subconjunto. No debe interpretarse como error de merge mientras el test esté limitado a 20 archivos.

## 4. Resultados principales: etiquetas discursivas `[[...]]`

La señal más clara del test aparece en las etiquetas discursivas. En la figura `etiquetas [[...]] vs variables relevantes (w30)` se observan asociaciones relevantes entre ciertas etiquetas y tareas cognitivas o lingüísticas.

Las asociaciones candidatas más importantes fueron:

| Etiqueta | Variable asociada | Dirección | Interpretación preliminar |
|---|---:|---:|---|
| `SIN_RESPUESTA` | Reading comprehension task | negativa | Más ausencia de respuesta se asocia con peor comprensión lectora. |
| `SIN_RESPUESTA` | COHERENCIA NARRATIVA | negativa | Más ausencia de respuesta se asocia con menor coherencia narrativa. |
| `EE` | Reading comprehension task | negativa | Mayor proporción de esta etiqueta se asocia con peor comprensión lectora. |
| `PAUSA` | Naming task | negativa | Más pausas se asocian con peor desempeño en denominación. |
| `PAUSA` | Conceptualization task | negativa | Más pausas se asocian con peor desempeño en conceptualización. |
| `IF` | COHERENCIA NARRATIVA | negativa | Mayor proporción de `IF` se asocia con menor coherencia narrativa. |
| `PS` | TOTAL_zscore | negativa | Mayor proporción de `PS` se asocia débilmente con menor puntaje total estandarizado. |

En el heatmap se observan valores aproximados como:

```text
SIN_RESPUESTA ~ Reading comprehension task: r ≈ -0.56
SIN_RESPUESTA ~ COHERENCIA NARRATIVA: r ≈ -0.49
SIN_RESPUESTA ~ Naming task: r ≈ -0.43
EE ~ Reading comprehension task: r ≈ -0.48
PAUSA ~ Naming task: r ≈ -0.48
PAUSA ~ Conceptualization task: r ≈ -0.47
IF ~ COHERENCIA NARRATIVA: r ≈ -0.40
PS ~ TOTAL_zscore: r ≈ -0.35
```

La lectura general es que las etiquetas no parecen ser ruido técnico. En este subconjunto, etiquetas como `SIN_RESPUESTA`, `EE`, `PAUSA` e `IF` capturan aspectos discursivos que podrían estar relacionados con dificultad cognitiva, menor desempeño lingüístico o menor coherencia narrativa.

Esta es una de las señales más prometedoras para revisar con todos los sujetos.

## 5. Resultados principales: métricas de grafos y variables cognitivas

Además de las etiquetas, aparecieron asociaciones preliminares entre métricas estructurales de grafos y variables cognitivas, especialmente `COHERENCIA NARRATIVA`.

Las señales candidatas más importantes fueron:

| Métrica NLP | Variable asociada | Dirección | Lectura preliminar |
|---|---:|---:|---|
| `mean_density` | COHERENCIA NARRATIVA | negativa | Mayor densidad local podría reflejar mayor cierre/repetición dentro de la ventana, no necesariamente mejor narrativa. |
| `mean_lsc_ratio` | COHERENCIA NARRATIVA | negativa | Una mayor proporción fuertemente conectada no necesariamente implica mayor coherencia narrativa en este subconjunto. |
| `mean_asp` | COHERENCIA NARRATIVA | positiva | Mayor distancia promedio podría asociarse con trayectorias discursivas más extendidas. |
| `mean_diameter` | COHERENCIA NARRATIVA | positiva | Mayor diámetro podría reflejar mayor amplitud estructural de la ventana narrativa. |
| `mean_clustering` | COHERENCIA NARRATIVA | negativa | Mayor agrupamiento local podría estar asociado con más cierre local o repetición. |
| `mean_nodes` | COHERENCIA NARRATIVA | positiva | Más nodos únicos en ventanas de 30 palabras podría indicar mayor diversidad léxica local. |

La interpretación no debe simplificarse como “más conectividad = mejor discurso”. En este test, la señal parece más matizada: los sujetos con mayor coherencia narrativa podrían mostrar ventanas con más nodos y trayectorias más largas, mientras que densidad, clustering y conectividad local fuerte podrían reflejar mayor recirculación o repetición dentro de ventanas cortas.

Esta interpretación debe validarse con todos los sujetos y, posteriormente, con baseline aleatorio.

## 6. Resultados principales: Barratt

Las señales con Barratt existen, pero son menos limpias que las observadas con etiquetas y tareas cognitivas.

Las asociaciones candidatas más importantes fueron:

| Métrica NLP | Variable Barratt | Dirección | Lectura preliminar |
|---|---:|---:|---|
| `mean_repeated_edges_ratio` | `NPLAN` | positiva | Mayor impulsividad no planificada podría asociarse con mayor repetición de transiciones. |
| `mean_lsc` | `NPLAN` | negativa | Mayor impulsividad no planificada podría asociarse con menor conectividad fuerte promedio. |

Esta señal es conceptualmente interesante porque combina dos aspectos: más repetición de transiciones y menor conectividad fuerte. Podría ser compatible con una hipótesis de menor organización discursiva o mayor recurrencia local en sujetos con mayor impulsividad no planificada.

Sin embargo, todavía no debe considerarse un hallazgo definitivo. El subconjunto es pequeño, no hay baseline aleatorio y se probaron muchas asociaciones.

## 7. Perfilamiento de sujetos

La figura `sujetos con mayor proxy de desorganización NLP (w30)` ordena a los sujetos según un proxy derivado de métricas de conectividad. En esa figura, valores positivos indican mayor proxy de desorganización y valores negativos indican perfil más organizado según ese índice.

Sujetos con mayor proxy de desorganización en este test:

```text
CDMS-10-5C-ICAPA | high_imp
CDMS-11-5A-MGTHAR | high_imp
CDMS-10-5C-MAORTO | high_imp
CDMS-10-5C-DAAUSA | low_imp
CDMS-10-5A-SZURO | high_imp
```

Sujetos con menor proxy de desorganización:

```text
CDMS-10-5A-HEPIHO | high_imp
CDMS-10-4A-JURAN | low_imp
CDMS-11-5A-EAARCA | high_imp
CDMS-10-6B-JJMOBE | low_imp
CDMS-10-6B-ASCACO | low_imp
```

Este ranking es útil para análisis cualitativo posterior. Permite seleccionar sujetos extremos y revisar sus transcripciones manualmente.

Una observación importante es que no hay separación perfecta entre `high_imp` y `low_imp`. Algunos `high_imp` aparecen con alto proxy de desorganización, pero otros aparecen con bajo proxy. Esto sugiere que el perfil NLP no está copiando directamente la clasificación de impulsividad. Podría aportar información complementaria, pero debe confirmarse con la muestra completa.

## 8. Interpretación general del test

El test `test_w30_s20` muestra que el pipeline funciona y que ya aparecen señales candidatas interpretables.

Las señales más prometedoras son:

1. Las etiquetas `SIN_RESPUESTA`, `PAUSA`, `EE` e `IF` parecen relacionarse con tareas cognitivas y lingüísticas.
2. `COHERENCIA NARRATIVA` muestra asociaciones con varias métricas de grafos, especialmente densidad, LSC ratio, ASP, diámetro, clustering y nodos.
3. Barratt, especialmente `NPLAN`, muestra una señal preliminar con repetición de aristas y menor conectividad fuerte.
4. El ranking por sujeto permite identificar casos extremos para revisión cualitativa.

## 9. Limitaciones de este resultado

Este test tiene limitaciones importantes:

- Solo incluye 20 transcripciones.
- No usa baseline aleatorio (`random-times = 0`).
- No incluye corrección por múltiples comparaciones.
- Las correlaciones son exploratorias.
- Las métricas todavía pueden estar influidas por longitud, diversidad léxica o estructura particular de las transcripciones.
- Las etiquetas raras deben controlarse por prevalencia antes de interpretarse.

Por tanto, el resultado debe presentarse como exploración técnica y analítica preliminar, no como conclusión final.

## 10. Conclusión del test `test_w30_s20`

El test fue exitoso técnicamente y útil analíticamente. Confirmó que el pipeline puede leer transcripciones, calcular grafos con ventana 30, integrar metadata y generar visualizaciones interpretables.

La señal más clara está en las etiquetas discursivas, especialmente `SIN_RESPUESTA`, `PAUSA` y `EE`, que parecen asociarse con desempeño cognitivo-lingüístico. También aparecen señales preliminares entre estructura de grafos y coherencia narrativa, y una posible relación entre impulsividad no planificada y repetición de aristas.

El siguiente paso correcto es correr todos los sujetos sin random baseline, usando la misma ventana 30:

```bash
py -m src.pipeline.mota_test --random-times 0 --window-sizes 30 --output-dir outputs/01_w30_no_random_all
```

Ese run permitirá ver si las señales observadas en `test_w30_s20` se mantienen en la muestra completa.
