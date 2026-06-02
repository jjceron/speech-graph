# Resultados NLP por actividad — Speech graphs, etiquetas discursivas y metadata

## 1. Objetivo del análisis
Este análisis evalúa transcripciones en español mediante una adaptación de la metodología de *speech graphs*: cada palabra se representa como un nodo y cada transición consecutiva como una arista dirigida. A diferencia de un análisis global por sujeto, aquí las métricas se calculan por actividad/clase, con el fin de identificar si la organización discursiva, la recurrencia y las etiquetas de transcripción se asocian diferencialmente con variables cognitivas, educativas y Barratt según el tipo de tarea.

Las actividades de interés principal fueron Actividad2, Actividad6 y Actividad7. Las actividades 1, 4 y 5 se mantuvieron como análisis secundario o de contraste. La Actividad3 se reporta como referencia exploratoria, aunque no constituye el foco principal.

## 2. Comandos ejecutados
```bash
py -m src.pipeline.mota_activity --random-times 0 --window-size 30 --output-dir outputs/02_w30_by_activity_no_random_all
```

```bash
py -m src.visualization --run-dir outputs/02_w30_by_activity_no_random_all --only activities --level activity --window-size 30 --min-n 30 --min-abs-r 0.15 --label-min-nonzero 20
```

Este run usa ventana fija de 30 palabras, sin baseline aleatorio. Por tanto, los resultados son exploratorios y no deben interpretarse todavía como métricas normalizadas contra permutaciones aleatorias.

## 3. Cobertura y calidad del dataset
El archivo de métricas por actividad contiene 2136 filas: 1869 filas por actividad y 267 filas globales a nivel archivo. El análisis con metadata utiliza los casos con correspondencia entre el código de transcripción y el Excel de metadata.

| Actividad | Filas extraídas | Con metadata | Válidas W30 |
|---|---:|---:|---:|
| Actividad1 | 269 | 254 | 181 |
| Actividad2 | 266 | 251 | 236 |
| Actividad3 | 265 | 250 | 215 |
| Actividad4 | 268 | 253 | 253 |
| Actividad5 | 266 | 251 | 249 |
| Actividad6 | 268 | 253 | 253 |
| Actividad7 | 266 | 251 | 250 |

La cobertura por actividad es adecuada para análisis correlacional con ventana 30, especialmente en Actividad2, Actividad4, Actividad5, Actividad6 y Actividad7. Actividad1 presenta una menor cantidad de segmentos válidos con ventana 30, por lo que sus resultados deben interpretarse con mayor cautela.

Además, se detectó al menos una fila con actividad no estandarizada (`Actividad`). Esta fila no se incluyó en la interpretación por actividad porque no corresponde al formato `Actividad1`–`Actividad7`.

Se detectaron 15 códigos sin metadata en el archivo de grafos. Estos casos no entran en los análisis correlacionales con variables cognitivas/Barratt. Códigos sin match: `CMU-12-6A-SPURUR, CMU-12-6C-FAPUIP, CMU-13-6A-CMEPUR, CMU-13-6A-VVSIEP, CMU-14-6A-DFNUPU, CMU-14-6A-WJSUIG, CMU-14-6A-YAIPBO, CMU-14-8B-KYURIP, CMU-16-11-SJEPIP, CMU-16-11A-DDMODU, CMU-16-8B-LLEPPU, CMU-XX-10A-DABAUR, CMU-XX-7A-RAEPEP, CMU-XX-7B-CEPPU, CMU-XX-9B-MBPUPU`

## 4. Métricas consideradas
Las métricas principales de grafos fueron: `mean_nodes`, `mean_lsc`, `mean_lsc_ratio`, `mean_density`, `mean_asp`, `mean_diameter`, `mean_clustering`, `mean_l2`, `mean_l3` y `mean_repeated_edges_ratio`. Se evitaron interpretaciones sustantivas basadas solo en métricas directamente mecánicas como `mean_edges` o `token_count`.

Las etiquetas discursivas fueron tratadas como features del discurso y no como ruido. Se analizaron proporciones como `label_ratio_SIN_RESPUESTA`, `label_ratio_PAUSA`, `label_ratio_EE`, `label_ratio_IF`, `label_ratio_PS`, `label_ratio_DI`, `label_ratio_DP`, `label_ratio_IM`, `label_ratio_ES` y `label_ratio_SIN_PREGUNTA`. La etiqueta `SIN_PREGUNTA` se interpreta con cautela porque representa una condición de administración/protocolo y no una conducta directa del participante.

## 5. Resultado transversal principal: edad y escolaridad modulan los grafos
El patrón más consistente aparece en la relación entre edad/curso escolar y métricas estructurales. En varias actividades, especialmente Actividad4 y Actividad6, mayor edad o mayor escolaridad se asocian con más nodos, mayor distancia media (`mean_asp`), mayor diámetro, menor densidad y menor proporción de aristas repetidas.

Interpretación: dentro de ventanas fijas de 30 palabras, los sujetos mayores o con mayor curso tienden a producir segmentos con mayor diversidad léxica/estructural y menor recurrencia local. Este resultado implica que edad y escolaridad deben considerarse covariables centrales en análisis posteriores, especialmente antes de atribuir diferencias a impulsividad o rendimiento cognitivo.

### Actividad2: principales asociaciones de grafos
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_repeated_edges_ratio` | `Age` | -0.318 | 5.98e-07 | 236 |
| `mean_repeated_edges_ratio` | `School year` | -0.295 | 3.97e-06 | 236 |
| `mean_nodes` | `Age` | 0.278 | 1.51e-05 | 236 |
| `mean_nodes` | `School year` | 0.252 | 8.87e-05 | 236 |
| `mean_density` | `Age` | -0.239 | 2.15e-04 | 236 |
| `mean_diameter` | `Age` | 0.230 | 3.58e-04 | 236 |
| `mean_asp` | `Age` | 0.230 | 3.75e-04 | 236 |
| `mean_l2` | `Age` | -0.227 | 4.44e-04 | 236 |
| `mean_asp` | `School year` | 0.222 | 5.93e-04 | 236 |
| `mean_diameter` | `School year` | 0.221 | 6.26e-04 | 236 |

### Actividad4: principales asociaciones de grafos
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_nodes` | `School year` | 0.487 | 1.65e-16 | 253 |
| `mean_asp` | `School year` | 0.480 | 5.12e-16 | 253 |
| `mean_density` | `School year` | -0.478 | 7.43e-16 | 253 |
| `mean_asp` | `Age` | 0.472 | 1.84e-15 | 253 |
| `mean_diameter` | `School year` | 0.465 | 5.79e-15 | 253 |
| `mean_diameter` | `Age` | 0.461 | 9.96e-15 | 253 |
| `mean_nodes` | `Age` | 0.460 | 1.23e-14 | 253 |
| `mean_density` | `Age` | -0.450 | 5.34e-14 | 253 |
| `mean_lsc_ratio` | `School year` | -0.406 | 1.94e-11 | 253 |
| `mean_lsc_ratio` | `Age` | -0.399 | 4.48e-11 | 253 |

### Actividad6: principales asociaciones de grafos
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_nodes` | `School year` | 0.368 | 1.52e-09 | 253 |
| `mean_nodes` | `Age` | 0.357 | 4.89e-09 | 253 |
| `mean_density` | `School year` | -0.354 | 6.70e-09 | 253 |
| `mean_density` | `Age` | -0.346 | 1.65e-08 | 253 |
| `mean_repeated_edges_ratio` | `Age` | -0.335 | 4.88e-08 | 253 |
| `mean_repeated_edges_ratio` | `School year` | -0.331 | 7.17e-08 | 253 |
| `mean_asp` | `School year` | 0.312 | 3.95e-07 | 253 |
| `mean_diameter` | `School year` | 0.307 | 6.21e-07 | 253 |
| `mean_asp` | `Age` | 0.289 | 2.88e-06 | 253 |
| `mean_diameter` | `Age` | 0.284 | 4.33e-06 | 253 |

### Actividad7: principales asociaciones de grafos
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_nodes` | `School year` | 0.266 | 1.99e-05 | 250 |
| `mean_density` | `School year` | -0.259 | 3.34e-05 | 250 |
| `mean_nodes` | `Age` | 0.250 | 6.50e-05 | 250 |
| `mean_density` | `Age` | -0.246 | 8.27e-05 | 250 |
| `mean_asp` | `School year` | 0.227 | 2.92e-04 | 250 |
| `mean_diameter` | `School year` | 0.216 | 5.99e-04 | 250 |
| `mean_asp` | `Age` | 0.204 | 0.001 | 250 |
| `mean_diameter` | `Age` | 0.192 | 0.002 | 250 |
| `mean_repeated_edges_ratio` | `Reading comprehension task` | -0.176 | 0.005 | 250 |
| `mean_lsc_ratio` | `School year` | -0.173 | 0.006 | 250 |

## 6. Actividad6: tarea más sensible al desarrollo narrativo
Actividad6 muestra el patrón de desarrollo más claro. Las asociaciones más relevantes son:

| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_nodes` | `School year` | 0.368 | 1.52e-09 | 253 |
| `mean_nodes` | `Age` | 0.357 | 4.89e-09 | 253 |
| `mean_density` | `School year` | -0.354 | 6.70e-09 | 253 |
| `mean_density` | `Age` | -0.346 | 1.65e-08 | 253 |
| `mean_repeated_edges_ratio` | `Age` | -0.335 | 4.88e-08 | 253 |
| `mean_repeated_edges_ratio` | `School year` | -0.331 | 7.17e-08 | 253 |
| `mean_asp` | `School year` | 0.312 | 3.95e-07 | 253 |
| `mean_diameter` | `School year` | 0.307 | 6.21e-07 | 253 |
| `mean_asp` | `Age` | 0.289 | 2.88e-06 | 253 |
| `mean_diameter` | `Age` | 0.284 | 4.33e-06 | 253 |
| `mean_lsc_ratio` | `School year` | -0.262 | 2.42e-05 | 253 |
| `mean_density` | `Naming task` | -0.246 | 7.61e-05 | 253 |

La lectura principal es que Actividad6 parece capturar variación en madurez discursiva: a mayor edad y curso, aparecen grafos con más nodos y trayectorias más extensas, pero menor densidad y menor repetición de transiciones. Este patrón coincide con la expectativa de que una tarea narrativa amplia sea sensible al desarrollo lingüístico y educativo.

## 7. Actividad4: señal fuerte de `SIN_RESPUESTA` y desempeño cognitivo-lingüístico
Aunque Actividad4 no era una actividad target inicial, resultó ser una de las tareas más informativas en etiquetas discursivas. La señal dominante fue `label_ratio_SIN_RESPUESTA`, que se asoció negativamente con comprensión lectora, fluidez verbal, coherencia narrativa y tareas semánticas.

| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `label_ratio_SIN_RESPUESTA` | `Reading comprehension task` | -0.383 | 2.76e-10 | 253 |
| `label_ratio_IF` | `School year` | -0.280 | 6.11e-06 | 253 |
| `label_ratio_SIN_RESPUESTA` | `Verbal fluency tasks` | -0.261 | 2.70e-05 | 253 |
| `label_ratio_IF` | `Age` | -0.260 | 2.80e-05 | 253 |
| `label_ratio_SIN_RESPUESTA` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.227 | 2.64e-04 | 253 |
| `label_ratio_SIN_RESPUESTA` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS` | -0.212 | 6.85e-04 | 253 |
| `label_ratio_SIN_RESPUESTA` | `COHERENCIA NARRATIVA` | -0.209 | 8.21e-04 | 253 |
| `label_ratio_SIN_RESPUESTA` | `FLUIDEZ VERBAL - PRUEBA FONETICA` | -0.177 | 0.005 | 253 |
| `label_ratio_EE` | `School year` | -0.169 | 0.007 | 253 |
| `label_ratio_EE` | `Age` | -0.166 | 0.008 | 253 |
| `label_ratio_SIN_RESPUESTA` | `Conceptualization task` | -0.153 | 0.015 | 253 |
| `label_ratio_PS` | `School year` | -0.150 | 0.017 | 253 |

Interpretación: en Actividad4, la ausencia de respuesta no parece ser un evento aislado sino un marcador de dificultad cognitivo-lingüística. Su asociación con comprensión lectora y fluidez verbal sugiere que esta etiqueta puede contribuir al perfilamiento discursivo, especialmente cuando se analiza por actividad.

## 8. Actividad7: relación entre `SIN_RESPUESTA` e impulsividad Barratt
Actividad7 concentra la señal más clara entre etiquetas y Barratt. `label_ratio_SIN_RESPUESTA` se asoció positivamente con `NPLAN` y con puntajes agregados de Barratt.

| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `label_ratio_SIN_PREGUNTA` | `Naming task` | 0.263 | 2.46e-05 | 250 |
| `label_ratio_SIN_RESPUESTA` | `NPLAN` | 0.260 | 3.18e-05 | 250 |
| `label_ratio_PAUSA` | `TOTAL_zscore` | -0.207 | 9.79e-04 | 250 |
| `label_ratio_PS` | `Reading comprehension task` | -0.199 | 0.002 | 250 |
| `label_ratio_PAUSA` | `COG_zscore` | -0.195 | 0.002 | 250 |
| `label_ratio_SIN_RESPUESTA` | `TOTAL` | 0.190 | 0.003 | 250 |
| `label_ratio_SIN_RESPUESTA` | `Barratt (pre)` | 0.189 | 0.003 | 250 |
| `label_ratio_PAUSA` | `MOT_zscore` | -0.188 | 0.003 | 250 |
| `label_ratio_PAUSA` | `TOTAL` | -0.187 | 0.003 | 250 |
| `label_ratio_PAUSA` | `Barratt (pre)` | -0.187 | 0.003 | 250 |
| `label_ratio_SIN_RESPUESTA` | `TOTAL_zscore` | 0.183 | 0.004 | 250 |
| `label_ratio_IF` | `Age` | -0.176 | 0.005 | 250 |

Interpretación: en Actividad7, la ausencia de respuesta parece relacionarse con impulsividad, especialmente impulsividad no planificada. Este hallazgo es relevante porque las métricas puras de grafos mostraron relaciones más débiles con Barratt; en cambio, las etiquetas discursivas parecen capturar un componente conductual/cognitivo más directamente interpretable.

## 9. Actividad5: pausas y estructura del grafo asociadas con fluidez verbal
Actividad5 mostró asociaciones entre métricas de grafos y fluidez verbal, particularmente fluidez semántica de animales. También se observó que `label_ratio_PAUSA` se asocia negativamente con tareas cognitivo-lingüísticas.

### Actividad5: grafos
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_lsc_ratio` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.365 | 2.86e-09 | 249 |
| `mean_lsc_ratio` | `Verbal fluency tasks` | -0.344 | 2.44e-08 | 249 |
| `mean_nodes` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | 0.341 | 3.25e-08 | 249 |
| `mean_density` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.322 | 2.11e-07 | 249 |
| `mean_asp` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | 0.291 | 3.11e-06 | 249 |
| `mean_diameter` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | 0.288 | 3.91e-06 | 249 |
| `mean_asp` | `Verbal fluency tasks` | 0.275 | 1.10e-05 | 249 |
| `mean_diameter` | `Verbal fluency tasks` | 0.268 | 1.76e-05 | 249 |
| `mean_nodes` | `Verbal fluency tasks` | 0.266 | 2.18e-05 | 249 |
| `mean_lsc_ratio` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS` | -0.265 | 2.23e-05 | 249 |
| `mean_density` | `Verbal fluency tasks` | -0.262 | 2.83e-05 | 249 |
| `mean_asp` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS` | 0.222 | 4.14e-04 | 249 |

### Actividad5: etiquetas
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `label_ratio_PAUSA` | `COHERENCIA NARRATIVA` | -0.221 | 4.53e-04 | 249 |
| `label_ratio_DP` | `FLUIDEZ VERBAL - PRUEBA FONETICA` | -0.207 | 0.001 | 249 |
| `label_ratio_PAUSA` | `Verbal fluency tasks` | -0.195 | 0.002 | 249 |
| `label_ratio_PAUSA` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.193 | 0.002 | 249 |
| `label_ratio_PAUSA` | `Conceptualization task` | -0.180 | 0.004 | 249 |
| `label_ratio_SIN_RESPUESTA` | `FLUIDEZ VERBAL - PRUEBA FONETICA` | -0.179 | 0.005 | 249 |
| `label_ratio_ES` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS` | 0.169 | 0.008 | 249 |
| `label_ratio_PAUSA` | `FLUIDEZ VERBAL - PRUEBA FONETICA` | -0.160 | 0.011 | 249 |
| `label_ratio_EE` | `Age` | 0.158 | 0.013 | 249 |
| `label_ratio_ES` | `COHERENCIA NARRATIVA` | 0.157 | 0.013 | 249 |
| `label_ratio_PAUSA` | `Naming task` | -0.156 | 0.013 | 249 |
| `label_ratio_PAUSA` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS` | -0.156 | 0.014 | 249 |

Interpretación: Actividad5 parece sensible al desempeño verbal. Los sujetos con mejor fluidez tienden a producir grafos con más nodos, mayor ASP/diámetro y menor densidad/LSC ratio. En paralelo, más pausas se asocian con peor desempeño en fluidez, coherencia narrativa y conceptualización.

## 10. Actividad2: señal moderada de edad/escolaridad y disfluencias
Actividad2 mostró asociaciones moderadas con edad y escolaridad. Mayor edad o curso se relaciona con más nodos y menor repetición/densidad. En etiquetas, `PAUSA`, `PS`, `DP` y `DI` aparecen vinculadas con coherencia narrativa o fluidez, aunque con tamaños de efecto más modestos.

### Actividad2: grafos
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_repeated_edges_ratio` | `Age` | -0.318 | 5.98e-07 | 236 |
| `mean_repeated_edges_ratio` | `School year` | -0.295 | 3.97e-06 | 236 |
| `mean_nodes` | `Age` | 0.278 | 1.51e-05 | 236 |
| `mean_nodes` | `School year` | 0.252 | 8.87e-05 | 236 |
| `mean_density` | `Age` | -0.239 | 2.15e-04 | 236 |
| `mean_diameter` | `Age` | 0.230 | 3.58e-04 | 236 |
| `mean_asp` | `Age` | 0.230 | 3.75e-04 | 236 |
| `mean_l2` | `Age` | -0.227 | 4.44e-04 | 236 |
| `mean_asp` | `School year` | 0.222 | 5.93e-04 | 236 |
| `mean_diameter` | `School year` | 0.221 | 6.26e-04 | 236 |
| `mean_density` | `School year` | -0.212 | 0.001 | 236 |
| `mean_l2` | `School year` | -0.209 | 0.001 | 236 |

### Actividad2: etiquetas
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `label_ratio_PAUSA` | `FLUIDEZ VERBAL - PRUEBA FONETICA` | -0.259 | 5.68e-05 | 236 |
| `label_ratio_PS` | `COHERENCIA NARRATIVA` | -0.235 | 2.63e-04 | 236 |
| `label_ratio_PS` | `School year` | -0.220 | 6.51e-04 | 236 |
| `label_ratio_PAUSA` | `COHERENCIA NARRATIVA` | -0.216 | 8.36e-04 | 236 |
| `label_ratio_PS` | `Age` | -0.208 | 0.001 | 236 |
| `label_ratio_DP` | `COHERENCIA NARRATIVA` | -0.190 | 0.003 | 236 |
| `label_ratio_PS` | `Reading comprehension task` | -0.181 | 0.005 | 236 |
| `label_ratio_PAUSA` | `COG` | -0.178 | 0.006 | 236 |
| `label_ratio_DI` | `COHERENCIA NARRATIVA` | -0.172 | 0.008 | 236 |
| `label_ratio_PAUSA` | `COG_zscore` | -0.164 | 0.012 | 236 |
| `label_ratio_EE` | `Age` | 0.156 | 0.016 | 236 |
| `label_ratio_EE` | `School year` | 0.153 | 0.019 | 236 |

## 11. Actividad1 y Actividad3: señales secundarias
Actividad1 presentó asociaciones pequeñas con edad, curso y algunas tareas de fluidez. Debido a su menor número de casos válidos con ventana 30, debe ser considerada principalmente como análisis secundario.

### Actividad1
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_nodes` | `School year` | 0.224 | 0.002 | 181 |
| `mean_density` | `Age` | -0.223 | 0.003 | 181 |
| `mean_density` | `School year` | -0.221 | 0.003 | 181 |
| `mean_nodes` | `Age` | 0.220 | 0.003 | 181 |
| `mean_lsc` | `School year` | 0.200 | 0.007 | 181 |
| `mean_lsc` | `Age` | 0.187 | 0.012 | 181 |
| `mean_repeated_edges_ratio` | `Age` | -0.178 | 0.017 | 181 |
| `mean_repeated_edges_ratio` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.165 | 0.026 | 181 |

| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `label_ratio_DI` | `COHERENCIA NARRATIVA` | -0.277 | 1.62e-04 | 181 |
| `label_ratio_DI` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.270 | 2.35e-04 | 181 |
| `label_ratio_DI` | `Verbal fluency tasks` | -0.260 | 3.99e-04 | 181 |
| `label_ratio_EE` | `Reading comprehension task` | -0.253 | 5.92e-04 | 181 |
| `label_ratio_DI` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS` | -0.198 | 0.008 | 181 |
| `label_ratio_EE` | `MOT_zscore` | -0.172 | 0.021 | 181 |
| `label_ratio_EE` | `MOT` | -0.166 | 0.025 | 181 |
| `label_ratio_PNC` | `Conceptualization task` | 0.160 | 0.031 | 181 |

Actividad3 mostró asociaciones moderadas con conceptualización, fluidez semántica y edad, pero su patrón fue menos central que el observado en Actividad4, Actividad5, Actividad6 y Actividad7.

### Actividad3
| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `mean_nodes` | `Conceptualization task` | -0.282 | 2.72e-05 | 215 |
| `mean_diameter` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.266 | 7.68e-05 | 215 |
| `mean_density` | `Conceptualization task` | 0.254 | 1.68e-04 | 215 |
| `mean_repeated_edges_ratio` | `Conceptualization task` | 0.243 | 3.15e-04 | 215 |
| `mean_asp` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.242 | 3.44e-04 | 215 |
| `mean_asp` | `Conceptualization task` | -0.239 | 4.15e-04 | 215 |
| `mean_nodes` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES` | -0.232 | 5.99e-04 | 215 |
| `mean_diameter` | `Conceptualization task` | -0.221 | 0.001 | 215 |

| Métrica | Variable | r Spearman | p | n |
|---|---:|---:|---:|---:|
| `label_ratio_SIN_RESPUESTA` | `Conceptualization task` | -0.219 | 0.001 | 215 |
| `label_ratio_SIN_PREGUNTA` | `Reading comprehension task` | -0.200 | 0.003 | 215 |
| `label_ratio_PS` | `School year` | -0.193 | 0.005 | 215 |
| `label_ratio_SIN_RESPUESTA` | `Verbal fluency tasks` | -0.188 | 0.006 | 215 |
| `label_ratio_SIN_RESPUESTA` | `NPLAN` | 0.183 | 0.007 | 215 |
| `label_ratio_SIN_PREGUNTA` | `FLUIDEZ VERBAL - PRUEBA FONETICA` | -0.172 | 0.012 | 215 |
| `label_ratio_SIN_RESPUESTA` | `FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS` | -0.171 | 0.012 | 215 |
| `label_ratio_SIN_PREGUNTA` | `Verbal fluency tasks` | -0.171 | 0.012 | 215 |

## 12. Prevalencia de etiquetas por actividad
### Actividad1
| Etiqueta | Sujetos con ratio > 0 | n actividad | % |
|---|---:|---:|---:|
| `EE` | 196 | 254 | 77.2% |
| `PS` | 173 | 254 | 68.1% |
| `DI` | 111 | 254 | 43.7% |
| `IF` | 57 | 254 | 22.4% |
| `PAUSA` | 33 | 254 | 13.0% |
| `DP` | 24 | 254 | 9.4% |
| `PNC` | 2 | 254 | 0.8% |
| `IM` | 1 | 254 | 0.4% |

### Actividad2
| Etiqueta | Sujetos con ratio > 0 | n actividad | % |
|---|---:|---:|---:|
| `PS` | 237 | 251 | 94.4% |
| `EE` | 187 | 251 | 74.5% |
| `IF` | 176 | 251 | 70.1% |
| `DI` | 76 | 251 | 30.3% |
| `PAUSA` | 33 | 251 | 13.1% |
| `DP` | 22 | 251 | 8.8% |
| `ES` | 6 | 251 | 2.4% |
| `IM` | 5 | 251 | 2.0% |

### Actividad3
| Etiqueta | Sujetos con ratio > 0 | n actividad | % |
|---|---:|---:|---:|
| `PS` | 199 | 250 | 79.6% |
| `EE` | 171 | 250 | 68.4% |
| `SIN_RESPUESTA` | 147 | 250 | 58.8% |
| `IF` | 82 | 250 | 32.8% |
| `DI` | 55 | 250 | 22.0% |
| `PAUSA` | 35 | 250 | 14.0% |
| `DP` | 11 | 250 | 4.4% |
| `SIN_PREGUNTA` | 8 | 250 | 3.2% |

### Actividad4
| Etiqueta | Sujetos con ratio > 0 | n actividad | % |
|---|---:|---:|---:|
| `PS` | 240 | 253 | 94.9% |
| `IF` | 227 | 253 | 89.7% |
| `EE` | 165 | 253 | 65.2% |
| `DI` | 35 | 253 | 13.8% |
| `PAUSA` | 22 | 253 | 8.7% |
| `SIN_RESPUESTA` | 18 | 253 | 7.1% |
| `DP` | 7 | 253 | 2.8% |
| `ES` | 4 | 253 | 1.6% |

### Actividad5
| Etiqueta | Sujetos con ratio > 0 | n actividad | % |
|---|---:|---:|---:|
| `EE` | 244 | 251 | 97.2% |
| `PAUSA` | 233 | 251 | 92.8% |
| `DI` | 159 | 251 | 63.3% |
| `PS` | 113 | 251 | 45.0% |
| `IF` | 54 | 251 | 21.5% |
| `DP` | 42 | 251 | 16.7% |
| `IM` | 14 | 251 | 5.6% |
| `ES` | 5 | 251 | 2.0% |

### Actividad6
| Etiqueta | Sujetos con ratio > 0 | n actividad | % |
|---|---:|---:|---:|
| `PS` | 247 | 253 | 97.6% |
| `IF` | 233 | 253 | 92.1% |
| `EE` | 219 | 253 | 86.6% |
| `SIN_PREGUNTA` | 108 | 253 | 42.7% |
| `DI` | 57 | 253 | 22.5% |
| `PAUSA` | 36 | 253 | 14.2% |
| `DP` | 17 | 253 | 6.7% |
| `ES` | 11 | 253 | 4.3% |

### Actividad7
| Etiqueta | Sujetos con ratio > 0 | n actividad | % |
|---|---:|---:|---:|
| `PS` | 237 | 251 | 94.4% |
| `EE` | 189 | 251 | 75.3% |
| `IF` | 154 | 251 | 61.4% |
| `SIN_PREGUNTA` | 115 | 251 | 45.8% |
| `SIN_RESPUESTA` | 50 | 251 | 19.9% |
| `DI` | 31 | 251 | 12.4% |
| `PAUSA` | 15 | 251 | 6.0% |
| `DP` | 8 | 251 | 3.2% |

## 13. Diferencias exploratorias entre high_imp y low_imp
Se calcularon comparaciones exploratorias entre `high_imp` y `low_imp` mediante pruebas no paramétricas sobre métricas de grafos y etiquetas. Estos resultados no deben interpretarse como inferencia final porque no están corregidos por múltiples comparaciones y el run no incluye baseline aleatorio.
### Actividad2
| Métrica | Mediana high_imp | Mediana low_imp | Dif. high-low | p | n high / low |
|---|---:|---:|---:|---:|---:|
| `label_ratio_PS` | 0.0692 | 0.0542 | 0.0150 | 0.032 | 118 / 118 |

### Actividad4
| Métrica | Mediana high_imp | Mediana low_imp | Dif. high-low | p | n high / low |
|---|---:|---:|---:|---:|---:|
| `mean_repeated_edges_ratio` | 0.0334 | 0.0296 | 0.0038 | 0.035 | 122 / 131 |
| `label_ratio_ES` | 0.0000 | 0.0000 | 0.0000 | 0.037 | 122 / 131 |

### Actividad5
No se detectaron diferencias high_imp vs low_imp con p < .05 bajo el filtro aplicado.

### Actividad6
| Métrica | Mediana high_imp | Mediana low_imp | Dif. high-low | p | n high / low |
|---|---:|---:|---:|---:|---:|
| `label_ratio_SIN_RESPUESTA` | 0.0000 | 0.0000 | 0.0000 | 0.040 | 121 / 132 |

### Actividad7
| Métrica | Mediana high_imp | Mediana low_imp | Dif. high-low | p | n high / low |
|---|---:|---:|---:|---:|---:|
| `label_ratio_SIN_RESPUESTA` | 0.0000 | 0.0000 | 0.0000 | 0.002 | 120 / 130 |
| `label_ratio_DP` | 0.0000 | 0.0000 | 0.0000 | 0.006 | 120 / 130 |
| `label_ratio_PAUSA` | 0.0000 | 0.0000 | 0.0000 | 0.028 | 120 / 130 |
| `mean_lsc` | 19.0498 | 18.6293 | 0.4206 | 0.028 | 120 / 130 |
| `mean_lsc_ratio` | 0.7863 | 0.7614 | 0.0249 | 0.037 | 120 / 130 |

## 14. Hallazgos candidatos principales
1. **Edad y escolaridad son moduladores principales del discurso**: las métricas de grafos cambian de forma consistente con edad y curso escolar, especialmente en Actividad4 y Actividad6.
2. **Actividad6 parece capturar desarrollo narrativo**: mayor edad/curso se asocia con más nodos, menor densidad, menor repetición y trayectorias más largas.
3. **Actividad4 destaca por `SIN_RESPUESTA`**: la ausencia de respuesta se asocia con menor comprensión lectora, menor fluidez verbal y menor coherencia narrativa.
4. **Actividad7 concentra la señal con Barratt**: `SIN_RESPUESTA` se asocia positivamente con impulsividad no planificada y puntajes agregados de Barratt.
5. **Actividad5 relaciona pausas y estructura con fluidez**: más pausas se asocian con menor rendimiento cognitivo-lingüístico, y las métricas de grafos se alinean con desempeño en fluidez verbal.
6. **Las etiquetas aportan información complementaria a los grafos**: en varias actividades, las etiquetas discursivas fueron más interpretables que las métricas puramente estructurales para explicar Barratt o desempeño cognitivo.

## 15. Limitaciones
- Este análisis no incluye baseline aleatorio; por tanto, las métricas estructurales todavía no están normalizadas contra grafos permutados.
- Las correlaciones no están corregidas por múltiples comparaciones; los hallazgos deben tratarse como señales candidatas.
- Edad y escolaridad modulan fuertemente los resultados, por lo que futuras inferencias sobre Barratt o cognición deben controlarlas explícitamente.
- `SIN_PREGUNTA` no debe interpretarse como rasgo del participante, sino como indicador de administración/protocolo.
- Actividad1 tiene menor cobertura válida con ventana 30, por lo que sus resultados deben considerarse secundarios.

## 16. Próximos pasos recomendados
1. Repetir el análisis por actividad con baseline aleatorio liviano: `--random-times 20`, manteniendo ventana 30.
2. Validar los hallazgos más importantes con `random100` solo para Actividad4, Actividad6 y Actividad7 si el costo computacional es alto.
3. Generar modelos parciales o regresiones controlando edad y `School year` para separar desarrollo de impulsividad/cognición.
4. Construir perfiles por sujeto y actividad integrando: métricas de grafos, etiquetas `SIN_RESPUESTA`, `PAUSA`, `IF`, `EE`, y variables Barratt/cognitivas.
5. Tratar Actividad4 como actividad secundaria prioritaria, aunque no haya sido target inicial, debido a su fuerte asociación con desempeño cognitivo-lingüístico.

## 17. Conclusión
El análisis por actividad sugiere que la estructura del discurso no es homogénea entre tareas. Las métricas de grafos capturan principalmente variación asociada a edad y escolaridad, mientras que las etiquetas discursivas, especialmente `SIN_RESPUESTA` y `PAUSA`, capturan señales más directamente vinculadas con desempeño cognitivo-lingüístico e impulsividad. En conjunto, los resultados respaldan una estrategia de perfilamiento NLP por actividad, en la que Actividad6 aporta sensibilidad al desarrollo narrativo, Actividad4 aporta marcadores de dificultad cognitiva y Actividad7 aporta señales relacionadas con impulsividad Barratt.