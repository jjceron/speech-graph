# README de análisis — `01_w30_no_random_all` / ventana 30

Este README resume cómo analizar, exponer y discutir los resultados del run completo con ventana de 30 palabras. Aunque en la conversación se mencionó como `random_all`, el run revisado aquí corresponde técnicamente a:

```bash
py -m src.pipeline.mota_test --random-times 0 --window-sizes 30 --output-dir outputs/01_w30_no_random_all
```

Es decir, es un análisis **completo**, con **todos los sujetos disponibles**, **ventana móvil de 30 palabras**, pero **sin baseline aleatorio**. Por eso debe interpretarse como un análisis exploratorio fuerte, no como el análisis final estilo Mota con z-scores randomizados.

La visualización se generó con:

```bash
py -m src.visualization --runs 01_w30_no_random_all --window-size 30 --level file --min-n 30 --min-abs-r 0.15
```

---

## 1. Objetivo del run

El objetivo de este run fue evaluar, en toda la muestra disponible, si las métricas de grafos de palabras y las etiquetas discursivas `[[...]]` se relacionan con:

- impulsividad Barratt (`TOTAL`, `NPLAN`, `MOT`, `COG`);
- edad y escolaridad;
- tareas cognitivas y lingüísticas;
- coherencia narrativa;
- perfiles extremos por sujeto.

La unidad de análisis fue la transcripción completa de cada sujeto (`level = file`) resumida mediante ventanas móviles de 30 palabras.

---

## 2. Archivos principales del run

Los archivos centrales para revisar este run son:

```text
outputs/01_w30_no_random_all/graph_metrics_all_windows.csv
outputs/01_w30_no_random_all/graph_metrics_all_windows_with_meta.csv
outputs/01_w30_no_random_all/analysis/correlations_by_window.csv
outputs/01_w30_no_random_all/analysis/metadata_unmatched_transcripts.csv
outputs/01_w30_no_random_all/figures/nlp_profile/core/relevant_correlations_w30.csv
outputs/01_w30_no_random_all/figures/nlp_profile/labels/label_prevalence_w30.csv
outputs/01_w30_no_random_all/figures/nlp_profile/labels/label_relevant_correlations_w30.csv
outputs/01_w30_no_random_all/figures/nlp_profile/subjects/subject_nlp_profile_w30.csv
outputs/01_w30_no_random_all/figures/nlp_profile/subjects/subject_disorganization_proxy_correlations_w30.csv
```

Figuras principales:

```text
outputs/01_w30_no_random_all/figures/nlp_profile/core/heatmap_barratt_w30.png
outputs/01_w30_no_random_all/figures/nlp_profile/core/heatmap_cognitive_w30.png
outputs/01_w30_no_random_all/figures/nlp_profile/core/heatmap_demographic_w30.png
outputs/01_w30_no_random_all/figures/nlp_profile/labels/label_correlations_heatmap_w30.png
outputs/01_w30_no_random_all/figures/nlp_profile/labels/label_nonzero_counts_w30.png
outputs/01_w30_no_random_all/figures/nlp_profile/subjects/subject_rank_disorganization_proxy_w30.png
outputs/01_w30_no_random_all/figures/nlp_profile/subjects/subject_disorganization_vs_Age_w30.png
outputs/01_w30_no_random_all/figures/nlp_profile/subjects/subject_disorganization_vs_School_year_w30.png
```

---

## 3. Control técnico del dataset

El pipeline procesó:

| Elemento | Resultado |
|---|---:|
| Transcripciones procesadas | 267 |
| Ventana usada | 30 palabras |
| Random baseline | No (`random_times = 0`) |
| Nivel | `file` |
| Sujetos con metadata usable | 252 |
| Transcripciones sin match en metadata | 15 |

Las 15 transcripciones sin match en el Excel fueron:

```text
CMU-12-6A-SPURUR
CMU-12-6C-FAPUIP
CMU-13-6A-CMEPUR
CMU-13-6A-VVSIEP
CMU-14-6A-DFNUPU
CMU-14-6A-WJSUIG
CMU-14-6A-YAIPBO
CMU-14-8B-KYURIP
CMU-16-11-SJEPIP
CMU-16-11A-DDMODU
CMU-16-8B-LLEPPU
CMU-XX-10A-DABAUR
CMU-XX-7A-RAEPEP
CMU-XX-7B-CEPPU
CMU-XX-9B-MBPUPU
```

Estos casos no deben entrar en interpretaciones con Barratt, edad, escolaridad o tareas cognitivas hasta corregir el match con la metadata. Sí aparecen en algunos rankings de sujeto porque tienen métricas NLP, pero no tienen variables clínicas/cognitivas asociadas.

Distribución de `Tipo` en los sujetos con metadata:

| Tipo | n |
|---|---:|
| `low_imp` | 131 |
| `high_imp` | 121 |
| Sin metadata | 15 |

---

## 4. Resumen ejecutivo de resultados

El resultado principal del run completo no es una asociación fuerte entre métricas de grafos y Barratt total. El patrón más claro fue este:

1. **Edad y escolaridad dominan las métricas estructurales de grafos.**  
   Sujetos mayores o con mayor curso tienden a tener más nodos por ventana, menor densidad, menor repetición de aristas y menor proxy de desorganización.

2. **Las asociaciones directas entre grafos y Barratt agregado son débiles.**  
   El patrón observado en el subset de 20 sujetos se debilitó al usar toda la muestra.

3. **La etiqueta `SIN_RESPUESTA` es el marcador discursivo más prometedor.**  
   Se asocia con mayor `NPLAN` y con peor desempeño en coherencia narrativa, fluidez verbal y conceptualización.

4. **El proxy de desorganización no debe interpretarse sin controlar edad y escolaridad.**  
   Su correlación con edad y curso es moderada y negativa.

5. **El ranking de sujetos sirve para revisión cualitativa, no para diagnóstico.**  
   Hay sobrerrepresentación de `high_imp` entre los perfiles más extremos, pero no separación limpia entre grupos.

---

## 5. Resultados demográficos: edad y escolaridad

Las asociaciones más fuertes del análisis aparecen con `Age` y `School year`.

| Métrica NLP | Variable | Spearman r | p | n |
|---|---:|---:|---:|---:|
| `mean_nodes` | `School year` | 0.483 | 3.92e-16 | 252 |
| `mean_density` | `School year` | -0.471 | 2.75e-15 | 252 |
| `mean_nodes` | `Age` | 0.468 | 4.06e-15 | 252 |
| `mean_density` | `Age` | -0.460 | 1.30e-14 | 252 |
| `mean_repeated_edges_ratio` | `Age` | -0.458 | 1.90e-14 | 252 |
| `mean_repeated_edges_ratio` | `School year` | -0.450 | 5.65e-14 | 252 |
| `mean_asp` | `School year` | 0.411 | 1.13e-11 | 252 |
| `mean_diameter` | `School year` | 0.400 | 4.06e-11 | 252 |
| `mean_asp` | `Age` | 0.380 | 4.28e-10 | 252 |
| `mean_diameter` | `Age` | 0.369 | 1.47e-09 | 252 |

### Interpretación

En ventanas fijas de 30 palabras, los sujetos de mayor edad o mayor curso tienden a mostrar:

- más palabras únicas por ventana (`mean_nodes` más alto);
- menor densidad (`mean_density` más bajo);
- menor proporción de aristas repetidas;
- trayectorias más largas (`mean_asp`, `mean_diameter` más altos).

Esto sugiere un patrón compatible con mayor diversidad léxica/estructural a medida que aumenta el desarrollo escolar. Sin embargo, hay que ser prudentes: en una ventana fija de 30 palabras, muchas métricas están matemáticamente conectadas. Si suben los nodos únicos, la densidad tiende a bajar. Por eso estos resultados deben leerse como un perfil estructural-descriptivo, no como causalidad.

---

## 6. Proxy de desorganización NLP

El archivo `subject_disorganization_proxy_correlations_w30.csv` mostró:

| Target | Spearman r | p | n |
|---|---:|---:|---:|
| `Age` | -0.469 | 3.37e-15 | 252 |
| `School year` | -0.456 | 2.28e-14 | 252 |

### Interpretación

El proxy de desorganización disminuye con edad y escolaridad. Esto significa que los sujetos más jóvenes tienden a aparecer con mayor desorganización según el proxy actual.

Este punto es central para la discusión: el proxy todavía no puede interpretarse como indicador clínico o de impulsividad sin controlar desarrollo y escolaridad.

### Recomendación

Para un perfilamiento más justo, conviene crear una versión residualizada del proxy controlando por:

```text
Age
School year
```

Así se podrá distinguir si un sujeto aparece extremo por rasgos discursivos propios o simplemente porque pertenece a un grupo de menor edad/curso.

---

## 7. Resultados Barratt

En el subset de 20 sujetos había señales entre `NPLAN` y algunas métricas estructurales. En la muestra completa esas señales se debilitaron.

Correlaciones exploratorias relevantes:

| Relación | Spearman r | p | n |
|---|---:|---:|---:|
| `disorganization_proxy` ~ `NPLAN` | 0.101 | 0.110 | 252 |
| `disorganization_proxy` ~ `TOTAL` | 0.046 | 0.472 | 252 |
| `disorganization_proxy` ~ `MOT` | 0.050 | 0.428 | 252 |
| `disorganization_proxy` ~ `COG` | -0.094 | 0.139 | 252 |

### Interpretación

No aparece una relación robusta entre el proxy estructural general y Barratt total o subescalas agregadas. Esto sugiere que el patrón de grafos no está simplemente reflejando impulsividad general.

No obstante, sí aparece una señal más clara en etiquetas discursivas, especialmente `SIN_RESPUESTA`, que se discute en la sección siguiente.

---

## 8. Etiquetas discursivas `[[...]]`

### 8.1 Prevalencia

Las etiquetas con mayor presencia fueron:

| Etiqueta | Proporción media | Sujetos con etiqueta | n total |
|---|---:|---:|---:|
| `IF` | 0.0122 | 267 | 267 |
| `PS` | 0.0331 | 266 | 267 |
| `EE` | 0.0210 | 263 | 267 |
| `PAUSA` | 0.0072 | 250 | 267 |
| `DI` | 0.0041 | 213 | 267 |
| `SIN_RESPUESTA` | 0.0014 | 168 | 267 |
| `SIN_PREGUNTA` | 0.0013 | 120 | 267 |
| `DP` | 0.0006 | 95 | 267 |
| `PNC` | 0.0003 | 30 | 267 |
| `IM` | 0.0004 | 27 | 267 |

### 8.2 Correlaciones relevantes

El heatmap de etiquetas en el run completo muestra menos etiquetas y menos variables que el subset de 20 sujetos porque el filtro conservador (`min-n = 30`, `min-abs-r = 0.15`) deja pasar solo señales estables.

Asociaciones relevantes:

| Feature | Variable | Spearman r | p | n |
|---|---:|---:|---:|---:|
| `label_ratio_SIN_RESPUESTA` | `COHERENCIA NARRATIVA` | -0.301 | 1.0e-06 | 252 |
| `label_ratio_IF` | `School year` | -0.288 | 3.0e-06 | 252 |
| `label_ratio_SIN_RESPUESTA` | `NPLAN` | 0.285 | 4.0e-06 | 252 |
| `label_ratio_SIN_RESPUESTA` | `Verbal fluency tasks` | -0.279 | 7.0e-06 | 252 |
| `label_ratio_IF` | `Age` | -0.276 | 9.0e-06 | 252 |
| `label_ratio_SIN_PREGUNTA` | `Naming task` | 0.274 | 1.0e-05 | 252 |
| `label_ratio_SIN_RESPUESTA` | `Conceptualization task` | -0.255 | 4.3e-05 | 252 |

### Interpretación

La señal más consistente es `SIN_RESPUESTA`:

- más `SIN_RESPUESTA` se asocia con mayor `NPLAN`;
- más `SIN_RESPUESTA` se asocia con menor coherencia narrativa;
- más `SIN_RESPUESTA` se asocia con menor fluidez verbal;
- más `SIN_RESPUESTA` se asocia con peor desempeño en conceptualización.

Esto convierte a `SIN_RESPUESTA` en un marcador discursivo muy relevante para el perfil NLP. A diferencia de algunas métricas estructurales de grafos, esta etiqueta tiene una interpretación directa: momentos en los que el sujeto no responde o no produce contenido esperado.

### Comparación high_imp vs low_imp

En el perfil por grupo, `SIN_RESPUESTA` también diferencia grupos:

| Variable | Media high_imp | Media low_imp | p Mann-Whitney |
|---|---:|---:|---:|
| `label_ratio_SIN_RESPUESTA` | 0.00165 | 0.00108 | 0.000842 |

Esto sugiere que el grupo `high_imp` presenta más proporción de `SIN_RESPUESTA` que el grupo `low_imp`.

Este resultado es más prometedor que las asociaciones directas entre grafos y Barratt total.

---

## 9. Resultados cognitivo-lingüísticos

Las asociaciones entre métricas de grafos y tareas cognitivas fueron más pequeñas que las demográficas. Aun así, hay algunas señales:

| Métrica NLP | Variable | Spearman r | p | n |
|---|---:|---:|---:|---:|
| `mean_l3` | `COHERENCIA NARRATIVA` | -0.174 | 0.00565 | 252 |
| `mean_diameter` | `Naming task` | 0.168 | 0.00749 | 252 |
| `mean_asp` | `Naming task` | 0.166 | 0.00834 | 252 |
| `mean_density` | `Naming task` | -0.162 | 0.00989 | 252 |
| `mean_nodes` | `Naming task` | 0.158 | 0.0122 | 252 |
| `mean_lsc_ratio` | `Naming task` | -0.152 | 0.0156 | 252 |

### Interpretación

El patrón es débil pero coherente: mejor desempeño en denominación parece asociarse con más nodos, mayor distancia promedio y menor densidad. Esto puede reflejar mayor diversidad en las ventanas de 30 palabras.

Sin embargo, estas correlaciones son pequeñas. Para exposición oral o presentación, deben quedar como resultados secundarios, no como eje principal.

---

## 10. Perfil por grupo high_imp / low_imp

Promedios relevantes:

| Variable | high_imp | low_imp |
|---|---:|---:|
| `disorganization_proxy` | 0.0319 | -0.0714 |
| `mean_nodes` | 24.1589 | 24.3176 |
| `mean_density` | 0.1013 | 0.1000 |
| `mean_lsc_ratio` | 0.7412 | 0.7362 |
| `mean_asp` | 4.4409 | 4.4968 |
| `label_ratio_SIN_RESPUESTA` | 0.00165 | 0.00108 |
| `Age` | 12.3884 | 12.3053 |
| `School year` | 6.9917 | 6.9008 |
| `NPLAN` | 13.7603 | 7.1603 |
| `TOTAL` | 43.6860 | 22.9237 |

Comparaciones exploratorias Mann-Whitney:

| Variable | p |
|---|---:|
| `disorganization_proxy` | 0.185 |
| `label_ratio_SIN_RESPUESTA` | 0.000842 |
| `label_ratio_IF` | 0.759 |
| `label_ratio_SIN_PREGUNTA` | 0.714 |
| `mean_density` | 0.173 |
| `mean_nodes` | 0.195 |

### Interpretación

La diferencia entre grupos no aparece fuerte en métricas estructurales globales, pero sí en `SIN_RESPUESTA`. Esto apoya una lectura combinada:

- los grafos capturan desarrollo/estructura general del discurso;
- las etiquetas capturan eventos discursivos específicos más cercanos al desempeño o dificultad durante la tarea;
- `SIN_RESPUESTA` parece especialmente sensible a impulsividad no planificada y perfil cognitivo-lingüístico.

---

## 11. Ranking de sujetos extremos

El ranking `subject_rank_disorganization_proxy_w30.png` identifica sujetos con mayor proxy de desorganización.

Top 15:

| Rank | Código | Tipo | Proxy | Edad | Curso |
|---:|---|---|---:|---:|---:|
| 1 | `CSO-10-5A-JLCARI` | high_imp | 2.002 | 10 | 5 |
| 2 | `CDMS-11-6B-JACUPE` | high_imp | 1.501 | 11 | 6 |
| 3 | `CDMS-8-4A-DJAVDA` | low_imp | 1.402 | 8 | 4 |
| 4 | `CMU-14-6A-DFNUPU` | sin metadata | 1.381 | NA | NA |
| 5 | `CDMS-10-5C-ICAPA` | high_imp | 1.336 | 10 | 5 |
| 6 | `CMU-12-6A-SPURUR` | sin metadata | 1.297 | NA | NA |
| 7 | `CSO-10-5A-DABRZA` | low_imp | 1.251 | 10 | 5 |
| 8 | `CMU-14-6A-YAIPBO` | sin metadata | 1.197 | NA | NA |
| 9 | `CDMS-8-4C-JDBUTU` | low_imp | 1.183 | 8 | 4 |
| 10 | `CDMS-9-5A-ABERO` | high_imp | 1.092 | 9 | 5 |
| 11 | `CSO-9-4A-SLOCA` | low_imp | 1.031 | 9 | 4 |
| 12 | `CSO-12-7A-DALRE` | high_imp | 1.008 | 12 | 7 |
| 13 | `CDMS-12-6B-SCLAG` | high_imp | 0.995 | 12 | 6 |
| 14 | `CDMS-9-5C-JDMADE` | high_imp | 0.934 | 9 | 5 |
| 15 | `CDMS-11-6B-JDCHBA` | high_imp | 0.878 | 11 | 6 |

En el top 30 hubo:

| Tipo | n |
|---|---:|
| high_imp | 17 |
| low_imp | 8 |
| sin metadata | 5 |

### Interpretación

Hay cierta sobrerrepresentación de `high_imp` entre los sujetos con mayor proxy, pero no hay separación limpia. También hay `low_imp` altos y `high_imp` bajos.

Esto es metodológicamente útil: el perfil NLP no parece ser una simple copia de la clasificación Barratt. Puede aportar información individual adicional. Sin embargo, por la fuerte influencia de edad/curso, el ranking debe usarse solo para revisión cualitativa hasta tener un proxy residualizado.

---

## 12. Cómo exponer estos resultados

Una forma clara de presentar estos hallazgos sería:

### Diapositiva 1 — Pipeline

- Transcripciones en español procesadas.
- Preservación de etiquetas `[[...]]`.
- Grafos dirigidos de palabras consecutivas.
- Ventanas móviles de 30 palabras.
- Métricas estructurales + etiquetas + metadata.

### Diapositiva 2 — Control de muestra

- 267 transcripciones procesadas.
- 252 con metadata válida.
- 15 sin match que deben revisarse.
- Grupos: 121 `high_imp`, 131 `low_imp`.

### Diapositiva 3 — Hallazgo estructural principal

- Edad y escolaridad explican una parte importante del perfil de grafos.
- A mayor edad/curso: más nodos, menor densidad, menos repetición, menor proxy de desorganización.

### Diapositiva 4 — Barratt

- Las métricas estructurales puras no muestran relación fuerte con Barratt total.
- El patrón más interesante aparece en `SIN_RESPUESTA` y `NPLAN`.

### Diapositiva 5 — Etiquetas discursivas

- `SIN_RESPUESTA` se asocia con:
  - mayor `NPLAN`;
  - menor coherencia narrativa;
  - menor fluidez verbal;
  - peor conceptualización.

### Diapositiva 6 — Perfilamiento de sujetos

- Ranking de sujetos extremos.
- Mezcla de high/low imp.
- Uso cualitativo, no diagnóstico.

### Diapositiva 7 — Conclusión y próximos pasos

- Controlar edad/escolaridad.
- Corregir sujetos sin match.
- Limpiar variantes de etiquetas.
- Validar con baseline random.

---

## 13. Cómo discutir los resultados

La discusión debe enfatizar tres ideas.

### 13.1 El discurso cambia con edad/escolaridad

El análisis muestra que muchas métricas de grafos están asociadas con desarrollo escolar. Esto es esperable: a mayor edad y curso, el sujeto puede producir discurso más variado, menos repetitivo y menos compacto localmente.

Por eso, cualquier análisis clínico o cognitivo debe controlar por edad y escolaridad.

### 13.2 Las etiquetas no son ruido

Las etiquetas `[[...]]` son informativas. El caso más claro es `SIN_RESPUESTA`, que se relaciona con impulsividad no planificada y tareas cognitivas. Esto justifica haber preservado las etiquetas en lugar de eliminarlas durante el preprocesamiento.

### 13.3 Las métricas estructurales y las etiquetas capturan cosas distintas

Las métricas de grafos parecen capturar estructura global/desarrollo del discurso. Las etiquetas capturan eventos discursivos puntuales: pausas, falta de respuesta, interrupciones o fenómenos anotados. Ambas capas son complementarias.

---

## 14. Limitaciones

Este run tiene varias limitaciones importantes:

1. **No usa baseline aleatorio.**  
   No hay z-scores contra grafos permutados. Por tanto, no se puede afirmar todavía que una estructura sea mayor o menor que lo esperado por azar.

2. **Edad y escolaridad confunden el proxy.**  
   El proxy de desorganización tiene correlaciones moderadas con edad y curso.

3. **Hay 15 transcripciones sin metadata.**  
   Deben resolverse antes del análisis final.

4. **Algunas etiquetas necesitan limpieza.**  
   Aparecen variantes como `SIN_RESPUESTA` / `SIN_REPUESTA`, `PAUSA` / `Pausa` / `PAUA`, `IF` / `if`, y etiquetas con `StarTime` o `StartTime`.

5. **Múltiples comparaciones.**  
   Se exploraron muchas métricas y muchas variables. Los resultados deben considerarse candidatos hasta validación posterior.

---

## 15. Recomendaciones para el siguiente análisis

### 15.1 Corrección de metadata

Resolver los 15 sujetos sin match en Excel.

### 15.2 Limpieza de etiquetas

Unificar variantes:

```text
SIN_RESPUESTA / SIN_REPUESTA
PAUSA / Pausa / PAUA
IF / if
EE / EEE
DI_StarTime_* → DI
PAUSA_StarTime_* → PAUSA
```

### 15.3 Proxy residualizado

Crear un proxy de desorganización controlado por edad y escolaridad:

```text
disorganization_proxy_resid ~ disorganization_proxy - Age - School year
```

Esto permitirá perfilar sujetos extremos sin que el ranking esté dominado por desarrollo escolar.

### 15.4 Random baseline liviano

Validar señales estructurales con:

```bash
py -m src.pipeline.mota_test --random-times 20 --window-sizes 30 --output-dir outputs/02_w30_random20_all
```

Luego visualizar:

```bash
py -m src.visualization --runs 02_w30_random20_all --window-size 30 --level file --min-n 30 --min-abs-r 0.15
```

### 15.5 Preguntas prioritarias para random20

- ¿Los z-scores estructurales siguen asociados con edad y escolaridad?
- ¿Aparece alguna relación robusta entre z-scores y `NPLAN`?
- ¿El patrón de `SIN_RESPUESTA` se mantiene?
- ¿El ranking de sujetos extremos cambia al usar z-scores?

---

## 16. Conclusión breve

El run completo con ventana 30 y sin baseline aleatorio muestra que las métricas estructurales de grafos están fuertemente moduladas por edad y escolaridad. Las asociaciones directas entre grafos y Barratt son débiles en la muestra completa. En cambio, las etiquetas discursivas, especialmente `SIN_RESPUESTA`, muestran un patrón más robusto: se asocian con mayor impulsividad no planificada y con peor desempeño narrativo/cognitivo-lingüístico.

La conclusión más defendible por ahora es que el perfil NLP combina dos capas: una capa estructural fuertemente relacionada con desarrollo escolar, y una capa de etiquetas discursivas que parece capturar dificultad funcional durante la tarea. El siguiente paso metodológico es controlar edad/escolaridad, limpiar etiquetas y validar con baseline aleatorio.
