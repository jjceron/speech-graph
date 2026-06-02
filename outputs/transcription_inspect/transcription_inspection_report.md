# Inspección de transcripciones corregidas y etiquetadas
**Fecha de ejecución:** 2026-06-01 16:35:18
**Carpeta inspeccionada:** `data\processed\Transcripciones`
**Salida generada:** este reporte Markdown y `transcription_inspection_summary.csv`.
## 1. Etiquetas esperadas según protocolo
### Etiquetas entre doble corchete `[[...]]`
- `[[EE]]`: elementos extralexicales.
- `[[IF]]`: inicio falso.
- `[[PS]]`: prolongación de sonidos.
- `[[PAUSA StartTime=MM:SS EndTime=MM:SS]]`: pausa de 3 segundos o más. Para pausas menores se usa `(.)`.
- `[[PNC]]`: palabra no clara.
- `[[DP StartTime=MM:SS EndTime=MM:SS]]`: diálogo entre las partes.
- `[[DI StartTime=MM:SS EndTime=MM:SS]]` o `[[DI]]`: diálogo interno; suele acompañarse de texto entre `<...>`.
- `[[SIN RESPUESTA]]`: ausencia de respuesta del entrevistado.
- `[[SIN PREGUNTA]]`: pregunta/estímulo omitido por el entrevistador.
- `[[IM]]`: inquietud motora.
- `[[ES=texto]]`: evento simple producido por el entrevistado.

### Marcaciones estructurales o no encerradas en `[[...]]`
- `<<NombreActividad=Actividad#>>`: inicio de actividad.
- `[StartTime=MM:SS EndTime=MM:SS]`: marca de tiempo de actividad o fragmento.
- Apóstrofo `'`: acentuación de sonido o sílaba.
- Paréntesis dentro de palabra, por ejemplo `(en)tonces` o `pu(d)o`: palabra incompleta.
- Circunflejo `^`, por ejemplo `car^nero`: término segmentado.
## 2. Resumen general
- Archivos `.txt` inspeccionados: **267**.
- Total de errores: **44**.
- Total de advertencias: **116**.
- Total de notas informativas: **15**.

### Estado QC por archivo
| Estado | n | % |
|---|---:|---:|
| OK | 170 | 63.7% |
| OK_CON_ADVERTENCIAS | 51 | 19.1% |
| REVISAR_ADVERTENCIAS | 2 | 0.7% |
| REVISAR_ERROR | 44 | 16.5% |

## 3. Inventario de etiquetas canónicas
| Etiqueta | apariciones totales | archivos con etiqueta | % archivos |
|---|---:|---:|---:|
| EE | 6488 | 263 | 98.5% |
| IF | 3663 | 267 | 100.0% |
| PS | 10099 | 266 | 99.6% |
| PAUSA | 1939 | 252 | 94.4% |
| PNC | 72 | 31 | 11.6% |
| DP | 179 | 96 | 36.0% |
| DI | 1212 | 213 | 79.8% |
| SIN_RESPUESTA | 355 | 169 | 63.3% |
| SIN_PREGUNTA | 354 | 121 | 45.3% |
| IM | 126 | 28 | 10.5% |
| ES | 73 | 20 | 7.5% |

## 4. Principales tipos de alerta
| Severidad / tipo | n | ejemplos |
|---|---:|---|
| warning::tag_missing_time | 56 | CDMS-10-6B-ASCACO L44: [[PAUSA]] sin StartTime/EndTime.; CDMS-11-6B-SCAPA L48: [[PAUSA]] sin StartTime/EndTime.; CDMS-12-6B-IJDAGA L47: [[DP]] sin StartTime/EndTime.; CDMS-13-6B-ASUMO L9: [[DP]] sin StartTime/EndTime.; CDMS-14-10A-JCVANI L47: [[DP]] sin StartTime/EndTime. |
| error::unbalanced_double_brackets | 43 | CDMS-10-5A-SZURO L: Cantidad de [[ (81) distinta de ]] (82).; CDMS-11-5A-EAARCA L: Cantidad de [[ (81) distinta de ]] (80).; CDMS-11-5A-MGTHAR L: Cantidad de [[ (59) distinta de ]] (60).; CDMS-11-6A-DFVEBE L: Cantidad de [[ (90) distinta de ]] (91).; CDMS-11-6B-ARAHE L: Cantidad de [[ (78) distinta de ]] (76). |
| warning::unknown_double_bracket_tag | 36 | CDMS-10-5A-KSRAGU L47: Etiqueta [[...]] no reconocida: [[[DI StartTime=12:59 EndTime=13:08]].; CDMS-11-5A-EAARCA L47: Etiqueta [[...]] no reconocida: [[EE] tiburón. Pez. [[DI StartTime=11:25 EndTime=11:31]].; CDMS-11-6B-ARAHE L35: Etiqueta [[...]] no reconocida: [[IF] y la cabra, un lobo viejo que ya no tenía la fuerza y la astucia de otras épocas (.) y al que por eso llamaban Tontolobo. Persiguió a una cabra que se puso a salvo subiéndose a una alta piedra. ¿Por qué te es [[IF]].; CDMS-11-6B-ARAHE L64: Etiqueta [[...]] no reconocida: [[EE] el gato porque le parece divertido cazar a los pájaros y el papá porque de pronto estaba viendo a la mamá y se le regaron los platos. Mientras (.) mientras que los niños pues le pidieron permiso al papá, pero como el papá no los ve entonces ellos están haciendo daños. spk_0: ¿Qué crees que está pensando el señor de la imagen? spk_1: [[EE]].; CDMS-11-6B-JACUPE L30: Etiqueta [[...]] no reconocida: [[PS][ de las emociones. spk_0: ¿En qué se parecen la libertad y la justicia? spk_1: [[DI StartTime=06:19 EndTime=06:34]]. |
| info::tag_variant | 14 | CDMS-10-6B-ASCACO L18: Variante normalizable: [[SIN REPUESTA]] -> SIN_RESPUESTA.; CDMS-10-6B-ASCACO L22: Variante normalizable: [[SIN REPUESTA]] -> SIN_RESPUESTA.; CDMS-10-6B-ASCACO L24: Variante normalizable: [[SIN REPUESTA]] -> SIN_RESPUESTA.; CDMS-10-6B-ASCACO L26: Variante normalizable: [[SIN REPUESTA]] -> SIN_RESPUESTA.; CDMS-10-6B-ASCACO L28: Variante normalizable: [[SIN REPUESTA]] -> SIN_RESPUESTA. |
| warning::sin_respuesta_with_extra_text | 13 | CDMS-15-10A-JJMACH L59: SIN_RESPUESTA aparece junto con texto adicional en la misma línea.; CDMS-15-10A-JJMACH L62: SIN_RESPUESTA aparece junto con texto adicional en la misma línea.; CDMS-15-10A-JJMACH L64: SIN_RESPUESTA aparece junto con texto adicional en la misma línea.; CSO-10-5A-RVMOIB L60: SIN_RESPUESTA aparece junto con texto adicional en la misma línea.; CSO-11-6B-LHFDLACH L60: SIN_RESPUESTA aparece junto con texto adicional en la misma línea. |
| warning::missing_header | 5 | CSO-15-10C-ALBAHO L: No se encontró <TransInfo ...>.; CSO-16-10A-JEGAHE L: No se encontró <TransInfo ...>.; CSO-16-10A-YKMUJO L: No se encontró <TransInfo ...>.; CSO-16-9A-JPROSO L: No se encontró <TransInfo ...>.; CSO-16-9B-SJPAJI L: No se encontró <TransInfo ...>. |
| warning::activity_without_timestamp_same_line | 4 | CDMS-9-4A-CRAAG L61: Actividad sin timestamp en la misma línea.; CMU-16-11A-DDMODU L21: Actividad sin timestamp en la misma línea.; CSO-9-4A-LCOMO L11: Actividad sin timestamp en la misma línea.; CSO-9-4A-MPAME L14: Actividad sin timestamp en la misma línea. |
| error::unknown_speaker | 1 | CDMS-XX-6A-MAGICA L12: Línea con hablante no esperado: spk_01. |
| warning::high_spk0_ratio | 1 | CSO-15-9B-SJAGRO L: Alta proporción de tokens spk_0: 38.29%. |
| info::filename_suffix | 1 | CSO-17-10B-YCQUVE L: El nombre del archivo no contiene CorrEtiq. |
| warning::malformed_activity_tag | 1 | CSO-9-4A-EGCAMO L: Hay 1 etiquetas NombreActividad con formato no estándar. |

## 5. Variantes y etiquetas desconocidas
### Variantes normalizables detectadas
- if -> IF (1): 2
- SIN REPUESTA -> SIN_RESPUESTA (6): 1
- EEE -> EE (1): 1
- DI]. [[EE -> DI (1): 1
- PS] si tenía la rana la fuera deja(d)o en la casa para que (.) no causara ese problema. spk_0: ¿Qué hubieras hecho tú en lugar del niño cuando intentan sacar a la rana del restaurante? spk_1: [[SIN PREGUNTA -> SIN_PREGUNTA (1): 1
- SIN REPUESTA -> SIN_RESPUESTA (1): 1
- EE] más o menos. Que ni bien ni mal. spk_0: ¿Qué hubieras hecho tú en lugar del niño cuando intentan sacar a la rana del restaurante? spk_1: [[SIN PREGUNTA -> SIN_PREGUNTA (1): 1
- PAUA StartTime=11:40 EndTime=11:47 -> PAUSA (1): 1

### Etiquetas desconocidas `[[...]]`
- [DI StartTime=12:59 EndTime=13:08 (1): 1
- EE] tiburón. Pez. [[DI StartTime=11:25 EndTime=11:31 (1): 1
- IF] y la cabra, un lobo viejo que ya no tenía la fuerza y la astucia de otras épocas (.) y al que por eso llamaban Tontolobo. Persiguió a una cabra que se puso a salvo subiéndose a una alta piedra. ¿Por qué te es [[IF (1): 1
- EE] el gato porque le parece divertido cazar a los pájaros y el papá porque de pronto estaba viendo a la mamá y se le regaron los platos. Mientras (.) mientras que los niños pues le pidieron permiso al papá, pero como el papá no los ve entonces ellos están haciendo daños. spk_0: ¿Qué crees que está pensando el señor de la imagen? spk_1: [[EE (1): 1
- PS][ de las emociones. spk_0: ¿En qué se parecen la libertad y la justicia? spk_1: [[DI StartTime=06:19 EndTime=06:34 (1): 1
- EE] spk_1: [[EE (1): 1
- EE] esa tarde con esa rana. spk_0: ¿Cómo te pareció el cuento? spk_1: Me pareció bien. Me pareció muy [[PS (1): 1
- EE] granadilla. Creo que ya. spk_1: Gato, perro, lobo, le(o)pardo, puma, [[EE (1): 1
- EE]. Pues los niños están como molestando. Uno de ellos se se está intentando caer. Y nadie le está prestando atención al perrito viendo que se está comiendo cosas del suelo. La mamá. En el celular, sin prestar atención a lo que pasa en casa. Qué sucede y qué sienten. Y el papá dedicado a los oficios. Se le ve. Que [[PS (1): 1
- [DI StartTime=10:29 EndTime=10:32 (1): 1
- IF] intentan sacar galletas. El niño casi se cae intentando sacar las galletas. Pues mientra que la hermana pues come una mientras que se ríe y pues sí, el perro abajo el [[PS (1): 1
- I (1): 1
- EE] bonito. spk_0: ¿Cómo te sentirías si fueras el niño de la historia? spk_1: Un poco mal [[EE (1): 1
- EE]. Mandarina. Limón. Lima. Borojó. <Creo que es una fruta. ¿Cierto?. El borojó es una fruta> [[DI (1): 1
- EE] manzana. Mandarina. Mariposa. [[EE (1): 1
- IF]. Me [[IF (1): 1
- EE] y después les cae a otra persona al lado que estaba el señor del saxofón ese. <Espérate, que no analice esta página bien> [[DI StartTime=22:41 EndTime=22:48 (1): 1
- PS] en su casa y no se ha dado cuenta porque es como que ciego y [[PS (1): 1
- EE]. Tontolobo y la cabra. Un lobo viejo. Que ya no tenía la [[IF (1): 1
- [PS (1): 1
- EE] naranja, mandarina, [[EE (1): 1
- EE] culebra. [[EE] elefante. Dinosaurio. [[EE] mariposa. Servi <!Ah, no¡> [[DI (1): 1
- EE] zancudo, mosco. [[EE (1): 1
- EE] [[PAUSA StartTime=11:09 EndTime=11:19 (1): 1
- EE] conejo. Nutria. Un gusano. Un mosquito. [[EE (1): 1
- EE] un hombre sin camisa. [[EE] un abrigo. Un rayo. [[EE (1): 1
- IF] se [[PS (1): 1
- PS] Don Juan arma [[PS (1): 1
- PS] [[EE (1): 1
- PS], empiezan a elegi [[IF (1): 1

## 6. Archivos que más requieren revisión
| code | status | errores | advertencias | variantes | desconocidas | resumen |
|---|---|---:|---:|---:|---:|---|
| CSO-13-8A-VHODU | REVISAR_ERROR | 1 | 3 | 0 | 2 | error:unbalanced_double_brackets:Cantidad de [[ (87) distinta de ]] (83). / warning:sin_respuesta_with_extra_text:L51: SIN_RESPUESTA aparece junto con texto adicional en la misma línea. / warning:unknown_double_bracket_tag:L47: Etiqueta [[...]] no reconocida: [[EE] culebra. [[EE] elefante. Dinosaurio. [[EE] mariposa. Servi <!Ah, no¡> [[DI]]. / warning:unknown_double_bracket_tag:L47: Etiqueta [[...]] no reconocida: [[EE] zancudo, mosco. [[EE]]. |
| CSO-15-10A-SFEGA | REVISAR_ERROR | 1 | 3 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (74) distinta de ]] (73). / warning:tag_missing_time:L28: [[DP]] sin StartTime/EndTime. / warning:tag_missing_time:L51: [[DP]] sin StartTime/EndTime. / warning:unknown_double_bracket_tag:L47: Etiqueta [[...]] no reconocida: [[EE] conejo. Nutria. Un gusano. Un mosquito. [[EE]]. |
| CDMS-11-6B-ARAHE | REVISAR_ERROR | 1 | 2 | 0 | 2 | error:unbalanced_double_brackets:Cantidad de [[ (78) distinta de ]] (76). / warning:unknown_double_bracket_tag:L35: Etiqueta [[...]] no reconocida: [[IF] y la cabra, un lobo viejo que ya no tenía la fuerza y la astucia de otras épocas (.) y al que por eso llamaban Tontolobo. Persiguió a una cabra que se puso a salvo subiéndose a una alta piedra. ¿Por qué te es [[IF]]. / warning:unknown_double_bracket_tag:L64: Etiqueta [[...]] no reconocida: [[EE] el gato porque le parece divertido cazar a los pájaros y el papá porque de pronto estaba viendo a la mamá y se le regaron los platos. Mientras (.) mientras que los niños pues le pidieron permiso al papá, pero como el papá no los ve entonces ellos están haciendo daños. spk_0: ¿Qué crees que está pensando el señor de la imagen? spk_1: [[EE]]. |
| CDMS-15-11B-JEMOLO | REVISAR_ERROR | 1 | 2 | 0 | 2 | error:unbalanced_double_brackets:Cantidad de [[ (151) distinta de ]] (150). / warning:unknown_double_bracket_tag:L47: Etiqueta [[...]] no reconocida: [[EE] spk_1: [[EE]]. / warning:unknown_double_bracket_tag:L51: Etiqueta [[...]] no reconocida: [[EE] esa tarde con esa rana. spk_0: ¿Cómo te pareció el cuento? spk_1: Me pareció bien. Me pareció muy [[PS]]. |
| CSO-11-6A-CAPAMU | REVISAR_ERROR | 1 | 2 | 0 | 2 | error:unbalanced_double_brackets:Cantidad de [[ (138) distinta de ]] (135). / warning:unknown_double_bracket_tag:L51: Etiqueta [[...]] no reconocida: [[EE] y después les cae a otra persona al lado que estaba el señor del saxofón ese. <Espérate, que no analice esta página bien> [[DI StartTime=22:41 EndTime=22:48]]. / warning:unknown_double_bracket_tag:L62: Etiqueta [[...]] no reconocida: [[PS] en su casa y no se ha dado cuenta porque es como que ciego y [[PS]]. |
| CSO-15-11C-JETALO | REVISAR_ERROR | 1 | 2 | 0 | 2 | error:unbalanced_double_brackets:Cantidad de [[ (136) distinta de ]] (133). / warning:unknown_double_bracket_tag:L51: Etiqueta [[...]] no reconocida: [[IF] se [[PS]]. / warning:unknown_double_bracket_tag:L9: Etiqueta [[...]] no reconocida: [[EE] un hombre sin camisa. [[EE] un abrigo. Un rayo. [[EE]]. |
| CSO-15-9B-ALRIHE | REVISAR_ERROR | 1 | 2 | 0 | 2 | error:unbalanced_double_brackets:Cantidad de [[ (143) distinta de ]] (141). / warning:unknown_double_bracket_tag:L12: Etiqueta [[...]] no reconocida: [[PS] Don Juan arma [[PS]]. / warning:unknown_double_bracket_tag:L12: Etiqueta [[...]] no reconocida: [[PS] [[EE]]. |
| CSO-9-2A-EXMOCO | REVISAR_ERROR | 1 | 2 | 0 | 2 | error:unbalanced_double_brackets:Cantidad de [[ (109) distinta de ]] (107). / warning:unknown_double_bracket_tag:L28: Etiqueta [[...]] no reconocida: [[PS].] Cuando uno está feliz, está feliz. Cuando uno está triste, no puede estar feliz. spk_0: ¿En qué se parecen la libertad y la justicia? spk_1: Nada, porque cuando uno está libre se siente libre y la justicia. No [[PS]]. / warning:unknown_double_bracket_tag:L60: Etiqueta [[...]] no reconocida: [[EE]. [[EE]]. |
| CSO-15-9B-KSBEJA | REVISAR_ERROR | 1 | 1 | 1 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (144) distinta de ]] (142). / warning:unknown_double_bracket_tag:L51: Etiqueta [[...]] no reconocida: [[PS], empiezan a elegi [[IF]]. / info:tag_variant:L55: Variante normalizable: [[EE] más o menos. Que ni bien ni mal. spk_0: ¿Qué hubieras hecho tú en lugar del niño cuando intentan sacar a la rana del restaurante? spk_1: [[SIN PREGUNTA]] -> SIN_PREGUNTA. |
| CSO-11-6B-LHFDLACH | REVISAR_ERROR | 1 | 1 | 1 | 0 | error:unbalanced_double_brackets:Cantidad de [[ (132) distinta de ]] (131). / warning:sin_respuesta_with_extra_text:L60: SIN_RESPUESTA aparece junto con texto adicional en la misma línea. / info:tag_variant:L53: Variante normalizable: [[PS] si tenía la rana la fuera deja(d)o en la casa para que (.) no causara ese problema. spk_0: ¿Qué hubieras hecho tú en lugar del niño cuando intentan sacar a la rana del restaurante? spk_1: [[SIN PREGUNTA]] -> SIN_PREGUNTA. |
| CDMS-11-5A-EAARCA | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (81) distinta de ]] (80). / warning:unknown_double_bracket_tag:L47: Etiqueta [[...]] no reconocida: [[EE] tiburón. Pez. [[DI StartTime=11:25 EndTime=11:31]]. |
| CDMS-11-6B-JACUPE | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (278) distinta de ]] (277). / warning:unknown_double_bracket_tag:L30: Etiqueta [[...]] no reconocida: [[PS][ de las emociones. spk_0: ¿En qué se parecen la libertad y la justicia? spk_1: [[DI StartTime=06:19 EndTime=06:34]]. |
| CDMS-15-11B-VSAHO | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (52) distinta de ]] (51). / warning:unknown_double_bracket_tag:L44: Etiqueta [[...]] no reconocida: [[EE] granadilla. Creo que ya. spk_1: Gato, perro, lobo, le(o)pardo, puma, [[EE]]. |
| CDMS-16-11B-NGABE | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (109) distinta de ]] (108). / warning:unknown_double_bracket_tag:L62: Etiqueta [[...]] no reconocida: [[EE]. Pues los niños están como molestando. Uno de ellos se se está intentando caer. Y nadie le está prestando atención al perrito viendo que se está comiendo cosas del suelo. La mamá. En el celular, sin prestar atención a lo que pasa en casa. Qué sucede y qué sienten. Y el papá dedicado a los oficios. Se le ve. Que [[PS]]. |
| CDMS-9-5C-JDMADE | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (125) distinta de ]] (124). / warning:unknown_double_bracket_tag:L62: Etiqueta [[...]] no reconocida: [[IF] intentan sacar galletas. El niño casi se cae intentando sacar las galletas. Pues mientra que la hermana pues come una mientras que se ríe y pues sí, el perro abajo el [[PS]]. |
| CSO-10-5A-ACAVA | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (147) distinta de ]] (146). / warning:unknown_double_bracket_tag:L53: Etiqueta [[...]] no reconocida: [[EE] bonito. spk_0: ¿Cómo te sentirías si fueras el niño de la historia? spk_1: Un poco mal [[EE]]. |
| CSO-10-5A-JANORO | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (97) distinta de ]] (96). / warning:unknown_double_bracket_tag:L46: Etiqueta [[...]] no reconocida: [[EE]. Mandarina. Limón. Lima. Borojó. <Creo que es una fruta. ¿Cierto?. El borojó es una fruta> [[DI]]. |
| CSO-10-5A-JLCARI | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (142) distinta de ]] (141). / warning:unknown_double_bracket_tag:L47: Etiqueta [[...]] no reconocida: [[EE] manzana. Mandarina. Mariposa. [[EE]]. |
| CSO-10-5A-MAOSMO | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (105) distinta de ]] (104). / warning:unknown_double_bracket_tag:L48: Etiqueta [[...]] no reconocida: [[IF]. Me [[IF]]. |
| CSO-12-7A-IMERO | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (172) distinta de ]] (171). / warning:unknown_double_bracket_tag:L35: Etiqueta [[...]] no reconocida: [[EE]. Tontolobo y la cabra. Un lobo viejo. Que ya no tenía la [[IF]]. |
| CSO-13-8A-MGAHE | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (116) distinta de ]] (115). / warning:unknown_double_bracket_tag:L46: Etiqueta [[...]] no reconocida: [[EE] naranja, mandarina, [[EE]]. |
| CSO-14-8B-JLLORA | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (52) distinta de ]] (51). / warning:unknown_double_bracket_tag:L45: Etiqueta [[...]] no reconocida: [[EE] [[PAUSA StartTime=11:09 EndTime=11:19]]. |
| CSO-16-10A-SARI | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (101) distinta de ]] (100). / warning:unknown_double_bracket_tag:L9: Etiqueta [[...]] no reconocida: [[EE] una camisa. Un trueno o un relámpago. [[DI StartTime=03:22 EndTime=03:27]]. |
| CSO-16-10B-JESUGU | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (72) distinta de ]] (71). / warning:unknown_double_bracket_tag:L35: Etiqueta [[...]] no reconocida: [[PS] saltaré dentro. El lobo abrió la boca. Y la cabra saltó. Al saltar, le dio tal cornada que derribó al lobo del suelo. Dejandos [[IF]]. |
| CSO-9-4A-FAGIGA | REVISAR_ERROR | 1 | 1 | 0 | 1 | error:unbalanced_double_brackets:Cantidad de [[ (76) distinta de ]] (75). / warning:unknown_double_bracket_tag:L62: Etiqueta [[...]] no reconocida: [[EE], ya. No sé. El agua se estaba regando y nadie le ayuda al papá [[EE]]. |
| CDMS-XX-6A-MAGICA | REVISAR_ERROR | 1 | 1 | 0 | 0 | error:unknown_speaker:L12: Línea con hablante no esperado: spk_01. / warning:tag_missing_time:L51: [[DP]] sin StartTime/EndTime. |
| CSO-14-9A-JSMAAL | REVISAR_ERROR | 1 | 1 | 0 | 0 | error:unbalanced_double_brackets:Cantidad de [[ (115) distinta de ]] (116). / warning:tag_missing_time:L45: [[DP]] sin StartTime/EndTime. |
| CSO-10-4A-KCOMO | REVISAR_ERROR | 1 | 0 | 1 | 0 | error:unbalanced_double_brackets:Cantidad de [[ (111) distinta de ]] (110). / info:tag_variant:L12: Variante normalizable: [[DI]. [[EE]] -> DI. |
| CDMS-10-5A-SZURO | REVISAR_ERROR | 1 | 0 | 0 | 0 | error:unbalanced_double_brackets:Cantidad de [[ (81) distinta de ]] (82). |
| CDMS-11-5A-MGTHAR | REVISAR_ERROR | 1 | 0 | 0 | 0 | error:unbalanced_double_brackets:Cantidad de [[ (59) distinta de ]] (60). |

## 7. Cómo usar este reporte
1. Revisar primero los archivos con `REVISAR_ERROR`.
2. Luego revisar variantes de etiquetas, especialmente si afectan `SIN_RESPUESTA`, `SIN_PREGUNTA`, `PAUSA`, `EE`, `IF`, `PS`, `DI` o `DP`.
3. Confirmar que las etiquetas con duración (`PAUSA`, `DP` y cuando aplique `DI`) tengan `StartTime` y `EndTime`.
4. Revisar archivos con alta proporción de `spk_0`, porque pueden conservar verbalizaciones del entrevistador que no deben entrar al análisis NLP.
5. Usar `transcription_inspection_summary.csv` para filtrar por `qc_status`, `issue_summary`, `unknown_tags` y columnas `tag_*_count`.
