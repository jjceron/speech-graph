# Activity-focused speech graph analysis: 02_w30_by_activity_no_random_all

Este reporte resume el análisis por actividad/clase usando grafos dirigidos de palabras y ventana móvil de 30 palabras. Las transcripciones no se modifican: el código normaliza etiquetas del protocolo y remueve timestamps técnicos durante el análisis.

## Parámetros
- `window_size`: 30
- `random_times`: 0
- Actividades target: Actividad2, Actividad6, Actividad7
- Actividades secundarias: Actividad1, Actividad4, Actividad5

## Control de calidad por actividad
| activity   | activity_role   |   n_rows |   n_with_metadata |   n_valid_window |   token_median |   window_count_median |
|:-----------|:----------------|---------:|------------------:|-----------------:|---------------:|----------------------:|
| Actividad1 | secondary       |      269 |               254 |              191 |           39   |                  10   |
| Actividad2 | target          |      266 |               251 |              246 |          135   |                 106   |
| Actividad3 | other           |      265 |               250 |              224 |           51   |                  22   |
| Actividad4 | secondary       |      268 |               253 |              268 |          188   |                 159   |
| Actividad5 | secondary       |      266 |               251 |              264 |           70   |                  41   |
| Actividad6 | target          |      268 |               253 |              268 |          424.5 |                 395.5 |
| Actividad7 | target          |      266 |               251 |              264 |          121   |                  92   |
| NA         | other           |        1 |                 1 |                1 |          111   |                  82   |

## Correlaciones de métricas de grafos
| activity   | metric                    | target                                       |         r |           p |   q_fdr_within_section |   n |
|:-----------|:--------------------------|:---------------------------------------------|----------:|------------:|-----------------------:|----:|
| Actividad4 | mean_nodes                | School year                                  |  0.487496 | 1.65404e-16 |            3.63888e-15 | 253 |
| Actividad4 | mean_asp                  | School year                                  |  0.480458 | 5.12345e-16 |            5.4513e-15  | 253 |
| Actividad4 | mean_density              | School year                                  | -0.478105 | 7.4336e-16  |            5.4513e-15  | 253 |
| Actividad4 | mean_asp                  | Age                                          |  0.472292 | 1.84225e-15 |            1.01324e-14 | 253 |
| Actividad4 | mean_diameter             | School year                                  |  0.4648   | 5.78505e-15 |            2.54542e-14 | 253 |
| Actividad4 | mean_diameter             | Age                                          |  0.461182 | 9.95541e-15 |            3.65032e-14 | 253 |
| Actividad4 | mean_nodes                | Age                                          |  0.459758 | 1.23042e-14 |            3.86702e-14 | 253 |
| Actividad4 | mean_density              | Age                                          | -0.449699 | 5.34488e-14 |            1.46984e-13 | 253 |
| Actividad4 | mean_lsc_ratio            | School year                                  | -0.40561  | 1.94205e-11 |            4.74723e-11 | 253 |
| Actividad4 | mean_lsc_ratio            | Age                                          | -0.398806 | 4.47903e-11 |            9.85387e-11 | 253 |
| Actividad6 | mean_nodes                | School year                                  |  0.368284 | 1.51618e-09 |            5.76148e-08 | 253 |
| Actividad5 | mean_lsc_ratio            | FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES | -0.36517  | 2.85584e-09 |            5.14051e-08 | 249 |
| Actividad6 | mean_nodes                | Age                                          |  0.357387 | 4.8918e-09  |            8.49037e-08 | 253 |
| Actividad6 | mean_density              | School year                                  | -0.354386 | 6.70293e-09 |            8.49037e-08 | 253 |
| Actividad4 | mean_l2                   | School year                                  | -0.350081 | 1.04705e-08 |            2.09411e-08 | 253 |
| Actividad6 | mean_density              | Age                                          | -0.345602 | 1.6537e-08  |            1.57102e-07 | 253 |
| Actividad5 | mean_lsc_ratio            | Verbal fluency tasks                         | -0.344333 | 2.43794e-08 |            1.94908e-07 | 249 |
| Actividad5 | mean_nodes                | FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES |  0.341426 | 3.24846e-08 |            1.94908e-07 | 249 |
| Actividad4 | mean_repeated_edges_ratio | School year                                  | -0.338968 | 3.21189e-08 |            5.88847e-08 | 253 |
| Actividad6 | mean_repeated_edges_ratio | Age                                          | -0.334688 | 4.88861e-08 |            3.71535e-07 | 253 |

## Correlaciones de etiquetas discursivas
| activity   | metric                    | target                                       |         r |           p |   q_fdr_within_section |   n |
|:-----------|:--------------------------|:---------------------------------------------|----------:|------------:|-----------------------:|----:|
| Actividad4 | label_ratio_SIN_RESPUESTA | Reading comprehension task                   | -0.383438 | 2.761e-10   |            3.3132e-09  | 253 |
| Actividad6 | label_ratio_SIN_PREGUNTA  | Naming task                                  |  0.302002 | 9.84989e-07 |            9.84989e-06 | 253 |
| Actividad1 | label_ratio_DI            | FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES | -0.284731 | 4.00186e-06 |            4.00186e-05 | 254 |
| Actividad4 | label_ratio_IF            | School year                                  | -0.280013 | 6.10803e-06 |            3.66482e-05 | 253 |
| Actividad7 | label_ratio_SIN_RESPUESTA | NPLAN                                        |  0.260649 | 2.89797e-05 |            0.000579593 | 251 |
| Actividad4 | label_ratio_SIN_RESPUESTA | Verbal fluency tasks                         | -0.260602 | 2.70049e-05 |            8.39009e-05 | 253 |
| Actividad4 | label_ratio_IF            | Age                                          | -0.260126 | 2.7967e-05  |            8.39009e-05 | 253 |
| Actividad2 | label_ratio_PAUSA         | FLUIDEZ VERBAL - PRUEBA FONETICA             | -0.258314 | 3.43449e-05 |            0.000549518 | 251 |
| Actividad7 | label_ratio_SIN_PREGUNTA  | Naming task                                  |  0.247828 | 7.22259e-05 |            0.000722259 | 251 |
| Actividad2 | label_ratio_PAUSA         | COHERENCIA NARRATIVA                         | -0.236617 | 0.000154407 |            0.00123526  | 251 |
| Actividad1 | label_ratio_DI            | Verbal fluency tasks                         | -0.231749 | 0.000194458 |            0.000887478 | 254 |
| Actividad4 | label_ratio_SIN_RESPUESTA | FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES | -0.22747  | 0.000264132 |            0.000633916 | 253 |
| Actividad1 | label_ratio_DI            | COHERENCIA NARRATIVA                         | -0.226905 | 0.000266243 |            0.000887478 | 254 |
| Actividad5 | label_ratio_PAUSA         | COHERENCIA NARRATIVA                         | -0.226442 | 0.000298376 |            0.00447564  | 251 |
| Actividad3 | label_ratio_SIN_RESPUESTA | NPLAN                                        |  0.216224 | 0.000576394 |            0.00375819  | 250 |
| Actividad3 | label_ratio_SIN_RESPUESTA | Conceptualization task                       | -0.213384 | 0.000683307 |            0.00375819  | 250 |
| Actividad4 | label_ratio_SIN_RESPUESTA | FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS   | -0.212099 | 0.000684667 |            0.00136933  | 253 |
| Actividad5 | label_ratio_PAUSA         | Verbal fluency tasks                         | -0.209739 | 0.000827079 |            0.00472407  | 251 |
| Actividad4 | label_ratio_SIN_RESPUESTA | COHERENCIA NARRATIVA                         | -0.209036 | 0.000821364 |            0.00140805  | 253 |
| Actividad5 | label_ratio_PAUSA         | FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES | -0.207462 | 0.000944814 |            0.00472407  | 251 |

## Diferencias entre grupos
| activity   | metric                    | target   | group_1   | group_2   |   group_1_n |   group_2_n |   group_1_median |   group_2_median |          p |   q_fdr_within_section |
|:-----------|:--------------------------|:---------|:----------|:----------|------------:|------------:|-----------------:|-----------------:|-----------:|-----------------------:|
| Actividad7 | label_ratio_SIN_RESPUESTA | Tipo     | high_imp  | low_imp   |         121 |         130 |        0         |       0          | 0.00193021 |               0.136844 |
| Actividad2 | label_ratio_PS            | Tipo     | high_imp  | low_imp   |         123 |         128 |        0.0689655 |       0.0521562  | 0.00530168 |               0.136844 |
| Actividad7 | label_ratio_DP            | Tipo     | high_imp  | low_imp   |         121 |         130 |        0         |       0          | 0.00570183 |               0.136844 |
| Actividad3 | label_ratio_SIN_RESPUESTA | Tipo     | high_imp  | low_imp   |         119 |         131 |        0.0181818 |       0.00884956 | 0.0141898  |               0.255417 |
| Actividad7 | label_ratio_PAUSA         | Tipo     | high_imp  | low_imp   |         121 |         130 |        0         |       0          | 0.0269098  |               0.387501 |
| Actividad7 | mean_lsc                  | Tipo     | high_imp  | low_imp   |         120 |         130 |       19.0498    |      18.6293     | 0.0280809  |               0.819667 |
| Actividad4 | mean_repeated_edges_ratio | Tipo     | high_imp  | low_imp   |         122 |         131 |        0.0333984 |       0.0295567  | 0.0345963  |               0.819667 |
| Actividad7 | mean_lsc_ratio            | Tipo     | high_imp  | low_imp   |         120 |         130 |        0.786289  |       0.761395   | 0.0365179  |               0.819667 |
| Actividad4 | label_ratio_ES            | Tipo     | high_imp  | low_imp   |         122 |         131 |        0         |       0          | 0.0374482  |               0.40942  |
| Actividad6 | label_ratio_SIN_RESPUESTA | Tipo     | high_imp  | low_imp   |         121 |         132 |        0         |       0          | 0.0398047  |               0.40942  |
| Actividad4 | mean_l3                   | Tipo     | high_imp  | low_imp   |         122 |         131 |        1.52196   |       1.41732    | 0.0542561  |               0.819667 |
| Actividad1 | label_ratio_PS            | Tipo     | high_imp  | low_imp   |         121 |         133 |        0.04      |       0.0285714  | 0.0542898  |               0.488609 |
| Actividad2 | label_ratio_ES            | Tipo     | high_imp  | low_imp   |         123 |         128 |        0         |       0          | 0.0905572  |               0.724458 |
| Actividad3 | mean_lsc                  | Tipo     | high_imp  | low_imp   |         100 |         115 |       17.7236    |      18.3878     | 0.0985836  |               0.819667 |
| Actividad7 | mean_repeated_edges_ratio | Tipo     | high_imp  | low_imp   |         120 |         130 |        0.0180316 |       0.0221596  | 0.103074   |               0.819667 |
| Actividad1 | label_ratio_EE            | Tipo     | high_imp  | low_imp   |         121 |         133 |        0.0512821 |       0.0606061  | 0.10622    |               0.764785 |
| Actividad7 | mean_l2                   | Tipo     | high_imp  | low_imp   |         120 |         130 |        0.254724  |       0.385114   | 0.110764   |               0.819667 |
| Actividad6 | mean_repeated_edges_ratio | Tipo     | high_imp  | low_imp   |         121 |         132 |        0.0289117 |       0.0255843  | 0.120209   |               0.819667 |
| Actividad3 | label_ratio_ES            | Tipo     | high_imp  | low_imp   |         119 |         131 |        0         |       0          | 0.138578   |               0.769108 |
| Actividad3 | label_ratio_PNC           | Tipo     | high_imp  | low_imp   |         119 |         131 |        0         |       0          | 0.138578   |               0.769108 |

## Sujetos extremos por actividad
### Actividad1
|   rank | code              | group    |   activity_disorganization_proxy |
|-------:|:------------------|:---------|---------------------------------:|
|      1 | CDMS-14-6B-KDCAHE | high_imp |                          2.52791 |
|      2 | CDMS-10-6B-JJMOBE | low_imp  |                          2.16537 |
|      3 | CDMS-8-4A-LGOPA   | low_imp  |                          1.63563 |
|      4 | CSO-12-7A-IMERO   | high_imp |                          1.59364 |
|      5 | CSO-8-4A-JRAAR    | low_imp  |                          1.57815 |
|      6 | CDMS-15-10A-SHEGA | high_imp |                          1.5373  |
|      7 | CDMS-10-5C-MAORTO | high_imp |                          1.4577  |
|      8 | CSO-12-8A-YTOME   | high_imp |                          1.40585 |
|      9 | CDMS-10-5A-SZURO  | high_imp |                          1.36979 |
|     10 | CDMS-11-6B-JMARME | low_imp  |                          1.33816 |

### Actividad2
|   rank | code              | group    |   activity_disorganization_proxy |
|-------:|:------------------|:---------|---------------------------------:|
|      1 | CDMS-9-5C-SCAAG   | high_imp |                          6.85632 |
|      2 | CDMS-8-4C-JDBUTU  | low_imp  |                          3.43003 |
|      3 | CDMS-9-4A-ACDUTR  | low_imp  |                          3.39533 |
|      4 | CSO-13-7A-KRAMA   | high_imp |                          2.12915 |
|      5 | CDMS-11-6B-JACUPE | high_imp |                          1.84481 |
|      6 | CDMS-8-4A-DJAVDA  | low_imp  |                          1.84234 |
|      7 | CDMS-9-5A-ABERO   | high_imp |                          1.7731  |
|      8 | CSO-9-4A-LCOMO    | low_imp  |                          1.68944 |
|      9 | CSO-9-4A-SLOCA    | low_imp  |                          1.66556 |
|     10 | CSO-12-7A-DALRE   | high_imp |                          1.63942 |

### Actividad4
|   rank | code             | group    |   activity_disorganization_proxy |
|-------:|:-----------------|:---------|---------------------------------:|
|      1 | CSO-10-5A-IPECA  | high_imp |                          2.80731 |
|      2 | CSO-9-4A-SLOCA   | low_imp  |                          2.13322 |
|      3 | CDMS-9-5A-PAGACO | high_imp |                          2.11383 |
|      4 | CDMS-10-5C-ICAPA | high_imp |                          2.05155 |
|      5 | CDMS-12-6B-SCLAG | high_imp |                          1.85168 |
|      6 | CSO-10-5A-JLCARI | high_imp |                          1.74328 |
|      7 | CDMS-10-5A-SZURO | high_imp |                          1.64361 |
|      8 | CSO-16-9B-SJPAJI | high_imp |                          1.60153 |
|      9 | CSO-9-4A-AGAPE   | low_imp  |                          1.51145 |
|     10 | CSO-10-4A-KCOMO  | low_imp  |                          1.49427 |

### Actividad5
|   rank | code              | group    |   activity_disorganization_proxy |
|-------:|:------------------|:---------|---------------------------------:|
|      1 | CSO-10-5A-JLCARI  | high_imp |                          2.88998 |
|      2 | CSO-9-4A-DSMOMU   | high_imp |                          2.83895 |
|      3 | CSO-14-9B-MFEGA   | high_imp |                          2.76852 |
|      4 | CDMS-12-6B-IJDAGA | high_imp |                          2.35796 |
|      5 | CSO-9-4A-DKVIMO   | low_imp  |                          2.15906 |
|      6 | CDMS-10-6B-ASCACO | low_imp  |                          1.90258 |
|      7 | CDMS-11-5A-MGTHAR | high_imp |                          1.82763 |
|      8 | CSO-10-5A-ACAVA   | low_imp  |                          1.54367 |
|      9 | CDMS-11-6B-EROCO  | high_imp |                          1.53936 |
|     10 | CSO-17-11C-JDAAG  | low_imp  |                          1.52817 |

### Actividad6
|   rank | code              | group    |   activity_disorganization_proxy |
|-------:|:------------------|:---------|---------------------------------:|
|      1 | CDMS-9-4A-CRAAG   | high_imp |                          2.87408 |
|      2 | CSO-13-8A-MJAGA   | low_imp  |                          2.86678 |
|      3 | CDMS-8-4A-DJAVDA  | low_imp  |                          2.6634  |
|      4 | CDMS-8-4C-JDBUTU  | low_imp  |                          2.33223 |
|      5 | CDMS-10-5C-ICAPA  | high_imp |                          2.22969 |
|      6 | CDMS-11-6B-JDCHBA | high_imp |                          2.14789 |
|      7 | CDMS-11-6B-JACUPE | high_imp |                          2.10288 |
|      8 | CDMS-9-5C-JDMADE  | high_imp |                          1.7199  |
|      9 | CSO-9-4A-YVDITR   | high_imp |                          1.70562 |
|     10 | CSO-12-6B-VPAMA   | high_imp |                          1.56013 |

### Actividad7
|   rank | code              | group    |   activity_disorganization_proxy |
|-------:|:------------------|:---------|---------------------------------:|
|      1 | CDMS-11-6B-EFRGR  | low_imp  |                          2.95951 |
|      2 | CDMS-13-6B-ASUMO  | low_imp  |                          2.81526 |
|      3 | CSO-9-4A-SLOCA    | low_imp  |                          2.4707  |
|      4 | CSO-12-7A-DALRE   | high_imp |                          2.2266  |
|      5 | CDMS-11-6B-JACUPE | high_imp |                          1.8171  |
|      6 | CDMS-11-6B-MEBRTA | high_imp |                          1.7895  |
|      7 | CSO-12-6B-VPAMA   | high_imp |                          1.73265 |
|      8 | CDMS-11-6B-EAFESA | low_imp  |                          1.67655 |
|      9 | CDMS-11-6B-JPSAMA | low_imp  |                          1.59847 |
|     10 | CDMS-12-6B-SCLAG  | high_imp |                          1.53208 |

## Lectura metodológica
- Actividad 2, 6 y 7 se consideran targets porque son más cercanas al perfil narrativo/discursivo que queremos estudiar.
- Actividad 1, 4 y 5 sirven como análisis secundario para ver si una señal depende del tipo de tarea o aparece de forma transversal.
- Sin baseline aleatorio, los resultados son exploratorios. Si una señal es consistente, conviene validarla con `--random-times 20` o `--random-times 100` en ventana 30.
- Edad y curso deben considerarse covariables importantes porque ya mostraron asociación fuerte con las métricas globales.
