# Detección de anomalías en logs HDFS con ML clásico

Proyecto semestral de Inteligencia Artificial · Track A (ML tabular) · TEC, Sede Central Cartago
Integrantes: Luis Fernando · Fabricio Herrera

Detector de bloques anómalos en logs del sistema de archivos distribuido HDFS, con un **árbol de decisión** como modelo principal, comparado contra baselines y explicado con SHAP. Este repositorio contiene la **Etapa 2** (modelado, explicabilidad y evaluación).

## Resumen

- **Problema:** clasificar cada bloque de HDFS como normal o anómalo a partir de cuántas veces aparece cada tipo de evento en su traza de logs.
- **Dataset:** HDFS_v1 de Loghub, con 575,061 bloques etiquetados (97.07 % normales, 2.93 % anómalos).
- **Modelo principal:** árbol de decisión, elegido por su interpretabilidad, su bajo costo y porque la literatura reporta un resultado comparable (F1 = 0.9410, Yu et al. 2024).
- **Referencias de comparación:** baseline trivial, regresión logística y el resultado de la literatura.

## Decisiones clave

| Decisión                                                 | Motivo                                                                             |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Se descartan las columnas `Label` y `Type`               | Revelan la etiqueta (_target leakage_)                                             |
| Se deduplican los bloques con el mismo vector de conteos | El 99.9 % de los bloques está repetido; sin deduplicar, las métricas miden memoria |
| Partición 60/20/20 por vectores, estratificada           | Ningún vector queda en dos particiones                                             |
| Prueba usada una sola vez, al final                      | Evita fuga por ajuste sobre la prueba                                              |
| Métricas: precisión, recall, F1, PR-AUC y MCC            | _Accuracy_ engaña con clases desbalanceadas                                        |

## Estructura

```
notebooks/Etapa2_Modelado.ipynb   cuaderno de la Etapa 2
src/etapa2_utils.py               carga, particiones y transformador de variables
data/                             datos (ver data/README.md)
results/                          métricas y resultados de la búsqueda
figures/                          curva de aprendizaje y gráficos SHAP
models/                           modelo entrenado y metadatos
```

## Cómo reproducirlo

1. Instalar dependencias: `pip install -r requirements.txt`
2. Descargar los datos siguiendo `data/README.md` y copiarlos a `data/`.
3. Ejecutar `notebooks/Etapa2_Modelado.ipynb` de arriba abajo. Semilla fija: `SEED = 42`.

**En Google Colab:** subir `Event_occurrence_matrix.csv` y `anomaly_label.csv` a `/content`, y ejecutar el cuaderno.


## Referencias

- W. Xu, L. Huang, A. Fox, D. Patterson, M. Jordan. _Detecting Large-Scale System Problems by Mining Console Logs._ SOSP 2009.
- J. Zhu, S. He, P. He, J. Liu, M. R. Lyu. _Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics._ ISSRE 2023.
- Yu et al. _Deep Learning or Classical Machine Learning? An Empirical Study on Log-Based Anomaly Detection._ ICSE 2024.
