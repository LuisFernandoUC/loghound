# Datos

Dataset: HDFS_v1 de Loghub (11,175,629 líneas, 575,061 bloques etiquetados).

## Cómo obtenerlo

1. Descargar https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1 (1.47 GiB).
2. Descomprimir y copiar a esta carpeta:
   - `Event_occurrence_matrix.csv`
   - `anomaly_label.csv`

Estos dos archivos no se versionan (ver `.gitignore`). `HDFS_templates.csv` sí está incluido.

## Columnas

- `E1`..`E29`: cuántas veces aparece cada tipo de evento en el bloque. Son las únicas variables de entrada.
- `Label` y `Type`: **no se usan** (revelan la etiqueta; ver el cuaderno).

Cita: Xu et al. (SOSP 2009) y Zhu et al., Loghub (ISSRE 2023).
