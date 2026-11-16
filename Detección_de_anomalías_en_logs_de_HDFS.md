# Detección de anomalías en logs de HDFS con aprendizaje automático clásico: metodología y resultados (Etapa 2)

Autores: Luis Fernando Ureña Corrales y Fabricio Herrera Rodríguez. Curso de Inteligencia Artificial, Tecnológico de Costa Rica, Track A (ML tabular).

## Resumen

Un árbol de decisión ajustado alcanza F1 = 0.949 (IC 95 %: 0.909 a 0.981) al detectar anomalías en HDFS deduplicado, por encima del 0.941 reportado en la literatura, pero por debajo de una regresión logística simple (0.968). Las diferencias entre los tres modelos no triviales caen dentro del intervalo de confianza, así que no se puede afirmar que uno sea mejor que otro en desempeño.

El trabajo usa los 575,061 bloques de HDFS\_v1, que se reducen a 589 vectores de conteo distintos. Se descartan dos columnas que revelan la etiqueta, se deduplica antes de partir y todo el preprocesamiento se ajusta solo con entrenamiento. El árbol se eligió como modelo principal por su interpretabilidad, y SHAP confirma que sus decisiones dependen de eventos de error con sentido operativo.

El hallazgo central coincide con la hipótesis de la Etapa 1: ningún modelo clásico domina a la vez en desempeño, eficiencia e interpretabilidad. La regresión logística obtiene el mejor F1, el árbol es el más fácil de explicar, y el costo de cómputo es de milisegundos para ambos.

## 1. Datos y hallazgos de calidad

Se usa HDFS\_v1 de Loghub en su versión ya procesada: una fila por bloque de HDFS, con el conteo de veces que aparece cada tipo de evento (E1 a E29) y la etiqueta normal o anomalía. Esto difiere de lo descrito en la Etapa 1, que partía de los logs crudos y de un parser como Drain; aquí se parte de la matriz de ocurrencia que Loghub distribuye, con 29 plantillas de evento en lugar de las 14 observadas en la muestra de 2,000 líneas.

| Característica | Valor |
| --- | --- |
| Bloques | 575,061 |
| Bloques normales | 558,223 (97.07 %) |
| Bloques anómalos | 16,838 (2.93 %) |
| Vectores de conteo distintos | 589 (99.90 % de los bloques son repetidos) |
| Vectores con etiqueta contradictoria | 9 |
| Columnas sin variación | E22 |
| Valores nulos | 0 |

Tres hallazgos condicionan todo el diseño experimental:

- **Fuga directa en el archivo.** La matriz incluye las columnas `Label` y `Type`. `Type` tiene valor solo en los 16,838 bloques anómalos y es nula en todos los normales, por lo que revela la etiqueta. En producción no estaría disponible al momento de predecir. Ambas columnas se descartan y solo entran E1 a E29.
- **Duplicación casi total.** Con 589 vectores distintos entre 575,061 bloques, una partición por bloque deja el mismo vector en entrenamiento y en prueba, y las métricas miden memoria en lugar de generalización.
- **Ruido de etiquetado.** Nueve vectores aparecen como normales y como anómalos. Ningún modelo basado en conteos puede acertar ambos casos, lo que fija un techo de desempeño inferior a 1.

## 2. Metodología definitiva

### 2.1 Deduplicación

Se conserva una fila por cada combinación distinta de vector de conteos y etiqueta. Quedan 598 filas: 589 vectores, de los cuales 9 aparecen dos veces por tener ambas etiquetas. Tras deduplicar, el 64.5 % de las filas son anomalías (386 de 598), porque los comportamientos anómalos son más variados que los normales.

La deduplicación es una operación fila a fila que no ajusta ninguna estadística, por lo que hacerla antes de partir no introduce fuga.

### 2.2 Partición

Los vectores se parten 60/20/20 de forma aleatoria con semilla fija (42), estratificando por tres estratos: normal, anomalía y contradictorio. La partición se hace por vector, de modo que las dos filas de un vector contradictorio caen siempre en el mismo conjunto. El código verifica con una aserción que ningún vector se comparte entre conjuntos.

| Conjunto | Filas | Anomalías | Uso |
| --- | --- | --- | --- |
| Entrenamiento | 358 | 231 (64.5 %) | Ajuste y búsqueda de hiperparámetros |
| Validación | 120 | 78 (65.0 %) | Comparación entre modelos |
| Prueba | 120 | 77 (64.2 %) | Una sola lectura, al final |

Se usa 60/20/20 y no 80/20 para disponer de una validación aparte donde comparar modelos sin tocar la prueba.

### 2.3 Cambio respecto a la Etapa 1

En la Etapa 1 se propuso una partición cronológica. Se probó sobre los vectores únicos, tomando el orden de primera aparición en el archivo como tiempo, y la prueba queda con 89 % de anomalías y solo 13 vectores normales. En esas condiciones predecir siempre "anomalía" obtiene el mejor F1, y las métricas dejan de distinguir entre modelos.

Por esa razón el protocolo principal es el estratificado, y la partición cronológica se reporta como prueba de estrés en la sección 7. La decisión es optimista: la tasa de anomalía cambia a lo largo del archivo, así que el resultado principal estima el desempeño ante comportamientos de la misma distribución, y la prueba de estrés lo estima ante comportamientos nuevos.

### 2.4 Métricas

La clase positiva es la anomalía. Se reportan precisión, recall, F1, PR-AUC y MCC.

- **F1** es la métrica de decisión, porque en detección de anomalías cuestan tanto las anomalías perdidas como las falsas alarmas, y es la métrica del baseline de literatura.
- **MCC** se agrega porque F1 no penaliza a un modelo que marca todo como anomalía: tras deduplicar, la clase mayoritaria obtiene F1 = 0.78 con MCC = 0.
- **Accuracy** no se usa: sobre los datos originales, predecir siempre "normal" acertaría el 97 %.

El F1 en prueba se acompaña de un intervalo de confianza al 95 % por bootstrap con 2,000 remuestreos, porque con 120 filas cada error mueve el F1 cerca de un punto.

## 3. Modelos

### 3.1 Puntos de referencia

| Referencia | Modelo | Pregunta que responde |
| --- | --- | --- |
| Trivial | Aleatorio estratificado y clase mayoritaria | ¿El modelo aprende algo? |
| Simple | Regresión logística sobre los conteos sin transformar | ¿Vale la pena la complejidad? |
| Interna | Árbol de decisión con parámetros por defecto | ¿El ajuste aporta algo? |
| Literatura | Árbol de decisión de Yu et al. \[1\], F1 = 0.9410 | ¿Cómo se ubica frente a lo publicado? |

La comparación con la literatura es aproximada. Yu et al. reportan su cifra sobre HDFS deduplicado con partición aleatoria 80/20 y ajuste bayesiano; aquí se usa 60/20/20 con validación aparte, búsqueda en rejilla y un conjunto de prueba distinto.

### 3.2 Modelo principal

El modelo principal es un árbol de decisión. Se eligió por tres razones: la literatura reporta un resultado directo para él sobre HDFS deduplicado, es el más interpretable de las familias del Track A, y es barato de entrenar y consultar, lo que importa para exponerlo como herramienta de un agente en la Etapa 3.

Todo el preprocesamiento vive en un `Pipeline` de scikit-learn con tres pasos, ajustado solo con entrenamiento:

1. `EventFeatures`: agrega de forma opcional variables derivadas por bloque (total de eventos, número de tipos de evento distintos y proporción de cada evento). Se calculan fila a fila, sin estadísticas globales, por lo que no pueden filtrar información entre conjuntos.
2. `VarianceThreshold`: elimina columnas sin variación, como E22.
3. `DecisionTreeClassifier` con semilla fija.

No se usa SMOTE. Tras deduplicar cada fila es un comportamiento real, y un vector de conteos sintético no tiene un significado operativo claro.

### 3.3 Búsqueda de hiperparámetros

Se explora una rejilla completa de 960 combinaciones con validación cruzada de 5 pliegues, estratificada y agrupada por vector, usando solo el conjunto de entrenamiento y optimizando F1.

| Hiperparámetro | Valores explorados | Elegido |
| --- | --- | --- |
| Variables de totales | no, sí | sí |
| Variables de proporciones | no, sí | no |
| Criterio | gini, entropy | gini |
| Profundidad máxima | 3, 5, 8, 12, sin límite | 8 |
| Mínimo de muestras por hoja | 1, 2, 5, 10 | 1 |
| Poda (`ccp_alpha`) | 0, 0.001, 0.01 | 0.001 |
| Pesos de clase | ninguno, balanceado | ninguno |

La rejilla es pequeña a propósito: con 358 filas cada ajuste toma milisegundos, y recorrerla entera costó 64 s de CPU. Una búsqueda aleatoria o bayesiana no se justifica a ese costo.

Se aplica una regla de parsimonia: entre las configuraciones a menos de 0.005 de F1 de la mejor, se elige la más simple (menor profundidad, menos variables derivadas, más poda). La interpretabilidad es parte de la pregunta de investigación, y una diferencia de 0.005 es menor que la variación entre pliegues. La configuración elegida obtuvo F1 = 0.9433 (±0.0204) en validación cruzada, que es también el mejor valor de toda la rejilla.

## 4. Resultados

### 4.1 Validación

En validación, con los modelos ajustados solo con entrenamiento, el árbol ajustado no supera al árbol por defecto (F1 0.894 frente a 0.918).

| Modelo | Precisión | Recall | F1 | PR-AUC | MCC |
| --- | --- | --- | --- | --- | --- |
| Aleatorio estratificado | 0.630 | 0.654 | 0.642 | 0.637 | -0.062 |
| Clase mayoritaria | 0.650 | 1.000 | 0.788 | 0.650 | 0.000 |
| Regresión logística | 0.897 | 0.897 | 0.897 | 0.950 | 0.707 |
| Árbol por defecto | 0.901 | 0.936 | 0.918 | 0.891 | 0.759 |
| Árbol con variables derivadas, sin ajustar | 0.859 | 0.936 | 0.896 | 0.845 | 0.682 |
| Árbol ajustado (principal) | 0.867 | 0.923 | 0.894 | 0.859 | 0.683 |

El modelo principal se mantuvo a pesar de este resultado, porque se había fijado antes de la comparación y cambiarlo según la validación habría sido ajustar la elección a 120 filas.

### 4.2 Prueba

En prueba, con los modelos reajustados con entrenamiento más validación (478 filas) y una sola lectura, el árbol ajustado alcanza F1 = 0.949 con un intervalo de confianza al 95 % de 0.909 a 0.981.

| Modelo | Precisión | Recall | F1 | PR-AUC | MCC | VP | FP | FN | VN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aleatorio estratificado | 0.654 | 0.688 | 0.671 | 0.650 | 0.038 | 53 | 28 | 24 | 15 |
| Clase mayoritaria | 0.642 | 1.000 | 0.782 | 0.642 | 0.000 | 77 | 43 | 0 | 0 |
| Regresión logística | 0.950 | 0.987 | 0.968 | 0.963 | 0.909 | 76 | 4 | 1 | 39 |
| Árbol por defecto | 0.947 | 0.935 | 0.941 | 0.935 | 0.838 | 72 | 4 | 5 | 39 |
| Árbol ajustado (principal) | 0.937 | 0.961 | 0.949 | 0.916 | 0.854 | 74 | 5 | 3 | 38 |
| Literatura, Yu et al. \[1\] |  |  | 0.941 |  |  |  |  |  |  |

### 4.3 Lectura frente a los puntos de referencia

- **Frente a los triviales, el modelo aprende.** El árbol ajustado supera al azar estratificado por 0.28 de F1 y tiene MCC = 0.854 frente a 0 de la clase mayoritaria.
- **Frente a la regresión logística, la complejidad no se paga en desempeño.** La regresión obtiene F1 = 0.968, con 5 errores frente a 8 del árbol. Su F1 cae dentro del intervalo del árbol, así que la diferencia no es concluyente, pero tampoco hay evidencia de que el árbol sea mejor.
- **Frente al árbol por defecto, el ajuste aporta poco.** La diferencia es de 0.008 de F1, equivalente a un error en 120 filas.
- **Frente a la literatura, el resultado está en el mismo rango.** El 0.949 queda apenas sobre el 0.941 de Yu et al., y ese valor también cae dentro del intervalo. No se puede afirmar que se supera el baseline, solo que se reproduce su orden de magnitud bajo un protocolo que controla la duplicación.

El orden entre modelos cambia de validación a prueba: la regresión logística pasa de 0.897 a 0.968. Con conjuntos de 120 filas, las diferencias de pocos puntos entre modelos son ruido de partición.

## 5. Explicabilidad con SHAP

Las decisiones del árbol dependen sobre todo de la diversidad de eventos del bloque y de cuatro eventos de error. Se aplicó `TreeExplainer` de SHAP \[2\] al árbol final sobre el conjunto de prueba, y cada variable se tradujo a su plantilla de log.

| Variable | SHAP medio absoluto | Plantilla de log | Efecto observado |
| --- | --- | --- | --- |
| `tipos_distintos` | 0.213 | Variable derivada: número de tipos de evento distintos en el bloque | Valores altos empujan hacia anomalía |
| E20 | 0.122 | Unexpected error trying to delete block, BlockInfo not found in volumeMap | Su presencia empuja con fuerza hacia anomalía |
| E28 | 0.061 | addStoredBlock request received, but it does not belong to any file | Su presencia empuja hacia anomalía |
| E3 | 0.044 | Served block | Valores bajos empujan hacia anomalía |
| E27 | 0.040 | Redundant addStoredBlock request received | Su presencia empuja hacia anomalía |
| E4 | 0.036 | Got exception while serving | Valores altos empujan hacia normal |
| `total_eventos` | 0.035 | Variable derivada: total de eventos del bloque | Efecto mixto |
| E2 | 0.033 | Verification succeeded | Valores altos empujan hacia normal |

### 5.1 ¿Tiene sentido que dominen?

Sí, en su mayoría. E20, E28 y E27 son mensajes de error o de operaciones redundantes sobre un bloque, y E2 es una verificación exitosa. Que un bloque con muchos tipos de evento distintos sea sospechoso también es razonable: un bloque normal sigue una secuencia corta y repetitiva de recepción, almacenamiento y borrado, y los eventos adicionales suelen ser excepciones.

### 5.2 Control de calidad

- **No hay señal de fuga.** Ninguna variable ajena a los conteos aparece, y la importancia se reparte entre varias variables. Con la columna `Type`, en cambio, un árbol pone toda su importancia en ella (sección 7).
- **E4 se comporta de forma inesperada.** Las excepciones al servir un bloque empujan hacia "normal". La causa probable es el ruido de etiquetado: el vector de prueba con E4 = 15 es uno de los nueve contradictorios, y aparece como normal y como anómalo. Es una variable cuya importancia revela un problema en los datos, no en el modelo.
- **La variable más importante es derivada.** `tipos_distintos` resume muchos eventos en un número, lo que es menos accionable que un evento concreto. Es el costo en interpretabilidad de haber aceptado las variables de totales en la búsqueda.

### 5.3 Hallazgos accionables

- Un monitor puede vigilar E20, E27 y E28 como alertas tempranas, sin esperar a tener el vector completo del bloque.
- Los bloques con E4 alto deberían revisarse manualmente, porque sus etiquetas son inconsistentes.
- En la Etapa 3, el agente puede explicar una alerta citando la plantilla del evento que más contribuyó, en lugar de un identificador como E20.

## 6. Análisis

### 6.1 Curva de aprendizaje y sobreajuste

El modelo tiene un sobreajuste leve y el tamaño de los datos no es su principal límite. En la curva de aprendizaje, el F1 de validación cruzada sube de 0.79 con 57 filas a 0.91 con 114, y termina en 0.94 con 286. El F1 de entrenamiento se mantiene entre 0.96 y 0.98, de modo que la brecha final es de unos 0.04.

La curva no es monótona: cae a 0.885 con 229 filas antes de volver a subir. Con pliegues de unas 70 filas, esa oscilación es ruido de muestreo y no una tendencia.

El modelo final confirma el diagnóstico: F1 = 0.965 sobre sus propios datos de ajuste y 0.949 en prueba, una brecha de 0.016. No hay subajuste, porque el F1 de entrenamiento es alto, ni sobreajuste fuerte, porque la brecha es pequeña. El tope de profundidad 8 es lo que contiene la complejidad.

### 6.2 Análisis de errores

El árbol comete 8 errores en 120 vectores de prueba: 5 falsas alarmas y 3 anomalías no detectadas.

| Tipo | Eventos del vector | Observación |
| --- | --- | --- |
| Anomalía no detectada | E4:15, E26:4, E5:3, E9:3, E11:3, E21:3, E23:3 | Vector contradictorio: también existe etiquetado como normal |
| Anomalía no detectada | E20:1, E5:4, E21:4, E26:4, resto típico | Caso límite: tiene E20 pero el árbol lo clasifica normal |
| Anomalía no detectada | E3:203, E5:12, E21:11, E26:11, E23:10, E7:1 | Bloque servido 203 veces, muy fuera del rango habitual |
| Falsa alarma | E20:2, E5:5, E21:5, E23:5, E26:5 | Etiquetado normal a pesar de tener dos errores de borrado |
| Falsa alarma | E4:2, E5:4, E21:4, E26:4, E2:1 | Vector contradictorio |
| Falsa alarma | E5:9, E6:6, E21:9, E23:9, E26:9 | Bloque con el triple de eventos de lo habitual |
| Falsa alarma | E4:2, E23:4, E26:5, E5:4 | Parecido a un bloque normal con una réplica extra |
| Falsa alarma | E26:5, E5:4, E21:1, E2:1 | Un solo borrado frente a cinco réplicas registradas |

Tres patrones explican los errores:

- **Ruido de etiquetas.** Dos de los 8 errores son vectores contradictorios, que ningún modelo puede acertar en ambas versiones.
- **E20 no es una regla perfecta.** Hay un bloque anómalo con E20 que el árbol deja pasar y un bloque normal con E20 = 2 que marca como anomalía. El evento es un indicador fuerte, no una definición.
- **Vectores atípicos.** El bloque con E3 = 203 está fuera de lo visto en entrenamiento, y el árbol, que solo compara contra umbrales, no tiene forma de tratarlo como caso especial.

### 6.3 Eficiencia computacional

La búsqueda de hiperparámetros, que se ejecuta una sola vez, costó 64 s de CPU para 960 combinaciones por 5 pliegues. Cada ajuste individual toma milisegundos porque el conjunto tiene 478 filas.

La conclusión esperada, a confirmar con esa tabla, es que la eficiencia no distingue entre modelos clásicos en HDFS deduplicado, y solo importa al compararlos con modelos de aprendizaje profundo. La literatura revisada en la Etapa 1 reporta tiempos de reentrenamiento de 13 a 131 s para Random Forest y de hasta 138,000 s para NeuralLog \[3\], medidos en otro equipo y sobre los datos sin deduplicar.

## 7. Diagnósticos de fuga y deriva

Estas pruebas cuantifican los riesgos identificados en la Etapa 1 y no influyen en ninguna decisión de modelado. Las cifras de esta sección provienen de la primera versión del cuaderno, que usaba otra semilla de partición y otra configuración del árbol.

| Diagnóstico | Resultado | Qué demuestra |
| --- | --- | --- |
| Árbol de profundidad 3 con `Type` como variable | F1 = 1.000 en validación, con toda la importancia en `Type` | `Type` es fuga directa de la etiqueta |
| Partición cronológica por bloque, sin deduplicar | 99.90 % de los bloques de prueba tienen un vector idéntico en entrenamiento | Sin deduplicar, la prueba repite el entrenamiento |
| Árbol ajustado, sin deduplicar | F1 = 0.991 con precisión 0.999 | La duplicación infla el F1 en varios puntos |
| Tasa de anomalía por décima parte del archivo | De 4.5 % al inicio a 1.4 % al final | Existe deriva temporal |

### 7.1 Prueba de estrés temporal

Ante comportamientos nuevos el modelo detecta menos anomalías. Al partir los vectores únicos por orden de primera aparición y repetir el ciclo completo, con su propia búsqueda, la prueba queda con 118 vectores, 105 de ellos anómalos.

| Modelo | Precisión | Recall | F1 | MCC |
| --- | --- | --- | --- | --- |
| Clase mayoritaria | 0.890 | 1.000 | 0.942 | 0.000 |
| Regresión logística | 0.977 | 0.800 | 0.880 | 0.455 |
| Árbol por defecto | 0.964 | 0.762 | 0.851 | 0.364 |
| Árbol ajustado | 0.964 | 0.771 | 0.857 | 0.374 |

La clase mayoritaria obtiene el mejor F1 porque la prueba tiene 89 % de anomalías, lo que confirma que este protocolo no sirve para comparar modelos con F1. El MCC sí los separa, y muestra una caída real: el recall baja de 0.96 a cerca de 0.77 y el MCC de 0.85 a 0.37.

El resultado principal de la sección 4 es entonces una estimación optimista, válida para comportamientos parecidos a los ya vistos. Esta prueba es la estimación pesimista. Se asume que el orden de las filas del archivo corresponde al orden de primera aparición de cada bloque en el log.

## 8. Limitaciones y conclusiones

### 8.1 Limitaciones

- **Conjunto de prueba pequeño.** Con 120 vectores, el intervalo del F1 abarca 7 puntos y no permite ordenar los tres modelos no triviales.
- **Cada vector pesa igual.** La deduplicación ignora cuántas veces aparece cada comportamiento, y la proporción de anomalías pasa de 2.93 % a 64.5 %. Las métricas describen comportamientos distintos, no el tráfico real de un clúster; en producción la precisión sería otra.
- **Partición optimista.** La partición aleatoria mezcla comportamientos tempranos y tardíos. La prueba de estrés temporal muestra que ante vectores nuevos el recall baja a cerca de 0.77.
- **Vecinos cercanos.** Los vectores no se repiten entre conjuntos, pero sí hay vectores casi iguales, lo que también favorece al modelo.
- **Una sola partición.** El resultado en prueba cambió de 0.917 a 0.949 entre dos semillas de partición durante el desarrollo. La cifra reportada corresponde a la partición final, y la variación entre particiones es del mismo orden que las diferencias entre modelos.
- **Ruido de etiquetas.** Nueve vectores contradictorios fijan un techo de desempeño que no depende del modelo.
- **Representación por conteos.** Se descartan el orden y los tiempos de los eventos, y no se parte de los logs crudos.
- **Un solo dataset.** HDFS es muy redundante y favorece a casi cualquier técnica \[3\]; los resultados no se extienden a otros sistemas.

### 8.2 Conclusiones

1. Un árbol de decisión de profundidad 8 detecta anomalías en HDFS deduplicado con F1 = 0.949 y MCC = 0.854, en el rango del baseline de literatura y bajo un protocolo que controla la duplicación y la fuga por `Type`.
2. El ajuste de hiperparámetros y las variables derivadas no producen una mejora medible frente al árbol por defecto, y una regresión logística sin transformar obtiene un F1 igual o mayor.
3. El valor del árbol está en la interpretabilidad: sus decisiones se explican con eventos de error concretos (E20, E27, E28), que además sirven como alertas por sí solos.
4. La hipótesis de la Etapa 1 se sostiene con estos datos: ningún modelo domina en los tres criterios a la vez.
5. Sin deduplicar, el mismo modelo reportaría F1 = 0.991. La diferencia con 0.949 es el tamaño del inflado que la literatura atribuye a la duplicación de secuencias.

Para la Etapa 3, el modelo queda guardado como herramienta para un agente que consulte la predicción y traduzca la explicación SHAP a lenguaje natural. La caída de recall ante comportamientos nuevos es el caso que la validación del agente debe cubrir.

## 9. Reproducibilidad

| Elemento | Valor |
| --- | --- |
| Semilla | 42, en partición, validación cruzada, modelos y bootstrap |
| Entorno de ejecución | Google Colab, Python 3 |
| Bibliotecas | scikit-learn, pandas, numpy, shap, matplotlib, joblib |
| Datos | HDFS\_v1 de Loghub \[4\]: `Event_occurrence_matrix.csv`, `anomaly_label.csv` y el archivo de plantillas |
| Artefactos guardados | Modelo (`.joblib`), mejores parámetros, métricas de validación y prueba, resultados completos de la búsqueda, figuras |

## 10. Declaración de uso de IA

Borrador a revisar y completar por los autores.

- **Herramienta:** Claude (Anthropic).
- **En qué partes:** generación de un primer esqueleto del código del cuaderno, revisión de la solución contra la rúbrica de la etapa, y redacción del primer borrador de este artículo a partir de las salidas del cuaderno.
- **Propósito:** acelerar la implementación y la redacción, y detectar vacíos frente a los requisitos.
- **Qué se verificó manualmente:** por completar. Como mínimo: que cada cifra del artículo coincide con la salida del cuaderno final, que el conjunto de prueba se lee una sola vez, que `Label` y `Type` no entran al modelo, y que las citas corresponden al contenido de los trabajos citados.

## Referencias

&#91;1\] B. Yu, J. Yao, Q. Fu, Z. Zhong, H. Xie, Y. Wu, Y. Ma y P. He, "Deep learning or classical machine learning? An empirical study on log-based anomaly detection," en *Proc. IEEE/ACM 46th Int. Conf. Software Engineering (ICSE)*, Lisboa, Portugal, 2024, doi: 10.1145/3597503.3623308.

&#91;2\] S. M. Lundberg y S.-I. Lee, "A unified approach to interpreting model predictions," en *Advances in Neural Information Processing Systems 30 (NIPS)*, 2017.

&#91;3\] S. Ali, C. Boufaied, D. Bianculli, P. Branco y L. Briand, "A comprehensive study of machine learning techniques for log-based anomaly detection," *Empirical Software Engineering*, vol. 30, n.º 129, 2025, doi: 10.1007/s10664-025-10669-3.

&#91;4\] J. Zhu, S. He, P. He, J. Liu y M. R. Lyu, "Loghub: A large collection of system log datasets for AI-driven log analytics," en *Proc. IEEE 34th Int. Symp. Software Reliability Engineering (ISSRE)*, 2023.

Las referencias \[1\] y \[3\] provienen del artículo de la Etapa 1. Las referencias \[2\] y \[4\] son nuevas y deben leerse y verificarse antes de la entrega.
