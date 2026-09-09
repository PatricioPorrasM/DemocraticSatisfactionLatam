# Cambios a realizar en el documento de tesis (Overleaf)

**Generado:** 2026-09-06 · **Revisado:** 2026-09-07 (auditoría de defendibilidad)
**Base:** corrida definitiva del servidor con GPU, 2026-09-05 11:58 → 2026-09-06 01:20
(`modo_ejecucion = real` verificado en `results/resultados_modelos.csv` y en los 20
registros de `models/hp_*.json`).
**Estado:** la Parte A de la versión anterior de este documento **ya está aplicada**
en el PDF. Este documento la reemplaza por completo y contiene únicamente lo que
falta cambiar, contrastando el PDF actual con los artefactos de la corrida nueva.

**Revisión del 2026-09-07.** Se auditó cada afirmación del documento contra los
archivos de `results/`, `models/` y `notebooks/output/`. Se corrigieron nueve
errores propios y se retiraron tres explicaciones que no tienen respaldo en la
corrida. Lo corregido:

| Apartado | Qué estaba mal | Ahora |
|---|---|---|
| 3.7 | Citaba 431.756 como «disponibles tras las exclusiones», en contradicción con las 430.895 de la Tabla 5.1 | Se explica que son dos totales distintos y cuál usar |
| 4.8 | Ganancias calculadas sobre cifras redondeadas (+0,0730; rango 0,0196–0,0337) | +0,0729; rango 0,0197–0,0336 |
| 4.8 | Atribuía la asimetría a que «los árboles capturan la estructura de clases minoritarias» | Retirado: la corrida no lo mide |
| 4.13 | Comparaba con la estrategia de pesos de clase usando un archivo que no existe | Retirado |
| 4.15 | Mezclaba configuraciones: Perú con «sin balanceo» y Costa Rica «supera 1,0» con pesos de clase | Cifras de la configuración principal; el peor país es Argentina (0,6678) |
| 4.15 | La inversión MAE/kappa se explicaba sin respaldo | Respaldada con ρ(concentración, MAE) = −1,00 |
| 4.17 | «Las cuatro primeras posiciones son fijas en el 100 %» | Son tres; la 4.ª y la 5.ª intercambian |
| 4.18 | «Dieciséis veces superior» | 16,9 veces |
| 4.20 | «Los mismos países que registran el peor MAE por subregión» | Falso; reformulado sobre lo que la tabla sí permite afirmar |
| 4.23 | Daba a Devine y a Easton veredictos apoyados en métricas distintas | Un solo criterio, con las dos métricas declaradas |

Y lo que se añadió porque estaba en la corrida y el documento no lo recogía: las
ocho variables con \|SHAP\| nulo (4.17), la desviación estándar de los pliegues con
presupuesto homogéneo (4.22) y la cautela sobre la W de Kendall (4.21).

---

## Cómo usar este documento

Está organizado **en el orden del PDF**, de modo que se pueda recorrer de arriba
abajo. Cada apartado indica:

- **PDF actual**: lo que dice hoy el documento.
- **Corrida nueva**: el valor real, con el archivo del que se lee.
- **Qué hacer**: la acción concreta.

Marcas usadas:

| Marca | Significado |
|---|---|
| 🔴 | Cambia una conclusión o una hipótesis. Revisar antes que nada. |
| 🟡 | Cambia cifras de una tabla o figura, sin alterar la conclusión. |
| 🟢 | Ya es correcto en el PDF. No tocar; se lista para que no se revise dos veces. |

## Nivel de evidencia de cada afirmación

El tribunal puede pedir la fuente de cualquier cifra. Todo lo que este documento
propone incorporar al PDF pertenece a uno de estos tres niveles, y conviene saber
en cuál está antes de defenderlo:

| Nivel | Qué es | Cómo se defiende |
|---|---|---|
| **A — Lectura directa** | La cifra está escrita en un archivo de `results/`, `models/` o `notebooks/output/`. | Se abre el archivo. El Anexo I dice cuál. |
| **B — Aritmética sobre el nivel A** | Diferencias, promedios, porcentajes y razones calculados a partir de valores de nivel A. | Se muestra la operación. Toda la aritmética de este documento está comprobada. |
| **C — Interpretación** | Explicaciones de *por qué* se observa un patrón: mecanismos, compensaciones entre métricas, lecturas sustantivas. | Se declara como interpretación del autor. No se presenta como resultado. |

**Regla que se ha seguido al redactar los pasajes propuestos:** ninguna
afirmación de nivel C aparece en indicativo como si fuera un hallazgo. Cuando el
patrón admite una comprobación empírica, se ha calculado y se ha citado el
estadístico (por ejemplo, la explicación de la inversión entre MAE y kappa por
subregión de 4.15). Cuando no la admite, el pasaje lo dice con un verbo de
compatibilidad («es consistente con», «sugiere») y no de causalidad.

**Advertencia sobre las comparaciones con la corrida anterior.** El documento de
tesis reporta **una** corrida. No hay ninguna necesidad —ni conviene— explicar en
él por qué las cifras difieren de una ejecución previa: obligaría a documentar
resultados que ya no existen en el repositorio de entrega. El Apéndice B recoge
esa comparación solo para uso interno.

> **La configuración principal cambió.** La selección en validación ya no es
> CatBoost con pesos de clase, sino **CatBoost con SMOTE-NC**
> (κw = 0,4840 en validación; `results/modelo_xai_seleccionado.json`). Todos los
> artefactos de explicabilidad, estabilidad regional y contraste teórico se
> generaron para esa configuración y sus archivos llevan el sufijo `_smotenc`.
> Esto afecta a las Tablas 5.7, 5.9, 5.10, 5.11, 5.12 y 5.13 y a todas las
> figuras 04, 05 y 06.

---

# BLOQUE 0 — Cinco alertas que cambian conclusiones

Estas cinco cosas no son ajustes de cifras: cambian lo que el documento afirma.
Conviene resolverlas antes de tocar las tablas.

## 🔴 0.1 El test KS entre validación y prueba **rechaza** la igualdad de distribuciones

**PDF actual (§5.2):**

> «La prueba de Kolmogorov–Smirnov entre los conjuntos de validación y prueba
> obtuvo un estadístico de 0,0549 y p = 0,787. En consecuencia, no se encontró
> evidencia suficiente para rechazar la igualdad de sus distribuciones.»

**Corrida nueva** (`notebooks/output/02_...ipynb`, §10.7):

```
KS(val, test): estadístico=0.0549, p=0.0000
```

El estadístico es el mismo (0,0549) pero el p-valor es esencialmente cero. Y es
lo correcto: con n_val = 17.219 y n_test = 35.084, el valor crítico al 5 % es
1,36·√((n₁+n₂)/(n₁·n₂)) ≈ 0,0127, muy por debajo de 0,0549. **El p = 0,787 que
figura hoy en el PDF es un error**: corresponde a aplicar el test sobre los
cuatro porcentajes de clase en lugar de sobre las muestras completas.

**Qué hacer** — sustituir el pasaje de §5.2 por:

> La prueba de Kolmogorov–Smirnov entre las distribuciones del target en
> validación y en prueba arroja un estadístico de 0,0549 con p < 0,001. Con
> tamaños de muestra de 17.219 y 35.084 observaciones, el test detecta como
> significativa una diferencia de esa magnitud, de modo que las dos
> distribuciones no son idénticas: la validación concentra más masa en «Muy
> satisfecho» (29,0 % frente a 25,5 %) y menos en «No muy satisfecho» (17,6 %
> frente a 21,1 %). La magnitud del desplazamiento es, no obstante, moderada —la
> clase mayoritaria es la misma y su peso apenas varía (44,0 % frente a 42,0 %)—
> y no invalida el uso de 2020 como conjunto de calibración, cuya elección
> responde a la restricción temporal descrita en §4.4.2 y no a un supuesto de
> igualdad distribucional.

**Y en §4.4.2**, eliminar cualquier apoyo del argumento en la igualdad de
distribuciones: la justificación válida de 2020 es que es el **único año
admisible** (no puede estar en entrenamiento ni en prueba, y 2019, 2021 y 2022
no tienen ola), más su pertinencia histórica. Eso ya está bien redactado; solo
hay que no añadirle el KS como respaldo.

## 🔴 0.2 H1 no se sostiene en kappa cuando se incorpora la incertidumbre; sí en MAE

**PDF actual (§5.3.1 y Cap. 6):**

> «Por tanto, los resultados respaldan la hipótesis H1 en el conjunto de
> prueba… La diferencia entre CatBoost y la regresión logística ordinal es de
> 0,0220 puntos.»

**Corrida nueva.** Ahora la línea base **sí es ordinal** (`mord.LogisticIT`) y
es mucho más competitiva. Bootstrap pareado de 1.000 réplicas sobre 32 clústeres
país-año (`results/tables/bootstrap_pareado_vs_principal.csv`):

| Comparación con CatBoost [SMOTE-NC] | Δκw | IC 95 % de Δ | Conclusión |
|---|---|---|---|
| OLO [SMOTE-NC] | +0,0077 | [−0,0074; +0,0212] | **no distinguible** |
| OLO [pesos de clase] | +0,0108 | [−0,0042; +0,0240] | **no distinguible** |
| OLO [sin balanceo] | +0,0807 | [+0,0642; +0,0956] | distinguible |

En cambio, en **MAE ordinal** la ventaja sobre OLO es grande y distinguible:

| Comparación | ΔMAE | IC 95 % | Conclusión |
|---|---|---|---|
| OLO [pesos de clase] | +0,1779 | [+0,1532; +0,2034] | distinguible |
| OLO [SMOTE-NC] | +0,1676 | [+0,1431; +0,1926] | distinguible |

**Qué hacer** — reformular H1 en §5.3.1 y en Cap. 6 con este texto:

> El respaldo de H1 depende de la métrica. En kappa cuadrático, la ventaja de
> CatBoost con SMOTE-NC sobre la regresión logística ordinal con su mejor
> estrategia es de 0,0077 puntos, con un intervalo de confianza de
> [−0,0074; +0,0212] que **incluye el cero**: con la variabilidad del conjunto de
> prueba, ambos modelos son indistinguibles en acuerdo ordinal corregido por
> azar. En MAE ordinal, en cambio, la diferencia es de 0,1676 puntos con un
> intervalo de [+0,1431; +0,1926] que excluye el cero por amplio margen. La
> lectura conjunta es que los modelos de *gradient boosting* no mejoran de forma
> detectable el acuerdo ordinal global respecto de una línea base ordinal bien
> especificada, pero sí reducen sustancialmente la distancia media del error: sus
> equivocaciones caen más cerca de la categoría real. H1 se respalda, por tanto,
> en su dimensión de precisión ordinal y no en la de acuerdo corregido por azar.

Conviene añadir que esta conclusión **es más exigente que la de la literatura
previa** porque la línea base es un logit ordinal acumulativo y no una logística
multinomial; ese es un aporte del diseño, no una debilidad.

## 🔴 0.3 Seis de las quince configuraciones son indistinguibles de la principal

**Corrida nueva** (`bootstrap_pareado_vs_principal.csv`, métrica kappa):

| Comparado con CatBoost [SMOTE-NC] | Δκw | IC 95 % | P(Δ>0) |
|---|---|---|---|
| TabNet [pesos de clase] | +0,0011 | [−0,0095; +0,0106] | 0,591 |
| CatBoost [pesos de clase] | +0,0013 | [−0,0129; +0,0166] | 0,526 |
| XGBoost [pesos de clase] | +0,0046 | [−0,0064; +0,0146] | 0,787 |
| LightGBM [pesos de clase] | +0,0066 | [−0,0056; +0,0173] | 0,865 |
| OLO [SMOTE-NC] | +0,0077 | [−0,0074; +0,0212] | 0,832 |
| OLO [pesos de clase] | +0,0108 | [−0,0042; +0,0240] | 0,912 |

Las ocho restantes sí son distinguibles (Δ de +0,0174 a +0,0807).

**Qué hacer** — añadir a §5.3.1, después de la tabla de kappa en prueba:

> Seis de las quince configuraciones evaluadas presentan diferencias de kappa
> respecto de la configuración principal cuyo intervalo de confianza incluye el
> cero. Entre ellas están las mejores configuraciones de los cinco modelos
> comparados. En consecuencia, la jerarquía entre familias no queda determinada
> por el tamaño efectivo de muestra disponible —32 clústeres país-año en el
> conjunto de prueba— y la elección de CatBoost con SMOTE-NC como configuración
> principal se sostiene en el criterio de selección en validación y en su ventaja
> en MAE ordinal, no en una superioridad estadísticamente distinguible en kappa.

Esto responde directamente a la observación del revisor sobre atribuir
superioridad a diferencias descriptivas.

## 🔴 0.4 El menor MAE ordinal **no** corresponde a la configuración principal

**PDF actual (Resumen, Abstract, §5.3.1, Cap. 6):** «CatBoost con pesos de clase
obtuvo el mayor kappa cuadrático (κw = 0,5418) y el menor MAE ordinal (0,6297)».

**Corrida nueva.** El MAE de la configuración principal es 0,5688; el mínimo
entre las quince configuraciones es **CatBoost sin balanceo con 0,5421**, y la
diferencia **sí es distinguible**: ΔMAE = −0,0268 con IC [−0,0376; −0,0165].

MAE ordinal en prueba, las quince configuraciones
(`results/resultados_modelos.csv`):

| Modelo | Sin balanceo | Pesos de clase | SMOTE-NC |
|---|---|---|---|
| OLO | 0,5601 | 0,7468 | 0,7363 |
| XGBoost | **0,5449** | 0,6415 | 0,5681 |
| CatBoost | **0,5421** | 0,6406 | 0,5688 |
| LightGBM | **0,5438** | 0,6463 | 0,5604 |
| TabNet | 0,5539 | 0,6576 | 0,6718 |

**Qué hacer** — corregir la afirmación en los cuatro sitios y añadir a §5.3.1:

> El menor MAE ordinal entre las quince configuraciones no corresponde a la
> seleccionada: CatBoost sin balanceo alcanza 0,5421 frente a 0,5688 de CatBoost
> con SMOTE-NC, y la diferencia es distinguible del ruido de muestreo
> (Δ = −0,0268; IC 95 % [−0,0376; −0,0165]). El patrón es sistemático: en los
> cinco modelos, la ausencia de balanceo produce el MAE más bajo y los pesos de
> clase el más alto. Al redistribuir masa predictiva hacia las clases
> minoritarias, el modelo mejora el acuerdo ordinal corregido por azar y, al
> mismo tiempo, se aleja más en promedio de la categoría exacta. SMOTE-NC ocupa
> una posición intermedia en esa compensación, lo que explica que sea la
> estrategia seleccionada: obtiene el mayor kappa sin degradar el MAE tanto como
> la ponderación de clases.

## 🔴 0.5 El desglose por categoría se invierte respecto de lo previsto

Con SMOTE-NC el modelo ya no sobrecorrige hacia las clases minoritarias como lo
hacía con pesos de clase. La categoría mayoritaria pasa a ser la **mejor**
recuperada, no la peor.

**Corrida nueva**
(`results/tables/metricas_por_clase_CatBoost_smotenc_ordinal_4clases_test.csv`):

| Clase | Precision | Recall | F1 | Soporte | % |
|---|---|---|---|---|---|
| 0 — Para nada satisfecho | 0,4489 | 0,3500 | 0,3933 | 3.969 | 11,31 |
| 1 — No muy satisfecho | 0,4321 | 0,3729 | 0,4003 | 7.414 | 21,13 |
| 2 — Más bien satisfecho | 0,5347 | **0,6328** | **0,5796** | 14.746 | 42,03 |
| 3 — Muy satisfecho | 0,5664 | 0,5149 | 0,5394 | 8.955 | 25,52 |
| Promedio macro | 0,4955 | 0,4676 | 0,4782 | 35.084 | 100 |
| Promedio ponderado | 0,5114 | 0,5158 | 0,5104 | 35.084 | 100 |

**Qué hacer** — al redactar la sección nueva de métricas por categoría (ver 4.7
más abajo), no reutilizar la narrativa de «el modelo deja de sobrepredecir la
moda»: ahora ocurre lo contrario. El texto correcto está en 4.7.

---

# BLOQUE 1 — Resumen y Abstract

## 🔴 1.1 Cifras del modelo principal

**PDF actual:** «CatBoost con pesos de clase obtuvo el mayor kappa cuadrático
(κw = 0,5418) y el menor MAE ordinal (0,6297), aunque ningún modelo dominó todas
las métricas.»

**Qué poner:**

> CatBoost con SMOTE-NC obtuvo el mayor kappa cuadrático (κw = 0,5386) y el menor
> MAE ordinal entre las configuraciones seleccionadas en validación (0,5688),
> aunque ningún modelo dominó todas las métricas y seis de las quince
> configuraciones evaluadas resultaron estadísticamente indistinguibles de la
> principal.

Aplicar el mismo cambio en el Abstract (κw = 0.5386, ordinal MAE 0.5688,
CatBoost with SMOTE-NC).

## 🔴 1.2 La afirmación sobre LIME cambia de variable

**PDF actual:** «LIME destacó la situación económica del país en los errores
extremos.» / «LIME highlighted the national economic situation in extreme
prediction errors.»

**Corrida nueva** (`results/tables/lime_CatBoost_smotenc.csv`, media de |peso|
en el grupo de error máximo): Confianza en el Gobierno 0,0461 · Distribución del
ingreso justa 0,0430 · Situación económica del país 0,0414 · Apoyo a la
democracia 0,0389 · Componente igualitario 0,0291.

**Qué poner:** «LIME situó la confianza en el Gobierno como la variable de mayor
peso en las aproximaciones locales de los errores extremos, seguida por la
percepción de justicia en la distribución del ingreso y por la situación
económica del país.» (Y su equivalente en inglés.)

## 🟡 1.3 Orden de los bloques SHAP

**PDF actual:** «percepción política como el bloque de mayor importancia, seguida
por la evaluación económica y la confianza institucional. El componente
igualitario fue el indicador contextual más relevante.» → **sigue siendo
correcto** con las cifras nuevas. Solo verificar que no se citen valores
numéricos de bloque en el Resumen (no los cita).

## 🟢 1.4 Tamaño de la muestra

«430.895 respuestas» ya está aplicado y es correcto. Única sugerencia: quitar
«aproximadamente», porque la cifra es exacta, y mencionar las 489.771 iniciales:
«Se integraron 489.771 respuestas individuales de 24 olas del Latinobarómetro, de
las cuales 430.895 se utilizan efectivamente en el modelado».

---

# BLOQUE 2 — Capítulo 1: hipótesis

## 🔴 2.1 H1

**Actual:** «Los modelos de *gradient boosting* superarán a la regresión logística
ordinal en las métricas predictivas, debido a su capacidad para capturar
interacciones no lineales.»

Con la línea base ordinal correcta, esta hipótesis se cumple en MAE y no en
kappa (ver 0.2). **No hace falta reescribir la hipótesis** —una hipótesis puede
respaldarse parcialmente—, pero sí su evaluación en §5.3.1 y en Cap. 6.

## 🔴 2.2 H2

**Actual:** «TabNet obtendrá un kappa cuadrático superior al de la regresión
logística ordinal, pero inferior al del modelo de *gradient boosting* con mejor
desempeño en el conjunto de prueba.»

**Corrida nueva:** κ(OLO best) = 0,5308 < κ(TabNet best) = 0,5374 <
κ(CatBoost best) = 0,5386. El ordenamiento se cumple, pero los márgenes son de
0,0066 y **0,0012** puntos, y la diferencia entre TabNet y CatBoost tiene un
intervalo de [−0,0095; +0,0106].

**Qué hacer** — en §5.3.1 y Cap. 6:

> El ordenamiento previsto por H2 se cumple en las estimaciones puntuales:
> κw = 0,5308 para la regresión logística ordinal, 0,5374 para TabNet y 0,5386
> para CatBoost. Sin embargo, los márgenes son de 0,0066 y 0,0012 puntos, y el
> intervalo de confianza de la diferencia entre TabNet y CatBoost
> ([−0,0095; +0,0106]) incluye el cero. H2 se respalda por tanto de forma
> descriptiva y no inferencial: el orden observado es compatible con la
> hipótesis, pero los datos no permiten distinguir a TabNet de la mejor
> configuración de *gradient boosting* ni de la línea base ordinal.

## 🟡 2.3 H3

**Actual:** «La confianza institucional, la percepción de la corrupción y la
evaluación económica concentrarán la mayor contribución predictiva atribuida por
SHAP.»

**Corrida nueva** (`results/tables/shap_bloques_ic_CatBoost_smotenc.csv`):
percepción política 1.º, evaluación económica 2.º, confianza institucional 3.º,
contexto democrático 4.º, corrupción y seguridad 5.º. El respaldo sigue siendo
**parcial**, igual que en el PDF actual: se confirman dos de los tres bloques y
la corrupción queda en penúltimo lugar. El texto de §5.5.1 solo necesita los
valores nuevos (ver 4.9).

---

# BLOQUE 3 — Capítulo 4: metodología

## 🟡 3.1 §4.2.3 y Tabla 4.1 — entorno computacional

**Corrida nueva** (registro de versiones del NB02):

| Componente | Valor |
|---|---|
| Python | 3.12.3 |
| Kernel | 7.0.0-28-generic |
| pandas / numpy / scikit-learn | 3.0.3 / 2.4.6 / 1.9.0 |
| xgboost / lightgbm / catboost | 3.3.0 / 4.6.0 / 1.2.10 |
| torch | 2.13.0+cu130 |
| optuna | 4.9.0 |
| mord | 0.7 |
| shap | 0.52.0 |
| imbalanced-learn | 0.14.2 |

**Qué hacer:** la Tabla 4.1 (hardware) no cambia. Conviene **añadir** las
versiones de librerías, en la propia Tabla 4.1 o en una nota al pie, porque son
el requisito de reproducibilidad que un revisor comprobaría. En particular
`mord 0.7`, que es la que sostiene que la línea base es ordinal.

## 🔴 3.2 §5.3.1 (primer párrafo) — duración de la ejecución

**PDF actual:** «La ejecución completa del proceso experimental requirió
aproximadamente seis horas.»

**Corrida nueva** (marcas de inicio y fin de cada notebook):

| Notebook | Duración |
|---|---|
| 01_carga_datos | 9 s |
| 02_preprocesamiento_entrenamiento | **13 h 19 min** |
| 03_evaluacion_comparativa | 2 min 40 s |
| 04_explicabilidad_xai | 19 s |
| 05_estabilidad_temporal_regional | 5 s |
| 06_contraste_teorico | 1 s |
| **Total** | **13 h 22 min** |

**Qué poner:**

> La ejecución completa del flujo requirió 13 h 22 min en el servidor descrito en
> la Tabla 4.1. El notebook de preprocesamiento y entrenamiento concentra el
> 99,6 % de ese tiempo (13 h 19 min), porque incluye la búsqueda de
> hiperparámetros de las quince configuraciones del experimento E1, las cinco de
> E2 y las quince de los tres pliegues temporales. Las etapas de evaluación,
> explicabilidad y contraste teórico, que operan sobre los modelos ya
> serializados, suman menos de tres minutos y medio.

Este desglose es más informativo que la cifra global y responde a la limitación
que el PDF reconoce hoy («los registros disponibles no conservan tiempos
separados para cada algoritmo»): sí los conserva por notebook.

## 🟡 3.3 §4.6 — multicolinealidad de V-Dem

**PDF actual:** «En la especificación inicial de diecinueve variables V-Dem se
identificaron 46 pares con correlaciones de Spearman superiores a 0,85 y una
correlación máxima de 0,990. Al conservar los cuatro indicadores… el número de
pares por encima de 0,85 se redujo a uno y la correlación máxima descendió a
0,864.»

**Corrida nueva** (`results/tables/correlaciones_resumen_matrices.csv`):

| Matriz | Variables | \|r\| máx | \|r\| medio | Pares \|r\|>0,7 | Pares totales |
|---|---|---|---|---|---|
| Latinobarómetro | 24 | 0,5581 | 0,1316 | 0 | 576 |
| V-Dem | 4 | **0,8552** | 0,7750 | 6 | 16 |
| Fusionado (LB × V-Dem) | 28 | **0,8618** | 0,1224 | 5 | 784 |

**Qué hacer:** sustituir «0,864» por **0,8552** (V-Dem) y, si se cita la matriz
fusionada, **0,8618**. La cifra de la especificación inicial (46 pares, máximo
0,990) proviene de un diagnóstico previo que el flujo actual ya no reejecuta;
conviene decir que corresponde al conjunto inicial de diecinueve indicadores y
citar como fuente el diagnóstico de selección, no la corrida.

**Y añadir** el dato nuevo, que refuerza el argumento: en la matriz del
Latinobarómetro **ningún** par supera 0,7 (máximo 0,5581), de modo que la
redundancia se concentra por completo en el bloque contextual.

## 🟡 3.4 Figuras 4.3 y 4.4 — matrices de correlación

Se regeneraron: `results/figures/eda2_matriz_correlacion_latinobarometro.png` y
`eda2_matriz_correlacion_vdem.png`. Hay además una tercera,
`eda2_matriz_correlacion_merge.png` (28 variables), que el PDF no incluye y que
conviene añadir como Figura 4.5: es la que documenta que la redundancia no se
propaga al conjunto fusionado.

**Valores a citar en el texto de la Figura 4.4** (V-Dem, nivel país-año): las
parejas y magnitudes que hoy cita el PDF (0,86 / 0,76 / −0,83 / 0,79) deben
leerse de la figura regenerada; el máximo global es 0,8552.

## 🟢 3.5 §4.9.1 — línea base ordinal

El PDF ya dice «implementada con `mord.LogisticIT`… bajo el supuesto de momios
proporcionales» y «20 ensayos» para OLO. Verificado en la corrida:

- `models/hp_OLO_smotenc_ordinal_4clases.json` → `implementacion:
  mord.LogisticIT`, `formulacion: logit acumulativo con umbrales de umbral
  inmediato`, `n_coeficientes: 28` (un único vector), `umbrales_theta:
  [−0,4073; −0,0442; +0,3970]`, `alpha = 0,00746`, 20 ensayos.
- El NB02 imprime `Backend OLO: mord.LogisticIT (logit acumulativo, ordinal)`.

**Único ajuste recomendado (nota al pie):** precisar el estimador, porque
`LogisticIT` minimiza la pérdida de umbral inmediato y no la verosimilitud del
modelo de odds proporcionales de McCullagh:

> La implementación empleada (`mord.LogisticIT`) estima el modelo minimizando la
> pérdida de umbral inmediato en lugar de la verosimilitud del modelo de odds
> proporcionales; conserva la estructura de umbrales ordenados y el vector de
> coeficientes compartido, con regularización L2 sobre este último. En el ajuste
> final se estimaron 28 coeficientes y tres umbrales
> (θ = −0,4073; −0,0442; +0,3970).

## 🔴 3.6 §4.9.4 y §6.2 — el balanceo de TabNet se aplica en la función de pérdida

El PDF describe hoy a TabNet sin precisar cómo recibe el desbalance, y §6.2 dice
que «TabNet no admite pesos para cada observación, lo que introduce una
asimetría».

**Corrida nueva** (`models/hp_TabNet_*_ordinal_4clases.json`,
`config_entrenamiento`):

| Estrategia | `weights` | `loss_fn` | Épocas entrenadas |
|---|---|---|---|
| sin_balanceo | 0 | `cross_entropy (por defecto)` | 43 de 200 |
| pesos_clase | 0 | `CrossEntropyLoss ponderada por frecuencia inversa de clase` | 42 de 200 |
| smotenc | 0 | `cross_entropy (por defecto)` | 62 de 200 |

Vector de pesos aplicado en la pérdida: [2,6524; 0,9245; 0,5756; 1,2439].

**Qué añadir a §4.9.4:**

> TabNet no admite un peso por observación en su interfaz de entrenamiento, de
> modo que el desbalance se corrige en la función de pérdida y no en el muestreo:
> la estrategia de pesos de clase se implementa como una entropía cruzada
> ponderada por la frecuencia inversa de clase estimada sobre el conjunto de
> entrenamiento, mientras que el muestreo se mantiene uniforme en las tres
> estrategias. La consecuencia que debe declararse es que **el factor de
> expansión muestral X_020 no interviene en el ajuste de TabNet**, a diferencia
> de los otros cuatro modelos, que lo reciben en `sample_weight`.

**Y reescribir la cuarta limitación de §6.2:**

> Cuarta, TabNet no admite pesos por observación: la estrategia de pesos de clase
> se implementa en su función de pérdida y el factor de expansión muestral no
> interviene en su ajuste. Esto introduce una asimetría en el tratamiento del
> desbalance respecto de los cuatro modelos restantes.

## 🔴 3.7 §4.9.3 — los pesos de clase se estiman sobre el conjunto de entrenamiento

La fórmula (4.1) del PDF usa N, el total del conjunto. El código ahora los
estima solo sobre las olas de entrenamiento.

**Corrida nueva** (salida del NB02):

```
Pesos inversos de clase (estimados sobre train: 378,592 registros de 431,756):
  Clase 0 (Para nada satisfecho): 2.6524
  Clase 1 (No muy satisfecho)   : 0.9245
  Clase 2 (Más bien satisfecho) : 0.5756
  Clase 3 (Muy satisfecho)      : 1.2439
```

**Qué hacer** — en la fórmula (4.1), sustituir N y n_k por sus equivalentes de
entrenamiento y añadir:

> Los pesos se estiman exclusivamente sobre las 21 olas de entrenamiento
> (378.592 registros), como
> w_c = n_train / (K · n_{c,train}). Calcularlos sobre el conjunto integrado
> introduciría las frecuencias marginales de validación y prueba en el ajuste del
> modelo. Los valores resultantes son 2,6524; 0,9245; 0,5756 y 1,2439 para las
> clases 0 a 3.

**Atención a una ambigüedad que el propio flujo genera.** El mensaje del NB02
imprime «378,592 registros de 431,756», y esa segunda cifra **no** es la del
Cuadro 5.1. Hay dos totales distintos y los dos son correctos:

| Cifra | Qué es | Dónde aparece |
|---|---|---|
| **431.756** | Dataset tras excluir el target NS/NR (−21.036) y Venezuela posterior a 2017 (−4.707). Es el `df` sobre el que se calculan los pesos y las correlaciones del EDA. | NB02 celdas 23, 25, 27 y 35 |
| **430.895** | Suma de los tres conjuntos de análisis, tras excluir además Nicaragua 2020 de la validación (−861). Es la muestra efectivamente modelada. | Tabla 5.1, `conjuntos_resumen_despues_exclusiones.csv` |

Los 861 registros de diferencia son los de Nicaragua en la ola 2020: existen en
`df` pero no entran en ningún conjunto. **En el documento debe citarse siempre
430.895 como muestra de modelado**, y si se cita 431.756 (por ejemplo, junto a
ρ año–target en 4.4) hay que decir a qué corresponde. Citar las dos sin
distinguirlas es el tipo de inconsistencia que se detecta al revisar.

## 🔴 3.8 §4.9.3 — SMOTE-NC no es *ceteris paribus* (ahora es la estrategia principal)

Esta advertencia pasa a ser central, porque la configuración seleccionada usa
SMOTE-NC. `SMOTENC.fit_resample` se aplica sobre la matriz **ya imputada con
MICE**, mientras que los brazos A y B entregan a los árboles los valores
ausentes sin imputar.

**Qué añadir a §4.9.3 (y como séptima limitación en §6.2):**

> SMOTE-NC no admite valores ausentes, de modo que la estrategia C se aplica
> sobre la matriz de entrenamiento ya imputada. Para los modelos de árboles, el
> brazo SMOTE-NC combina por tanto dos factores —sobremuestreo sintético e
> imputación previa— frente a los brazos A y B, que reciben los valores faltantes
> nativos. Dado que la configuración principal de este trabajo pertenece a ese
> brazo, la advertencia es especialmente pertinente: su ventaja en kappa no puede
> atribuirse exclusivamente al remuestreo. Separar ambos efectos requeriría un
> cuarto brazo de control con la matriz imputada y sin remuestreo, que no forma
> parte de este diseño.

## 🟢 3.9 §4.10 — criterio de selección

Ya aplicado y verificado: `results/modelo_xai_seleccionado.json` registra
`conjunto_seleccion: "val"` y `valor_metrica_seleccion: 0.48395`. No tocar.

## 🔴 3.10 §4.11.2 y §5.5.2 — solo se generan cinco curvas ALE, no seis

**PDF actual:** «Se generan gráficos ALE para un máximo de seis variables».

**Corrida nueva.** La selección produce seis variables (dos por bloque), pero
solo cinco llegan a graficarse:

```
Variables para ALE (2 por bloque, ordenadas por |SHAP|):
  Confianza Gobierno            [Confianza institucional]  |SHAP|=0.1697
  Confianza Partidos Políticos  [Confianza institucional]  |SHAP|=0.0366
  Situación económica país      [Evaluación económica]     |SHAP|=0.1637
  Distribución ingreso justa    [Evaluación económica]     |SHAP|=0.1174
  Apoyo a la democracia         [Percepción política]      |SHAP|=0.2136
  Aprobación gobierno           [Percepción política]      |SHAP|=0.1436
```

Figuras existentes en `results/figures/`: `04_ale_H_002_031.png`,
`04_ale_H_002_241.png`, `04_ale_D_001_001.png`, `04_ale_C_006_003_011.png`,
`04_ale_A_001_001.png`. **Falta la de `B_006_061` (Aprobación gobierno)** porque
es una variable binaria tras la recodificación y el filtro de ALE exige al menos
tres valores distintos para estimar la curva.

**Qué hacer:**

1. En §4.11.2, añadir la condición explícita: «Se excluyen las variables con
   menos de tres valores distintos, para las que la curva acumulada no está
   definida; por esa razón la aprobación del Gobierno, recodificada como
   binaria, no aparece entre las curvas.»
2. **Sustituir por completo las Figuras 5.5–5.10** (hoy son seis curvas, todas de
   confianza institucional) por las cinco nuevas, y reescribir §5.5.2 sobre
   ellas. Las curvas ahora cubren los tres bloques sustantivos, que es lo que
   §4.11.2 describe.
3. El párrafo de §2.5.3 que usa como ejemplo las curvas de «Confianza en el
   Congreso» y «Confianza en la televisión» (Figura 2.7) también hay que
   revisarlo: esas dos variables **ya no tienen curva ALE**. Opciones: sustituir
   el ejemplo por dos de las cinco curvas nuevas, o dejar la Figura 2.7 como
   ilustración conceptual declarando que proviene de un diagnóstico previo.

## 🟢 3.11 §4.11.3 — LIME sobre 200 casos

Verificado en la corrida: 100 representativos + 50 de error máximo + 50
discordantes = 200, y los tres grupos se explican íntegros
(`results/tables/lime_CatBoost_smotenc.csv` contiene 200 índices únicos). El
texto del PDF es correcto. Añadir la cifra al pie de la tabla de LIME.

## 🟡 3.12 Tabla 4.12 — pie de TabNet

Añadir al pie, además de los 20 ensayos: «Entrenamiento con `max_epochs = 200` y
paciencia 20 sobre el conjunto de validación; las tres estrategias detuvieron el
ajuste entre las épocas 42 y 62.»

---

# BLOQUE 4 — Capítulo 5: resultados

## 🟢 4.1 Tabla 5.1 y §5.1 — trazabilidad de la muestra

Coincide exactamente con `results/tables/conjuntos_*.csv`: 489.771 → 457.499 →
436.463 (−21.036) → 431.756 (−4.707) → 430.895 (−861). No tocar.

**Dato nuevo disponible** (`conjuntos_efecto_exclusiones.csv`), útil como nota:
el efecto de las exclusiones es muy desigual entre conjuntos —train −20.284
(5,09 %), validación −2.985 (14,77 %), prueba −3.335 (8,68 %)— porque Venezuela
y Nicaragua pesan más en los conjuntos pequeños.

## 🟡 4.2 §5.1 — cifras de Venezuela y Nicaragua

**PDF actual:** «el 73,7 % de las personas encuestadas se ubicó ese año [2018] en
la categoría "Muy satisfecho"… En 2024, la categoría de máxima satisfacción
todavía concentra el 54,5 %… un índice de poliarquía electoral de 0,196 en 2024,
frente a un promedio de 0,741».

**Problema de trazabilidad:** el flujo actual excluye Venezuela posterior a 2017
**antes** del análisis exploratorio, así que
`results/tables/eda_venezuela_anomalia.csv` solo llega a 2017 y esas tres cifras
ya no son reproducibles desde la corrida.

**Lo que la corrida sí produce y respalda el corte:**

| Año | n | % «Muy satisfecho» | KS vs. otros países | p |
|---|---|---|---|---|
| 2013 | 1.187 | 21,9 | 0,0779 | <0,001 |
| 2015 | 1.195 | 39,7 | 0,2096 | <0,001 |
| 2016 | 1.192 | 49,2 | 0,2742 | <0,001 |
| 2017 | 1.173 | 49,9 | 0,2552 | <0,001 |

Nicaragua 2020: KS = 0,1114, p = 2,6 × 10⁻⁹.

**Qué hacer** — reescribir el pasaje apoyándolo en la escalada **observable
dentro del rango conservado**, que es igual de convincente y sí es reproducible:

> La proporción de personas que declara la máxima satisfacción en Venezuela pasa
> del 21,9 % en 2013 al 39,7 % en 2015, el 49,2 % en 2016 y el 49,9 % en 2017,
> mientras el índice de poliarquía electoral del país desciende de forma
> sostenida. El test de Kolmogorov–Smirnov entre la distribución venezolana y la
> del resto de los países alcanza un estadístico de 0,2552 en 2017 (p < 0,001),
> frente a valores inferiores a 0,08 en las olas de los años noventa. La
> divergencia respalda la decisión de conservar a Venezuela únicamente hasta 2017
> en entrenamiento y excluir sus olas posteriores de la evaluación.

Si se quieren conservar las cifras de 2018 y 2024, hay que declararlas como
provenientes de un diagnóstico previo a la exclusión y no de los resultados del
flujo.

## 🟢 4.3 Tabla 5.2 y Figura 5.1 — distribución del target

Coincide con `results/tables/eda_desbalance_clases.csv`: train 9,4/27,0/43,4/20,1
(razón 4,61); validación 9,3/17,6/44,0/29,0 (4,71); prueba
11,3/21,1/42,0/25,5 (3,72). No tocar. **Salvo** el pasaje del KS, ver 0.1.

## 🟡 4.4 §5.2 — correlación año–target

**PDF actual:** ρ = 0,027.
**Corrida nueva** (`results/tables/eda_correlacion_año_target.csv`):
**ρ = 0,0217**, p < 0,001, n = 431.756 (el conjunto previo a la exclusión de
Nicaragua 2020; véase la nota de 3.7).

Nótese que con ese n el p-valor es significativo aunque la magnitud sea
despreciable. Conviene redactarlo así: «ρ = 0,0217; con n = 431.756 el contraste
resulta significativo, pero la magnitud de la asociación es despreciable, lo que
respalda su uso como criterio de partición y no como predictor».

## 🟡 4.5 §5.2 — correlaciones V-Dem a nivel país-año

**PDF actual:** v2x_egal ρ = −0,384 país-año; v2xcl_rol −0,284; v2x_corr +0,168.

**Corrida nueva** (`results/tables/eda_correlaciones_features.csv`, columna
`r_pais_año`):

| Variable | ρ individual | ρ país-año |
|---|---|---|
| v2x_egal | −0,1703 | **−0,4652** |
| v2xcl_rol | −0,1320 | **−0,3527** |
| v2x_polyarchy | −0,0854 | **−0,2034** |
| v2x_corr | +0,0821 | **+0,1888** |

Las asociaciones son más fuertes que las reportadas, lo que **refuerza** el
argumento de los ciudadanos críticos. Actualizar las tres cifras y añadir
v2x_polyarchy.

## 🟢 4.6 Tabla 5.3 — correlaciones de Spearman individuales

Verificada línea por línea contra `eda_correlaciones_features.csv`: coincide en
las 28 variables. No tocar.

## 🟢 4.7 Tabla 5.4 — kappa en validación

Coincide exactamente con la corrida
(`results/tables/metricas_kappa_pivot_val.csv`). No tocar.

**Sí hay que corregir el texto que la acompaña**, que compara con la Tabla 5.5:

**PDF actual:** «XGBoost y LightGBM conservan la misma mejor estrategia en
validación y en prueba, mientras que CatBoost, OLO y TabNet cambian de estrategia
al considerar el máximo en prueba. La configuración seleccionada en validación
—CatBoost con SMOTE-NC— alcanza un kappa de 0,5386 en prueba, aunque el mayor
valor de esa tabla corresponde a CatBoost con pesos de clase (0,5418).»

**Corrida nueva** (`results/tables/mejor_estrategia_por_modelo.csv`):

| Modelo | Mejor en val | κ val | κ test de esa config. | Mejor en test | Coincide |
|---|---|---|---|---|---|
| CatBoost | SMOTE-NC | 0,4840 | 0,5386 | SMOTE-NC | **sí** |
| LightGBM | Pesos de clase | 0,4796 | 0,5320 | Pesos de clase | **sí** |
| OLO | SMOTE-NC | 0,4804 | 0,5308 | SMOTE-NC | **sí** |
| TabNet | Pesos de clase | 0,4783 | 0,5374 | Pesos de clase | **sí** |
| XGBoost | Pesos de clase | 0,4799 | 0,5340 | Pesos de clase | **sí** |

**Los cinco modelos coinciden**, y la configuración global seleccionada en
validación (CatBoost SMOTE-NC) es también la de mayor kappa en prueba (0,5386).
Este es un resultado mucho más fuerte que el del PDF actual.

**Qué poner:**

> La comparación con la Tabla 5.5 permite verificar que la separación entre
> selección y evaluación no altera los resultados sustantivos: los cinco modelos
> conservan la misma mejor estrategia en validación y en prueba, y la
> configuración global seleccionada en validación —CatBoost con SMOTE-NC— es
> también la de mayor kappa cuadrático en prueba (0,5386). Los valores de
> validación (0,3933–0,4840) son sistemáticamente inferiores a los de prueba
> (0,4578–0,5386) en las quince combinaciones, lo que indica que el conjunto de
> prueba no resulta más exigente que el de calibración; esa diferencia no basta,
> por sí sola, para descartar sobreajuste a la validación.
>
> Promediando por estrategia entre los cinco modelos, los pesos de clase (0,4796)
> superan a SMOTE-NC (0,4730) y a la ausencia de balanceo (0,4472). Que la
> configuración seleccionada pertenezca al brazo SMOTE-NC pese a que esa
> estrategia no lidera el promedio ilustra que la interacción entre modelo y
> estrategia de balanceo no es aditiva.

## 🔴 4.8 Tabla 5.5 — kappa en prueba (reemplazar completa)

```latex
\begin{table}[htbp]
\centering
\caption{Kappa cuadrático por modelo y estrategia de balanceo en el conjunto de prueba.}
\label{tab:kappa-test}
\begin{tabular}{lcccl}
\hline
\textbf{Modelo} & \textbf{Sin balanceo} & \textbf{Pesos de clase} & \textbf{SMOTE-NC} & \textbf{Mejor estrategia} \\
\hline
OLO (línea base) & 0,4578 & 0,5278 & \textbf{0,5308} & SMOTE-NC \\
XGBoost  & 0,5010 & \textbf{0,5340} & 0,5150 & Pesos de clase \\
CatBoost & 0,5049 & 0,5373 & \textbf{0,5386} & SMOTE-NC \\
LightGBM & 0,5050 & \textbf{0,5320} & 0,5051 & Pesos de clase \\
TabNet   & 0,5178 & \textbf{0,5374} & 0,5208 & Pesos de clase \\
\hline
\end{tabular}
\end{table}
```

Promedios por estrategia en prueba: sin balanceo 0,4973 · pesos de clase 0,5337 ·
SMOTE-NC 0,5220.

**Texto que hay que reescribir a continuación.** El pasaje actual sobre las
estrategias sigue siendo válido en su estructura pero con otras cifras:

> Los pesos de clase generan el mejor kappa para XGBoost, LightGBM y TabNet,
> mientras que SMOTE-NC lo hace para CatBoost y para la regresión logística
> ordinal. La mejora más pronunciada respecto de la ausencia de balanceo se
> observa en la línea base ordinal, cuyo kappa aumenta de 0,4578 a 0,5308
> (+0,0729); en los cuatro modelos restantes la ganancia se sitúa entre 0,0197
> (TabNet) y 0,0336 puntos (CatBoost). El patrón es, por tanto, que la línea base
> ordinal depende del tratamiento del desbalance en un grado que los modelos no
> lineales no requieren.

> ⚠️ **Nivel C.** El PDF no debe afirmar *por qué* ocurre eso. Una explicación
> del tipo «los árboles capturan la estructura de las clases minoritarias sin
> reponderación» no se puede sostener con los artefactos de esta corrida: haría
> falta un experimento que aislara la capacidad de la frontera de decisión, y no
> forma parte del diseño. Describir la asimetría es suficiente y es lo
> defendible.

## 🔴 4.9 Tabla 5.6 — métricas complementarias (reemplazar completa)

```latex
\begin{table}[htbp]
\centering
\caption{Métricas complementarias por modelo con la mejor estrategia de balanceo
seleccionada en validación, sobre el conjunto de prueba.}
\label{tab:metricas-comp}
\begin{tabular}{llccccc}
\hline
\textbf{Modelo} & \textbf{Estrategia} & \textbf{Accuracy} & \textbf{F1 Macro} & \textbf{MAE Ordinal} & \textbf{AUROC OvR} & \textbf{Bal. acc.} \\
\hline
OLO (línea base) & SMOTE-NC       & 0,4198 & 0,3977 & 0,7363 & 0,7143 & 0,4590 \\
XGBoost          & Pesos de clase & 0,4905 & 0,4793 & 0,6415 & \textbf{0,7688} & 0,5084 \\
CatBoost         & SMOTE-NC       & \textbf{0,5158} & 0,4782 & \textbf{0,5688} & 0,7582 & 0,4676 \\
LightGBM         & Pesos de clase & 0,4909 & \textbf{0,4798} & 0,6463 & 0,7672 & \textbf{0,5115} \\
TabNet           & Pesos de clase & 0,4747 & 0,4607 & 0,6576 & 0,7577 & 0,4949 \\
\hline
\end{tabular}
\end{table}
```

**Texto:**

> No existe un único modelo dominante en todos los criterios. CatBoost con
> SMOTE-NC obtiene la mayor exactitud (0,5158) y el menor MAE ordinal (0,5688)
> entre las configuraciones seleccionadas; LightGBM alcanza el mejor F1 macro
> (0,4798) y la mayor exactitud balanceada (0,5115); y XGBoost, el AUROC OvR más
> alto (0,7688). La regresión logística ordinal queda por detrás en las cinco
> métricas, con la diferencia más marcada en MAE ordinal (0,7363 frente a 0,5688)
> y en AUROC (0,7143 frente a 0,7688).
>
> Conviene notar la tensión entre exactitud y exactitud balanceada: CatBoost con
> SMOTE-NC lidera la primera (0,5158) y queda última entre los modelos no
> lineales en la segunda (0,4676), mientras que LightGBM con pesos de clase
> invierte esa relación. Es la misma compensación que se observa entre kappa y
> MAE: las estrategias que redistribuyen masa hacia las clases minoritarias
> mejoran el equilibrio entre categorías a costa del acierto agregado.

## 🔴 4.10 Tabla 5.7 — hiperparámetros de CatBoost (reemplazar completa)

`models/hp_CatBoost_smotenc_ordinal_4clases.json`:

```latex
\begin{table}[htbp]
\centering
\caption{Hiperparámetros de CatBoost con SMOTE-NC seleccionados en validación (2020)
y evaluados en prueba (2023--2024).}
\label{tab:hp-catboost}
\begin{tabular}{llp{7.2cm}}
\hline
\textbf{Hiperparámetro} & \textbf{Valor óptimo} & \textbf{Descripción} \\
\hline
iterations           & 300       & Número de iteraciones de boosting o árboles construidos. \\
depth                & 4         & Profundidad máxima de los árboles. \\
learning\_rate       & 0,010033  & Tasa que regula la contribución de cada nuevo árbol al ensamble. \\
l2\_leaf\_reg        & 9,272779  & Coeficiente de regularización L2 aplicado a los valores de las hojas. \\
bagging\_temperature & 0,055943  & Intensidad de la aleatoriedad aplicada mediante el bootstrap bayesiano. \\
border\_count        & 68        & Número de intervalos utilizados para discretizar las variables numéricas. \\
random\_strength     & 6,448171  & Nivel de aleatoriedad incorporado al evaluar las divisiones de los árboles. \\
\hline
\end{tabular}
\end{table}
```

**Pie:** «Búsqueda bayesiana TPE con 50 ensayos, semilla 42, objetivo kappa
cuadrático en validación; el kappa alcanzado en validación fue 0,4840. Función de
pérdida `MultiClass`, entrenamiento en GPU (`task_type = GPU`).»

**Texto que sigue a la tabla — hay que reescribirlo por completo**, porque el
modelo seleccionado es ahora mucho más pequeño y regularizado:

> Los valores describen un modelo deliberadamente conservador: 300 iteraciones,
> profundidad 4 y una tasa de aprendizaje de 0,010, combinadas con una
> regularización L2 elevada (9,27). Es una configuración muy distinta de la que
> maximizaría el ajuste sobre el entrenamiento, y resulta coherente con el brazo
> SMOTE-NC: el sobremuestreo sintético amplía el conjunto de entrenamiento con
> observaciones interpoladas, de modo que la búsqueda de hiperparámetros
> privilegia un ensamble de baja varianza que no memorice los ejemplos
> sintéticos.

## 🆕 4.11 Nueva Tabla 5.7b — intervalos de confianza por bootstrap de clústeres

**Insertar después de la Tabla 5.6.** Fuente:
`results/tables/bootstrap_ic_modelos.csv`. Las quince configuraciones tienen
ahora intervalo (en la versión anterior de este documento solo nueve lo tenían).

| Modelo | Estrategia | κw | IC 95 % | EE bootstrap |
|---|---|---|---|---|
| CatBoost | SMOTE-NC | 0,5386 | [0,4937; 0,5741] | 0,0209 |
| TabNet | Pesos de clase | 0,5374 | [0,4965; 0,5692] | 0,0187 |
| CatBoost | Pesos de clase | 0,5373 | [0,4931; 0,5716] | 0,0202 |
| XGBoost | Pesos de clase | 0,5340 | [0,4914; 0,5690] | 0,0199 |
| LightGBM | Pesos de clase | 0,5320 | [0,4910; 0,5658] | 0,0193 |
| OLO | SMOTE-NC | 0,5308 | [0,4922; 0,5634] | 0,0180 |
| OLO | Pesos de clase | 0,5278 | [0,4891; 0,5601] | 0,0180 |
| TabNet | SMOTE-NC | 0,5212 | [0,4832; 0,5516] | 0,0179 |
| TabNet | Sin balanceo | 0,5178 | [0,4738; 0,5522] | 0,0207 |
| XGBoost | SMOTE-NC | 0,5150 | [0,4662; 0,5518] | 0,0218 |
| LightGBM | SMOTE-NC | 0,5051 | [0,4598; 0,5408] | 0,0208 |
| LightGBM | Sin balanceo | 0,5050 | [0,4584; 0,5390] | 0,0206 |
| CatBoost | Sin balanceo | 0,5049 | [0,4642; 0,5374] | 0,0197 |
| XGBoost | Sin balanceo | 0,5010 | [0,4583; 0,5337] | 0,0193 |
| OLO | Sin balanceo | 0,4578 | [0,4162; 0,4876] | 0,0185 |

**Pie:** «Bootstrap de 1.000 repeticiones sobre 32 clústeres país-año del
conjunto de prueba.»

## 🆕 4.12 Nueva Tabla 5.7c — comparación pareada contra la configuración principal

**Insertar inmediatamente después.** Fuente:
`results/tables/bootstrap_pareado_vs_principal.csv`.

| Comparado | Δκw | IC 95 % de Δ | P(Δ>0) | Conclusión |
|---|---|---|---|---|
| OLO [sin balanceo] | +0,0807 | [+0,0642; +0,0956] | 1,000 | distinguible |
| XGBoost [sin balanceo] | +0,0375 | [+0,0278; +0,0482] | 1,000 | distinguible |
| CatBoost [sin balanceo] | +0,0336 | [+0,0238; +0,0426] | 1,000 | distinguible |
| LightGBM [sin balanceo] | +0,0335 | [+0,0253; +0,0422] | 1,000 | distinguible |
| LightGBM [SMOTE-NC] | +0,0334 | [+0,0192; +0,0469] | 1,000 | distinguible |
| XGBoost [SMOTE-NC] | +0,0236 | [+0,0181; +0,0294] | 1,000 | distinguible |
| TabNet [sin balanceo] | +0,0208 | [+0,0152; +0,0263] | 1,000 | distinguible |
| TabNet [SMOTE-NC] | +0,0174 | [+0,0040; +0,0303] | 0,997 | distinguible |
| **OLO [pesos de clase]** | **+0,0108** | **[−0,0042; +0,0240]** | 0,912 | **no distinguible** |
| **OLO [SMOTE-NC]** | **+0,0077** | **[−0,0074; +0,0212]** | 0,832 | **no distinguible** |
| **LightGBM [pesos de clase]** | **+0,0066** | **[−0,0056; +0,0173]** | 0,865 | **no distinguible** |
| **XGBoost [pesos de clase]** | **+0,0046** | **[−0,0064; +0,0146]** | 0,787 | **no distinguible** |
| **CatBoost [pesos de clase]** | **+0,0013** | **[−0,0129; +0,0166]** | 0,526 | **no distinguible** |
| **TabNet [pesos de clase]** | **+0,0011** | **[−0,0095; +0,0106]** | 0,591 | **no distinguible** |

**Texto de acompañamiento** (es el argumento central frente al revisor):

> Para evitar atribuir superioridad a diferencias puramente descriptivas, las
> comparaciones se acompañan de intervalos de confianza obtenidos por remuestreo
> de clústeres. Los registros de un mismo país-año comparten los indicadores de
> V-Dem y provienen de la misma muestra nacional, de modo que no son
> independientes; remuestrear registros individuales subestimaría el error
> estándar, por lo que se remuestrean los 32 clústeres país-año completos con
> reemplazo (Cameron y Miller, 2015; Efron y Tibshirani, 1993). La comparación
> entre pares es pareada: en cada réplica se evalúan ambas configuraciones sobre
> exactamente las mismas observaciones, lo que elimina la variabilidad común del
> remuestreo.
>
> El resultado matiza la lectura de la Tabla 5.5 en una dirección precisa. La
> ventaja de la configuración principal sobre las mejores configuraciones de los
> otros cuatro modelos oscila entre 0,0011 y 0,0108 puntos de kappa, y en los
> cinco casos el intervalo **incluye el cero**. Lo que sí resulta distinguible es
> la ventaja frente a la ausencia de balanceo: entre +0,0208 y +0,0807 puntos, con
> intervalos que excluyen el cero en las cinco familias. La conclusión defendible
> es por tanto que el tratamiento del desbalance produce una mejora detectable
> del acuerdo ordinal, mientras que la elección entre familias de modelos, dado
> el tamaño efectivo de muestra, no lo hace.
>
> Este resultado también acota el alcance de H1 y H2 (véase la discusión en los
> apartados correspondientes): con 32 clústeres país-año el diseño no tiene
> potencia para distinguir diferencias de una centésima de kappa entre modelos.

## 🆕 4.13 Nueva §5.3.3 — métricas por categoría y matriz de confusión

**Insertar como subsección nueva.** Fuentes:
`metricas_por_clase_CatBoost_smotenc_ordinal_4clases_test.csv`,
`matriz_confusion_CatBoost_smotenc_ordinal_4clases_test.csv` y su versión en
porcentajes; figuras `03_metricas_por_clase.png` y
`03_confusion_modelo_principal.png`.

Precision, recall y F1 por categoría (ver tabla en 0.5).

> ⚠️ **No comparar con la estrategia de pesos de clase.** La corrida solo genera
> `metricas_por_clase_*` para la configuración principal (CatBoost con SMOTE-NC);
> no existe el archivo equivalente para CatBoost con pesos de clase. Cualquier
> frase del tipo «a diferencia de lo que produciría una ponderación explícita de
> clases» sería una conjetura sin respaldo en los artefactos. Si se quiere hacer
> esa comparación, hay que generar el desglose por clase de la configuración con
> pesos y citarlo.

Matriz de confusión, porcentaje por clase real:

| Real \ Predicho | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 — Para nada satisfecho | **35,0** | 30,1 | 29,1 | 5,8 |
| 1 — No muy satisfecho | 16,0 | **37,3** | 41,4 | 5,2 |
| 2 — Más bien satisfecho | 2,8 | 14,2 | **63,3** | 19,7 |
| 3 — Muy satisfecho | 1,1 | 3,9 | 43,5 | **51,5** |

Conteos absolutos: 1389/1195/1154/231 · 1190/2765/3070/389 ·
418/2087/9331/2910 · 97/352/3895/4611.

Distribución de la distancia ordinal del error: 0 → 18.096 (51,58 %);
1 → 14.347 (40,89 %); 2 → 2.313 (6,59 %); 3 → 328 (0,93 %). Errores con
distancia ≥ 2: **2.641 (7,53 %)**.

**Texto:**

> Las métricas agregadas resumen el rendimiento en una sola cifra y ocultan el
> comportamiento diferencial entre categorías. El desglose muestra tres patrones.
>
> Primero, el rendimiento se ordena con la frecuencia de la clase: la categoría
> mayoritaria «Más bien satisfecho» es la mejor recuperada (F1 = 0,5796;
> recall = 63,3 %) y las dos categorías de insatisfacción son las más débiles
> (F1 = 0,3933 y 0,4003), con una brecha de 0,1863 puntos entre la mejor y la
> peor. El sobremuestreo sintético equilibra el conjunto de entrenamiento, pero
> el modelo resultante sigue prediciendo con mayor acierto las categorías que
> dominan la distribución real del conjunto de prueba.
>
> Segundo, la precisión y la exhaustividad se comportan de forma asimétrica entre
> los extremos. En «Para nada satisfecho», la precisión (0,4489) supera al recall
> (0,3500): el modelo es relativamente conservador al asignar esa categoría y
> deja sin recuperar dos tercios de los casos. En «Muy satisfecho» ocurre lo
> contrario en menor medida (precisión 0,5664 frente a recall 0,5149).
>
> Tercero, los errores se concentran entre categorías contiguas. El 51,58 % de
> las predicciones es exacto y el 40,89 % se desvía una sola categoría, de modo
> que el 92,47 % cae en la categoría real o en una adyacente. Los errores con
> distancia ordinal mayor o igual a dos son 2.641 (7,53 %) y los de distancia
> máxima, 328 (0,93 %). El desplazamiento dominante es hacia el centro de la
> escala: el 29,1 % de los insatisfechos extremos se clasifica como «Más bien
> satisfecho» y el 43,5 % de los muy satisfechos, también.
>
> Este desglose acota el alcance de las conclusiones: el modelo separa
> razonablemente satisfacción de insatisfacción —lo que la variante binaria
> confirma con un F1 macro de 0,7381— pero distingue con dificultad los grados
> dentro de cada polo.

**Y añadir a §6.2** como limitación:

> Séptima, el rendimiento no es homogéneo entre categorías: el F1 de las dos
> clases de insatisfacción (0,3933 y 0,4003) es sustancialmente inferior al de la
> categoría mayoritaria (0,5796). Las conclusiones sobre los determinantes de la
> insatisfacción deben leerse teniendo en cuenta que esas son, precisamente, las
> categorías que el modelo recupera peor.

## 🔴 4.14 Tabla 5.8 — experimento E2 (reemplazar completa)

`results/resultados_modelos.csv`, variante binaria, estrategia `pesos_clase`
(la mejor en promedio en E1):

| Modelo | Ordinal de 4 clases (κw) | Binaria (F1 macro) | Binaria (accuracy) | Binaria (AUROC) |
|---|---|---|---|---|
| OLO | 0,5308 | 0,7278 | 0,7483 | 0,8199 |
| XGBoost | 0,5340 | 0,7374 | 0,7624 | 0,8252 |
| CatBoost | 0,5386 | **0,7381** | **0,7644** | 0,8246 |
| LightGBM | 0,5320 | 0,7370 | 0,7631 | 0,8253 |
| TabNet | 0,5374 | **0,7385** | 0,7627 | **0,8273** |
| Rosa et al. (2023), referencia | — | 0,84–0,85 | 0,74–0,76 | 0,74–0,77 |

> 🔴 **La fila de Rosa et al. no procede de la corrida y hay que comprobarla en
> la fuente original antes de defenderla.** Los tres intervalos (F1 0,84–0,85;
> exactitud 0,74–0,76; AUROC 0,74–0,77) se han transcrito del PDF actual, no de
> ningún archivo de `results/`. Y presentan una anomalía que un tribunal puede
> señalar: un F1 de 0,84 con una exactitud de 0,75 es una combinación inusual en
> clasificación binaria, y sugiere que la cifra citada puede ser un F1 de la clase
> positiva y no un F1 macro, o proceder de una partición distinta. Antes de
> mantener la comparación hay que abrir el artículo y verificar (a) qué variante
> de F1 reporta, (b) sobre qué conjunto, y (c) si su target binario es
> equivalente al de aquí. Si no se puede verificar, la salida limpia es citar solo
> el orden de magnitud y declarar que la comparación no es estricta porque el
> diseño de evaluación difiere.

**Nota importante para el pie:** la columna ordinal recoge el kappa de la mejor
estrategia de cada modelo (Tabla 5.5), mientras que la columna binaria se
entrenó con la estrategia de pesos de clase, que es la de mayor kappa promedio en
validación en E1. Conviene declararlo para que la comparación entre columnas no
se lea como *ceteris paribus*.

**Texto:**

> En la variante binaria, el conjunto de prueba contiene un 32,2 % de personas
> insatisfechas y un 67,8 % de satisfechas, con una razón de 1,7 en el conjunto
> de entrenamiento (138.058 frente a 240.534 registros). TabNet obtiene el mayor
> F1 macro (0,7385) y el AUROC más alto (0,8273), seguido de cerca por CatBoost
> (0,7381 y 0,8246); las cinco configuraciones quedan en un rango de 0,0107
> puntos de F1. Los valores permanecen por debajo del intervalo de 0,84–0,85 que
> Rosa et al. reportan para Brasil, lo que es consistente con la mayor
> heterogeneidad regional y temporal del presente diseño: aquel trabajo evalúa
> una sola ola de un solo país, mientras que este predice 16 países en olas
> posteriores a las de entrenamiento. Nótese, en cambio, que la exactitud
> (0,7483–0,7644) y el AUROC (0,8199–0,8273) obtenidos aquí superan los de esa
> referencia (0,74–0,76 y 0,74–0,77), de modo que la comparación no es
> desfavorable en su conjunto.

## 🔴 4.15 Tabla 5.9 y §5.4 — análisis por subregión (reemplazar completa)

Dos cambios: las cifras son nuevas y **la asignación de subregiones cambió** —
Perú pasa de Cono Sur a Región Andina (`utils/config.py`). Fuente:
`results/tables/mae_subregiones.csv`.

| Subregión | Países | n | MAE ordinal | κw | Accuracy | Mejor país | Peor país |
|---|---|---|---|---|---|---|---|
| Región Andina | Bolivia, Colombia, Ecuador, **Perú** | 9.389 | **0,4908** | 0,4519 | **0,5697** | Perú | Colombia |
| Brasil | Brasil | 2.331 | 0,5457 | 0,5364 | 0,5148 | Brasil | Brasil |
| Cono Sur | Argentina, Chile, Uruguay, Paraguay | 9.403 | 0,5788 | 0,5042 | 0,4973 | Chile | Argentina |
| México y Caribe | México, República Dominicana | 4.306 | 0,5945 | 0,5217 | 0,5065 | México | Rep. Dominicana |
| Centroamérica | Costa Rica, El Salvador, Guatemala, Honduras, Panamá | 9.655 | **0,6292** | **0,5480** | 0,4858 | El Salvador | Costa Rica |

**Todas las cifras por país de esta sección deben leerse de la configuración
principal.** El error más fácil de cometer aquí es mezclar configuraciones: en
`mae_por_pais_test.csv` hay 240 filas (5 modelos × 3 estrategias × 16 países) y
las cifras cambian mucho entre ellas. Dos ejemplos de valores que **no**
pertenecen a la configuración principal y que no deben citarse como si lo
fueran: el MAE mínimo de Perú es 0,4304 pero con CatBoost **sin balanceo** (con
SMOTE-NC es 0,4469), y el único valor de toda la tabla que supera 1,0 es Costa
Rica con **pesos de clase** (1,0010; con SMOTE-NC es 0,6627). Bajo la
configuración principal, ningún país llega a 0,67.

**Texto — y aquí hay un hallazgo nuevo que conviene destacar:**

> La Región Andina obtiene el menor MAE ordinal (0,4908) y Centroamérica el más
> elevado (0,6292), una diferencia de 0,1384 puntos que indica que la capacidad
> predictiva no es geográficamente uniforme. El ordenamiento por kappa
> cuadrático, sin embargo, **se invierte en los extremos**: Centroamérica alcanza
> el valor más alto (0,5480) y la Región Andina el más bajo (0,4519). Las dos
> métricas discrepan por tanto sobre qué subregión predice mejor: la correlación
> de Spearman entre MAE y kappa a través de las cinco subregiones es de +0,70, de
> signo contrario al que tendría si ambas ordenaran la calidad del mismo modo.
> La causa está en la composición del target: el índice de concentración de
> Herfindahl de la distribución por subregión ordena el MAE de forma perfecta
> (ρ = −1,00 sobre las cinco subregiones), es decir, donde la distribución está
> más concentrada el error medio es menor, pero una porción mayor de ese acuerdo
> es atribuible al azar y el kappa la descuenta (ρ = −0,70 entre concentración y
> kappa). La Región Andina concentra el 48,6 % de sus casos en la categoría modal
> y Centroamérica el 39,0 %. La lectura conjunta es que Centroamérica es la
> subregión donde el modelo aporta más información por encima del azar, aunque
> sus errores caigan en promedio más lejos de la categoría real.
>
> A escala de país y bajo la configuración principal, el mejor desempeño
> corresponde a Perú (MAE = 0,4469) y el peor a Argentina (0,6678), seguida de
> cerca por República Dominicana (0,6677) y Costa Rica (0,6627). República
> Dominicana y Guatemala son además los países que aparecen con más frecuencia
> entre los más afectados por errores graves (véase la Tabla 5.12).

**Además:** el PDF lista hoy a Perú en Cono Sur y deja la Región Andina con tres
países. Hay que corregir la asignación en la Tabla 5.9, en §5.4 y en cualquier
otra mención (la muestra estratificada de LIME y el contraste de H4/H5 del NB05
usan la misma agrupación).

## 🔴 4.16 Figura 5.2 y §5.5.1 — importancia por bloque

Fuente: `results/tables/shap_bloques_ic_CatBoost_smotenc.csv`. Figura
regenerada: `results/figures/04_shap_bloques_CatBoost_smotenc.png`.

| # | Bloque | Variables | \|SHAP\| total | IC 95 % |
|---|---|---|---|---|
| 1 | Percepción política | 4 | 0,4818 | [0,4759; 0,4874] |
| 2 | Evaluación económica | 5 | 0,3853 | [0,3754; 0,3944] |
| 3 | Confianza institucional | 7 | 0,2601 | [0,2558; 0,2644] |
| 4 | Contexto democrático | 4 | 0,1267 | [0,1015; 0,1562] |
| 5 | Corrupción y seguridad | 3 | 0,0938 | [0,0920; 0,0956] |
| 6 | Características sociodemográficas | 5 | 0,0034 | [0,0034; 0,0034] |

**Qué hacer:** actualizar los tres valores que cita el texto (0,6290 → 0,4818;
0,5613 → 0,3853; 0,4697 → 0,2601) y **añadir la columna de intervalos**, con este
párrafo:

> A diferencia del ranking por variable, la jerarquía por bloque es completamente
> estable: ningún par de bloques consecutivos presenta intervalos solapados. El
> orden —percepción política, evaluación económica, confianza institucional,
> contexto democrático, corrupción y seguridad, características
> sociodemográficas— es por tanto robusto al remuestreo de clústeres, mientras
> que el orden interno entre variables próximas no siempre lo es. Esto respalda
> la decisión metodológica de §4.11.1 de reportar la importancia en dos niveles y
> de basar la discusión sustantiva en la agregación por bloque. El único bloque
> con un intervalo apreciablemente ancho es el contexto democrático
> ([0,1015; 0,1562]), y la razón es estructural: sus cuatro indicadores toman un
> único valor por país-año, de modo que al remuestrear países su contribución
> varía mucho más que la de las variables individuales, cuya variación se
> promedia sobre miles de encuestados.

## 🔴 4.17 Tabla 5.10 y Figura 5.3 — ranking de variables (reemplazar completa)

Fuente: `results/tables/shap_importancia_ic_CatBoost_smotenc.csv`. La versión
ampliada con incertidumbre sustituye a la Tabla 5.10 actual.

| # | Variable | \|SHAP\| | IC 95 % | Rango (IC) | % top-5 |
|---|---|---|---|---|---|
| 1 | Apoyo a la democracia | 0,2136 | [0,2102; 0,2170] | 1 (1–1) | 100 |
| 2 | Confianza Gobierno | 0,1697 | [0,1661; 0,1733] | 2 (2–2) | 100 |
| 3 | Situación económica país | 0,1637 | [0,1602; 0,1673] | 3 (3–3) | 100 |
| 4 | Aprobación gobierno | 0,1436 | [0,1417; 0,1452] | 4 (4–5) | 100 |
| 5 | País para todos / poderosos | 0,1247 | [0,1225; 0,1267] | 5 (5–6) | 68,7 |
| 6 | Componente igualitario | 0,1177 | **[0,0930; 0,1469]** | 6 **(4–7)** | 29,8 |
| 7 | Distribución ingreso justa | 0,1174 | [0,1122; 0,1223] | 7 (6–7) | 1,5 |
| 8 | Economía país vs. año anterior | 0,0576 | [0,0539; 0,0615] | 8 (8–8) | 0 |
| 9 | Progreso contra corrupción | 0,0439 | [0,0427; 0,0450] | 9 (9–9) | 0 |
| 10 | Expectativa económica país | 0,0393 | [0,0371; 0,0415] | 10 (10–10) | 0 |
| 11 | Confianza Partidos Políticos | 0,0366 | [0,0363; 0,0368] | 11 (11–11) | 0 |
| 12 | Conoce caso de corrupción | 0,0331 | [0,0326; 0,0336] | 12 (12–12) | 0 |
| 13 | Confianza Congreso | 0,0291 | [0,0286; 0,0296] | 13 (13–13) | 0 |
| 14 | Confianza Poder Judicial | 0,0247 | [0,0243; 0,0251] | 14 (14–14) | 0 |
| 15 | Victimización delictiva | 0,0168 | [0,0158; 0,0178] | 15 (15–15) | 0 |

Posiciones 16 a 20: Expectativa económica personal 0,0073 · **Democracia
electoral 0,00695** · Sexo 0,0034 · **Igualdad ante la ley 0,0013** ·
**Integridad institucional (corrupción) 0,0008**.

**Hallazgo que el PDF no reporta y conviene incorporar.** Las ocho variables
restantes tienen |SHAP| **exactamente cero**, empatadas en la posición 21:
confianza en las Fuerzas Armadas, en la televisión y en la policía, interés en
política, nivel socioeconómico, situación ocupacional, nivel educativo y edad.
Es decir, **el modelo principal utiliza efectivamente 20 de las 28 variables**;
CatBoost nunca divide sobre las otras ocho. El dato es coherente con los
hiperparámetros seleccionados —300 iteraciones, profundidad 4 y regularización
L2 de 9,27, un ensamble deliberadamente pequeño— y tiene dos consecuencias que
merecen una frase cada una:

1. La ausencia del bloque sociodemográfico del ranking no es un efecto de escala:
   cuatro de sus cinco variables tienen contribución nula y la quinta (sexo)
   0,0034. Esto refuerza, con más fuerza que el orden de bloques, la conclusión
   de que la satisfacción con la democracia se explica por percepciones y no por
   posición social en esta especificación.
2. Da contenido empírico a la limitación sobre selección de variables: el modelo
   descarta por sí mismo ocho de las veintiocho, incluidas tres de las siete
   medidas de confianza institucional.

**Cambio de fondo respecto del PDF actual.** En la tabla vigente, los cuatro
indicadores de V-Dem ocupan las posiciones 7, 12 y 15 con importancias entre
0,056 y 0,128. Ahora **solo el componente igualitario mantiene una contribución
apreciable (0,1177)**; los otros tres caen a las posiciones 17, 19 y 20 con
valores de 0,0070, 0,0013 y 0,0008 — dos órdenes de magnitud por debajo. Con
SMOTE-NC el modelo prácticamente prescinde de tres de los cuatro indicadores
contextuales.

**Texto de acompañamiento (responde a la observación del revisor sobre la
identificabilidad del ranking):**

> El ranking de importancia es una estimación puntual y su orden no es igualmente
> identificable en todos los tramos. Para cuantificar esa incertidumbre se
> remuestrean con reemplazo los 32 clústeres país-año del conjunto de prueba
> (1.000 réplicas) y se recalculan tanto el valor |SHAP| medio como el rango de
> cada variable.
>
> El resultado es notablemente estable: la amplitud media del intervalo de rango
> en las veinte variables principales es de 0,4 posiciones y solo cuatro pares
> consecutivos presentan intervalos de importancia solapados (confianza en el
> Gobierno con situación económica del país; país para todos con componente
> igualitario; componente igualitario con distribución del ingreso justa; y
> expectativa económica personal con democracia electoral). Las **tres** primeras
> posiciones son idénticas en todas las réplicas (intervalo de rango 1–1, 2–2 y
> 3–3), y las cinco primeras variables permanecen dentro del top-5 en el 100 % de
> ellas, aunque la cuarta y la quinta intercambian posición entre sí.
>
> La única variable con incertidumbre sustantiva es el componente igualitario,
> cuyo intervalo de rango abarca de la 4.ª a la 7.ª posición y que permanece en el
> top-5 en el 29,8 % de las réplicas. La razón es estructural: los índices de
> V-Dem toman un único valor por país-año, de manera que al remuestrear países su
> contribución varía mucho más que la de las variables individuales. En
> consecuencia, la afirmación de que el componente igualitario es el indicador
> contextual de mayor importancia se sostiene —su intervalo no se solapa con el
> de los otros tres índices de V-Dem, que quedan dos órdenes de magnitud por
> debajo—, pero su posición exacta dentro del ranking global no debe
> interpretarse como un ordinal preciso.

## 🔴 4.18 Tabla 5.11 — análisis micro/macro (reemplazar completa)

| Pos. micro | Variable LB | \|SHAP\| | Pos. macro | Índice V-Dem | \|SHAP\| |
|---|---|---|---|---|---|
| 1 | A_001_001 | 0,2136 | 1 | v2x_egal | 0,1177 |
| 2 | H_002_031 | 0,1697 | 2 | v2x_polyarchy | 0,00695 |
| 3 | D_001_001 | 0,1637 | 3 | v2xcl_rol | 0,0013 |
| 4 | B_006_061 | 0,1436 | 4 | v2x_corr | 0,0008 |
| 5 | B_001_101 | 0,1247 | — | — | — |

**Cambia el orden del nivel macro:** hoy el PDF lista v2x_egal, v2xcl_rol,
v2x_corr, v2x_polyarchy; ahora es v2x_egal, v2x_polyarchy, v2xcl_rol, v2x_corr.

**Texto:**

> El apoyo a la democracia y la confianza en el Gobierno dominan el nivel
> individual, con importancias de 0,2136 y 0,1697. En el nivel contextual, el
> componente igualitario alcanza 0,1177, un valor comparable al de las variables
> individuales mejor posicionadas y **casi diecisiete veces superior** (16,9) al del segundo
> indicador contextual (democracia electoral, 0,00695). La igualdad ante la ley y
> la integridad institucional quedan por debajo de 0,0015. La comparación
> confirma que el nivel macro aporta señal propia, pero que esa señal está
> concentrada casi por completo en una sola dimensión —la igualitaria— y no
> distribuida entre los cuatro indicadores.

**Consecuencia para la discusión de §2.7 y §6.3:** el argumento de que los
índices de V-Dem «compiten artificialmente» por estar correlacionados se refuerza
con este resultado: el modelo elige uno y descarta los otros tres.

## 🔴 4.19 §5.5.2 y Figuras 5.5–5.10 — curvas ALE

Reemplazar las seis figuras por las cinco nuevas (ver 3.10) y reescribir el
análisis. Las curvas nuevas son, por bloque:

| Bloque | Variable | Figura |
|---|---|---|
| Confianza institucional | Confianza en el Gobierno | `04_ale_H_002_031.png` |
| Confianza institucional | Confianza en los partidos políticos | `04_ale_H_002_241.png` |
| Evaluación económica | Situación económica del país | `04_ale_D_001_001.png` |
| Evaluación económica | Distribución del ingreso justa | `04_ale_C_006_003_011.png` |
| Percepción política | Apoyo a la democracia | `04_ale_A_001_001.png` |

Las curvas se calcularon con ALE nativo de `alibi` (el notebook confirma
`✓ alibi disponible para ALE`), no con el respaldo de PDP. Al redactar, leer de
cada figura el nivel marcado con la línea vertical punteada, que identifica el
intervalo de mayor cambio local.

**Aporte que conviene destacar:** ahora las curvas cubren los tres bloques
sustantivos, de modo que §5.5.2 puede comparar la forma funcional **entre**
mecanismos (institucional, económico y político) y no solo dentro del bloque de
confianza, que era la limitación de la versión anterior.

## 🔴 4.20 Tabla 5.12 — errores graves (reemplazar completa)

Fuente: `results/tables/errores_graves_CatBoost_smotenc.csv`. La tabla actual
tiene dos filas (solo distancia 3); la nueva cubre las seis combinaciones con
distancia ≥ 2.

| Predicho | Real | Distancia | Tipo | Casos | % prueba | Países más frecuentes |
|---|---|---|---|---|---|---|
| 2 | 0 | 2 | Sobreestimación | **1.154** | 3,29 | Guatemala (130), Rep. Dominicana (125) |
| 0 | 2 | 2 | Subestimación | 418 | 1,19 | Brasil (67), Rep. Dominicana (54) |
| 3 | 1 | 2 | Sobreestimación | 389 | 1,11 | Argentina (111), Chile (32) |
| 1 | 3 | 2 | Subestimación | 352 | 1,00 | Costa Rica (45), México (35) |
| 3 | 0 | 3 | Sobreestimación | 231 | 0,66 | Honduras (38), Guatemala (32) |
| 0 | 3 | 3 | Subestimación | 97 | 0,28 | Rep. Dominicana (12), Panamá (12) |
| **Total ≥ 2** | | | | **2.641** | **7,53** | |

> ⚠️ La columna «Países más frecuentes» recoge **los dos primeros de cada fila**,
> no un total por país. No se puede afirmar con ella que un país concentre los
> errores graves en conjunto; para eso habría que agregar por país, lo que la
> corrida no produce. La formulación defendible es la del texto: qué países
> reaparecen en más tipos de error.

**Pie:** «Variable dominante en las explicaciones LIME del grupo de error
máximo: confianza en el Gobierno (|peso| medio 0,0461), seguida por distribución
del ingreso justa (0,0430) y situación económica del país (0,0414).»

**Texto:**

> Los errores con distancia ordinal mayor o igual a dos son 2.641, el 7,53 % del
> conjunto de prueba, y los de distancia máxima 328, el 0,93 %. La asimetría es
> clara: la sobreestimación (1.774 casos) predomina sobre la subestimación (867),
> y el caso más frecuente con diferencia es clasificar como «Más bien satisfecho»
> a quien declara no estar satisfecho en absoluto (1.154 casos, 3,29 %). Estos
> errores afectan con mayor frecuencia a República Dominicana, que figura entre
> los dos países más frecuentes en tres de los seis tipos de error grave, y a
> Guatemala, que lo hace en dos.
>
> Las explicaciones LIME del grupo de error máximo señalan la confianza en el
> Gobierno como la variable de mayor peso (|peso| medio 0,0461), seguida por la
> percepción de justicia en la distribución del ingreso (0,0430) y la situación
> económica del país (0,0414). Las tres primeras variables son **las mismas y en
> el mismo orden** en los tres grupos de casos explicados, con magnitudes menores
> en los representativos (0,0306; 0,0285; 0,0256) y en los discordantes (0,0243;
> 0,0243; 0,0225), de modo que no se trata de un patrón exclusivo de los errores:
> la confianza en el Gobierno es la señal que domina las aproximaciones locales
> del modelo en los tres grupos.

## 🆕 4.21 Nueva subsección — concordancia entre modelos indistinguibles

Fuentes: `results/tables/shap_concordancia_modelos.csv`,
`shap_top5_por_modelo.csv`; figura `04_shap_concordancia_modelos.png`.

ρ de Spearman entre rankings (28 variables):

| | CatBoost | XGBoost | LightGBM |
|---|---|---|---|
| CatBoost | 1,0000 | 0,9625 | 0,8614 |
| XGBoost | 0,9625 | 1,0000 | 0,9243 |
| LightGBM | 0,8614 | 0,9243 | 1,0000 |

W de Kendall = **0,9367** (χ² = 75,87; p = 2,0 × 10⁻⁶; n = 28). Verificado en
la salida del NB04, celda 14.

> ⚠️ **Cautela que conviene declarar en el pie.** El ranking de CatBoost contiene
> ocho variables empatadas en importancia exactamente cero (véase 4.17). Un
> coeficiente de concordancia calculado sobre rangos con empates masivos en la
> cola tiende a ser más alto que el que se obtendría solo entre las variables que
> los modelos sí usan. La conclusión cualitativa —los tres rankings concuerdan en
> lo sustantivo— no depende de eso, pero el valor 0,9367 no debe presentarse como
> una medida de acuerdo en el tramo informativo del ranking. Si se quiere una
> cifra sin ese sesgo, hay que recalcular W sobre las 20 variables con
> importancia no nula; la corrida actual no lo hace.

Top-5 por modelo:

| # | CatBoost | XGBoost | LightGBM |
|---|---|---|---|
| 1 | Apoyo a la democracia | Confianza Gobierno | Apoyo a la democracia |
| 2 | Confianza Gobierno | Apoyo a la democracia | Confianza Gobierno |
| 3 | Situación económica país | Aprobación gobierno | País para todos / poderosos |
| 4 | Aprobación gobierno | País para todos / poderosos | Aprobación gobierno |
| 5 | País para todos / poderosos | Distribución ingreso justa | Distribución ingreso justa |

**Texto:**

> Como las mejores configuraciones de los tres modelos de *gradient boosting*
> presentan diferencias de kappa que el bootstrap pareado no distingue del ruido
> (Tabla 5.7c), cabe preguntarse si el ranking de importancia describe los datos
> o la elección de algoritmo. La concordancia es alta pero no perfecta: la W de
> Kendall entre los tres rankings es 0,9367, y las correlaciones de Spearman por
> pares van de 0,8614 (CatBoost–LightGBM) a 0,9625 (CatBoost–XGBoost).
>
> Cuatro de las cinco variables principales son comunes a los tres modelos
> —apoyo a la democracia, confianza en el Gobierno, aprobación del Gobierno y
> país para todos o para los poderosos—, pero la quinta difiere: CatBoost sitúa
> ahí la situación económica del país, mientras que XGBoost y LightGBM colocan la
> percepción de justicia en la distribución del ingreso. El orden interno también
> varía: XGBoost invierte las dos primeras posiciones respecto de los otros dos.
> La lectura sustantiva —el predominio de la percepción política y de la
> evaluación económica sobre la confianza institucional— no depende por tanto del
> algoritmo elegido; la identidad y el orden precisos del top-5, sí.

## 🆕 4.22 Nueva §5.3.4 — estabilidad temporal del rendimiento

Fuentes: `results/tables/validacion_temporal_resumen.csv` y
`validacion_temporal_folds.csv`. **Es material completamente nuevo** y responde a
la observación del revisor sobre la validación temporal.

Pliegues (ventana de entrenamiento expansiva, origen fijo en 1995):

| Pliegue | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| 1 | 1995–2007 (12 olas) | 2008 | 2009–2010 |
| 2 | 1995–2010 (15 olas) | 2011 | 2013–2015 |
| 3 | 1995–2015 (18 olas) | 2016 | 2017–2018 |
| Definitivo | 1995–2018 (21 olas) | 2020 | 2023–2024 |

Kappa cuadrático en prueba, media ± desviación estándar entre los cuatro cortes:

| Modelo | κw medio ± sd | κw mín–máx | MAE ordinal medio ± sd |
|---|---|---|---|
| XGBoost | 0,4860 ± **0,0320** | 0,4677–0,5340 | 0,6587 ± 0,0360 |
| CatBoost | 0,4842 ± 0,0381 | 0,4467–0,5373 | 0,6620 ± 0,0402 |
| LightGBM | 0,4824 ± 0,0331 | 0,4622–0,5320 | 0,6635 ± 0,0347 |
| TabNet | 0,4574 ± 0,0763 | 0,3535–0,5374 | 0,7131 ± 0,0727 |
| OLO | 0,4505 ± **0,0821** | 0,3343–0,5278 | 0,8380 ± 0,1109 |

**Texto — y este es el hallazgo que rescata H1 por otra vía:**

> El diseño principal descansa en un único corte temporal, de modo que la métrica
> reportada no lleva asociada una medida de su variabilidad entre períodos. Para
> acotarla se replicó el mismo esquema en tres pliegues históricos con ventana de
> entrenamiento expansiva, reajustando en cada uno la imputación, el escalado,
> los pesos de clase y los hiperparámetros exclusivamente con su propia ventana
> de entrenamiento y aplicando las mismas reglas de exclusión de países.
>
> El resultado introduce una distinción que el corte único no permite ver. En el
> conjunto de prueba definitivo, las mejores configuraciones de los cinco modelos
> quedan dentro de un rango de 0,0078 puntos de kappa y el bootstrap pareado no
> las distingue entre sí. Entre períodos, en cambio, la diferencia es marcada:
> los tres modelos de *gradient boosting* presentan desviaciones estándar de
> 0,0320 a 0,0381, mientras que TabNet y la regresión logística ordinal duplican
> esa dispersión (0,0763 y 0,0821). El caso extremo es la línea base ordinal en el
> primer pliegue, donde su kappa cae a 0,3343 frente a 0,4778 de CatBoost: una
> brecha de 0,1435 puntos. En MAE ordinal la brecha es aún mayor
> (0,9957 frente a 0,6162).
>
> La lectura conjunta es que la ventaja de los modelos de *gradient boosting* no
> está en el nivel de rendimiento alcanzado en un corte favorable, donde resultan
> indistinguibles de las alternativas, sino en su **estabilidad**: mantienen un
> desempeño comparable con ventanas de entrenamiento de 12, 15, 18 y 21 olas,
> mientras que la línea base ordinal y la red tabular dependen mucho más de la
> cantidad de historia disponible. Para un uso aplicado —predecir la ola
> siguiente con la información acumulada— esa propiedad es más relevante que una
> diferencia de una centésima en el corte final.

**Alcance que hay que declarar explícitamente**, porque limita la lectura:

- Los pliegues se corren con una sola estrategia de balanceo, **pesos de clase**,
  no con las tres. En consecuencia la fila «Definitivo» de esta tabla corresponde
  a la configuración con pesos de clase de cada modelo y **no** a la
  configuración principal del trabajo (CatBoost con SMOTE-NC). El objetivo es
  medir estabilidad temporal, no repetir el experimento E1.
- El presupuesto de Optuna en los pliegues es de 15 ensayos frente a 50 en el
  corte definitivo, así que sus valores absolutos son ligeramente pesimistas. **Lo
  interpretable es la dispersión, no el nivel.**

**Objeción previsible, y la respuesta la refuerza.** La desviación estándar de la
tabla se calcula sobre los cuatro cortes, y uno de ellos —el definitivo— se
ajustó con 50 ensayos y los otros tres con 15. Un revisor puede señalar que esa
mezcla contamina la medida de dispersión. Conviene adelantarse, porque al
restringirla a los tres pliegues con presupuesto homogéneo la separación **se
agranda**:

| Modelo | sd sobre los 4 cortes | sd sobre los 3 pliegues (15 ensayos) | κw medio de los 3 |
|---|---|---|---|
| XGBoost | 0,0320 | **0,0022** | 0,4701 |
| LightGBM | 0,0331 | **0,0032** | 0,4659 |
| CatBoost | 0,0381 | **0,0172** | 0,4665 |
| TabNet | 0,0763 | **0,0668** | 0,4307 |
| OLO | 0,0821 | **0,0783** | 0,4248 |

Con presupuesto homogéneo, la mayor dispersión de un árbol (0,0172) es cuatro
veces menor que la menor de los otros dos modelos (0,0668): los dos grupos no se
solapan. **Recomendación:** reportar la columna de los tres pliegues como medida
principal de estabilidad y la de los cuatro cortes como referencia, declarando la
diferencia de presupuesto. Es más honesto y el argumento sale más fuerte.

- Los modelos de los pliegues no sustituyen a los reportados: sus
  hiperparámetros se registran con el sufijo `_foldN` en `models/` y no se
  persisten pipelines.

## 🔴 4.23 Tabla 5.13 y §5.6 — contraste con la teoría democrática

Fuentes: `results/tables/contraste_teorico_CatBoost.csv` y
`tabla_convergencias_CatBoost.csv`.

Porcentaje de las diez variables principales que pertenece al bloque que cada
teoría predice como dominante:

| Marco teórico | Bloque predicho | Variables en el top-10 | % convergencia |
|---|---|---|---|
| Lewis-Beck & Stegmaier (2000) | Evaluación económica | 4 de 10 | **40 %** |
| Norris (2011) | Percepción política | 3 de 10 | 30 % |
| Easton (1975) | Confianza institucional | 1 de 10 | 10 % |
| Devine (2024) | Confianza institucional | 1 de 10 | 10 % |

Clasificación variable a variable en el top-10 (columna `convergencia` de
`tabla_convergencias_CatBoost.csv`): 4 convergencias (apoyo a la democracia,
confianza en el Gobierno, aprobación del Gobierno, país para todos), 4 parciales
(las cuatro de evaluación económica) y 2 divergencias (componente igualitario y
progreso contra la corrupción).

> 🔴 **Aquí hay una trampa que hay que resolver antes de redactar.** La corrida
> produce **dos** operacionalizaciones distintas del contraste, y dan órdenes
> distintos. No son contradictorias —miden cosas diferentes— pero mezclarlas
> produce una tabla que se refuta a sí misma.
>
> | Teoría | `contraste_teorico` (un bloque por teoría, top-10) | `tabla_convergencias` (variable a variable, top-10) |
> |---|---|---|
> | Lewis-Beck y Stegmaier | 4 de 10 → 40 % | 7 convergen, 1 parcial, 2 divergen |
> | Norris | 3 de 10 → 30 % | 4 convergen, 1 parcial, 5 divergen |
> | Easton | 1 de 10 → 10 % | 5 convergen, 3 parciales, 2 divergen |
> | Devine | 1 de 10 → 10 % | 4 convergen, 5 parciales, 1 divergen |
>
> La primera métrica asigna **un solo bloque predicho** a cada teoría
> (`utils/config.py`), y por eso penaliza a los marcos cuya predicción abarca más
> de un bloque: a Easton, que cubre apoyo difuso y específico, y a Devine, que
> cubre confianza y percepción política. La segunda evalúa cada variable contra
> cada teoría por separado y no tiene ese sesgo.
>
> **Decisión recomendada:** construir la Tabla 5.13 sobre la clasificación
> variable a variable, y declarar en el pie que la métrica agregada por bloque
> —también disponible— asigna una sola dimensión a cada marco y por eso
> subestima los que son multidimensionales. Lo que **no** se puede hacer es lo
> que hace el borrador de la tabla nueva más abajo: dar a Devine un veredicto
> «Sí» apoyado en la lectura variable a variable y a Easton un «Parcial» apoyado
> en la métrica de bloque. Con un solo criterio, Easton (5 convergencias) queda
> por delante de Devine (4) y de Norris (4), y solo por detrás de Lewis-Beck (7).

**Esto altera la Tabla 5.13 del PDF**, que hoy declara convergencia completa con
Easton, Lewis-Beck y Devine, y parcial solo con Norris.

**Tabla 5.13 nueva.** Construida sobre **un solo criterio**: la clasificación
variable a variable de `tabla_convergencias_CatBoost.csv` aplicada a las diez
variables principales. Las tres columnas de recuento suman 10 en todas las filas,
de modo que la tabla es autoverificable.

| Marco teórico | Predicción dominante | Convergen | Parciales | Divergen | Convergencia |
|---|---|---|---|---|---|
| Lewis-Beck y Stegmaier (voto económico) | Evaluación económica | **7** | 1 | 2 | **Sí** |
| Easton (apoyo difuso y específico) | Confianza institucional | 5 | 3 | 2 | **Parcial** |
| Norris (ciudadanos críticos) | Percepción política y déficit democrático | 4 | 1 | 5 | **Parcial** |
| Devine (satisfacción expresiva) | Confianza y percepción política | 4 | 5 | 1 | **Parcial** |

**Pie obligatorio:** «Clasificación variable a variable de las diez variables de
mayor \|SHAP\|. La métrica agregada por bloque, disponible en
`contraste_teorico_CatBoost.csv`, asigna una única dimensión a cada marco y arroja
40 % (Lewis-Beck), 30 % (Norris) y 10 % (Easton y Devine); subestima por
construcción los marcos cuya predicción abarca más de un bloque.»

**Texto:**

> El contraste produce un resultado más matizado que el previsto, y su lectura
> depende del criterio de agregación. Evaluando cada una de las diez variables
> principales frente a cada marco, la convergencia más fuerte es con el enfoque
> de evaluación económica de Lewis-Beck y Stegmaier, con siete convergencias
> sobre diez; el bloque económico ocupa además la segunda posición en la
> agregación. Los tres marcos restantes convergen de forma parcial: Easton con
> cinco, y Norris y Devine con cuatro cada uno, si bien Devine acumula cinco
> clasificaciones parciales frente a las cinco divergencias de Norris.
>
> La convergencia con Easton es, en cambio, **parcial y no completa** como se
> anticipaba: la confianza institucional es el tercer bloque en importancia
> agregada, pero solo una de sus siete variables —la confianza en el Gobierno—
> entra en el top-10. De las seis restantes, los partidos políticos ocupan la
> posición 11, el Congreso la 13 y el Poder Judicial la 14, mientras que la
> confianza en la policía, en la televisión y en las Fuerzas Armadas tiene
> importancia nula: el modelo no las utiliza. El patrón es compatible con que lo
> que el modelo aprovecha no sea la confianza institucional como dimensión difusa
> sino la evaluación del **Ejecutivo en funciones**, que es el apoyo específico de
> la distinción de Easton y no el difuso.
>
> Con Norris la convergencia también es parcial, por una razón distinta: la
> percepción política es el bloque dominante, lo que la respalda, pero el
> contexto democrático —el «déficit» que su marco enfatiza— queda en cuarto lugar
> y, dentro de él, tres de los cuatro indicadores tienen importancias inferiores
> a 0,008. Solo el componente igualitario mantiene una contribución apreciable.
>
> La convergencia entre un patrón predictivo y una teoría indica compatibilidad
> descriptiva; no valida el mecanismo causal que la teoría postula.

## 🟡 4.24 §5.5.1 — resultados de H4 y H5 (NB05)

Estos resultados existen en la corrida y el PDF no los reporta de forma
cuantitativa.

**H5** (`results/tables/spearman_subregiones.csv`): ρ mínima entre pares de
subregiones = **0,9669**; media = **0,9899**. Todas las parejas superan con
holgura el umbral de 0,7. **H5 confirmada.**

**H4** (salida del NB05): rango de la importancia por bloque entre subregiones —
contexto democrático 0,0373 · evaluación económica 0,0083 · percepción política
0,0047 · confianza institucional 0,0025 · corrupción y seguridad 0,0019 ·
características sociodemográficas 0,0000.

**Qué hacer** — añadir a §5.4 o §5.5.1:

> La correlación de Spearman entre los rankings de importancia SHAP de las cinco
> subregiones tiene un mínimo de 0,9669 y una media de 0,9899, muy por encima del
> umbral de 0,7 fijado para H5: los determinantes identificados son robustos
> entre subregiones. H4 se respalda de forma parcial. El bloque cuya importancia
> varía más entre subregiones es el contexto democrático (rango 0,0373), no la
> confianza institucional ni la corrupción, que presentan rangos de 0,0025 y
> 0,0019; las características sociodemográficas no varían en absoluto
> (rango 0,0000). Es decir, la variación regional se concentra en el nivel
> contextual —lo esperable, porque sus indicadores difieren por país— y no en las
> percepciones individuales, que se comportan de forma notablemente homogénea.

## 🟡 4.25 §5.5.1 — nota sobre la atención de TabNet

`results/tables/tabnet_atencion_smotenc.csv` da un ranking de atención muy
distinto del SHAP de CatBoost: encabeza «Conoce caso de corrupción» (0,0984),
seguido de «Apoyo a la democracia» (0,0807) e «Igualdad ante la ley» (0,0685).
Si §5.5.1 menciona el análisis de atención de TabNet, conviene señalar la
discrepancia como evidencia de que las medidas de importancia internas de una
arquitectura no son intercambiables con las atribuciones post hoc.

---

# BLOQUE 5 — Capítulo 6: conclusiones y limitaciones

## 🔴 5.1 OE1 — enumerar en lugar de contar

**PDF actual:** «El proceso documentó 24 transformaciones de escala, seis
recodificaciones transversales…»

Esas cifras no son verificables contra el código. Lo auditable es: armonización
de tres variables económicas comparativas (`D_001_021`, `D_001_041`,
`D_001_091`), recodificación binaria de tres variables efectivas (`B_006_061`,
`B_001_101`, `S_001`), armonización longitudinal de la victimización
`I_001_001` sobre tres esquemas de codificación y el caso especial de
`G_002_011` en la ola 2013. **Sustituir los conteos por esa enumeración.**

## 🔴 5.2 OE2 — reescribir

**Qué poner:**

> En relación con el objetivo específico 2 (OE2), la configuración seleccionada
> en validación —CatBoost con SMOTE-NC— obtuvo el mayor kappa cuadrático del
> conjunto de prueba (κw = 0,5386) y el menor MAE ordinal entre las
> configuraciones seleccionadas (0,5688). Su ventaja sobre la regresión logística
> ordinal es de 0,0077 puntos de kappa, con un intervalo de confianza que incluye
> el cero, y de 0,1676 puntos de MAE ordinal, con un intervalo que lo excluye. En
> las métricas complementarias no hay dominancia: LightGBM alcanza el mejor F1
> macro (0,4798) y la mayor exactitud balanceada (0,5115), y XGBoost el AUROC OvR
> más alto (0,7688). Seis de las quince configuraciones evaluadas resultan
> estadísticamente indistinguibles de la principal, de modo que la superioridad
> de CatBoost se establece de manera descriptiva y no inferencial para el corte
> temporal de prueba disponible.

## 🔴 5.3 OE3 — reescribir

**Qué poner:**

> En relación con el objetivo específico 3 (OE3), el experimento E1 mostró que
> los pesos de clase producen el mayor kappa cuadrático para XGBoost, LightGBM y
> TabNet, mientras que SMOTE-NC lo hace para CatBoost y para la regresión
> logística ordinal; los cinco modelos conservan la misma mejor estrategia en
> validación y en prueba. El efecto del balanceo sobre el acuerdo ordinal es
> distinguible del ruido de muestreo en las cinco familias (entre +0,0208 y
> +0,0807 puntos frente a la ausencia de balanceo), a diferencia del efecto de la
> elección de familia. En la formulación binaria, TabNet obtuvo el mejor F1 macro
> (0,7385), seguido por CatBoost (0,7381), con las cinco configuraciones en un
> rango de 0,0107 puntos. El análisis por subregión mostró que la capacidad
> predictiva no es geográficamente uniforme y que los ordenamientos por MAE y por
> kappa discrepan entre subregiones.

> ⚠️ La comparación con Rosa et al. se ha retirado de este pasaje a propósito.
> Mantenerla en OE3 obliga a defenderla dos veces (aquí y en §5.3.2) y su
> verificación está pendiente (véase el aviso de 4.14). Si tras comprobar la
> fuente la comparación se sostiene, basta con dejarla en §5.3.2 y referenciarla
> desde aquí; si no se sostiene, no habrá que tocar el Capítulo 6.

## 🔴 5.4 OE4 — reescribir

**Qué poner:**

> En relación con el objetivo específico 4 (OE4), SHAP identificó la percepción
> política como el bloque de mayor importancia global (0,4818), seguida por la
> evaluación económica (0,3853) y la confianza institucional (0,2601); la
> jerarquía entre bloques es robusta al remuestreo, sin intervalos solapados
> entre bloques consecutivos. En el nivel de variable, las tres primeras
> posiciones son idénticas en el 100 % de las réplicas bootstrap y la amplitud
> media del intervalo de rango es de 0,4 posiciones. El modelo utiliza
> efectivamente veinte de las veintiocho variables: las ocho restantes —tres
> medidas de confianza institucional, el interés en política y cuatro de las
> cinco características sociodemográficas— reciben una contribución nula. El
> componente igualitario es el único indicador contextual con contribución
> apreciable (0,1177), casi diecisiete veces superior a la del siguiente. Las
> curvas ALE confirmaron que los efectos de la confianza, la evaluación económica
> y el apoyo a la democracia no son estrictamente lineales. LIME mostró que la
> confianza en el Gobierno es la variable con mayor peso en las aproximaciones
> locales, tanto en los casos representativos como en los errores extremos. Los
> errores con distancia ordinal mayor o igual a dos representan el 7,53 % del
> conjunto de prueba y afectan con más frecuencia a República Dominicana y
> Guatemala.

## 🔴 5.5 OE5 — reescribir

**Qué poner** (ajustando el pasaje actual a la Tabla 5.13 nueva):

> En relación con el objetivo específico 5 (OE5), la convergencia más fuerte es
> con el enfoque de evaluación económica de Lewis-Beck y Stegmaier, que converge
> con siete de las diez variables principales. Los tres marcos restantes
> convergen de forma parcial: Easton con cinco variables, y Norris y Devine con
> cuatro cada uno. La convergencia con Easton resulta parcial porque la
> confianza institucional es el tercer bloque en importancia, pero solo la
> confianza en el Gobierno entra en el top-10, lo que apunta al apoyo específico
> —la evaluación del Ejecutivo en funciones— más que al difuso. Con Norris la
> convergencia también es parcial: la percepción política domina el ranking, pero
> tres de los cuatro indicadores contextuales tienen importancias inferiores a
> 0,008, de modo que el déficit democrático no se expresa en el modelo a través de
> los índices institucionales, salvo el componente igualitario. Estas
> correspondencias deben interpretarse como asociaciones predictivas y no como
> evidencia causal de los mecanismos propuestos.

## 🔴 5.6 §6.1 Recomendaciones — añadir dos

A las cuatro recomendaciones metodológicas actuales conviene añadir:

> Quinta, acompañar toda comparación entre modelos de un intervalo de confianza
> obtenido por remuestreo de la unidad de agrupación relevante: en este trabajo,
> seis de las quince configuraciones resultaron indistinguibles de la principal,
> algo que las estimaciones puntuales no revelan.
>
> Sexta, complementar el corte temporal único con pliegues históricos: la
> comparación entre familias de modelos cambió sustantivamente al evaluar la
> dispersión entre períodos, que resultó ser el criterio más discriminante.

## 🔴 5.7 §6.2 Limitaciones — actualizar y ampliar

Las seis limitaciones actuales siguen siendo válidas, con estos ajustes:

- **Cuarta** (TabNet): reescribir según 3.6.
- **Séptima (nueva)**: SMOTE-NC e imputación, según 3.8.
- **Octava (nueva)**: rendimiento desigual entre categorías, según 4.13.
- **Novena (nueva)**: estabilidad multi-semilla no cuantificada:

> Novena, la incertidumbre del ranking de importancias se cuantifica por
> remuestreo del conjunto de prueba, lo que captura la variabilidad muestral, pero
> no se cuantifica la variabilidad debida a la inicialización aleatoria del
> entrenamiento, que requeriría reajustar el modelo con varias semillas.

- **Décima (nueva)**: potencia del contraste entre modelos:

> Décima, el conjunto de prueba contiene 32 clústeres país-año, un tamaño
> efectivo que no permite distinguir diferencias de kappa del orden de una
> centésima entre configuraciones. Las comparaciones entre familias de modelos
> deben leerse con esa limitación de potencia.

## 🟡 5.8 §6.3 Trabajo futuro

Las cinco líneas siguen vigentes. La cuarta («heterogeneidad mediante efectos de
interacción SHAP») gana pertinencia a la luz de 4.18: el modelo concentra la
señal contextual en un solo indicador, y las interacciones podrían explicar por
qué. Y conviene añadir:

> Sexta, separar el efecto del remuestreo sintético del de la imputación previa
> mediante un cuarto brazo de control, dado que la configuración principal de este
> trabajo pertenece al brazo SMOTE-NC.

---

# ANEXO I — De dónde se lee cada cifra

Todos los archivos son relativos a la raíz del repositorio y provienen de la
corrida del 2026-09-05/06. **Antes de leer un valor, comprobar que la columna o el
campo `modo_ejecucion` dice `real`.**

## Tablas

| Archivo | Contenido | Sección del PDF |
|---|---|---|
| `results/resultados_modelos.csv` | 8 métricas × 5 modelos × 3 estrategias × 2 splits × 2 variantes (40 filas) | Tablas 5.4, 5.5, 5.6, 5.8 |
| `results/modelo_xai_seleccionado.json` | Configuración principal y criterio de selección | §4.10, §5.3.1 |
| `results/tables/metricas_kappa_pivot_val.csv` | κw en validación, modelo × estrategia | Tabla 5.4 |
| `results/tables/metricas_kappa_pivot.csv` | κw en prueba, modelo × estrategia | Tabla 5.5 |
| `results/tables/mejor_estrategia_por_modelo.csv` | Coincidencia val/test por modelo | §5.3.1 (4.7) |
| `results/tables/comparativa_estrategias_balanceo.csv` | Las 15 configuraciones con las 8 métricas | Tabla 5.6, §5.3.1 |
| `results/tables/bootstrap_ic_modelos.csv` | IC 95 % de 4 métricas × 15 configuraciones | Nueva Tabla 5.7b |
| `results/tables/bootstrap_pareado_vs_principal.csv` | Δ con IC y P(Δ>0) frente a la principal | Nueva Tabla 5.7c |
| `results/tables/bootstrap_ic_principal_ponderado.csv` | IC ponderados por `X_020` | §5.3.1 |
| `results/tables/metricas_ponderadas_vs_no_CatBoost_smotenc.csv` | Lectura muestral frente a poblacional | §2.4, §5.3 |
| `results/tables/metricas_por_clase_CatBoost_smotenc_*_test.csv` | Precision, recall y F1 por categoría | Nueva §5.3.3 |
| `results/tables/matriz_confusion_CatBoost_smotenc_*_test.csv` | Conteos | Nueva §5.3.3 |
| `results/tables/matriz_confusion_pct_CatBoost_smotenc_*_test.csv` | Porcentajes por fila | Nueva §5.3.3 |
| `results/tables/validacion_temporal_folds.csv` | κw, MAE y F1 de cada modelo en cada pliegue | Nueva §5.3.4 |
| `results/tables/validacion_temporal_resumen.csv` | Media, sd, mínimo y máximo entre pliegues | Nueva §5.3.4 |
| `results/tables/metricas_e2_variantes_target.csv` | Comparación de formulaciones del target | Tabla 5.8 |
| `results/tables/mae_subregiones.csv` | MAE, κw y accuracy por subregión | Tabla 5.9 |
| `results/tables/mae_por_pais_test.csv` | MAE por país, todas las configuraciones | §5.4 |
| `results/tables/spearman_subregiones.csv` | ρ entre rankings SHAP de subregiones | §5.4 (H5) |
| `results/tables/shap_bloques_ic_CatBoost_smotenc.csv` | Importancia por bloque con IC | Figura 5.2, §5.5.1 |
| `results/tables/shap_importancia_ic_CatBoost_smotenc.csv` | \|SHAP\| con IC, IC del rango y % top-k | Tabla 5.10 |
| `results/tables/shap_importancias_CatBoost_smotenc.csv` | Ranking simple, sin IC | Figura 5.3 |
| `results/tables/shap_concordancia_modelos.csv` | ρ de Spearman entre modelos | Nueva subsección (4.21) |
| `results/tables/shap_top5_por_modelo.csv` | Top-5 comparado entre modelos | Nueva subsección (4.21) |
| `results/tables/errores_graves_CatBoost_smotenc.csv` | Errores con distancia ≥ 2 por tipo y país | Tabla 5.12 |
| `results/tables/lime_CatBoost_smotenc.csv` | Pesos LIME de los 200 casos explicados | Tabla 5.12, §5.5.3 |
| `results/tables/tabnet_atencion_smotenc.csv` | Atención interna de TabNet | §5.5.1 (4.25) |
| `results/tables/contraste_teorico_CatBoost.csv` | % de convergencia por teoría y top-N | Tabla 5.13 |
| `results/tables/tabla_convergencias_CatBoost.csv` | Convergencia variable a variable | Tabla 5.13, §5.6 |
| `results/tables/tabla_maestra_xai_CatBoost.csv` | Importancias con bloque y rangos teóricos | Cap. 6 |
| `results/tables/hiperparametros_modelos.csv` | Registro completo de HP (20 filas) | Tablas 4.9–4.12, 5.7 |
| `results/tables/conjuntos_resumen_*.csv` | Trazabilidad antes y después de las exclusiones | Tabla 5.1 |
| `results/tables/conjuntos_efecto_exclusiones.csv` | Efecto de las exclusiones por conjunto | §5.1 |
| `results/tables/eda_desbalance_clases.csv` | Distribución del target por conjunto | Tabla 5.2 |
| `results/tables/eda_correlaciones_features.csv` | ρ individual y país-año de las 28 variables | Tabla 5.3, §5.2 |
| `results/tables/eda_correlacion_año_target.csv` | ρ año–target | §5.2 |
| `results/tables/eda_venezuela_anomalia.csv` | % clase 3 y KS por año en Venezuela | §5.1 |
| `results/tables/correlaciones_resumen_matrices.csv` | Máximos y pares por matriz | §4.6 |

## Figuras

| Archivo | Contenido | Sección |
|---|---|---|
| `results/figures/eda1_target_split.png` | Distribución del target por conjunto | Figura 5.1 |
| `eda2_matriz_correlacion_latinobarometro.png` | Correlaciones LB | Figura 4.3 |
| `eda2_matriz_correlacion_vdem.png` | Correlaciones V-Dem | Figura 4.4 |
| `eda2_matriz_correlacion_merge.png` | Correlaciones del conjunto fusionado | **Nueva Figura 4.5** |
| `03_metricas_comparativas.png` | Comparativa de métricas entre modelos | §5.3.1 |
| `03_val_vs_test.png` | Validación frente a prueba | §5.3.1 |
| `03_metricas_por_clase.png` | Precision, recall y F1 por categoría | Nueva §5.3.3 |
| `03_confusion_modelo_principal.png` | Matriz de confusión, dos paneles | Nueva §5.3.3 |
| `03_matrices_confusion.png` | Las 15 matrices normalizadas | §5.3.1 |
| `03_mae_por_pais.png` / `03_mae_subregion_heatmap.png` | MAE por país y subregión | §5.4 |
| `04_shap_bloques_CatBoost_smotenc.png` | Importancia por bloque | Figura 5.2 |
| `04_shap_bar_CatBoost_smotenc.png` | Ranking de variables | Figura 5.3 |
| `04_shap_importancia_ic_CatBoost_smotenc.png` | Ranking con barras de error | **Nueva**, junto a la Tabla 5.10 |
| `04_shap_beeswarm_CatBoost_smotenc.png` | Dirección e intensidad por variable | Figura 5.4 |
| `04_ale_H_002_031.png`, `04_ale_H_002_241.png`, `04_ale_D_001_001.png`, `04_ale_C_006_003_011.png`, `04_ale_A_001_001.png` | Curvas ALE (5, no 6) | Figuras 5.5–5.9 |
| `04_shap_concordancia_modelos.png` | Mapa de calor de concordancia | Nueva subsección (4.21) |
| `04_confusion_errores_CatBoost_smotenc.png` | Errores graves | §5.5.3 |
| `04_lime_errores_CatBoost_smotenc.png` | Variables dominantes en los errores | §5.5.3 |
| `04_shap_vs_tabnet_smotenc.png` | SHAP frente a atención de TabNet | §5.5.1 |
| `05_shap_por_subregion.png` / `05_spearman_subregiones.png` | Estabilidad regional | §5.4 |
| `06_convergencias_teoricas_CatBoost.png`, `06_tabla_convergencias_CatBoost.png`, `06_ranking_empirico_vs_teorico_CatBoost.png` | Contraste teórico | §5.6 |

## Registros de trazabilidad (no van al documento; sirven para verificar)

| Archivo | Qué comprobar |
|---|---|
| `models/hp_OLO_smotenc_ordinal_4clases.json` | `implementacion: mord.LogisticIT`, `n_coeficientes: 28`, `umbrales_theta` con 3 valores |
| `models/hp_TabNet_*_ordinal_4clases.json` | `weights: 0` en las tres; `loss_fn` ponderada solo en `pesos_clase` |
| `models/hp_CatBoost_smotenc_ordinal_4clases.json` | Los siete hiperparámetros de la Tabla 5.7 y `task_type: GPU` |
| `models/hp_*_fold[123].json` | Confirman que los pliegues no sobrescribieron los registros del corte definitivo |
| `data/processed/nan_audit.json` | Variables con 100 % de ausentes en prueba, para §6.2 |
| `notebooks/output/*.ipynb` | Salida completa con las cifras impresas celda a celda |

---

# ANEXO II — Parámetros de la corrida

Todos en `PARAMETERS` de `utils/config.py`, perfil `"real"`. Las afirmaciones
numéricas del documento deben coincidir con estos valores.

| Afirmación del documento | Clave | Valor |
|---|---|---|
| Semilla de reproducibilidad | `SEED` | 42 |
| Ensayos de Optuna, árboles de gradiente | `N_TRIALS_OPTUNA` | 50 |
| Ensayos de Optuna, línea base ordinal | `N_TRIALS_OLO` | 20 |
| Ensayos de Optuna, TabNet | `N_TRIALS_TABNET` | 20 |
| Épocas máximas de TabNet | `EPOCAS_TABNET` | 200 |
| Paciencia del early stopping de TabNet | `PACIENCIA_TABNET` | 20 |
| Métrica de selección | `METRICA_PRINCIPAL` | `kappa_cuadratico` |
| Conjunto donde se selecciona | `CONJUNTO_SELECCION` | `val` |
| Repeticiones del bootstrap | `N_BOOTSTRAP` | 1000 |
| Nivel de confianza | `NIVEL_CONFIANZA` | 0,95 |
| Unidad de remuestreo | `NIVEL_CLUSTER` | `pais_anio` (32 clústeres) |
| Casos LIME representativos | `CASOS_LIME_REPRESENTATIVOS` | 100 |
| Casos LIME de error máximo | `CASOS_LIME_ERRORES` | 50 |
| Casos LIME discordantes | `CASOS_LIME_DISCORDANTES` | 50 |
| Variables con curva ALE por bloque | `VARS_ALE_POR_BLOQUE` | 2 |
| Repeticiones del bootstrap del ranking | `N_BOOTSTRAP_SHAP` | 1000 |
| Estrategia de los pliegues temporales | `ESTRATEGIA_FOLDS` | `pesos_clase` |
| Ensayos de Optuna en los pliegues | `N_TRIALS_FOLDS` | 15 |

Las olas de cada conjunto están en `SPLIT` y los pliegues históricos en
`SPLITS_TEMPORALES`, en el mismo archivo.

---

# CHECKLIST

## Bloque 0 — resolver primero (cambian conclusiones)

- [ ] 0.1 §5.2 y §4.4.2: el KS entre validación y prueba da p < 0,001, no 0,787
- [ ] 0.2 §5.3.1 y Cap. 6: H1 se respalda en MAE, no en kappa
- [ ] 0.3 §5.3.1: seis configuraciones son indistinguibles de la principal
- [ ] 0.4 Resumen, Abstract, §5.3.1, Cap. 6: el menor MAE no es de la config. principal
- [ ] 0.5 No reutilizar la narrativa de sobrecorrección de clases minoritarias

## Resumen y Abstract

- [ ] 1.1 κw = 0,5386 y MAE = 0,5688 con CatBoost + SMOTE-NC (los dos idiomas)
- [ ] 1.2 LIME: confianza en el Gobierno, no situación económica (los dos idiomas)
- [ ] 1.4 Quitar «aproximadamente» y mencionar las 489.771 iniciales

## Capítulo 1

- [ ] 2.1 y 2.2: evaluación de H1 y H2 con márgenes e intervalos
- [ ] 2.3 H3: respaldo parcial, con los valores nuevos de bloque

## Capítulo 4

- [ ] 3.1 Tabla 4.1: añadir versiones de librerías (incluida `mord 0.7`)
- [ ] 3.2 §5.3.1: duración 13 h 22 min con desglose por notebook
- [ ] 3.3 §4.6: máximo 0,8552 (V-Dem) y 0,8618 (fusionado); 0 pares > 0,7 en LB
- [ ] 3.4 Añadir Figura 4.5 (matriz del conjunto fusionado)
- [ ] 3.5 §4.9.1: nota al pie sobre el estimador de `LogisticIT` y los umbrales
- [ ] 3.6 §4.9.4 y §6.2: TabNet con pérdida ponderada; `X_020` no interviene
- [ ] 3.7 §4.9.3: fórmula del peso de clase sobre train + los cuatro valores
- [ ] 3.8 §4.9.3 y §6.2: SMOTE-NC e imputación (ahora es la estrategia principal)
- [ ] 3.10 §4.11.2, §2.5.3 y Figuras 5.5–5.10: cinco curvas ALE, no seis
- [ ] 3.11 §4.11.3: confirmar los 200 casos LIME en el pie de tabla
- [ ] 3.12 Tabla 4.12: pie con épocas y paciencia de TabNet

## Capítulo 5

- [ ] 4.2 §5.1: reescribir las cifras de Venezuela con datos reproducibles
- [ ] 4.4 §5.2: ρ año–target = 0,0217
- [ ] 4.5 §5.2: ρ país-año de los cuatro índices de V-Dem
- [ ] 4.7 Texto de la Tabla 5.4: los cinco modelos coinciden val/test
- [ ] 4.8 **Reemplazar la Tabla 5.5** (kappa en prueba) + texto de estrategias
- [ ] 4.9 **Reemplazar la Tabla 5.6** (métricas complementarias) + texto
- [ ] 4.10 **Reemplazar la Tabla 5.7** (hiperparámetros de CatBoost) + texto
- [ ] 4.11 **Insertar la Tabla 5.7b** (IC por bootstrap, 15 configuraciones)
- [ ] 4.12 **Insertar la Tabla 5.7c** (comparación pareada) + texto
- [ ] 4.13 **Insertar la §5.3.3** (métricas por clase, confusión, 2 figuras)
- [ ] 4.14 **Reemplazar la Tabla 5.8** (E2) + nota sobre la estrategia
- [ ] 4.15 **Reemplazar la Tabla 5.9** (subregiones) + corregir Perú + texto
- [ ] 4.16 Figura 5.2 y §5.5.1: bloques con IC
- [ ] 4.17 **Reemplazar la Tabla 5.10** (ranking con IC y rango) + texto
- [ ] 4.18 **Reemplazar la Tabla 5.11** (micro/macro) + texto
- [ ] 4.19 §5.5.2: reescribir sobre las cinco curvas ALE nuevas
- [ ] 4.20 **Reemplazar la Tabla 5.12** (errores graves, seis filas) + texto
- [ ] 4.21 **Insertar la subsección** de concordancia entre modelos
- [ ] 4.22 **Insertar la §5.3.4** (estabilidad temporal) + su alcance
- [ ] 4.23 **Reemplazar la Tabla 5.13** (contraste teórico) + §5.6
- [ ] 4.24 Añadir los resultados cuantitativos de H4 y H5
- [ ] 4.25 Nota sobre la discrepancia entre la atención de TabNet y SHAP
- [ ] 4.17 Añadir las ocho variables con \|SHAP\| nulo (20 de 28 usadas)
- [ ] 4.23 Elegir **una** operacionalización del contraste teórico y declararla

## Capítulo 6

- [ ] 5.1 OE1: enumerar transformaciones en lugar de contarlas
- [ ] 5.2 OE2, 5.3 OE3, 5.4 OE4, 5.5 OE5: reescribir los cuatro
- [ ] 5.6 §6.1: añadir dos recomendaciones
- [ ] 5.7 §6.2: actualizar la cuarta limitación y añadir cuatro nuevas
- [ ] 5.8 §6.3: añadir la sexta línea de trabajo futuro

## Revisión final de coherencia

- [ ] Ninguna tabla o figura cita todavía `pesos_clase` como configuración principal
- [ ] Los nombres de archivo de las figuras insertadas terminan en `_smotenc`
- [ ] Perú aparece en Región Andina en todas las menciones
- [ ] Ninguna cifra proviene de un artefacto con `modo_ejecucion = humo`
- [ ] El número de casos LIME del texto coincide con `CASOS_LIME_*`
- [ ] El número de ensayos de Optuna del texto coincide con `N_TRIALS_*` (50/20/20)
- [ ] κw = 0,5386 y MAE = 0,5688 en Resumen, Abstract, Cap. 5 y Cap. 6
- [ ] La muestra de modelado se cita como 430.895 y no como 431.756 (nota de 3.7)
- [ ] Ninguna cifra por país mezcla configuraciones (aviso de 4.15)
- [ ] Los intervalos de Rosa et al. están verificados en la fuente original (4.14)
- [ ] Ningún pasaje afirma un mecanismo que la corrida no mide (nivel C)

---

# APÉNDICE A — La discrepancia de 0,0004 en TabNet con SMOTE-NC

`results/resultados_modelos.csv` registra para TabNet con SMOTE-NC un kappa de
**0,5208**, mientras que `bootstrap_ic_modelos.csv` y
`bootstrap_pareado_vs_principal.csv` registran **0,5212**.

**Qué se ha comprobado.** Se compararon las quince configuraciones entre los dos
archivos. En catorce, la diferencia es de 5 × 10⁻⁵ o menor, que es exactamente lo
que produce el redondeo a cuatro decimales con que se guardan las tablas de
bootstrap. La única excepción es TabNet con SMOTE-NC, cuya diferencia es de
4,2 × 10⁻⁴, un orden de magnitud mayor: es una discrepancia real y no un
redondeo.

**Qué no se ha determinado.** La causa. Afecta a una sola de las tres
configuraciones de TabNet y a ninguna de las tres de SMOTE-NC de los demás
modelos, de modo que no se explica ni por la arquitectura ni por el brazo de
balanceo por separado. No se ha investigado más porque no hace falta: la
diferencia es 1/47 del error estándar bootstrap de esa configuración (0,0179) y
las dos cifras caen dentro del mismo intervalo de confianza.

**Qué hacer en el documento.** Citar **una sola** de las dos. Recomendación:
0,5208 en la Tabla 5.5, que es la métrica calculada en el entrenamiento y
coherente con el resto de esa tabla, y 0,5212 en las tablas de bootstrap. Si un
revisor lo señala, la respuesta correcta es que las tablas de bootstrap se
almacenan redondeadas a cuatro decimales y que la diferencia es dos órdenes de
magnitud inferior a la incertidumbre de la propia estimación. **No inventar un
mecanismo.**

---

# APÉNDICE B — Comparación con la corrida anterior (uso interno)

**Este apéndice no va al documento de tesis.** Está aquí porque durante la
revisión surgió la pregunta de por qué cambiaron las cifras, y la respuesta
verificada es útil para no repetir afirmaciones incorrectas.

Los artefactos de la corrida anterior están en el historial de git: el commit
`943eaee` (2026-08-31) contiene `results/` y `models/` de la ejecución con
`sklearn.LogisticRegression` como línea base. Se pueden recuperar con
`git show 943eaee:results/resultados_modelos.csv`.

**Lo que la comparación demuestra.** De los seis valores de kappa que *no podían*
cambiar con las modificaciones del código —los tres brazos SMOTE-NC de los
árboles y `sin_balanceo` de los tres— cuatro son **idénticos bit a bit** (por
ejemplo XGBoost sin balanceo, 0,5010191493879517 en las dos corridas) y dos
presentan residuos de 2,1 × 10⁻³ (CatBoost) y 1,2 × 10⁻³ (LightGBM) en el brazo
`sin_balanceo`. Un diff normalizado de `utils/models.py` entre los dos commits
confirma que la lógica de ajuste de los tres árboles es idéntica. Los residuos
son, por tanto, variabilidad entre ejecuciones y no efecto de los cambios.

Además, el `alpha` óptimo de la línea base ordinal es **el mismo valor exacto**
(0,0074593432857265485) con 50 ensayos en la corrida anterior y con 20 en la
actual, lo que respalda que reducir el presupuesto de Optuna para un espacio
unidimensional no degradó la búsqueda.

**Dos afirmaciones que se hicieron y que hay que descartar.** Primera, que los
seis deltas del brazo `pesos_clase` eran todos negativos y que eso constituía «la
firma» de haber eliminado un sesgo optimista: son cinco de seis (LightGBM en
prueba sube +0,0004) y su magnitud es comparable al ruido entre ejecuciones que
se acaba de medir. Segunda, cualquier explicación mecanicista de los residuos
—nodeterminación de las reducciones en GPU, tratamiento de los valores
ausentes— no está verificada y no debe usarse.

**Nota sobre el PDF actual.** Tres celdas de la Tabla 5.5 vigente no coinciden
con ninguna corrida del repositorio: LightGBM con pesos de clase figura como
0,5367 cuando la corrida anterior dio 0,5315; CatBoost sin balanceo como 0,5045
frente a 0,5070; y LightGBM sin balanceo como 0,5076 frente a 0,5062. Son
errores de transcripción. Quedan resueltos al reemplazar la tabla completa
(apartado 4.8), pero explican por qué algunas comparaciones antes/después
parecían no cuadrar.
