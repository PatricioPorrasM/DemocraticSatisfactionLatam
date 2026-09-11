# Plan de cambios sobre la tesis — v2

Cotejo número por número del PDF de la tesis (octubre de 2026) contra los artefactos de la
corrida real, tras la reejecución de NB03–NB06. Todos los valores nuevos son trazables a
`results/`.

**No quedan decisiones pendientes: el plan es ejecutable de principio a fin.**

Destinatarios:

- **Agente IA**: secciones 2 a 6.
- **Trabajo manual del investigador**: sección 7 (figuras) — sin acciones, todas están al día.
- **Estado del repositorio**: sección 8.
- **Trazabilidad de las decisiones**: sección 9.

---

## 1. Correcciones a la versión anterior de este plan

**Leer antes de aplicar nada.** La v1 de este documento contenía cuatro errores propios,
detectados al cotejar contra el texto completo del PDF. Si el agente IA ya aplicó alguno,
hay que revertirlo.

| # | Lo que decía la v1 | Lo correcto |
|---|---|---|
| 1 | En §5.3.4, cambiar "TabNet y la regresión logística ordinal duplican esa dispersión **(0,0763 y 0,0821)**" por "(0,0668 y 0,0783)". | **No cambiar.** 0,0763 y 0,0821 son las desviaciones de la Tabla 5.13 (cuatro cortes) y son **correctas**. 0,0668 y 0,0783 son otro estadístico —la dispersión de los tres pliegues con presupuesto homogéneo— y ya aparecen, correctamente, en la frase siguiente del mismo párrafo. Aplicar el cambio habría mezclado dos estadísticos distintos. |
| 2 | Tabla de §4.6 con la columna "ρ en el PDF" (−0,027 / −0,001 / +0,030 / +0,059 …). | Esos valores **no son los del PDF**. Los reales del PDF están en la sección 2 de este documento. |
| 3 | Quitar del §4.6 el "**94 % de categorías nuevas en test**" de `X_004`. | Esa cifra **no existe en el PDF**. El PDF dice "muchas de ellas presentes en prueba pero ausentes en entrenamiento". No hay nada que quitar; se puede precisar con el valor real (50,4 %). |
| 4 | Sección "Figuras a regenerar" con cinco ítems (heatmap de kappa, forest plot, figura E1 vs. E2, figura de pliegues, figura de métricas complementarias). | **Ninguna de esas figuras existe en el documento.** El índice de figuras del PDF contiene solo 2.1–2.8, 4.1–4.5 y 5.1–5.10. Ver sección 7: no hay ninguna figura que regenerar. |

---

## 2. Capítulo 4 — Datos y diseño

### §4.6 — exclusiones de Latinobarómetro

Base de cálculo: conjunto de entrenamiento del flujo, **378.592 registros**, Spearman con los
códigos de NS/NR tratados como ausentes.

| Variable | ρ en el PDF | ρ real | p real |
|---|---|---|---|
| `A_007_071` Escala izquierda-derecha | −0,0082 | **−0,0412** | < 0,001 |
| `S_700` Religión | −0,0077 | **−0,0096** | < 0,001 |
| `S_701` Práctica religiosa | +0,0233 | **+0,0137** | < 0,001 |
| `C_001_031` Problema más importante del país | +0,0395 | **+0,0081** | < 0,001 |
| `H_002_101` Confianza en la Iglesia | +0,0411 | **+0,0480** | < 0,001 |
| `C_003_003_011` Preocupación por quedar sin trabajo | −0,0442 | **−0,0520** | < 0,001 |
| `H_001_011` Confianza interpersonal | 0,112 | **+0,1169** | < 0,001 |
| `X_004` Región o área geográfica | +0,019 (p = 0,132) | **−0,0017** | **0,318** |
| `X_008` Tamaño de la ciudad | +0,048 | **+0,0448** | < 0,001 |

**Cambio estructural: el grupo de señal baja pasa de seis variables a cinco.** Con ρ = −0,0520,
`C_003_003_011` supera el umbral de |ρ| < 0,05 y ya no pertenece a ese grupo. Texto de reemplazo
del primer párrafo:

> El proceso descartó doce variables del Latinobarómetro. **Cinco** se excluyeron por señal baja
> (|ρ| < 0,05 con la variable objetivo en el conjunto de entrenamiento): la escala
> izquierda-derecha (`A_007_071`, ρ = −0,0412), la religión (`S_700`, ρ = −0,0096), la práctica
> religiosa (`S_701`, ρ = +0,0137), el problema más importante del país (`C_001_031`,
> ρ = +0,0081) y la confianza en la Iglesia (`H_002_101`, ρ = +0,0480).
>
> Las otras **siete** exclusiones responden a criterios específicos de solapamiento con la
> variable objetivo, disponibilidad de información, delimitación conceptual, cardinalidad y señal
> predictiva:

`C_003_003_011` pasa a la lista con viñetas, con el mismo criterio que ya aplica el PDF a
`X_008`. Viñeta nueva:

> **`C_003_003_011`** (preocupación por quedar sin trabajo): su asociación con la variable
> objetivo (ρ = −0,0520) apenas supera el umbral de |ρ| < 0,05, de modo que no entra en el grupo
> de señal baja. Se excluye porque esa señal sigue siendo débil y el marco conceptual adoptado no
> aporta una justificación teórica suficiente para conservarla como predictor de la satisfacción
> democrática.

Conviene colocarla junto a la de `X_008`, que responde al mismo criterio, para que el lector vea
que se trata de una regla aplicada de forma consistente y no de una decisión ad hoc.

Ajustes en las viñetas existentes:

- **`X_004`**: reemplazar "**627 categorías únicas**" por **696** (cardinalidad en el conjunto de
  entrenamiento; la cifra 627 no se reproduce desde ninguna base: todas las olas con país y
  target válidos dan 839 y el archivo base completo 883). Reemplazar "(r = 0,019; p = 0,132)"
  por **"(ρ = −0,002; p = 0,32)"**. Opcionalmente, precisar "muchas de ellas presentes en prueba
  pero ausentes en entrenamiento" como **"el 50,4 % de las presentes en prueba (139 de 276) está
  ausente del entrenamiento"**.
- **`X_008`**: reemplazar "r = 0,048" por **"ρ = +0,045"**. El encuadre del PDF —señal baja más
  ausencia de cobertura en las cinco primeras olas— es correcto y se mantiene.
- **`H_001_011`**: reemplazar "r = 0,112" por **"ρ = +0,117"**.

### §4.6, resto — sin cambios

Verificado y correcto: las Figuras 4.3, 4.4 y 4.5 y todos los valores del texto que las
acompaña (LB máx. 0,56 y ningún par sobre 0,85; V-Dem a nivel país-año 0,86 / 0,76 / −0,83 /
0,79 / −0,72 / −0,71; fusionado 0,86 / 0,75 / 0,80 / −0,82 / −0,69 / −0,71 y magnitudes
micro-macro por debajo de 0,15); los 46 pares con |ρ| > 0,85 y el máximo de 0,9901 en la
especificación inicial de 19 variables; la reducción a un par y el máximo de 0,8641 sobre las
540 observaciones país-año; el máximo de 0,8552 de la Figura 4.4 sobre las 360 de entrenamiento.
La tabla `results/tables/correlaciones_resumen_matrices.csv` confirma 0,5581 / 0,8552 / 0,8618.

### §4.9.3 — sin cambios

Los pesos de clase (2,6524 / 0,9245 / 0,5756 / 1,2439) están verificados contra
`data/processed/train.parquet`.

### §4.9.4 y Tabla 4.14 — reestructuración completa

El diseño cambió: **E2 ya no entrena cinco modelos en binario ni usa pesos de clase**. Entrena
exactamente una configuración adicional —la ganadora de E1— reajustada sobre el target binario,
con la estrategia ganadora re-aplicada a ese target.

Texto de reemplazo de los dos primeros párrafos de §4.9.4:

> El segundo experimento contrasta dos formulaciones del target sobre una única configuración: la
> que resulta ganadora en el Experimento 1 según el criterio declarado (mayor kappa cuadrático en
> la validación de 2020 sobre la formulación ordinal de cuatro clases). Esa configuración se
> reajusta sobre el target binario, obtenido al colapsar las categorías 0 y 1 en "insatisfecho" y
> las categorías 2 y 3 en "satisfecho", y la estrategia de balanceo ganadora se vuelve a aplicar
> sobre la distribución del target binario, en lugar de reutilizar el conjunto remuestreado del
> caso ordinal. Los hiperparámetros del brazo binario se optimizan de nuevo con el mismo
> presupuesto de Optuna, porque la función de pérdida y el número de clases cambian. El diseño
> evita, por construcción, comparar formulaciones sobre modelos distintos: la única variable que
> cambia entre los dos brazos es la definición del target. En total el Experimento 2 añade un
> pipeline a los quince del Experimento 1, de modo que la corrida completa entrena dieciséis.

**Tabla 4.14** pasa a describir **1 configuración × 2 formulaciones** (no 5 × 2):

| Elemento | Valor |
|---|---|
| Configuración | La ganadora de E1 (criterio: kappa cuadrático en validación 2020, formulación ordinal) |
| Formulaciones | Ordinal de 4 clases; binaria de 2 clases (0–1 → insatisfecho, 2–3 → satisfecho) |
| Balanceo | La estrategia ganadora de E1, re-aplicada sobre la distribución del target de cada formulación |
| Optimización | Optuna TPE, mismo presupuesto de trials que en E1, objetivo en validación 2020 |
| Pipelines entrenados | 1 (adicional a los 15 de E1) |
| Conjunto de evaluación | Prueba 2023–2024 (35.084 registros) |

**No perder** el párrafo final de §4.9.4, el que explica que TabNet no admite un peso por
observación y que `X_020` no interviene en su ajuste. Ese contenido sigue siendo correcto y
aplica a E1 y E2; si §4.9.4 se reescribe, hay que reubicarlo (por ejemplo al final de §4.9.3, que
es donde se define la estrategia de pesos de clase).

### §4.11.1 — precisión recomendada

El PDF afirma: *"Para la regresión logística ordinal se usa `KernelExplainer` con 500
observaciones y 50 centroides de referencia obtenidos mediante k-means."* La descripción es
exacta respecto del código (`utils`/NB04 implementan esa rama con `N_MUESTRAS_SHAP_OLO = 500` y
`shap.kmeans(..., 50)`), pero **esa rama no se ejecutó**: SHAP se calcula para la configuración
principal y para los tres modelos de concordancia, todos de gradient boosting, de modo que
`results/shap/` contiene únicamente CatBoost, XGBoost y LightGBM. Un lector inferiría que hay
valores SHAP de OLO, y no los hay. Reformulación sugerida:

> Para los modelos de árboles se utiliza TreeSHAP, que calcula valores de Shapley exactos de
> manera eficiente. El flujo prevé además una rama con `KernelExplainer` —500 observaciones y 50
> centroides de referencia obtenidos mediante k-means— para el caso en que la configuración
> seleccionada fuese la regresión logística ordinal; al resultar seleccionada una configuración
> de gradient boosting, esa rama no se ejecuta en esta corrida.

---

## 3. Capítulo 5 — Resultados

### §5.1 — dos valores

| En el PDF | Reemplazo |
|---|---|
| Venezuela 2024: "la categoría de máxima satisfacción todavía concentra el **54,5 %**" | **46,5 %** |
| "frente a valores comprendidos entre **0,059 y 0,085** en las olas de los años noventa" (KS) | **"entre 0,056 y 0,063"** (1995: 0,0592; 1996: 0,0555; 1997: 0,0567; 1998: 0,0632). El 0,085 no se reproduce desde `eda_venezuela_anomalia.csv`. |

**Verificado y sin cambios en §5.1:** cobertura del 100 % de la fusión; la serie 21,9 % (2013) →
39,7 % (2015) → 49,2 % (2016) → 49,9 % (2017); el KS de 0,2552 en 2017; la ola de 2018 con 1.184
respuestas válidas, 726 (61,3 %) en "Muy satisfecho" y 148 (12,5 %) de insatisfacción repartidas
en 57 y 91; los valores de poliarquía 0,327 → 0,233 → 0,196 y la media de 0,634 para los
diecisiete restantes; Nicaragua 2020 con 0,215 frente a 0,633; toda la Tabla 5.1 (489.771 →
457.499 → 436.463 → 431.756 → 430.895, con pérdidas de 32.272 / 21.036 / 4.707 / 861 y total
58.876); los faltantes 12,84 % global, 13,8 % / 6,3 % / 5,3 % por conjunto, y en prueba
`H_002_131` 50,69 %, `G_002_011` 50,66 % y `D_001_001` 0,33 %.

### §5.2 y Tabla 5.3 — sin cambios

KS validación–prueba 0,0549; ρ(año, target) = 0,0217 con n = 431.756; razón 4,7. Las 28
correlaciones de la Tabla 5.3 coinciden una a una con
`results/tables/eda_correlaciones_features.csv`, igual que los valores agregados por país-año
del texto (v2x_egal −0,4652; v2xcl_rol −0,3527; v2x_polyarchy −0,2034; v2x_corr +0,1888).
La Tabla 5.2 y la Figura 5.1 también coinciden.

### §5.3.1 — conteos y afirmaciones

| En el PDF | Reemplazo |
|---|---|
| "La ejecución completa del flujo requirió **13 h 22 min**" | **"12 h 16 min"**, medidos como la suma de las duraciones de los seis cuadernos (44.154 s = 12 h 15 min 54 s). Conviene declarar el criterio de medición en el texto, porque los cuadernos no se ejecutaron en una única sesión continua. |
| "concentra el **99,6 %** de ese tiempo" | **99,4 %** (12 h 11 min 34 s de 12 h 15 min 54 s) |
| "Las etapas de evaluación, explicabilidad y contraste teórico … suman **menos de cuatro minutos**" | **"suman cuatro minutos"** (4 min 08 s exactos). El NB03 pasó de aproximadamente un minuto a 3 min 49 s porque el apartado §7.2 añade el bootstrap pareado del contraste de H2. |
| "las quince configuraciones del experimento E1, **las cinco de E2** y las quince de los tres pliegues temporales" | **"la de E2"** (las quince de E1 y las quince de los pliegues son correctas) |

Desglose por cuaderno, tomado de las marcas "Inicio del programa" y "Fin del programa" de
`notebooks/output/`, por si se quiere incluir como tabla o nota al pie:

| Cuaderno | Duración | % del total |
|---|---|---|
| NB01 carga de datos | 11 s | 0,03 % |
| NB02 preprocesamiento y entrenamiento | 12 h 11 min 34 s | 99,41 % |
| NB03 evaluación comparativa | 3 min 49 s | 0,52 % |
| NB04 explicabilidad XAI | 13 s | 0,03 % |
| NB05 estabilidad temporal y regional | 5 s | 0,01 % |
| NB06 contraste teórico | 2 s | 0,00 % |
| **Total** | **12 h 15 min 54 s** | 100 % |
| top-3 en validación: CatBoost SMOTE-NC (0,4840), CatBoost pesos (0,4829), **OLO SMOTE-NC (0,4804)** | CatBoost SMOTE-NC (0,4840), CatBoost pesos (0,4829), **LightGBM pesos de clase (0,4810)** |
| top-3 por mejor configuración de cada modelo: CatBoost SMOTE-NC (0,4840), **OLO SMOTE-NC (0,4804)**, **XGBoost pesos (0,4799)** | CatBoost SMOTE-NC (0,4840), **LightGBM pesos de clase (0,4810)**, **TabNet SMOTE-NC (0,4805)** |
| medias por estrategia: pesos 0,4796 > SMOTE-NC 0,4730 > sin balanceo 0,4472 | pesos **0,4799** > SMOTE-NC **0,4737** > sin balanceo **0,4478** |
| "los cinco modelos conservan la misma mejor estrategia en validación y en prueba" | **"cuatro de los cinco modelos conservan la misma mejor estrategia en validación y en prueba; TabNet es la excepción: SMOTE-NC en validación (0,4805 frente a 0,4783) y pesos de clase en prueba (0,5374 frente a 0,5248)"** |
| "La ventaja de la configuración principal sobre las mejores configuraciones de los otros cuatro modelos oscila entre 0,0011 y 0,0108 puntos de kappa, y en los cinco casos el intervalo incluye el cero." | **"Seis de las catorce comparaciones no son distinguibles del cero, con diferencias entre 0,0011 y 0,0108 puntos. Entre las mejores configuraciones de los otros cuatro modelos, tres no son distinguibles de la principal (OLO con SMOTE-NC, +0,0077; LightGBM con pesos de clase, +0,0069; XGBoost con pesos de clase, +0,0046) y una sí lo es (TabNet con SMOTE-NC, +0,0137, con intervalo [+0,0017; +0,0250])."** |

**Verificado y sin cambios en §5.3.1:** rangos de validación 0,3933–0,4840 y de prueba
0,4578–0,5386, con val < test en las quince combinaciones; la ganancia de la línea base ordinal
frente a no balancear, +0,0729 (0,4578 → 0,5308); las ganancias de los otros cuatro modelos entre
0,0197 (TabNet) y 0,0336 (CatBoost); la ventaja distinguible frente a no balancear entre +0,0208
y +0,0807 en las cinco familias.

### Tabla 5.4 — kappa cuadrático en validación

Cambian 5 celdas y una columna de "Mejor (val)".

| Modelo | Sin balanceo | Pesos de clase | SMOTE-NC | Mejor (val) |
|---|---|---|---|---|
| CatBoost | 0,4582 | 0,4829 | **0,4840** | SMOTE-NC |
| LightGBM | **0,4607** ← 0,4578 | **0,4810** ← 0,4796 | 0,4599 | Pesos de clase |
| OLO | 0,3933 | 0,4772 | **0,4801** ← 0,4804 | SMOTE-NC |
| TabNet | 0,4692 | 0,4783 | **0,4805** ← 0,4766 | **SMOTE-NC** ← Pesos de clase |
| XGBoost | 0,4578 | **0,4799** | 0,4641 | Pesos de clase |

En la fila de TabNet, la negrita pasa de "Pesos de clase" a "SMOTE-NC".

### Tabla 5.5 — kappa cuadrático en prueba

Cambian 3 celdas. La columna "Mejor estrategia" no cambia en ninguna fila.

| Modelo | Sin balanceo | Pesos de clase | SMOTE-NC | Mejor estrategia |
|---|---|---|---|---|
| OLO (línea base) | 0,4578 | 0,5278 | **0,5308** | SMOTE-NC |
| XGBoost | 0,5010 | **0,5340** | 0,5150 | Pesos de clase |
| CatBoost | 0,5049 | 0,5373 | **0,5386** | SMOTE-NC |
| LightGBM | **0,5065** ← 0,5050 | **0,5316** ← 0,5320 | 0,5051 | Pesos de clase |
| TabNet | 0,5178 | **0,5374** | **0,5248** ← 0,5208 | Pesos de clase |

**Cambio de encabezado:** renombrar la última columna de "Mejor estrategia" a
**"Mejor (prueba)"**, para distinguirla del "Mejor (val)" de la Tabla 5.4. Con la corrida nueva
las dos columnas discrepan en TabNet, y sin esa distinción la diferencia parece un descuido.

### Tabla 5.5b — criterio de selección por modelo (tabla nueva, recomendada)

Insertar inmediatamente después de la Tabla 5.5. Fuente:
`results/tables/mejor_estrategia_por_modelo.csv`. Título sugerido: *"Estrategia seleccionada en
validación por modelo y su desempeño en prueba, frente a la estrategia que maximiza el kappa en
prueba."*

| Modelo | Mejor estrategia (val) | κw val | κw prueba de esa config. | Mejor estrategia (prueba) | Coincide |
|---|---|---|---|---|---|
| CatBoost | SMOTE-NC | 0,4840 | 0,5386 | SMOTE-NC | sí |
| LightGBM | Pesos de clase | 0,4810 | 0,5316 | Pesos de clase | sí |
| OLO | SMOTE-NC | 0,4801 | 0,5308 | SMOTE-NC | sí |
| TabNet | SMOTE-NC | 0,4805 | **0,5248** | **Pesos de clase** | **no** |
| XGBoost | Pesos de clase | 0,4799 | 0,5340 | Pesos de clase | sí |

Esta tabla es lo que hace auditable el criterio: deja explícito por qué TabNet aparece con
SMOTE-NC en las Tablas 5.6 y 5.10 —selección en validación, que es el protocolo declarado en
§4.10— y con pesos de clase en la columna "Mejor (prueba)" de la Tabla 5.5, que es solo
descriptiva.

### Tabla 5.6 — métricas complementarias

Se aplica el criterio de validación, por lo que TabNet entra con SMOTE-NC. Ajustar el epígrafe a
*"…con la estrategia seleccionada en validación"*.

| Modelo | Estrategia | Accuracy | F1 macro | MAE ordinal | AUROC OvR | Bal. acc. |
|---|---|---|---|---|---|---|
| OLO (línea base) | SMOTE-NC | **0,4196** ← 0,4198 | **0,3975** ← 0,3977 | **0,7364** ← 0,7363 | **0,7142** ← 0,7143 | **0,4588** ← 0,4590 |
| XGBoost | Pesos de clase | 0,4905 | **0,4793** (pasa a ser el máximo) | 0,6415 | **0,7688** | 0,5084 |
| CatBoost | SMOTE-NC | **0,5158** | 0,4782 | **0,5688** | 0,7582 | 0,4676 |
| LightGBM | Pesos de clase | **0,4893** ← 0,4909 | **0,4785** ← 0,4798 | **0,6472** ← 0,6463 | **0,7675** ← 0,7672 | **0,5104** ← 0,5115 (sigue siendo el máximo) |
| TabNet | **SMOTE-NC** ← Pesos de clase | **0,4671** ← 0,4747 | **0,4507** ← 0,4607 | **0,6639** ← 0,6576 | **0,7401** ← 0,7577 | **0,4755** ← 0,4949 |

**Cambio de negrita:** el mejor F1 macro pasa de LightGBM (0,4798) a **XGBoost (0,4793)**. Los
demás máximos no se mueven: accuracy y MAE en CatBoost, AUROC en XGBoost, exactitud balanceada
en LightGBM.

**Texto asociado de §5.3.1** (el párrafo que sigue a la tabla): "LightGBM alcanza el mejor F1
macro (0,4798) y la mayor exactitud balanceada (0,5115)" → **"XGBoost alcanza el mejor F1 macro
(0,4793) y el AUROC OvR más alto (0,7688); LightGBM, la mayor exactitud balanceada (0,5104)"**.
La frase sobre la línea base ordinal cambia solo en el MAE: 0,7363 → **0,7364**.

### Tabla 5.7 — intervalos bootstrap

Cambian 3 filas y el orden del ranking.

| Configuración | En el PDF | Reemplazo |
|---|---|---|
| LightGBM · Pesos de clase | 0,5320 [0,4910; 0,5658] ee 0,0193 | **0,5316 [0,4911; 0,5647] ee 0,0191** |
| TabNet · SMOTE-NC | 0,5212 [0,4832; 0,5516] ee 0,0179 | **0,5248 [0,4870; 0,5559] ee 0,0179** |
| LightGBM · Sin balanceo | 0,5050 [0,4584; 0,5390] ee 0,0206 | **0,5065 [0,4625; 0,5406] ee 0,0206** |

**Reordenamiento:** las filas 11 y 12 se invierten — LightGBM sin balanceo (0,5065) queda por
encima de LightGBM SMOTE-NC (0,5051).

### Tabla 5.8 — diferencias pareadas contra la configuración principal

Cambian 3 filas, el orden y una afirmación.

| Comparación | En el PDF | Reemplazo |
|---|---|---|
| LightGBM [sin balanceo] | +0,0335 [+0,0253; +0,0422] | **+0,0320 [+0,0254; +0,0395]** |
| TabNet [SMOTE-NC] | +0,0174 [+0,0040; +0,0303], P 0,997 | **+0,0137 [+0,0017; +0,0250], P 0,988** |
| LightGBM [pesos de clase] | +0,0066 [−0,0056; +0,0173], P 0,865 | **+0,0069 [−0,0058; +0,0189], P 0,855** |

**Reordenamiento:** LightGBM [SMOTE-NC] (+0,0334) pasa por encima de LightGBM [sin balanceo]
(+0,0320).

La conclusión de TabNet [SMOTE-NC] sigue siendo "distinguible", ahora con un intervalo que
excluye el cero por menos margen. Las seis filas "no distinguible" no cambian de categoría.

### Tabla 5.9 — sin cambios

Verificada contra `models/hp_CatBoost_smotenc_ordinal_4clases.json`: iterations 300, depth 4,
learning_rate 0,010033, l2_leaf_reg 9,272779, bagging_temperature 0,055943, border_count 68,
random_strength 6,448171, y el pie con κw = 0,4840 en validación.

### Tabla 5.10 — hiperparámetros

Cambian la fila de LightGBM completa, la fila de TabNet completa y el pie.

**LightGBM, pesos de clase:**

| Hiperparámetro | En el PDF | Reemplazo |
|---|---|---|
| n_estimators | 600 | **300** |
| num_leaves | 38 | **50** |
| max_depth | 6 | 6 (sin cambio) |
| learning_rate | 0,026616 | **0,052016** |
| subsample | 0,767667 | **0,692828** |
| colsample_bytree | 0,759975 | **0,830985** |
| reg_alpha | 0,020212 | **7,5853 × 10⁻⁵** |
| reg_lambda | 0,016194 | **0,024375** |
| min_child_samples | 43 | **57** |

**TabNet** — cambia además la estrategia: **SMOTE-NC** ← Pesos de clase

| Hiperparámetro | En el PDF | Reemplazo |
|---|---|---|
| n_d | 16 | **32** |
| n_a | 32 | **24** |
| n_steps | 6 | 6 (sin cambio) |
| gamma | 1,024487 | **1,139494** |
| lambda_sparse | 1,0223 × 10⁻⁶ | **7,5237 × 10⁻⁶** |
| momentum | 0,161620 | **0,152881** |
| mask_type | entmax | entmax (sin cambio) |
| lr | 0,000649 | **0,00025081** |

OLO (alpha = 0,007459) y XGBoost (las ocho filas) no cambian.

**Pie de tabla**, kappa de validación del ajuste final: OLO **0,4801** ← 0,4804; XGBoost 0,4799
(sin cambio); LightGBM **0,4810** ← 0,4796; TabNet **0,4805** ← 0,4783.

### Tabla 5.11 y §5.3.2 — reemplazo íntegro

Fuente: `results/tables/metricas_e2_variantes_target.csv`. Título sugerido: *"Experimento 2:
formulaciones del target sobre la configuración principal (CatBoost con SMOTE-NC, conjunto de
prueba 2023–2024, n = 35.084)."*

| Métrica | Ordinal (4 clases) | Binaria (2 clases) |
|---|---|---|
| Exactitud | 0,5158 | 0,7649 |
| Exactitud balanceada | 0,4676 | 0,7435 |
| F1 macro | 0,4782 | 0,7377 |
| F1 ponderado | 0,5104 | 0,7674 |
| Kappa cuadrático | 0,5386 | 0,4759 |
| MAE ordinal | 0,5688 | 0,2351 |
| AUROC (OvR macro) | 0,7582 | 0,8261 |

Notas de pie que conviene incluir:

- En la formulación binaria el kappa lineal y el cuadrático coinciden (0,4759): con dos clases
  existe un único tipo de desacuerdo, así que la ponderación cuadrática no distingue nada.
- El MAE y el kappa no son directamente comparables entre formulaciones, porque cambian el número
  de clases y la escala del error. La lectura es descriptiva, no un test de superioridad.

**Texto de reemplazo para §5.3.2** (los dos primeros párrafos):

> El Experimento 2 compara las dos formulaciones del target sobre una sola configuración
> —CatBoost con SMOTE-NC, ganadora del Experimento 1— y evalúa ambas sobre el mismo conjunto de
> prueba de 2023–2024. El brazo binario reoptimiza sus hiperparámetros con el mismo presupuesto
> de Optuna (50 trials, mejor kappa en validación 0,4211) y vuelve a aplicar SMOTE-NC sobre la
> distribución del target binario, lo que expande el entrenamiento de 378.592 a 481.068 registros
> con 240.534 casos por clase.
>
> En la variante binaria, el conjunto de prueba contiene un 32,4 % de personas insatisfechas
> (11.383 registros) y un 67,6 % de satisfechas (23.701 registros); la razón entre satisfechas e
> insatisfechas aumenta de 1,7 en entrenamiento (240.534 frente a 138.058 registros) a 2,1 en
> prueba. La formulación binaria mejora todas las métricas de clasificación: la exactitud pasa de
> 0,5158 a 0,7649, el F1 macro de 0,4782 a 0,7377 y el AUROC de 0,7582 a 0,8261. Ese salto no
> indica un modelo mejor, sino una tarea distinta: al colapsar cuatro categorías en dos
> desaparecen los desacuerdos entre niveles adyacentes de la escala, que son precisamente los que
> concentran el error en la formulación ordinal. El kappa cuadrático se mueve en dirección
> contraria (0,5386 frente a 0,4759), y la razón es la misma: con dos clases la ponderación
> cuadrática ya no puede premiar los aciertos parciales. La formulación de cuatro clases se
> conserva como principal porque el objetivo de la tesis es explicar la gradación de la
> satisfacción con la democracia, información que la dicotomización elimina.

**El tercer párrafo de §5.3.2, el de Rosa et al., se mantiene sin cambios.** Su afirmación —que
el intervalo de F1 de 0,84–0,85 corresponde a la clase de insatisfechos y no al promedio macro—
es correcta y sigue siendo pertinente.

Hiperparámetros del brazo binario, por si se citan: iterations 800, depth 7,
learning_rate 0,076532, l2_leaf_reg 7,800333, bagging_temperature 0,338405, border_count 109,
random_strength 3,376493; función de pérdida `Logloss`.

### §5.3.3 — un valor

| En el PDF | Reemplazo |
|---|---|
| "lo que la variante binaria confirma con un F1 macro de **0,7381**" | **0,7377** |

**Verificado y sin cambios:** F1 por clase 0,5796 / 0,3933 / 0,4003 y recall 63,3 % de la clase
mayoritaria; brecha de 0,1863; precisión 0,4489 frente a recall 0,3500 en la clase 0 y 0,5664
frente a 0,5149 en la clase 3; 51,58 % exactas, 40,89 % a una categoría y 92,47 % acumulado;
2.641 errores de distancia ≥ 2 (7,53 %) y 328 de distancia máxima (0,93 %); 29,1 % y 43,5 % de
desplazamiento hacia el centro.

### Tabla 5.12 — sin cambios

La definición de los pliegues está correcta.

### Tabla 5.13 y §5.3.4 — validación temporal

**Tabla 5.13**, dos filas:

| Modelo | En el PDF | Reemplazo |
|---|---|---|
| CatBoost | 0,4842 ± 0,0381 · rango 0,4467–0,5373 · MAE 0,6620 ± 0,0402 | 0,4842 ± **0,0380** · rango **0,4471**–0,5373 · MAE **0,6646 ± 0,0429** |
| LightGBM | 0,4824 ± 0,0331 · rango 0,4622–0,5320 · MAE 0,6635 ± 0,0347 | **0,4818** ± **0,0333** · rango **0,4624**–**0,5316** · MAE **0,6647 ± 0,0328** |

XGBoost (0,4860 ± 0,0320 · 0,4677–0,5340 · 0,6587 ± 0,0360), TabNet (0,4574 ± 0,0763 ·
0,3535–0,5374 · 0,7131 ± 0,0727) y OLO (0,4505 ± 0,0821 · 0,3343–0,5278 · 0,8380 ± 0,1109) no
cambian, ni el orden de filas.

**§5.3.4:**

| En el PDF | Reemplazo |
|---|---|
| "los tres modelos de gradient boosting presentan desviaciones estándar de 0,0320 a **0,0381**" | **"de 0,0320 a 0,0380"** |
| "TabNet y la regresión logística ordinal duplican esa dispersión (0,0763 y 0,0821)" | **sin cambio** (ver corrección 1 de la sección 1) |
| presupuesto homogéneo: "0,0022 para XGBoost, **0,0032** para LightGBM y **0,0172** para CatBoost, frente a 0,0668 para TabNet y 0,0783 para la regresión logística ordinal" | **"0,0022 para XGBoost, 0,0025 para LightGBM y 0,0169 para CatBoost"**; 0,0668 y 0,0783 no cambian |
| "casi cuatro veces la mayor de los modelos de árboles" | sin cambio (0,0668 / 0,0169 = 3,95) |
| "una brecha de **0,1435** puntos" | **0,1434** (0,4778 y 0,3343 no cambian, ni "0,9957 frente a 0,6162" en MAE) |
| "las mejores configuraciones de los cinco modelos quedan dentro de un rango de **0,0078** puntos de kappa" | **"de 0,0138 puntos"** (de 0,5386 en CatBoost a 0,5248 en TabNet, con la estrategia seleccionada en validación) |

**Precisión recomendada en la Tabla 5.13.** Los cuatro cortes se evalúan con la estrategia de
**pesos de clase** en los cinco modelos, constante entre pliegues. El epígrafe no lo dice, y eso
genera una aparente contradicción: el máximo de CatBoost en la Tabla 5.13 es 0,5373 mientras que
su mejor configuración en la Tabla 5.5 es 0,5386. Añadir al epígrafe o al pie: *"Los cuatro
cortes se entrenan con la estrategia de pesos de clase, que se mantiene constante entre pliegues
para que la comparación aísle el efecto del período."*

### §5.4 y Tabla 5.14 — sin cambios

Verificado contra `results/tables/mae_subregiones.csv`: las cinco filas, los n, MAE, kappa,
accuracy y los países extremos. También el texto: diferencia de 0,1384; ρ(MAE, κ) = +0,70;
ρ(HHI, MAE) = −1,00 y ρ(HHI, κ) = −0,70; concentración 48,6 % y 39,0 %; Perú 0,4469 como mejor y
Argentina 0,6678, República Dominicana 0,6677 y Costa Rica 0,6627 como peores.

### §5.5 completa y Tablas 5.15 a 5.19 — sin cambios

Todo verificado contra los artefactos de la reejecución de NB04 y NB05:

- Tabla 5.15: los seis bloques con 0,4818 / 0,3853 / 0,2601 / 0,1267 / 0,0938 / 0,0034 y sus
  intervalos.
- Tabla 5.16: las veinte variables, sus |SHAP|, intervalos, rangos y porcentajes de top-5,
  incluido el 68,7 % de `B_001_101` y el 29,8 % de `v2x_egal`.
- Tabla 5.17: 0,2136 / 0,1697 / 0,1637 / 0,1436 / 0,1247 en el nivel micro y 0,1177 / 0,00695 /
  0,0013 / 0,0008 en el macro, con la razón de 16,9.
- Las ocho variables de contribución nula son exactamente las que describe el texto: confianza en
  FF. AA., en televisión y en la policía; interés en política; y edad, nivel educativo, nivel
  socioeconómico y situación ocupacional. Los porcentajes de ausencias citados también coinciden
  (0,01 % / 0,36 % / 0,49 % / 1,32 % y 32,9 % / 40,8 %).
- Tabla 5.18: las seis filas de errores graves, con 1.154 / 418 / 389 / 352 / 231 / 97 y los
  países más frecuentes.
- §5.5.3: los pesos LIME 0,0461 / 0,0430 / 0,0414 en errores máximos, 0,0306 / 0,0285 / 0,0256 en
  representativos y 0,0243 / 0,0243 / 0,0225 en discordantes.
- Tabla 5.19 y Figura 5.10: los tres top-5 y las correlaciones 0,9625 / 0,8614 / 0,9243, con
  W de Kendall 0,9367.

### §5.6 y Tabla 5.20 — sin cambios

El contraste teórico coincide con `results/tables/contraste_teorico_CatBoost.csv`, igual que el
texto sobre Easton (una sola variable del bloque en el top-10; partidos en 11, Congreso en 13,
Poder Judicial en 14; policía, televisión y FF. AA. con importancia nula) y sobre Norris (tres de
los cuatro indicadores contextuales por debajo de 0,008).

---

## 4. Capítulo 5 — subsecciones nuevas para H4 y H5

Las dos hipótesis están enunciadas en §1.2 del PDF con el alcance correcto ("sobre la
configuración seleccionada en su formulación ordinal de cuatro clases"), pero **el capítulo 5 no
presenta su evidencia y §6.1 no las contrasta**. Ubicación propuesta: al final de §5.5, después
de §5.5.4.

### §5.5.5 — "Variación de la importancia por bloque temático entre subregiones" (H4)

> El contraste de H4 requiere comparar cuánto varía la importancia de cada bloque temático al
> pasar de una subregión a otra. El estadístico empleado es el coeficiente de variación de la
> importancia SHAP media del bloque entre las cinco subregiones, y no su rango absoluto: los
> bloques difieren en un orden de magnitud entre sí, de modo que el rango crece con la magnitud
> del bloque y no mide variación comparable. El bloque de características sociodemográficas se
> toma como referencia por ser el de contribución más baja y el menos ligado al contexto
> institucional.
>
> Los bloques de confianza institucional y de corrupción y seguridad presentan coeficientes de
> variación de 2,61 % y 2,50 %, frente al 1,26 % del bloque de referencia, lo que arroja una razón
> media de 2,03 a 1. La hipótesis se sostiene en la dirección predicha: la importancia atribuida
> a las variables institucionales es más sensible al contexto subregional que la de las variables
> sociodemográficas. La magnitud absoluta de esa variación es, sin embargo, pequeña —la
> importancia media del bloque de confianza institucional se mueve entre 0,0356 en Brasil y
> 0,0381 en la Región Andina—, de modo que la evidencia respalda la existencia del efecto pero no
> su relevancia práctica.
>
> El bloque de contexto democrático presenta un coeficiente de variación de 52,01 %, el más alto
> del conjunto, pero no forma parte del contraste: sus variables provienen de V-Dem y son
> constantes dentro de cada país-año, por lo que su dispersión entre subregiones refleja las
> diferencias de régimen político registradas en la fuente y no heterogeneidad en el
> comportamiento del modelo.

**Primera tabla de la subsección.** Fuente: `results/tables/shap_bloques_por_subregion.csv`.
Permite verificar aritméticamente los coeficientes de variación. Valores multiplicados por 1000
para legibilidad.

| Bloque | Cono Sur | R. Andina | Brasil | Centroamérica | México y Caribe |
|---|---|---|---|---|---|
| Percepción política | 119,47 | 123,48 | 118,91 | 119,58 | 118,74 |
| Evaluación económica | 71,96 | 77,41 | 77,63 | 80,25 | 79,91 |
| Contexto democrático | 47,29 | 22,65 | 9,97 | 34,81 | 21,89 |
| Confianza institucional | 36,67 | 38,09 | 35,55 | 37,41 | 36,52 |
| Corrupción y seguridad | 31,44 | 32,29 | 30,81 | 30,54 | 30,41 |
| Características sociodemográficas | 0,69 | 0,68 | 0,68 | 0,68 | 0,67 |

**Segunda tabla de la subsección.** Fuente:
`results/tables/h4_variacion_bloques_subregion.csv`.

| Bloque | Variables | Importancia media | CV (%) | Mín. | Máx. |
|---|---|---|---|---|---|
| Contexto democrático | 4 | 0,0273 | 52,01 | 0,0100 (Brasil) | 0,0473 (Cono Sur) |
| Evaluación económica | 5 | 0,0774 | 4,29 | 0,0720 (Cono Sur) | 0,0803 (Centroamérica) |
| Confianza institucional | 7 | 0,0368 | 2,61 | 0,0356 (Brasil) | 0,0381 (R. Andina) |
| Corrupción y seguridad | 3 | 0,0311 | 2,50 | 0,0304 (México y Caribe) | 0,0323 (R. Andina) |
| Percepción política | 4 | 0,1200 | 1,63 | 0,1187 (México y Caribe) | 0,1235 (R. Andina) |
| Características sociodemográficas | 5 | 0,0007 | 1,26 | 0,0007 (México y Caribe) | 0,0007 (Cono Sur) |

Figura disponible y no usada hoy: `results/figures/05_shap_por_subregion.png`.

### §5.5.6 — "Concordancia de los rankings de importancia entre subregiones" (H5)

> H5 se contrasta correlacionando, por pares de subregiones, los rankings completos de
> importancia SHAP de las variables. Las diez correlaciones de Spearman fuera de la diagonal se
> sitúan entre 0,9669 y 0,9989, con una media de 0,9899, de modo que todas superan con amplitud
> el umbral de 0,7 declarado en la hipótesis. El par más discrepante es Brasil frente al Cono Sur
> (0,9669) y el más concordante la Región Andina frente a México y el Caribe (0,9989). La
> hipótesis se confirma: la jerarquía de determinantes que aprende el modelo es prácticamente
> invariante entre subregiones, aunque el nivel de error predictivo sí difiera entre ellas, como
> muestra §5.4. En otras palabras, el modelo no cambia qué variables considera importantes al
> cambiar de subregión; cambia solo cuán bien acierta.

**Tabla de la subsección.** Fuente: `results/tables/spearman_subregiones.csv`.

| | Cono Sur | R. Andina | Brasil | Centroamérica | México y Caribe |
|---|---|---|---|---|---|
| Cono Sur | 1,0000 | 0,9905 | 0,9669 | 0,9955 | 0,9894 |
| Región Andina | 0,9905 | 1,0000 | 0,9905 | 0,9972 | 0,9989 |
| Brasil | 0,9669 | 0,9905 | 1,0000 | 0,9821 | 0,9905 |
| Centroamérica | 0,9955 | 0,9972 | 0,9821 | 1,0000 | 0,9972 |
| México y Caribe | 0,9894 | 0,9989 | 0,9905 | 0,9972 | 1,0000 |

Figura disponible y no usada hoy: `results/figures/05_spearman_subregiones.png`.

---

## 5. Capítulo 5 — tablas nuevas del contraste de H2 y del OE3

### Tabla del contraste de H2

Fuente: `results/tables/h2_tabnet_vs_referencias.csv`, que produce el **NB03 §7.2** (apartado
añadido en esta reejecución). Bootstrap pareado de clústeres país-año: kappa cuadrático, 32
conglomerados, B = 1000, semilla 42.

Esta tabla **resuelve una limitación que el propio PDF declara** en §6.1: *"la Tabla 5.8 no
presenta un contraste directo entre TabNet y OLO"*. Ahora sí existe.

Título sugerido: *"Contraste de H2: kappa cuadrático de TabNet frente a la línea base ordinal y
al mejor modelo de gradient boosting, con la estrategia seleccionada en validación. Bootstrap
pareado por conglomerados país-año, B = 1000."*

| Comparación (A − B) | κw A | κw B | Δ | IC 95 % | P(Δ > 0) | Lectura |
|---|---|---|---|---|---|---|
| TabNet [SMOTE-NC] − OLO [SMOTE-NC] | 0,5248 | 0,5308 | −0,0060 | [−0,0158; +0,0050] | 0,149 | no distinguible del cero |
| TabNet [SMOTE-NC] − CatBoost [SMOTE-NC] | 0,5248 | 0,5386 | −0,0137 | [−0,0250; −0,0017] | 0,012 | TabNet inferior, distinguible |

Filas de sensibilidad, con la estrategia que maximiza el kappa de TabNet en prueba:

| Comparación (A − B) | κw A | κw B | Δ | IC 95 % | P(Δ > 0) | Lectura |
|---|---|---|---|---|---|---|
| TabNet [Pesos de clase] − OLO [SMOTE-NC] | 0,5374 | 0,5308 | +0,0066 | [−0,0065; +0,0180] | 0,847 | no distinguible del cero |
| TabNet [Pesos de clase] − CatBoost [SMOTE-NC] | 0,5374 | 0,5386 | −0,0011 | [−0,0106; +0,0095] | 0,409 | no distinguible del cero |

### Tabla del efecto del balanceo sobre la clase minoritaria — evidencia del OE3

Fuente: `results/tables/efecto_balanceo_clase_minoritaria.csv`. Es evidencia del OE3 —el impacto
de las tres estrategias de manejo del desbalance sobre el rendimiento— y **no** el contraste de
una hipótesis. Ubicación sugerida: §5.3.1, después de la Tabla 5.8, o junto a la Tabla 5.5b.

Título sugerido: *"Efecto de las estrategias de balanceo sobre el F1 de la clase minoritaria
(clase 0), conjunto de prueba. Bootstrap pareado por conglomerados país-año, B = 1000."*

| Modelo | Estrategia | F1 clase 0 | F1 base | Δ | IC 95 % | Conclusión |
|---|---|---|---|---|---|---|
| OLO | Pesos de clase | 0,3944 | 0,2118 | +0,1826 | [+0,1194; +0,2389] | mejora distinguible |
| OLO | SMOTE-NC | 0,3937 | 0,2118 | +0,1819 | [+0,1183; +0,2379] | mejora distinguible |
| CatBoost | Pesos de clase | 0,4363 | 0,2636 | +0,1727 | [+0,1364; +0,2049] | mejora distinguible |
| XGBoost | Pesos de clase | 0,4395 | 0,3024 | +0,1370 | [+0,1139; +0,1559] | mejora distinguible |
| CatBoost | SMOTE-NC | 0,3933 | 0,2636 | +0,1297 | [+0,0941; +0,1585] | mejora distinguible |
| LightGBM | Pesos de clase | 0,4394 | 0,3118 | +0,1276 | [+0,1030; +0,1554] | mejora distinguible |
| TabNet | Pesos de clase | 0,4269 | 0,3035 | +0,1234 | [+0,0895; +0,1650] | mejora distinguible |
| TabNet | SMOTE-NC | 0,3897 | 0,3035 | +0,0862 | [+0,0543; +0,1285] | mejora distinguible |
| XGBoost | SMOTE-NC | 0,3848 | 0,3024 | +0,0823 | [+0,0575; +0,1058] | mejora distinguible |
| LightGBM | SMOTE-NC | 0,2984 | 0,3118 | −0,0134 | [−0,0422; +0,0220] | no distinguible |

Texto de acompañamiento sugerido:

> El efecto de las estrategias de balanceo sobre la clase minoritaria es inequívoco: nueve de las
> diez comparaciones pareadas muestran una mejora distinguible del F1 de la clase 0, con
> incrementos entre +0,0823 y +0,1826 puntos. La única excepción es LightGBM con SMOTE-NC, cuya
> diferencia de −0,0134 tiene un intervalo de [−0,0422; +0,0220] que incluye el cero. Ninguna
> combinación de modelo y estrategia produce un deterioro distinguible, de modo que el balanceo
> mejora la detección de la clase de menor prevalencia sin costo medible en ningún caso.

---

## 6. Capítulo 6 — Conclusiones

### OE1, OE4 y OE5 — sin cambios

Verificados: la armonización documentada, el uso de veinte de las veintiocho variables, el
componente igualitario como único indicador contextual apreciable con razón de 16,9, el 7,53 % de
errores graves y los cuatro marcos teóricos con sus conteos de convergencia.

### OE2 — tres cambios

| En el PDF | Reemplazo |
|---|---|
| "el MAE ordinal es de 0,5688 frente a **0,7363** en la línea base" | **0,7364** |
| "**Seis de las quince configuraciones** evaluadas resultan estadísticamente indistinguibles de la principal" | **"Seis de las catorce configuraciones comparadas resultan estadísticamente indistinguibles de la principal"** |
| "LightGBM alcanza el mejor F1 macro (0,4798) y la mayor exactitud balanceada (0,5115), y XGBoost el AUROC OvR más alto (0,7688)" | **"XGBoost alcanza el mejor F1 macro (0,4793) y el AUROC OvR más alto (0,7688), y LightGBM la mayor exactitud balanceada (0,5104)"** |

### OE3 — dos cambios

El enunciado del OE3 en §1.3.2 ya está actualizado con las tres cláusulas; no hay que tocarlo.
Sí el párrafo del capítulo 6:

| En el PDF | Reemplazo |
|---|---|
| "los cinco modelos conservan la misma mejor estrategia en validación y en prueba" | **"cuatro de los cinco modelos conservan la misma mejor estrategia en validación y en prueba; TabNet cambia de SMOTE-NC en validación a pesos de clase en prueba"** |
| "En la formulación binaria, TabNet obtuvo el mejor F1 macro (0,7385), seguido por CatBoost (0,7381), con las cinco configuraciones en un rango de 0,0107 puntos." | **eliminar por completo** y sustituir por el texto de abajo |

> En la formulación binaria, la única configuración entrenada —la ganadora del Experimento 1—
> alcanza un F1 macro de 0,7377 y un AUROC de 0,8261, frente a 0,4782 y 0,7582 en la formulación
> ordinal; el kappa cuadrático se mueve en dirección contraria (0,4759 frente a 0,5386), porque
> con dos clases la ponderación cuadrática deja de premiar los aciertos parciales.

### §6.1 — contraste de las hipótesis

**Frase de apertura:** "Las **tres** hipótesis planteadas en la introducción se contrastan sin
modificar su formulación a partir de los resultados. Para H1 y H2 se consideran las
configuraciones seleccionadas en validación y evaluadas en prueba; para H3, la importancia SHAP
del modelo principal." → **"Las cinco hipótesis planteadas en la introducción se contrastan sin
modificar su formulación a partir de los resultados. Para H1 y H2 se consideran las
configuraciones seleccionadas en validación y evaluadas en prueba; para H3, H4 y H5, la
importancia SHAP del modelo principal."**

**H1** — un valor: el MAE de la línea base pasa de 0,7363 a **0,7364**. El intervalo
[−0,0074; +0,0212] y el MAE de 0,5688 no cambian. La frase sobre la menor dispersión de los
árboles en los tres pliegues con presupuesto homogéneo se mantiene, con los dos valores
corregidos de §5.3.4.

**H2** — reescritura completa. El veredicto es **cumplimiento parcial** y ahora está respaldado
por la tabla nueva de la sección 5 de este plan. Texto de reemplazo:

> **H2: se cumple parcialmente.** La segunda cláusula se confirma: con la estrategia seleccionada
> en validación, TabNet alcanza un kappa cuadrático de 0,5248 en prueba frente a 0,5386 de
> CatBoost, el modelo de gradient boosting con mejor desempeño, y el bootstrap pareado por
> conglomerados país-año sitúa esa diferencia en −0,0137 con un intervalo del 95 % de
> [−0,0250; −0,0017], que excluye el cero. TabNet es, por tanto, distinguiblemente inferior al
> mejor modelo de gradient boosting. La primera cláusula no se cumple: TabNet no supera a la
> regresión logística ordinal, sino que queda 0,0060 puntos por debajo de ella (0,5248 frente a
> 0,5308), con un intervalo de [−0,0158; +0,0050] que incluye el cero. Lo que sostienen los datos
> es equivalencia estadística entre la red neuronal tabular y la línea base ordinal, no la
> superioridad que anticipaba la hipótesis. El resultado no depende del criterio de selección en
> su parte sustantiva: si en lugar de la estrategia seleccionada en validación se toma la que
> maximiza el kappa de TabNet en prueba (pesos de clase, 0,5374), el orden nominal que predice H2
> sí se observa, pero ninguna de las dos diferencias resulta distinguible del cero (+0,0066 con
> [−0,0065; +0,0180] frente a OLO; −0,0011 con [−0,0106; +0,0095] frente a CatBoost). Bajo ese
> criterio alternativo H2 tampoco encuentra respaldo estadístico. La conclusión sustantiva es que
> el costo computacional adicional de TabNet —el entrenamiento más largo de los cinco modelos y
> la mayor dispersión entre pliegues— no compra desempeño ni frente a la línea base lineal ni
> frente al gradient boosting.

**H3** — sin cambios.

**H4 y H5** — párrafos nuevos, que se añaden después de H3:

> **H4: se sostiene.** Sobre la configuración seleccionada, la importancia media de los bloques de
> confianza institucional y de corrupción y seguridad varía entre subregiones con coeficientes de
> variación de 2,61 % y 2,50 %, frente al 1,26 % del bloque de características sociodemográficas
> tomado como referencia: una razón media de 2,03 a 1. Los dos bloques que predice la hipótesis
> superan la variación relativa del bloque de referencia, de modo que la dirección anticipada se
> confirma. La magnitud absoluta de esa variación es, no obstante, pequeña —la importancia media
> del bloque institucional se mueve entre 0,0356 y 0,0381 puntos—, por lo que la evidencia
> respalda la existencia del efecto y no su relevancia práctica.
>
> **H5: se confirma.** Las diez correlaciones de Spearman entre los rankings de importancia SHAP
> de las cinco subregiones se sitúan entre 0,9669 y 0,9989, con una media de 0,9899; todas
> superan con amplitud el umbral de 0,7 declarado en la hipótesis. El par más discrepante es
> Brasil frente al Cono Sur y el más concordante la Región Andina frente a México y el Caribe. La
> jerarquía de determinantes que aprende el modelo es, en consecuencia, estable a través de las
> subregiones, aun cuando el nivel de error predictivo difiera entre ellas.

### §6.2 — un cambio

| En el PDF | Reemplazo |
|---|---|
| recomendación 5: "en este trabajo, **seis de las quince configuraciones** resultaron indistinguibles de la principal" | **"seis de las catorce configuraciones comparadas"** |

### §6.3 — limitaciones

Las diez limitaciones son correctas y se mantienen. **Falta una**, que conviene añadir dado el
alcance declarado en OE3, H4 y H5:

> **Alcance de la explicabilidad.** Los valores SHAP, y con ellos los contrastes de H3, H4 y H5,
> se calculan únicamente sobre la configuración seleccionada en el Experimento 1 y en su
> formulación ordinal de cuatro clases. La comparación de rankings de §5.5.4 se extiende a los
> otros dos modelos de gradient boosting con la misma estrategia de balanceo, pero el estudio no
> dispone de valores SHAP para la regresión logística ordinal ni para TabNet, de modo que no puede
> establecer si el perfil de determinantes identificado es propio de la familia de árboles.

### §6.4 — sin cambios

---

## 7. Figuras

**Conclusión: ninguna figura del documento requiere regeneración.**

El índice de figuras del PDF contiene 2.1–2.8, 4.1–4.5 y 5.1–5.10. De ellas, doce provienen del
flujo computacional y todas se derivan de artefactos que **no cambiaron** en esta reejecución:

| Figura del PDF | Archivo del flujo | Por qué no cambia |
|---|---|---|
| 2.7 (a) y (b) | `04_ale_A_001_001.png`, `04_ale_D_001_001.png` | ALE de la configuración principal, que sigue siendo CatBoost con SMOTE-NC |
| 4.3 | `eda2_matriz_correlacion_latinobarometro.png` | Se calcula sobre el entrenamiento (378.592), que no cambió |
| 4.4 | `eda2_matriz_correlacion_vdem.png` | Ídem, 360 observaciones país-año |
| 4.5 | `eda2_matriz_correlacion_merge.png` | Ídem, 28 variables |
| 5.1 | `eda1_target_split.png` | Distribución del target, sin cambios |
| 5.2 | `04_shap_bloques_CatBoost_smotenc.png` | SHAP de la configuración principal |
| 5.3 | `04_shap_bar_CatBoost_smotenc.png` | Ídem |
| 5.4 | `04_shap_beeswarm_CatBoost_smotenc.png` | Ídem |
| 5.5–5.9 | `04_ale_*.png` (5 archivos) | Ídem |
| 5.10 | `04_shap_concordancia_modelos.png` | Concordancia entre los tres modelos de gradient boosting, sin cambios |

Verifiqué además las etiquetas numéricas legibles en el propio PDF: la Figura 5.1 muestra
9,4/27,0/43,4/20,1 con n = 378.592, 9,3/17,6/44,0/29,0 con n = 17.219 y 11,3/21,1/42,0/25,5 con
n = 35.084; la Figura 5.3 lista las veinte variables en el mismo orden que la Tabla 5.16; y la
Figura 5.10 muestra 0,963 / 0,861 / 0,924, que son los valores reales redondeados. Las tres
coinciden con la corrida.

**Figuras disponibles y no usadas**, por si se adoptan las subsecciones de H4 y H5:
`05_shap_por_subregion.png` y `05_spearman_subregiones.png`.

**Ninguna figura del PDF corresponde a las Tablas 5.4–5.8, 5.10, 5.11 ni 5.13**, que son
precisamente las que cambian. Por eso la sección equivalente de la v1 de este plan quedó sin
objeto.

---

## 8. Estado del repositorio

### Aplicado y verificado

| Archivo | Cambio | Verificación |
|---|---|---|
| `utils/config.py` | Correlaciones de `VARS_EXCLUIR_LB` medidas sobre el entrenamiento exacto del flujo (378.592) y cardinalidad de `X_004` = 696 | Recalculado en esta sesión |
| `utils/config.py` | `MODELO_H2`, `MODELO_BASE_H2`, `MODELOS_GB_H2` en `PARAMETERS` | — |
| `README.md` | H2 con la formulación del PDF y apuntando a NB03 §7.2; NB03 descrito en §7.1 (OE3) y §7.2 (H2) | — |
| `notebooks/03_evaluacion_comparativa.ipynb` | §7.1 reetiquetada como respaldo del OE3; salida renombrada a `efecto_balanceo_clase_minoritaria.csv`; **§7.2 nueva** con el contraste de H2 | `h2_tabnet_vs_referencias.csv` generado, 4 filas, veredicto 1/2 cláusulas |
| `notebooks/05_estabilidad_temporal_regional.ipynb` | Persistencia de la matriz bloque × subregión | `shap_bloques_por_subregion.csv` generado, 6 × 5, reproduce los CV al centésimo |

La reejecución dejó `efecto_balanceo_clase_minoritaria.csv` **idéntico** a
`h2_balanceo_clase_minoritaria.csv`, lo que confirma que el renombrado no alteró el cálculo.

### Pendiente

- Borrar el huérfano `results/tables/h2_balanceo_clase_minoritaria.csv`, que ya no se regenera
  bajo ese nombre y cuyo prefijo apunta a una hipótesis que no es la de la tesis.

---

## 9. Decisiones resueltas

No quedan preguntas abiertas. Las tres que planteaba la versión anterior están cerradas:

| # | Pregunta | Resolución |
|---|---|---|
| P1 | Duración de la corrida | Tomada de las marcas "Inicio del programa" y "Fin del programa" de `notebooks/output/`: **12 h 15 min 54 s** en total, con el NB02 en 12 h 11 min 34 s (99,41 %). Aplicado en §5.3.1 de la sección 3, con el desglose por cuaderno. |
| P2 | Qué imágenes se actualizaron | ALE, SHAP y las matrices de correlación, todas tomadas de la última corrida. Eso cubre las Figuras 2.7, 4.3, 4.4, 4.5, 5.2, 5.3, 5.4, 5.5–5.9 y 5.10, es decir, **todas las figuras del documento que provienen del flujo**. Confirma la conclusión de la sección 7: no queda ninguna figura por regenerar. Las dos de subregión siguen disponibles por si se adoptan las subsecciones de H4 y H5. |
| P3 | Razón de exclusión de `C_003_003_011` | Señal predictiva baja sin justificación teórica que la compense, el mismo criterio que el PDF ya aplica a `X_008`. Redactado en la viñeta de §4.6 de la sección 2 y corregido en el comentario de `utils/config.py`. |

---

## 10. Resumen de escala

- **§4.6**: nueve valores de ρ, una cardinalidad, y un cambio estructural —el grupo de señal baja
  pasa de seis variables a cinco— con una viñeta nueva para `C_003_003_011`.
- **§4.9.4 y Tabla 4.14**: reestructuración completa por el rediseño de E2, cuidando de no perder
  el párrafo sobre la ponderación de TabNet.
- **Capítulo 5**: 22 valores cambian en las Tablas 5.4–5.8, 5.10 y 5.13, más dos filas completas
  de hiperparámetros; la Tabla 5.11 y §5.3.2 se reemplazan; tres valores en §5.1 y §5.3.3; siete
  correcciones de conteos y afirmaciones en §5.3.1 y §5.3.4.
- **Tablas nuevas**: 5.5b (criterio de selección), contraste de H2, efecto del balanceo (OE3),
  bloque × subregión, CV por bloque (H4) y Spearman entre subregiones (H5).
- **Subsecciones nuevas**: §5.5.5 (H4) y §5.5.6 (H5), con texto redactado.
- **Capítulo 6**: tres cambios en OE2, dos en OE3, la frase de apertura de §6.1, la reescritura
  completa del párrafo de H2, dos párrafos nuevos para H4 y H5, un valor en §6.2 y una limitación
  nueva en §6.3.
- **Figuras**: ninguna.
- **Repositorio**: todo aplicado; queda borrar un archivo huérfano.
