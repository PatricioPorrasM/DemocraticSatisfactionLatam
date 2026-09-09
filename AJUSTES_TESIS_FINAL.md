# Ajustes pendientes en el documento final de tesis

**Revisión:** 2026-09-07
**Base de comprobación:** corrida del servidor con GPU 2026-09-05/06 (`modo_ejecucion = real`),
carpetas `results/`, `models/`, `notebooks/output/` y código de `utils/`.
**Alcance:** se revisó el PDF completo (142 páginas) contra los artefactos. Cada punto
indica lo que dice el PDF, lo que dicen los artefactos y el archivo donde se comprueba.

Nivel de evidencia de cada ajuste:

| Nivel | Significado |
|---|---|
| **A** | La cifra correcta está escrita en un archivo de la corrida. Se abre y se lee. |
| **B** | Aritmética sobre valores de nivel A. Se muestra la operación. |
| **C** | Interpretación. No se propone como hallazgo, solo se marca lo que hay que retirar. |

**Segunda revisión (2026-09-07, tarde).** Tras la observación de que §4.9.1 sí contiene las
tablas de hiperparámetros, se reexaminó ese apartado y todo el capítulo 4. Resultado:

- Las **Tablas 4.9–4.12 son correctas**: los 32 rangos y escalas coinciden uno a uno con el
  `espacio_busqueda` registrado en los `models/hp_*.json`. El apartado A3 se reformuló para
  precisar qué es lo que falta (los valores óptimos, no los espacios de búsqueda).
- **Se retira una objeción anterior:** la multicolinealidad de las diecinueve variables V-Dem
  de §4.6 **sí es reproducible** y con las cifras exactas del PDF. Ver el apartado F1.
- **Dos hallazgos nuevos:** las dos cifras de poliarquía comparativa de §5.1 (A4) y las ocho
  exclusiones de variables del Latinobarómetro que §4.6 no documenta (B6).

Resumen de lo encontrado: **4 correcciones críticas**, **12 correcciones de cifra o
coherencia**, **5 afirmaciones sin respaldo en la corrida**, **3 refuerzos verificados** y
**1 objeción retirada**.

---

# BLOQUE A — Críticos (cambian una cifra reportada o una conclusión)

## 🔴 A1 · §5.2 — el test KS sigue con el p-valor equivocado

**Nivel A.** Este cambio estaba señalado y **no se aplicó**.

**PDF actual (pág. 101):**

> «La prueba de Kolmogorov–Smirnov entre los conjuntos de validación y prueba obtuvo un
> estadístico de 0,0549 y p = 0,787. En consecuencia, no se encontró evidencia suficiente
> para rechazar la igualdad de sus distribuciones.»

**Corrida** (`notebooks/output/02_preprocesamiento_entrenamiento.ipynb`, §10.7):

```
KS(val, test): estadístico=0.0549, p=0.0000
```

El estadístico coincide; el p-valor no. Y el valor del PDF es imposible: con
n₁ = 17.219 y n₂ = 35.084 el valor crítico al 5 % es
1,36·√((n₁+n₂)/(n₁·n₂)) ≈ 0,0127, muy por debajo de 0,0549. El p = 0,787 corresponde a
aplicar el test sobre los cuatro porcentajes de clase, no sobre las muestras.

**Texto de reemplazo:**

> La prueba de Kolmogorov–Smirnov entre las distribuciones del target en validación y en
> prueba arroja un estadístico de 0,0549 con p < 0,001. Con tamaños de muestra de 17.219 y
> 35.084 observaciones, el test detecta como significativa una diferencia de esa magnitud,
> de modo que las dos distribuciones no son idénticas: la validación concentra más masa en
> «Muy satisfecho» (29,0 % frente a 25,5 %) y menos en «No muy satisfecho» (17,6 % frente a
> 21,1 %). La magnitud del desplazamiento es, no obstante, moderada —la clase mayoritaria
> es la misma y su peso apenas varía (44,0 % frente a 42,0 %)— y no invalida el uso de 2020
> como conjunto de calibración, cuya elección responde a la restricción temporal descrita en
> §4.4.2 y no a un supuesto de igualdad distribucional.

**Y en el párrafo siguiente** (comentario de la Figura 5.1), sustituir «la distribución de
prueba conserva una estructura suficientemente próxima para evaluar el modelo» por «la
distribución de prueba difiere de la de validación en las categorías extremas, aunque
conserva la misma clase mayoritaria». §4.4.2 ya está bien: no apoya la elección de 2020 en
el KS.

## 🔴 A2 · §4.9.1 y §4.9.2 — la línea base ordinal usó 20 ensayos, no 50

**Nivel A.** `models/hp_OLO_*_ordinal_4clases.json` → `n_trials_optuna: 20` en las tres
estrategias. Confirmado también en `results/tables/hiperparametros_modelos.csv`:

| Modelo | Ensayos de Optuna |
|---|---|
| XGBoost, CatBoost, LightGBM | 50 |
| **OLO** | **20** |
| TabNet | 20 |

**PDF actual (§4.9.1, pág. 89):** «Se ejecutan 50 ensayos por configuración en la regresión
logística ordinal, XGBoost, CatBoost y LightGBM, y 20 ensayos en TabNet…»

**Texto de reemplazo:**

> Se ejecutan 50 ensayos por configuración en XGBoost, CatBoost y LightGBM, y 20 ensayos en
> la regresión logística ordinal y en TabNet. En la línea base ordinal el espacio de
> búsqueda es unidimensional —un único hiperparámetro de regularización—, de modo que un
> presupuesto menor es suficiente; en TabNet la reducción responde al costo por ensayo, que
> es sustancialmente mayor al entrenarse por épocas.

**Corregir el mismo error en §4.9.2** (pág. 92): «se ejecutan 50 ensayos de Optuna por
configuración para la regresión logística ordinal, XGBoost, CatBoost y LightGBM, y 20 para
TabNet» → misma redistribución.

## 🔴 A3 · Faltan los valores óptimos de cuatro de los cinco modelos

**Primero, lo que sí está y está bien.** §4.9.1 documenta los hiperparámetros de los cuatro
modelos no lineales con las Tablas 4.9–4.12, y de la línea base ordinal en prosa. **Los 33
rangos y escalas son correctos**: se compararon uno a uno con el campo `espacio_busqueda`
que la corrida registra en cada `models/hp_*.json` y no hay ni una discrepancia.

| Tabla | Modelo | Hiperparámetros | Coinciden con el código |
|---|---|---|---|
| 4.9 | XGBoost | 8 | 8 de 8 ✓ |
| 4.10 | CatBoost | 7 | 7 de 7 ✓ |
| 4.11 | LightGBM | 9 | 9 de 9 ✓ |
| 4.12 | TabNet | 8 | 8 de 8 ✓ |
| §4.9.1 (prosa) | OLO | 1 (`alpha` en [10⁻⁴, 10] logarítmico) | 1 de 1 ✓ |

**Lo que falta es otra cosa.** Esas tablas describen el **espacio de búsqueda** (los
intervalos dentro de los que Optuna exploró), no los **valores que resultaron
seleccionados**. Los valores óptimos solo se publican para CatBoost, en la Tabla 5.9 —cuyos
siete valores también son correctos, comprobados dígito a dígito contra
`models/hp_CatBoost_smotenc_ordinal_4clases.json`—. De OLO, XGBoost, LightGBM y TabNet el
documento **no dice qué configuración quedó seleccionada**.

Esto afecta a OE1 («base analítica integrada y reproducible»): con el espacio de búsqueda,
la semilla y el número de ensayos un revisor puede *repetir* la búsqueda, pero no puede
*reproducir directamente* los cuatro modelos ni comprobar que los valores caen dentro de los
intervalos declarados. La Tabla 5.9 sienta el precedente para CatBoost; conviene cerrar el
mismo círculo con los otros cuatro.

> **Comprobado además:** los 25 valores seleccionados caen todos dentro de los intervalos de
> las Tablas 4.9–4.12, y los tres discretos respetan su paso (TabNet `n_d`=16 y `n_a`=32 son
> múltiplos de 8; XGBoost `n_estimators`=400 y LightGBM `n_estimators`=600 son múltiplos de
> 100; CatBoost `iterations`=300 también). No hay ninguna inconsistencia entre el espacio
> declarado y el resultado obtenido.

**Insertar una tabla nueva** tras la Tabla 5.9, con los hiperparámetros seleccionados de las
configuraciones que la Tabla 5.6 reporta (la mejor estrategia de cada modelo). Fuente:
`results/tables/hiperparametros_modelos.csv`, columna `hp_optimizados`.

**Tabla 5.9b — Hiperparámetros seleccionados en validación para los cuatro modelos
restantes.**

| Modelo | Estrategia | Hiperparámetro | Valor óptimo |
|---|---|---|---|
| **OLO** | SMOTE-NC | alpha | 0,007459 |
| **XGBoost** | Pesos de clase | n_estimators | 400 |
| | | max_depth | 8 |
| | | learning_rate | 0,016104 |
| | | subsample | 0,705101 |
| | | colsample_bytree | 0,778709 |
| | | min_child_weight | 8 |
| | | reg_alpha | 0,009247 |
| | | reg_lambda | 8,038061 |
| **LightGBM** | Pesos de clase | n_estimators | 600 |
| | | num_leaves | 38 |
| | | max_depth | 6 |
| | | learning_rate | 0,026616 |
| | | subsample | 0,767667 |
| | | colsample_bytree | 0,759975 |
| | | reg_alpha | 0,020212 |
| | | reg_lambda | 0,016194 |
| | | min_child_samples | 43 |
| **TabNet** | Pesos de clase | n_d | 16 |
| | | n_a | 32 |
| | | n_steps | 6 |
| | | gamma | 1,024487 |
| | | lambda_sparse | 1,0223 × 10⁻⁶ |
| | | momentum | 0,161620 |
| | | mask_type | entmax |
| | | lr | 0,000649 |

**Pie:** «Búsqueda bayesiana TPE con semilla 42 y objetivo kappa cuadrático en validación;
50 ensayos en XGBoost y LightGBM, 20 en la regresión logística ordinal y en TabNet. Kappa
alcanzado en validación: 0,4804 (OLO), 0,4799 (XGBoost), 0,4797 (LightGBM) y 0,4783
(TabNet). Los registros completos, incluidas las quince configuraciones y los tres pliegues
temporales, están en `results/tables/hiperparametros_modelos.csv`.»

> **Atención a un detalle del pie.** Los dos archivos guardan cantidades distintas y en
> LightGBM **no coinciden**: `resultados_modelos.csv` registra el kappa de validación del
> ajuste final (0,4795636) y `hiperparametros_modelos.csv` el valor objetivo del mejor ensayo
> de Optuna (0,479724). La diferencia es de 1,6 × 10⁻⁴ y solo aparece en LightGBM; en los
> otros cuatro modelos los dos archivos son idénticos hasta el sexto decimal.
>
> | Modelo | Ajuste final (Tabla 5.4) | Objetivo del mejor ensayo | ¿Coinciden? |
> |---|---|---|---|
> | OLO [SMOTE-NC] | 0,4803524 | 0,480352 | sí |
> | XGBoost [pesos] | 0,4798943 | 0,479894 | sí |
> | CatBoost [SMOTE-NC] | 0,4839514 | 0,483951 | sí |
> | TabNet [pesos] | 0,4782915 | 0,478292 | sí |
> | **LightGBM [pesos]** | **0,4795636** | **0,479724** | **no** |
>
> **Citar siempre el valor del ajuste final**, que es el de la Tabla 5.4: 0,4804 (OLO),
> 0,4799 (XGBoost), **0,4796** (LightGBM) y 0,4783 (TabNet). No se ha determinado la causa de
> la diferencia en LightGBM y no hace falta: es dos órdenes de magnitud inferior al error
> estándar bootstrap (0,0193) y ambos valores redondean a la misma tercera cifra decimal. Lo
> que no debe hacerse es mezclar los dos archivos en la misma tabla.

## 🔴 A4 · §5.1 — las dos cifras de poliarquía comparativa no salen de la fuente

**Nivel A.** Hallazgo de la segunda revisión. Los valores de cada país son exactos; **los
promedios con los que se comparan, no**. Comprobado sobre `data/base/v-dem.csv` (V-Dem v16,
18 países, 1995–2024).

| Afirmación del PDF | Valor del país | Promedio citado | Promedio real | Veredicto |
|---|---|---|---|---|
| §5.1: «un índice de poliarquía electoral de 0,196 en 2024, frente a un promedio de **0,741** para los demás países analizados» | 0,1960 ✓ | 0,741 | **0,6341** | ✗ el 0,741 no corresponde a ninguna cantidad de la fuente |
| §5.1: «En 2020, su índice de poliarquía fue de 0,215 frente a un promedio de **0,610** en los demás países» (Nicaragua) | 0,2150 ✓ | 0,610 | **0,6330** | ⚠ el 0,610 es el promedio de los **18** países **incluyendo** Nicaragua (0,6098), no el de «los demás» |

Sobre el 0,741 se hizo una búsqueda sistemática: promedio de los 18, de los 17 sin
Venezuela, de los 16 sin Venezuela ni Nicaragua, mediana y máximo, para cada uno de los 30
años, y el promedio del período completo. **Ninguna de las 123 cantidades evaluadas da
0,741**; las más próximas son los promedios sin Venezuela ni Nicaragua de 2004 y 2005
(0,7147 y 0,7153). La cifra parece provenir de otra versión de V-Dem o de otro conjunto de
países.

**Texto de reemplazo para el pasaje de Venezuela:**

> En 2024, la categoría de máxima satisfacción todavía concentra el 54,5 % de las respuestas.
> Este patrón contrasta con un índice de poliarquía electoral de 0,196 en 2024, frente a un
> promedio de 0,634 para los diecisiete países restantes [14]. El descenso es sostenido: el
> índice pasa de 0,327 en 2013 a 0,233 en 2017 y 0,196 en 2024, mientras los demás países de
> la muestra se mantienen entre 0,63 y 0,67 en el mismo período.

**Texto de reemplazo para el pasaje de Nicaragua:**

> En 2020, su índice de poliarquía fue de 0,215 frente a un promedio de 0,633 en los
> diecisiete países restantes.

La serie de Venezuela que se cita arriba está verificada: 0,327 (2013) · 0,312 (2014) ·
0,300 (2015) · 0,293 (2016) · 0,233 (2017) · 0,213 (2018) · 0,206 (2020 y 2023) · 0,196
(2024). Y el promedio de los otros diecisiete se mantiene entre 0,6335 y 0,6676 de 2017 a
2024, lo que sostiene la afirmación de contraste con más fuerza que un único año.

---

# BLOQUE B — Correcciones de cifra

## 🟡 B1 · §5.1 — el missingness del 6,9 % no es el de la matriz de modelado

**Nivel A + B.** El PDF cierra §5.1 con «El *missingness* global del conjunto resultante es
del 6,9 %», justo después de la Tabla 5.1, que describe la muestra modelada. Las dos cifras
no se refieren a lo mismo:

| Cifra | Sobre qué se calcula | Fuente |
|---|---|---|
| **6,9 %** | Las 51 columnas del dataset tras las exclusiones, antes de la selección de variables | NB02 celda 27: `Missingness global: 6.9%` |
| **12,84 %** | Las **28 variables predictoras** que entran a los modelos, sobre los 430.895 registros | calculado sobre `data/processed/{train,val,test}.parquet` |
| **13,8 % / 6,3 % / 5,3 %** | Las 28 predictoras, por conjunto (train / validación / prueba) | NB02 celda 52: `NaN train: 13.8% \| NaN val: 6.3% \| NaN test: 5.3%` |

**Texto de reemplazo:**

> Sobre las veintiocho variables predictoras que entran efectivamente a los modelos, la
> proporción de valores ausentes es del 13,8 % en entrenamiento, el 6,3 % en validación y el
> 5,3 % en prueba. La concentración es muy desigual entre variables: en el conjunto de prueba,
> la confianza en la televisión y el conocimiento de un caso de corrupción superan ambas el
> 50 % de ausentes, mientras que la situación económica del país no llega al 0,4 %. Este
> patrón justifica el tratamiento diferenciado descrito en §4.8.

## 🟡 B2 · §5.3.2 — distribución de la variante binaria en prueba

**Nivel A.** El PDF dice «el conjunto de prueba contiene un 32,2 % de personas
insatisfechas y un 67,8 % de satisfechas».

Calculado sobre `data/processed/test.parquet` con el umbral de la Tabla 4.14
(insatisfecho = clases 0 y 1):

| Conjunto | Insatisfechos | Satisfechos | Razón |
|---|---|---|---|
| Entrenamiento | 138.058 (36,47 %) | 240.534 (63,53 %) | 1,74 |
| **Prueba** | **11.383 (32,44 %)** | **23.701 (67,56 %)** | **2,08** |

Los conteos de entrenamiento del PDF (138.058 y 240.534) son correctos y la razón de 1,7
también. Corregir los dos porcentajes de prueba a **32,4 %** y **67,6 %**, y considerar
añadir que la razón en prueba sube a 2,1: es un dato relevante porque explica parte de la
diferencia de F1 entre conjuntos.

## 🟡 B3 · Figuras 5.2 y 5.4 — el pie dice «pesos de clase» y la figura dice «smotenc»

**Nivel A.** Contradicción visible dentro de la misma página.

| Elemento | Pie / texto del PDF | Título dentro de la imagen | Archivo |
|---|---|---|---|
| Figura 5.2 | «Importancia SHAP global de **CatBoost** agregada por bloque temático» y el texto la introduce como «para CatBoost entrenado con **pesos de clase**» | `Importancia por bloque temático — CatBoost · smotenc` | `04_shap_bloques_CatBoost_smotenc.png` |
| Figura 5.4 | «Valores SHAP para el modelo CatBoost entrenado con **pesos de clase**» | `SHAP Beeswarm — CatBoost · smotenc` | `04_shap_beeswarm_CatBoost_smotenc.png` |

La configuración principal es **SMOTE-NC**. Sustituir «entrenado con pesos de clase» por
«entrenado con SMOTE-NC» en los dos pies y en la frase de §5.5.1 que introduce la Figura
5.2. Es el residuo más visible de la corrida anterior.

## 🟡 B4 · Figura 5.3 — el pie dice quince variables y la figura muestra veinte

**Nivel A.** `04_shap_bar_CatBoost_smotenc.png` grafica las veinte variables con
importancia no nula: desde «Apoyo a la democracia» hasta «Integridad institucional
(corrupción)». El pie actual («Quince variables con mayor importancia SHAP media») y la
frase que la introduce («La Figura 5.3 muestra las quince variables individuales con mayor
importancia SHAP media») describen otra figura.

**Pie de reemplazo:** «Veinte variables con importancia SHAP media no nula en el modelo
CatBoost con SMOTE-NC. Las ocho restantes reciben una contribución exactamente igual a
cero.»

## 🟡 B5 · Tabla 5.15 — el texto habla de veinte variables y la tabla tiene quince

**Nivel A.** El párrafo de §5.5.1 afirma que «la amplitud media del intervalo de rango en
las **veinte** variables principales es de 0,4 posiciones» y cita como par solapado
«expectativa económica personal con democracia electoral» — que son las posiciones **16 y
17**, ausentes de la Tabla 5.15. El lector no puede verificar ni la media ni ese par.

Dos salidas, cualquiera sirve:

1. **Ampliar la Tabla 5.15 a veinte filas.** Los cinco valores que faltan, leídos de
   `results/tables/shap_importancia_ic_CatBoost_smotenc.csv`:

   | # | Variable | \|SHAP\| | IC 95 % | Rango (IC) | % top-5 |
   |---|---|---|---|---|---|
   | 16 | Expectativa económica personal | 0,0073 | [0,0066; 0,0081] | 16 (16–17) | 0 |
   | 17 | Democracia electoral | 0,00695 | [0,0062; 0,0081] | 17 (16–17) | 0 |
   | 18 | Sexo | 0,0034 | [0,0034; 0,0034] | 18 (18–18) | 0 |
   | 19 | Igualdad ante la ley | 0,0013 | [0,0011; 0,0015] | 19 (19–19) | 0 |
   | 20 | Integridad institucional (corrupción) | 0,0008 | [0,0006; 0,0011] | 20 (20–20) | 0 |

   Con esto el pie de la tabla pasa a «Veinte variables principales…» y la amplitud media de
   0,4 posiciones queda comprobable (suma de amplitudes = 8 sobre 20 filas).

2. **Dejar quince filas** y cambiar «las veinte variables principales» por «las quince
   variables de la Tabla 5.15», recalculando la amplitud media sobre ellas y retirando de
   la lista de pares solapados el de expectativa económica personal con democracia
   electoral.

La opción 1 es preferible: sostiene además la afirmación de OE4 sobre las ocho variables de
contribución nula.

## 🟡 B6 · §4.6 — se documentan cuatro exclusiones de doce

**Nivel A.** Hallazgo de la segunda revisión. §4.6 dice:

> «**cuatro** variables con |ρ| < 0,05 respecto de la variable objetivo y sin justificación
> teórica sustantiva se descartaron por combinación de señal baja y ausencia de fundamento
> conceptual (confianza en la Iglesia Católica, preocupación por el desempleo, escala
> izquierda-derecha y tamaño del municipio).»

Hay dos problemas, y el segundo es el importante.

**Primero: la lista tiene doce variables, no cuatro.** `VARS_EXCLUIR_LB` de `utils/config.py`
contiene doce códigos, y el NB02 los imprime en la celda 23 (`Variables LB excluidas: [...]`).

**Segundo: el criterio declarado no explica la mitad de las exclusiones, y una de las cuatro
variables que el PDF nombra no lo cumple.** Se recalculó la correlación de Spearman de las
doce con la variable objetivo, exactamente como §4.6 declara —solo sobre las 21 olas de
entrenamiento, sobre `data/base/latinobarometro.csv`—:

| Código | Etiqueta | ρ | \|ρ\| < 0,05 | ¿La nombra §4.6? |
|---|---|---|---|---|
| `A_007_071` | Escala izquierda-derecha | −0,0082 | sí | **sí** |
| `S_700` | Religión | −0,0077 | sí | no |
| `S_701` | Práctica religiosa | +0,0233 | sí | no |
| `C_001_031` | Problema más importante del país | +0,0395 | sí | no |
| `H_002_101` | Confianza en la Iglesia | +0,0411 | sí | **sí** |
| `C_003_003_011` | Preocupación por quedar sin trabajo | −0,0442 | sí | **sí** |
| `X_004` | Región / área geográfica | −0,0520 | **no** | no |
| `D_001_131` | Ingreso subjetivo | +0,0850 | **no** | no |
| **`X_008`** | **Tamaño de la ciudad** | **+0,0971** | **no** | **sí** |
| `H_001_011` | Confianza interpersonal | +0,1214 | **no** | no |
| `D_001_061` | Situación económica personal actual | +0,1846 | **no** | no |
| `A_003_021` | Escala de desarrollo de la democracia | −0,2286 | **no** | no |

Dos consecuencias:

1. **`X_008` (tamaño de la ciudad) tiene ρ = 0,0971**, casi el doble del umbral, y §4.6 la
   presenta como uno de los cuatro casos de «señal baja». Ese ejemplo hay que sustituirlo o
   la justificación hay que cambiarla. Las otras tres que el PDF nombra sí cumplen
   (−0,0082, +0,0411, −0,0442).
2. **Seis de las doce se excluyeron por razones que el documento no da.** Tres de ellas
   tienen correlaciones sustantivas y un lector de ciencia política preguntará por las tres:
   - **`A_003_021` (escala de desarrollo de la democracia), ρ = −0,2286.** Es la más
     delicada: mide una valoración del régimen conceptualmente solapada con la satisfacción
     declarada. Excluirla es probablemente lo correcto —incluirla arriesgaría
     circularidad—, pero **es la variable individual con mayor correlación absoluta de todo
     el conjunto candidato descartado**, y si no se justifica, la omisión se lee como
     conveniencia.
   - **`D_001_061` (situación económica personal actual), ρ = +0,1846.** El bloque económico
     conserva cinco variables, incluida la *expectativa* económica personal (`D_001_091`,
     ρ = −0,1880 según la Tabla 5.3), pero no la situación personal *actual*, cuya
     correlación es prácticamente idéntica en magnitud.
   - **`H_001_011` (confianza interpersonal), ρ = +0,1214.** Predictor clásico de la
     literatura de capital social y apoyo democrático.

**Qué hacer** — sustituir la frase de §4.6 por una que cubra las doce, cite las
correlaciones reales y separe los criterios. Estructura sugerida:

> El proceso descartó doce variables del Latinobarómetro. Seis se excluyeron por señal baja
> (|ρ| < 0,05 con la variable objetivo en el conjunto de entrenamiento): la escala
> izquierda-derecha (−0,0082), la religión (−0,0077), la práctica religiosa (+0,0233), el
> problema más importante del país (+0,0395), la confianza en la Iglesia (+0,0411) y la
> preocupación por quedar sin trabajo (−0,0442). Las seis restantes presentan correlaciones
> superiores al umbral y se descartaron por otros motivos: [MOTIVO] en el caso de la escala
> de desarrollo de la democracia (−0,2286), [MOTIVO] en la situación económica personal
> actual (+0,1846) y el ingreso subjetivo (+0,0850), [MOTIVO] en la confianza interpersonal
> (+0,1214) y [MOTIVO] en la región geográfica (−0,0520) y el tamaño de la ciudad (+0,0971).

> ⚠️ **Los `[MOTIVO]` los tiene que completar el autor; no los inventes ni los tomes de aquí.**
> Ningún artefacto de la corrida registra la razón de cada exclusión: `utils/config.py`
> guarda la lista pero no el motivo. Y hay que descartar dos justificaciones que podrían
> parecer naturales pero que los datos **no** sostienen:
>
> - **No fue la disponibilidad longitudinal.** El criterio de §4.6 de «al menos 15 de las 24
>   olas» no excluyó a ninguna variable: las doce descartadas y las veinticuatro conservadas
>   aparecen **todas en las 24 olas** (comprobado sobre
>   `data/base/lb_frecuencia_valores_por_ola.csv`). Ese criterio está bien declarado como
>   requisito de elegibilidad, pero no discriminó en la práctica y no puede usarse para
>   explicar ninguna exclusión concreta.
> - **No fue la cardinalidad** en la mayoría de los casos: la religión, la práctica religiosa
>   y el tamaño de la ciudad son de cardinalidad baja o media.
>
> La justificación defendible de las seis restantes es conceptual —circularidad con el
> target, redundancia con variables conservadas del mismo bloque, o no comparabilidad de la
> categorización entre países— y el autor es quien puede afirmar cuál se aplicó a cada una.

---

# BLOQUE C — Afirmaciones que ningún artefacto respalda

## 🔴 C1 · §5.3.2 y Tabla 5.10 — la comparación con Rosa et al. está invertida

**Nivel B sobre el propio PDF.** Este es el hallazgo más importante del bloque, y **corrige
la conclusión en favor de la tesis**.

La Tabla 5.10 compara la columna **«Binaria (F1 macro)»** de los cinco modelos contra la
fila «Rosa et al. (2023) — 0,84–0,85», y §5.3.2 concluye:

> «Los valores permanecen por debajo del intervalo de 0,84–0,85 que Rosa et al. reportan
> para Brasil…»

Pero el propio documento, en la Tabla 3.1 y en §3.3, dice qué son esas cifras:

> «el F1 de la clase insatisfecha alcanzó el 84 %, 85 % y 84 %, pero para la clase satisfecha
> descendió al 55 %, 46 % y 57 %.»

El 0,84–0,85 es el **F1 de una sola clase**, no el F1 macro. El F1 macro de Rosa et al. se
obtiene promediando las dos clases:

| Modelo de Rosa et al. | F1 insatisfechos | F1 satisfechos | **F1 macro** |
|---|---|---|---|
| SVC | 0,84 | 0,55 | **0,695** |
| Random Forest | 0,85 | 0,46 | **0,655** |
| ANN / MLP | 0,84 | 0,57 | **0,705** |

Frente a los 0,7278–0,7385 de esta tesis. **La comparación se invierte: los cinco modelos de
este trabajo superan a la referencia en F1 macro**, no quedan por debajo. Y también en
AUROC (0,8199–0,8273 frente a 0,74–0,77) y en exactitud (0,7483–0,7644 frente a
0,74–0,76).

**Qué hacer:**

1. **Tabla 5.10 — corregir la fila de referencia** para que compare lo comparable:

   | Modelo | Ordinal de 4 clases (κw) | Binaria (F1 macro) | Binaria (accuracy) | Binaria (AUROC) |
   |---|---|---|---|---|
   | … | … | … | … | … |
   | Rosa et al. (2023) [8] | — | 0,655–0,705 | 0,74–0,76 | 0,74–0,77 |

   **Pie:** «El F1 macro de Rosa et al. se obtiene promediando el F1 por clase que reportan
   (0,84–0,85 en la clase insatisfecha y 0,46–0,57 en la satisfecha; véase la Tabla 3.1). La
   comparación no es estricta: aquel trabajo evalúa 872 casos de una sola ola de un solo
   país con validación cruzada aleatoria, mientras que este evalúa 35.084 casos de 16 países
   en olas posteriores a las de entrenamiento.»

2. **§5.3.2 — texto de reemplazo del párrafo:**

   > En la variante binaria, el conjunto de prueba contiene un 32,4 % de personas
   > insatisfechas y un 67,6 % de satisfechas; en el conjunto de entrenamiento la razón es de
   > 1,7 (138.058 frente a 240.534 registros). TabNet obtiene el mayor F1 macro (0,7385) y el
   > AUROC más alto (0,8273), seguido de cerca por CatBoost (0,7381 y 0,8246); las cinco
   > configuraciones quedan en un rango de 0,0107 puntos de F1. Los cinco modelos superan el
   > F1 macro de la referencia de Rosa et al., situado entre 0,655 y 0,705 al promediar el F1
   > de sus dos clases, así como su AUROC (0,74–0,77), y alcanzan una exactitud comparable o
   > ligeramente superior (0,7483–0,7644 frente a 0,74–0,76). La comparación debe leerse con
   > cautela porque los diseños de evaluación difieren de forma sustantiva: aquel trabajo
   > analiza 872 casos de una sola ola de Brasil con validación cruzada aleatoria, mientras
   > que este predice 35.084 casos de dieciséis países en olas posteriores a las de
   > entrenamiento. La diferencia más informativa no es la magnitud sino la naturaleza de la
   > tarea: el desequilibrio entre clases de Rosa et al. penaliza fuertemente su clase
   > minoritaria (F1 de 0,46 a 0,57), mientras que aquí la brecha entre las dos clases
   > binarias es sensiblemente menor.

3. **Verificar la fila en la fuente original antes de publicar.** Las cifras de la Tabla 3.1
   se transcribieron del artículo; conviene reconfirmar en él que 0,84–0,85 corresponde a la
   clase insatisfecha y que las exactitudes son 74–76 %. Todo el razonamiento anterior
   descansa en esa lectura.

## 🔴 C2 · §5.2 — el coeficiente de la regresión ordinal no está en ningún archivo

**Nivel A (por ausencia).** §5.2 afirma:

> «En cambio, v2x_corr, cuya escala aumenta con la corrupción, registra una asociación
> positiva de ρ = 0,1888 a nivel país-año y **un coeficiente positivo en la regresión
> logística ordinal**.»

La primera mitad es verificable (`eda_correlaciones_features.csv`, columna `r_pais_año`).
La segunda no: `models/hp_OLO_smotenc_ordinal_4clases.json` guarda el número de
coeficientes (`n_coeficientes: 28`) y los tres umbrales
(`umbrales_theta: [−0,407294; −0,0442; +0,396951]`), **pero no los coeficientes**. No existe
ningún artefacto con el signo de β para v2x_corr.

**Dos salidas:**

- **Retirar la mención al coeficiente** y dejar solo la correlación, que sí se sostiene:
  «…registra una asociación positiva de ρ = 0,1888 a nivel país-año».
- **O exportar los coeficientes** de `clf.coef_` en `entrenar_olo` y añadir una tabla en
  anexo. Es una línea de código, pero obliga a reejecutar el NB02 completo.

Recomendación: retirar la mención. La afirmación sustantiva —que la asociación con la
corrupción es positiva— ya queda respaldada por la correlación.

## 🔴 C3 · §4.3.4 — los errores estándar agrupados por país no se calculan

**Nivel A (por ausencia).** §4.3.4 afirma:

> «Para la regresión logística ordinal **se utilizan** errores estándar agrupados por país
> como diagnóstico de la magnitud de este efecto [49, 50].»

No hay ningún artefacto de errores estándar agrupados en `results/`. Lo que la corrida sí
hace es un **bootstrap de clústeres país-año sobre el conjunto de prueba**, aplicado a los
cinco modelos por igual (Tablas 5.7 y 5.8), que es un procedimiento distinto y se documenta
correctamente en §5.3.1.

**Texto de reemplazo:**

> Para acotar el efecto de esta dependencia sobre la incertidumbre de las métricas, la
> comparación entre configuraciones se realiza mediante un bootstrap que remuestrea los
> clústeres país-año completos en lugar de los registros individuales (véase §5.3.1);
> remuestrear observaciones individuales subestimaría el error estándar (Moulton, 1990;
> Cameron y Miller, 2015). Para los modelos de árboles, la validación temporal garantiza
> además que ningún par país-año aparezca simultáneamente en entrenamiento y prueba.

El párrafo de §2.6 que menciona los errores estándar optimistas es correcto como afirmación
general de la literatura y no hay que tocarlo.

## 🟡 C4 · §1.1 — la atribución a Ergun et al. no coincide con §3.2

**Coherencia interna.** §1.1 afirma:

> «Los estudios de Ergun et al. **[7]** y Rosa et al. [8] han demostrado que los modelos de
> *gradient boosting* y las redes neuronales superan sistemáticamente a la regresión
> logística cuando se aplican a variables actitudinales de encuestas…»

Pero §3.2 describe el trabajo de Ergun et al. **[52]** exactamente al contrario:

> «Por ello, el estudio corresponde a una regresión ordinal y no a una tarea de
> clasificación predictiva; los autores estiman distintas especificaciones de logit ordenado
> multinivel. […] En lugar de métricas como accuracy, F1 o AUC, el artículo reporta
> coeficientes, errores estándar robustos y significancia estadística.»

Son dos entradas bibliográficas distintas de los mismos autores: [7] («Machine learning
approaches to predict voting behavior and political attitudes», *Journal of Information
Technology & Politics*, 2019) y [52] («Satisfaction with democracy in Latin America»,
*Desarrollo y Sociedad*, 2019). Es posible que ambas existan, pero **hay que verificar [7]
en la fuente** y comprobar que sostiene la afirmación de §1.1. Si [7] no existe o no
contiene ese resultado, la frase de §1.1 debe atribuirse solo a Rosa et al. [8] y a los
antecedentes de la Tabla 3.1 que sí aplican aprendizaje automático (Tauil et al. [54],
Ferreyra et al. [53]).

## 🟡 C5 · Referencia [55] — el apellido citado no coincide con el de la bibliografía

**Coherencia interna.** El texto cita «Tipanluisa y Aguinda [55]» en §3.2, §3.5 y la Tabla
3.1, mientras que la bibliografía registra «E. A. T. Naranjo and D. G. A. Salazar». Se trata
del mismo trabajo con los apellidos invertidos en el campo `author` del BibTeX (nombres
compuestos ecuatorianos: Tipanluisa Naranjo y Aguinda Salazar). Corregir el BibTeX a
`{Tipanluisa Naranjo}, E. A. and {Aguinda Salazar}, D. G.` para que la cita y la entrada
coincidan.

---

# BLOQUE D — Coherencia interna

## 🟡 D1 · §2.4 — «cinco métricas» frente a una tabla de ocho

El texto que introduce la Tabla 2.1 dice: «La Tabla 2.1 resume la función de las **cinco**
métricas utilizadas en esta investigación». La tabla describe **ocho**: accuracy, F1 macro,
MAE ordinal, kappa cuadrático, AUROC OvR macro, balanced accuracy, F1 ponderado y kappa
lineal. Y son ocho exactamente las que calcula la corrida
(`results/resultados_modelos.csv`: `accuracy, balanced_accuracy, f1_macro, f1_weighted,
kappa_lineal, kappa_cuadratico, mae_ordinal, auroc_macro`).

Cambiar «cinco» por «ocho».

## 🟡 D2 · §4.10 — enumera cinco métricas pero la Tabla 5.6 reporta seis

§4.10 dice: «Todas las configuraciones se evalúan sobre el mismo conjunto de prueba mediante
kappa cuadrático, accuracy, F1 macro, MAE ordinal y AUROC OvR macro». La Tabla 5.6 incluye
además **exactitud balanceada**, y el párrafo que la sigue construye sobre ella un argumento
central («la tensión entre exactitud y exactitud balanceada»).

**Texto de reemplazo:** «…mediante kappa cuadrático, accuracy, F1 macro, MAE ordinal, AUROC
OvR macro y exactitud balanceada. El registro completo incluye además F1 ponderado y kappa
lineal, que se conservan en los resultados por trazabilidad y no se discuten en el texto.»

## 🟡 D3 · La Tabla 5.7 no se referencia en ningún punto del texto

Entre el final del párrafo de la Tabla 5.6 («…a costa del acierto agregado») y el párrafo
que introduce el bootstrap («Para evitar atribuir superioridad a diferencias puramente
descriptivas…») aparecen las Tablas 5.7 y 5.8 seguidas. La 5.8 se referencia después
(«matiza la lectura de la Tabla 5.5» y, en §5.5.4, «Tabla 5.8»), pero **la Tabla 5.7 no se
menciona en ninguna parte del documento**.

**Frase de enlace a insertar antes de la Tabla 5.7:**

> Las estimaciones puntuales de la Tabla 5.5 no llevan asociada una medida de su
> incertidumbre. La Tabla 5.7 la añade para las quince configuraciones mediante un bootstrap
> de 1.000 repeticiones sobre los 32 clústeres país-año del conjunto de prueba, y la Tabla
> 5.8 traduce esa incertidumbre a comparaciones pareadas contra la configuración principal.

## 🟡 D4 · La Figura 2.7 son dos de las curvas del capítulo de resultados

La Figura 2.7 del marco teórico ilustra la lectura de gráficos ALE con «Apoyo a la
democracia» y «Situación económica del país». Son **exactamente** las Figuras 5.5 y 5.8:
mismos archivos (`04_ale_A_001_001.png` y `04_ale_D_001_001.png`), mismos valores
(−0,0087 → 0,0050 con «Mayor cambio: 1.00»; máximo 0,03 en el código 3 con «Mayor cambio:
2.00»). Sin embargo el texto afirma:

> «El objetivo de esta comparación no es adelantar la discusión de resultados, sino mostrar
> cómo ALE permite reconocer umbrales, mesetas y cambios de dirección…»

La afirmación no se sostiene: la figura **es** el resultado, presentado ochenta páginas
antes. Tres salidas, en orden de preferencia:

1. **Declararlo.** Cambiar la frase por: «Las dos curvas provienen del modelo seleccionado y
   se retoman con su interpretación sustantiva en §5.5.2 (Figuras 5.5 y 5.8); aquí se usan
   únicamente para fijar las convenciones de lectura.» Es honesto y no cuesta nada.
2. Sustituir el ejemplo por dos de las tres curvas que no se repiten (confianza en los
   partidos políticos, distribución del ingreso o confianza en el Gobierno) y declararlo
   igualmente.
3. Construir una figura sintética con datos simulados y decirlo en el pie.

## 🟡 D5 · Resumen y Abstract — ALE ya no cubre solo confianza

El Resumen dice «ALE mostró relaciones no lineales en las variables de confianza» y el
Abstract «ALE revealed nonlinear relationships in trust variables». Pero las cinco curvas
publicadas cubren tres bloques, y el Capítulo 6 (OE4) lo dice bien: «Las curvas ALE
confirmaron que los efectos de la confianza, la evaluación económica y el apoyo a la
democracia no son estrictamente lineales».

**Resumen:** «ALE mostró relaciones no lineales en la confianza institucional, la evaluación
económica y el apoyo a la democracia».
**Abstract:** «ALE revealed nonlinear relationships in institutional trust, economic
evaluations, and support for democracy».

## 🟡 D6 · §4.10 — el conjunto de validación también es `eval_set` de XGBoost y CatBoost

§4.10 afirma: «El conjunto de validación cumple dos funciones simultáneas: es el `eval_set`
del *early stopping* de LightGBM (50 rondas) y de TabNet (paciencia 20)…».

Leído en `utils/models.py`, el conjunto de validación se pasa como `eval_set` **a los cuatro
modelos**: XGBoost (`eval_set=[(X_val, y_val)]`), CatBoost (`eval_set=pool_val`), LightGBM y
TabNet. La diferencia es que solo en los dos últimos activa una parada temprana; el registro
de XGBoost lo declara explícitamente (`"early_stopping": "no (n_estimators lo fija Optuna)"`)
y CatBoost no fija `od_wait`.

**Texto de reemplazo:** «El conjunto de validación cumple dos funciones simultáneas: es el
criterio de selección de hiperparámetros y de configuración, y es el conjunto de evaluación
que reciben los cuatro modelos no lineales durante el ajuste. Solo en LightGBM (50 rondas) y
TabNet (paciencia 20) activa una parada temprana; en XGBoost el número de árboles lo fija
Optuna y en CatBoost no se configura criterio de detención, de modo que en esos dos casos el
conjunto interviene únicamente como registro de seguimiento. Esto introduce un sesgo
optimista sobre la validación, no sobre la prueba, que permanece reservada.»

## 🟢 D7 · Encabezados «N estimado» en las Tablas 4.6 y 5.2

Las dos tablas rotulan la columna de tamaño como «N estimado», pero los valores son conteos
exactos leídos de los archivos (398.876 / 20.204 / 38.419 en la Tabla 4.6 y 378.592 / 17.219
/ 35.084 en la Tabla 5.2). Cambiar el encabezado a «N» o «Registros». Es cosmético, pero
«estimado» invita a una pregunta innecesaria.

## 🟢 D8 · Abstract — separadores decimales mezclados

En la misma oración del Abstract: «quadratic weighted kappa (κw = 0,5386) and the lowest
ordinal MAE (0.5688)». Unificar a punto decimal en todo el texto en inglés: `0.5386` y
`0.5688`. El Resumen en español usa coma de forma consistente y está bien.

---

# BLOQUE E — Refuerzos verificados que conviene incorporar

Nada de esto corrige un error; los tres puntos añaden respaldo comprobable a
afirmaciones que ya están en el documento y se adelantan a objeciones previsibles.

## 🆕 E1 · Las ocho variables de contribución nula no se explican por los valores ausentes

El Capítulo 6 (OE4) afirma que «el modelo utiliza efectivamente veinte de las veintiocho
variables». La objeción inmediata de un revisor es que esas ocho quizá tengan demasiados
ausentes. Los datos la descartan: **cuatro de las ocho tienen menos del 1,4 % de valores
ausentes** en el conjunto completo.

| Variable con \|SHAP\| = 0 | % de ausentes (global) |
|---|---|
| Edad | 0,01 % |
| Nivel socioeconómico | 0,36 % |
| Situación ocupacional | 0,49 % |
| Confianza en la Policía | 1,32 % |
| Nivel educativo | 5,21 % |
| Confianza en las Fuerzas Armadas | 10,61 % |
| Confianza en la televisión | 22,66 % |
| Interés en política | 35,93 % |

Y a la inversa: variables con importancia alta tienen ausencias mucho mayores —«País para
todos o para los poderosos» (posición 5) tiene el 32,9 % y «Progreso contra la corrupción»
(posición 9) el 40,8 %—. Fuente: `data/processed/nan_audit.json` y los tres `.parquet`.

**Frase para §5.5.1 o para la limitación octava de §6.2:**

> La contribución nula de estas ocho variables no se explica por la disponibilidad del dato:
> cuatro de ellas presentan menos del 1,4 % de valores ausentes, mientras que dos de las diez
> variables más importantes superan el 32 %. El modelo prescinde de ellas por su escaso
> aporte predictivo y no por falta de información.

## 🆕 E2 · La estabilidad temporal se sostiene mejor con presupuesto homogéneo

La Tabla 5.12 calcula la desviación estándar sobre los cuatro cortes, y uno de ellos —el
definitivo— se ajustó con 50 ensayos de Optuna frente a 15 en los otros tres. Un revisor
puede objetar que la mezcla contamina la medida. Conviene adelantarse, porque al restringirla
a los tres pliegues homogéneos **la separación se agranda**:

| Modelo | sd sobre los 4 cortes | sd sobre los 3 pliegues (15 ensayos) | κw medio de los 3 |
|---|---|---|---|
| XGBoost | 0,0320 | **0,0022** | 0,4701 |
| LightGBM | 0,0331 | **0,0032** | 0,4659 |
| CatBoost | 0,0381 | **0,0172** | 0,4665 |
| TabNet | 0,0763 | **0,0668** | 0,4307 |
| OLO | 0,0821 | **0,0783** | 0,4248 |

Con presupuesto homogéneo, la mayor dispersión de un árbol (0,0172) es cuatro veces menor
que la menor de los otros dos modelos (0,0668): los grupos no se solapan. Calculado sobre
`results/tables/validacion_temporal_folds.csv`.

**Frase para §5.3.4:** «La desviación estándar de la Tabla 5.12 se calcula sobre los cuatro
cortes, uno de los cuales dispuso de 50 ensayos de Optuna frente a 15 en los tres pliegues
históricos. Restringida a los tres pliegues con presupuesto homogéneo, la separación entre
familias se acentúa: los árboles quedan entre 0,0022 y 0,0172 y los otros dos modelos en
0,0668 y 0,0783, sin solapamiento entre grupos.»

## 🆕 E3 · Pie de la Tabla 5.9 — declarar el entrenamiento en GPU

El pie actual termina en «Función de pérdida `MultiClass`». El registro
`models/hp_CatBoost_smotenc_ordinal_4clases.json` documenta además `task_type: GPU`.
Añadirlo cierra el círculo con la Tabla 4.1, que documenta la RTX 4090, y es un dato de
reproducibilidad que un revisor comprobaría: «Función de pérdida `MultiClass`, entrenamiento
en GPU (`task_type = GPU`)».

---

# BLOQUE F — Objeción retirada

## ✅ F1 · §4.6 — la multicolinealidad de las diecinueve variables V-Dem **sí es reproducible**

En la revisión anterior de este proyecto se marcó este pasaje como «proveniente de un
diagnóstico previo que el flujo actual ya no reejecuta», con la recomendación de declararlo
como tal. **Esa objeción era incorrecta y queda retirada.** El pasaje es reproducible desde
los datos de la entrega y con las cifras exactas que el PDF cita.

**PDF actual (§4.6):**

> «En la especificación inicial de diecinueve variables V-Dem se identificaron 46 pares con
> correlaciones de Spearman superiores a 0,85 y una correlación máxima de 0,990. Al conservar
> los cuatro indicadores representados en la Figura 4.4, el número de pares por encima de
> 0,85 se redujo a uno y la correlación máxima mostrada es de aproximadamente 0,86.»

**Comprobación** sobre `data/base/v-dem.csv` (540 observaciones país-año, 1995–2024):

| Afirmación | Valor del PDF | Valor calculado | Veredicto |
|---|---|---|---|
| Variables de la especificación inicial | 19 | 23 indicadores `v2*` cargados − 4 excluidos por `VARS_EXCLUIR_VDEM` = **19** | ✓ |
| Pares con \|ρ\| > 0,85 sobre las 19 | 46 | **46** de 171 pares | ✓ exacto |
| Correlación máxima | 0,990 | **0,9901** (`v2x_libdem` con `v2x_accountability_osp`) | ✓ exacto |
| Pares > 0,85 al conservar cuatro | 1 | **1** | ✓ |
| Máximo al conservar cuatro | ≈ 0,86 | **0,8641** sobre 540 país-año · **0,8552** sobre los 360 de entrenamiento | ✓ |

Las cuatro variables excluidas de la especificación inicial son `v2x_neopat`,
`v2xnp_regcorr`, `v2xpe_exlsocgr` y `v2xpe_exlecon`.

**Único ajuste recomendado, y es menor:** declarar la base de cálculo, porque el 46 y el
0,990 se obtienen sobre las 540 observaciones país-año del período completo, mientras que la
Figura 4.4 se calcula sobre las 360 del conjunto de entrenamiento. Añadir al final de la
frase: «…calculadas sobre las 540 observaciones país-año del período 1995–2024; la Figura
4.4, restringida al conjunto de entrenamiento, muestra un máximo de 0,8552 entre los cuatro
indicadores conservados». Con eso el pasaje es íntegramente verificable y **puede defenderse
citando las cifras exactas**, no como una transcripción de un diagnóstico anterior.

---

# Lo que se comprobó y está correcto (no tocar)

Para no revisar dos veces. Todo lo siguiente se verificó celda a celda contra los
artefactos y **coincide**:

| Elemento | Comprobación |
|---|---|
| Tabla 4.1 (entorno) | Coincide con el registro de versiones del NB02, incluida `mord 0.7` |
| Tablas 4.4 y 4.5 (fusión) | 489.771 → 457.499; las 24 olas y los 32.272 registros eliminados cuadran |
| Tablas 4.6 y 4.7 (partición) | Totales por país y por conjunto coinciden con `conjuntos_por_pais_antes_exclusiones.csv` |
| Tabla 4.8 (28 variables) | Coincide con las 28 features del modelo |
| **Tablas 4.9–4.12 (espacios de búsqueda)** | **Los 33 rangos y escalas coinciden uno a uno** con el `espacio_busqueda` de los `models/hp_*.json` (8+7+9+8 en las tablas, más `alpha` de la OLO en [10⁻⁴, 10] logarítmico). Los 25 valores seleccionados caen dentro de sus intervalos y respetan los pasos discretos |
| §2.2.1 (cobertura por país) | 8 países en 1995, 17 desde 1996 y 18 desde 2004 · verificado contando países por ola |
| §4.3.2 y Tabla 4.4 (V-Dem) | 540 registros = 18 países × 30 años (1995–2024) y 28 columnas ✓ |
| **§4.6 (multicolinealidad V-Dem)** | **46 pares > 0,85 y máximo 0,9901 sobre las 19 variables; 1 par y 0,8641 al conservar cuatro** · reproducible (ver F1) |
| Figura 4.3 (matriz LB) | n = 378.592 · máximo 0,5581 entre variables distintas → «0,56» ✓ |
| **Figura 4.4 (matriz V-Dem)** | **n = 360 pares país-año exactos** en el conjunto de entrenamiento ✓ |
| Figura 4.5 (matriz fusionada) | n = 378.592 y 28 features ✓ |
| §5.1 (poliarquía de cada país) | Venezuela 0,1960 en 2024 y Nicaragua 0,2150 en 2020 ✓ · la serie 2013–2018 desciende de 0,327 a 0,213 ✓ (los **promedios** comparativos no, ver A4) |
| Tabla 5.1 (muestra) | 489.771 → 457.499 → 436.463 → 431.756 → 430.895; las cinco restas cuadran |
| Tabla 5.2 (target) | 9,4/27,0/43,4/20,1 · 9,3/17,6/44,0/29,0 · 11,3/21,1/42,0/25,5 y las razones 4,6/4,7/3,7 |
| Tabla 5.3 (Spearman) | **Las 28 filas coinciden exactamente** con `eda_correlaciones_features.csv` |
| Tablas 5.4 y 5.5 (kappa) | Las 30 celdas coinciden; los cinco modelos conservan estrategia val/test ✓ |
| §5.3.1 (ganancias del balanceo) | +0,0729 en OLO y 0,0197–0,0336 en los otros cuatro ✓ |
| Tabla 5.6 (complementarias) | Las 25 celdas coinciden con `resultados_modelos.csv` |
| Tabla 5.7 (bootstrap) | Las 15 filas con IC y EE coinciden con `bootstrap_ic_modelos.csv` |
| Tabla 5.8 (pareado) | Las 14 filas con Δ, IC y P(Δ>0) coinciden con `bootstrap_pareado_vs_principal.csv` |
| Tabla 5.9 (HP CatBoost) | Los siete valores coinciden dígito a dígito |
| §5.3.1 (duración) | 13 h 22 min y el 99,6 % concentrado en el NB02 ✓ |
| Tablas 5.11 y 5.12 (pliegues) | Coinciden con `validacion_temporal_{folds,resumen}.csv`; la brecha de 0,1435 y el MAE 0,9957/0,6162 ✓ |
| Tabla 5.13 (subregiones) | Coincide con `mae_subregiones.csv`; ρ(MAE,κ)=+0,70 y ρ(HHI,MAE)=−1,00 comprobados |
| §5.4 (MAE por país) | Perú 0,4469 y Argentina 0,6678 bajo la configuración principal ✓ |
| Tabla 5.14 (bloques SHAP) | Las seis filas con IC coinciden; ningún par consecutivo se solapa ✓ |
| Tabla 5.15 (15 variables) | Los valores de las quince filas coinciden (el problema es el alcance, ver B5) |
| Tabla 5.16 (micro/macro) | Coincide; la razón 16,9 entre v2x_egal y v2x_polyarchy ✓ |
| Figuras 5.5–5.9 (ALE) | Los cinco archivos existen y corresponden a las cinco variables citadas |
| Tabla 5.17 (errores graves) | Las seis filas y el total 2.641 (7,53 %) coinciden; 1.774 vs 867 ✓ |
| §5.5.3 (LIME) | 0,0461 / 0,0430 / 0,0414 en error máximo y las magnitudes de los otros dos grupos ✓ |
| Figura 5.10 y Tabla 5.18 | W de Kendall 0,9367 (χ²=75,87; p=2,0×10⁻⁶) impreso en el NB04; matriz y top-5 ✓ |
| Tabla 5.19 (teoría) | 7/1/2 · 5/3/2 · 4/1/5 · 4/5/1 coinciden con `tabla_convergencias_CatBoost.csv` |
| §5.5.4 y §5.6 | Las posiciones 11, 13 y 14 de la confianza institucional y las tres de importancia nula ✓ |
| Capítulo 6 (OE1–OE5) | Todas las cifras coinciden con los capítulos 4 y 5, salvo lo indicado en D5 |
| §6.2 (diez limitaciones) | Las diez están redactadas y son coherentes con el diseño |

---

# Checklist de aplicación

## Críticos

- [ ] **A1** §5.2: el KS da p < 0,001, no 0,787 · reescribir el pasaje y el comentario de la Figura 5.1
- [ ] **A2** §4.9.1 y §4.9.2: la OLO usó 20 ensayos, no 50
- [ ] **A3** Insertar la Tabla 5.9b con los **valores óptimos** de OLO, XGBoost, LightGBM y TabNet (las Tablas 4.9–4.12 ya están bien)
- [ ] **A4** §5.1: el promedio de poliarquía de 2024 es 0,634, no 0,741 · y el de 2020 es 0,633 para «los demás países»
- [ ] **C1** Tabla 5.10 y §5.3.2: el 0,84–0,85 de Rosa et al. es F1 de una clase, no macro · la comparación se invierte
- [ ] **C2** §5.2: retirar «un coeficiente positivo en la regresión logística ordinal»
- [ ] **C3** §4.3.4: los errores estándar agrupados no se calculan · reescribir sobre el bootstrap de clústeres

## Cifras y figuras

- [ ] **B1** §5.1: missingness 13,8 / 6,3 / 5,3 % sobre las 28 predictoras, no 6,9 %
- [ ] **B2** §5.3.2: 32,4 % / 67,6 % en prueba
- [ ] **B3** Figuras 5.2 y 5.4: los pies dicen «pesos de clase» y las imágenes dicen «smotenc»
- [ ] **B4** Figura 5.3: el pie dice quince y la figura muestra veinte
- [ ] **B5** Tabla 5.15: ampliar a veinte filas (o ajustar el texto a quince)
- [ ] **B6** §4.6: documentar las doce exclusiones de variables LB, no cuatro · y sustituir el ejemplo del tamaño de la ciudad, que tiene ρ = 0,0971
- [ ] **F1** §4.6: declarar la base de cálculo de los 46 pares y del máximo 0,990 (540 país-año) · el pasaje **es reproducible**, no hay que retirarlo

## Coherencia

- [ ] **C4** §1.1: verificar la referencia [7] de Ergun et al. o reatribuir la afirmación
- [ ] **C5** Bibliografía [55]: apellidos invertidos frente a la cita en el texto
- [ ] **D1** §2.4: «cinco métricas» → ocho
- [ ] **D2** §4.10: añadir exactitud balanceada a la enumeración
- [ ] **D3** Insertar la frase que referencia la Tabla 5.7
- [ ] **D4** §2.5.3: declarar que la Figura 2.7 son las Figuras 5.5 y 5.8
- [ ] **D5** Resumen y Abstract: ALE cubre tres bloques, no solo confianza
- [ ] **D6** §4.10: el conjunto de validación es `eval_set` de los cuatro modelos no lineales
- [ ] **D7** Tablas 4.6 y 5.2: encabezado «N estimado» → «N»
- [ ] **D8** Abstract: unificar el separador decimal

## Refuerzos opcionales

- [ ] **E1** Las ocho variables de \|SHAP\| nulo no se explican por los valores ausentes
- [ ] **E2** §5.3.4: sd sobre los tres pliegues con presupuesto homogéneo
- [ ] **E3** Pie de la Tabla 5.9: añadir `task_type = GPU`

## Revisión final

- [ ] Ningún pie de figura o tabla menciona ya «pesos de clase» como configuración principal
- [ ] Ninguna cifra de la variante binaria compara F1 macro contra F1 de una clase
- [ ] Toda afirmación sobre coeficientes o errores estándar tiene un archivo que la respalde
- [ ] El número de ensayos de Optuna del texto coincide con `hiperparametros_modelos.csv` (50/50/50/20/20)
- [ ] Los kappa de validación citados en pies de tabla coinciden con la Tabla 5.4
