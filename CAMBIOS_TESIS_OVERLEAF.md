# Cambios a realizar en el documento de tesis (Overleaf)

**Generado:** 2026-09-05
**Alcance:** respuesta completa a las observaciones de los revisores (bloques 1
a 4 del plan de trabajo).
**Estado del código:** aplicado y verificado en el repositorio. Este documento
recoge únicamente lo que hay que cambiar en el `.tex`.

---

## Cómo usar este documento

El trabajo se organiza en **dos fases**, porque los notebooks se van a
reejecutar completos en un servidor con GPU y esa corrida regenera todas las
cifras.

### Fase 1 — Editar ahora, antes de la reejecución

Son cambios de redacción, terminología, criterio metodológico y estructura que
**no dependen de ningún número**:

- **Parte A** — correcciones de redacción, terminología y criterio.
- **Parte C** — incongruencias documento ↔ código que requieren una decisión
  tuya sobre cómo redactar.
- **Parte E** — entradas BibTeX nuevas.

### Fase 2 — Completar después de la reejecución

Son las tablas, figuras y afirmaciones numéricas. La **estructura** de cada una
está definida aquí (columnas, título, pie, texto que la acompaña) y se puede
maquetar ya; los **valores** se rellenan leyendo los CSV que genera la corrida:

- **Parte B** — material nuevo: qué tabla o figura insertar, en qué sección y
  con qué texto.
- **Parte D** — cambios que la corrida nueva introduce y que obligan a reescribir
  pasajes concretos del documento.
- **Anexo I** — mapeo de cada tabla y figura del documento al archivo exacto de
  `results/` del que se leen sus valores.

> ### Aviso sobre las cifras de este documento
>
> Las cifras marcadas **[run anterior]** provienen de la corrida del servidor
> del 31 de agosto de 2026, que se hizo con el código **antes** de los cambios
> de esta fase. Sirven para dimensionar y maquetar las tablas, pero
> **deben reemplazarse** por las de la corrida nueva. Cambian porque:
>
> 1. La línea base OLO pasa de una logística multinomial a un modelo ordinal
>    (`mord.LogisticIT`): todas sus métricas cambian.
> 2. TabNet ya no aplica el muestreo ponderado en las tres estrategias, así que
>    sus tres brazos dejan de dar el mismo resultado.
> 3. Los pesos de clase se estiman solo sobre el conjunto de entrenamiento, lo
>    que desplaza ligeramente todas las métricas de la estrategia `pesos_clase`.
> 4. El bootstrap pasa de 16 clústeres país a 32 país-año, de modo que los
>    intervalos de confianza se estrechan.
> 5. Se añaden las métricas ponderadas por el factor de expansión `X_020`.
>
> Las cifras marcadas **[estructural]** no dependen de la corrida: son conteos
> del diseño (número de olas, de países, de variables, de casos LIME) o
> decisiones fijadas en `utils/config.py`.

Convención: `§` remite a la sección del PDF actual; los números entre paréntesis
son las tablas/figuras actuales.

---

# PARTE 0 — Antes de lanzar la reejecución

`clean_process_folders()` vacía `data/base`, `data/processed`, `models`,
`results` y `notebooks/output`. Hay dos cosas que conviene preservar antes:

1. **Archivar la evidencia del problema de la línea base.** Los archivos
   `models/hp_OLO_*.json` de la corrida anterior contienen
   `"implementacion": "sklearn.LogisticRegression"`, que es la prueba
   documental de que la línea base no era ordinal. Si quieres poder citarla en
   la defensa o en un anexo, cópialos fuera del repositorio antes de reejecutar,
   junto con `notebooks/output/`.

2. **Confirmar que `results/` está comiteado.** La carpeta está bajo control de
   versiones; un `git status` limpio antes de la limpieza garantiza que las
   figuras y tablas actuales queden recuperables con `git checkout -- results`.

3. **Correr primero la prueba de humo.** El proyecto tiene un único
   interruptor, `MODO_EJECUCION` en `utils/config.py`, que ajusta de una vez
   todos los parámetros que dependen del tamaño de la corrida:

   ```bash
   MODO_EJECUCION=humo bash run_all.sh    # minutos: verifica que todo corre
   MODO_EJECUCION=real bash run_all.sh    # horas: la corrida definitiva
   ```

   Si ejecutas notebook a notebook en lugar de con `run_all.sh`, el modo se
   fija editando `MODO = "humo"` en `utils/config.py` y **reiniciando el
   kernel** (Python conserva el módulo ya importado). Al cambiar de modo hay
   que reejecutar también el NB01, porque es el que genera la muestra reducida
   con las cuotas del perfil activo.

   El perfil de humo recorta el volumen de cada etapa pero **no desactiva
   ninguna**: los pliegues temporales, el bootstrap, SHAP, LIME y ALE se
   ejecutan igual, así que la prueba recorre las mismas rutas de código. Si la
   prueba de humo termina sin errores, la corrida real solo puede fallar por
   volumen (memoria o tiempo), no por código.

Comprobaciones rápidas antes de lanzar la corrida definitiva:

| Qué verificar | Cómo | Resultado esperado |
|---|---|---|
| Modo activo | banner de la primera celda de cualquier notebook | `MODO CORRIDA REAL — dataset completo y presupuestos completos` |
| `mord` instalado | la celda §2 del NB02 imprime el backend | `Backend OLO: mord.LogisticIT (ordinal)` |
| GPU visible | misma celda | `GPU solicitada / disponible : True / True` |
| Presupuesto de Optuna | banner de la primera celda | 50 árboles / 20 OLO / 20 TabNet |
| Pliegues temporales activos | banner de la primera celda | `EJECUTAR_FOLDS_TEMPORALES : True` |

> **Cómo distinguir después los artefactos de una prueba de los de la corrida
> real:** el modo queda estampado en la columna `modo_ejecucion` de
> `results/resultados_modelos.csv` y en el campo `modo_ejecucion` de cada
> `models/hp_*.json`. Antes de leer una cifra para el documento, comprobar que
> dice `real`.

Si `mord` no estuviera instalado, el NB02 **se detiene con un error explícito**
en lugar de sustituir la línea base por un modelo multinomial. Es
deliberado: ese silencio es justamente lo que causó la incongruencia que
señalaron los revisores.

---

# PARTE A — Cambios de redacción y cifras (aplicables ya)

## A.1 Corregir el tamaño de la muestra: «465.000» no corresponde a ninguna etapa

**Dónde:** Resumen, Abstract, §4.3.1 y Cap. 6 (primer párrafo).

La cifra «aproximadamente 465.000» no coincide con ninguna etapa del pipeline.
Cadena real **[run anterior]**:

| Etapa | Registros | Pérdida |
|---|---|---|
| Carga inicial del Latinobarómetro (24 olas) | 489.771 | — |
| Eliminación de registros sin país/año | 457.499 | −32.272 |
| Fusión con V-Dem (cobertura 100 %) | 457.499 | 0 |
| Exclusión por target NS/NR (`A_003_031`) | 436.463 | **−21.036** |
| Exclusión de Venezuela posterior a 2017 | 431.756 | −4.707 |
| Olas fuera del rango del split | 431.756 | 0 |
| Nicaragua fuera de validación | 430.895 | **−861** |
| **Total efectivamente modelado** | **430.895** | |

430.895 = 378.592 (train) + 17.219 (val) + 35.084 (test), que son los valores
de la Tabla 5.1.

**Sustituir** en los cuatro sitios:

> «Se integraron aproximadamente 465.000 respuestas individuales…»

por

> «Se integraron 489.771 respuestas individuales de 24 olas del Latinobarómetro,
> de las cuales 430.895 se utilizan efectivamente en el modelado tras la
> armonización y las exclusiones documentadas en la Tabla 4.5b.»

## A.2 §4.8 — La imputación es **simple**, no múltiple

El texto dice «imputación múltiple por ecuaciones encadenadas (MICE)». El código
usa `IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)`
con `sample_posterior=False` (valor por defecto), que produce **una única
imputación determinista**: no genera varios conjuntos imputados ni combina
estimaciones por las reglas de Rubin. El revisor pregunta exactamente esto.

**Reemplazar el primer párrafo de §4.8 por:**

> Para la regresión logística ordinal y TabNet, que no manejan valores faltantes
> de forma nativa, se aplica imputación iterativa por ecuaciones encadenadas en
> el estilo de MICE, mediante `IterativeImputer` con `BayesianRidge` de
> scikit-learn, diez iteraciones de cadena y semilla fija. A diferencia de la
> imputación múltiple propiamente dicha (van Buuren y Groothuis-Oudshoorn, 2011),
> el procedimiento se ejecuta con `sample_posterior=False` y produce **un único
> conjunto de datos imputado**, por lo que la incertidumbre de imputación no se
> propaga a las métricas. El imputador se ajusta exclusivamente sobre el conjunto
> de entrenamiento y se aplica sin reajuste a validación y prueba, lo que evita
> la fuga de información.

**Añadir a §6.2 (Limitaciones):**

> Sexta, la imputación de valores faltantes es simple y no múltiple: las métricas
> reportadas no incorporan la varianza asociada al proceso de imputación.

## A.3 §4.9.1 — Terminología de Optuna y número de ensayos de TabNet

Dos correcciones en el mismo párrafo.

1. Optuna no realiza una búsqueda ciega: `TPESampler` es un optimizador bayesiano
   secuencial (Tree-structured Parzen Estimator).
2. TabNet **no** usa 50 ensayos: el código aplica `min(n_trials, 20)` **[run anterior]**.

**Reemplazar:**

> «Los hiperparámetros se optimizan con Optuna (muestreador TPE, 50 ensayos),
> utilizando el conjunto de validación de 2020 y el kappa cuadrático como función
> objetivo.»

**por:**

> Los hiperparámetros se optimizan mediante búsqueda bayesiana secuencial con el
> muestreador TPE de Optuna (Bergstra et al., 2011; Akiba et al., 2019), usando
> el conjunto de validación de 2020 y el kappa cuadrático como función objetivo.
> Se ejecutan 50 ensayos por configuración en la regresión logística ordinal,
> XGBoost, CatBoost y LightGBM, y 20 ensayos en TabNet, cuyo coste por ensayo es
> sustancialmente mayor al entrenarse por épocas. El procedimiento **no emplea
> validación cruzada**: la evaluación de cada ensayo se realiza sobre el holdout
> temporal de 2020, porque una partición cruzada aleatoria rompería el orden
> temporal y mezclaría olas de encuesta, que es precisamente la fuga de
> información que el protocolo de §4.4 evita (Roberts et al., 2017).

**Añadir también, al final de §4.9.1 o en §4.10:**

> El conjunto de validación cumple dos funciones simultáneas: es el `eval_set`
> del *early stopping* de LightGBM (50 rondas) y de TabNet (paciencia 20), y es
> el criterio de selección de hiperparámetros y de configuración. Esto introduce
> un sesgo optimista **sobre la validación**, no sobre la prueba, que permanece
> reservada.

**Y corregir el pie de la Tabla 4.12 (TabNet):** añadir «20 ensayos de Optuna».

## A.4 §4.9.2 — «cada modelo se optimiza mediante 50 ensayos»

Misma corrección: «50 ensayos, excepto TabNet con 20».

## A.5 Tabla 4.11 (LightGBM) — está incompleta y con un rango erróneo

Cinco hiperparámetros aparecen con «-» aunque el código sí los optimiza, falta
`n_estimators` y `min_child_samples` figura como 5–100 cuando el código usa
**20–100** **[run anterior]**. Es la tabla que un revisor usaría para replicar.

**Reemplazar la Tabla 4.11 por:**

```latex
\begin{table}[htbp]
\centering
\caption{Espacio de búsqueda de hiperparámetros para LightGBM.}
\label{tab:hp-lightgbm}
\begin{tabular}{lll}
\hline
\textbf{Hiperparámetro} & \textbf{Rango} & \textbf{Escala de búsqueda} \\
\hline
n\_estimators      & 200--1000        & Entera, paso 100 \\
num\_leaves        & 20--150          & Entera uniforme \\
max\_depth         & 3--8             & Entera uniforme \\
learning\_rate     & 0,01--0,30       & Logarítmica \\
subsample          & 0,60--1,00       & Continua uniforme \\
colsample\_bytree  & 0,60--1,00       & Continua uniforme \\
reg\_alpha         & $10^{-8}$--10,0  & Logarítmica \\
reg\_lambda        & $10^{-8}$--10,0  & Logarítmica \\
min\_child\_samples & 20--100         & Entera uniforme \\
\hline
\end{tabular}
\end{table}
```

**Añadir al pie:** «Early stopping de 50 rondas sobre el conjunto de validación.»

## A.6 Tablas 4.9 y 4.10 — indicar el paso de la grilla

`n_estimators` (XGBoost) e `iterations` (CatBoost) se muestrean con `step=100`,
es decir 9 y 8 valores discretos, no una uniforme continua **[run anterior]**.

- Tabla 4.9, fila `n_estimators`: «Entera uniforme» → «Entera, paso 100».
- Tabla 4.10, fila `iterations`: «Entera uniforme» → «Entera, paso 100».

## A.7 §4.10 — Criterio de selección (observación central del revisor)

El texto actual dice: «Se prioriza el mayor kappa cuadrático **en el conjunto de
prueba** y se contrasta la decisión con el MAE ordinal». Eso es exactamente lo
que el revisor objeta. El código ya se corrigió: la selección se hace en
validación y la prueba solo se reporta.

**Reemplazar el último párrafo de §4.10 por:**

> La selección de la configuración principal se realiza **exclusivamente sobre el
> conjunto de validación (2020)**: se elige la combinación de modelo y estrategia
> de balanceo que maximiza el kappa cuadrático en validación, y el resultado se
> registra en `results/modelo_xai_seleccionado.json`, que es la fuente única para
> los análisis de explicabilidad, estabilidad regional y contraste teórico. El
> conjunto de prueba (2023–2024) se utiliza únicamente para **reportar** el
> desempeño de la configuración ya seleccionada, de modo que no interviene en
> ninguna decisión de modelado. La selección no exige dominar todas las métricas
> de forma simultánea; el MAE ordinal, el F1 macro, la exactitud y el AUROC OvR
> se usan para documentar compensaciones entre acierto exacto, equilibrio entre
> clases y discriminación.

## A.8 Tabla 5.5 — Título

**Actual:** «Hiperparámetros óptimos de CatBoost en el conjunto de prueba.»

**Nuevo:** «Hiperparámetros de CatBoost seleccionados en validación (2020) y
evaluados en prueba (2023–2024).»

**Añadir al pie:** «Búsqueda bayesiana TPE con 50 ensayos, semilla 42, objetivo
kappa cuadrático en validación; el kappa alcanzado en validación fue 0,4883
**[run anterior]**.»

## A.9 §2.4 — Número de métricas

El texto habla de «las cinco métricas utilizadas» (Tabla 2.1) mientras el código
calcula ocho y las persiste todas en `results/resultados_modelos.csv`
(añade *balanced accuracy*, F1 ponderado y kappa lineal).

**Añadir al final de §2.4.6:**

> El flujo computacional calcula y almacena ocho métricas agregadas —a las cinco
> anteriores se suman la exactitud balanceada, el F1 ponderado y el kappa
> lineal—; en el capítulo de resultados se reportan las cinco discutidas en esta
> sección, y las tres restantes quedan disponibles en los archivos de resultados
> para verificación.

## A.10 §4.6 — Contradicción sobre el umbral de correlación

El texto afirma: «Estas correlaciones **no se utilizan como un umbral automático
de exclusión**». Sin embargo, `utils/config.py` documenta cuatro variables
excluidas explícitamente por señal baja con el criterio |ρ| < 0,05
**[run anterior]**: `H_002_101` (0,047), `C_003_003_011` (0,039), `A_007_071`
(0,021) y `X_008` (0,042).

**Elige una de las dos redacciones y aplícala también al comentario del código:**

*Opción 1 (recomendada, refleja lo que se hizo):*

> Las correlaciones no operan como único criterio de exclusión, pero sí como
> criterio complementario: cuatro variables con |ρ| < 0,05 respecto del target y
> sin justificación teórica sustantiva se descartaron por combinación de señal
> baja y ausencia de fundamento conceptual (confianza en la Iglesia Católica,
> preocupación por el desempleo, escala izquierda-derecha y tamaño del
> municipio). Las variables sociodemográficas, en cambio, se conservan pese a su
> baja correlación univariada por su relevancia teórica.

*Opción 2:* eliminar el umbral del comentario de `config.py` y justificar esas
cuatro exclusiones solo por criterio sustantivo.

## A.11 §5.3.1 — TabNet ya no repite el mismo valor

> **Superado por el código.** Ver **D.2**.

En la corrida anterior TabNet mostraba κw = 0,5127 y exactitud 0,4698
**idénticos** en «sin balanceo» y «pesos de clase» **[run anterior]**. La causa
era que `fit(..., weights=1)` activaba el muestreo por frecuencia inversa de
clase en las tres estrategias, con lo que los tres brazos eran, en realidad, la
misma configuración.

El código actual desactiva ese muestreo y aplica el peso de clase en la función
de pérdida solo donde corresponde, así que los tres brazos pasan a ser
distintos. **No hay que añadir la explicación del valor repetido**: hay que
describir el mecanismo nuevo, que es lo que detalla D.2, y declarar en §6.2 la
asimetría que sí persiste (el factor de expansión muestral no interviene en el
ajuste de TabNet).

## A.12 §5.3.1 y Cap. 6 — Matizar la afirmación sobre el MAE

El texto afirma que «CatBoost … presenta el menor MAE ordinal (0,6297)». Es
correcto **solo** al comparar cada modelo con su mejor estrategia. En el conjunto
completo de 15 configuraciones, el MAE más bajo lo obtiene **CatBoost sin
balanceo (0,5413)**, seguido de LightGBM sin balanceo (0,5435) **[run anterior]**.
Las estrategias de balanceo mejoran el kappa y **empeoran** el MAE, que es un
resultado sustantivo interesante y hoy queda oculto.

**Añadir a §5.3.1:**

> Conviene precisar que el menor MAE ordinal entre las quince configuraciones no
> corresponde a la seleccionada: CatBoost sin balanceo alcanza 0,5413 y LightGBM
> sin balanceo 0,5435, frente a 0,6297 de CatBoost con pesos de clase. Las
> estrategias de balanceo mejoran el acuerdo ordinal corregido por azar y, al
> mismo tiempo, incrementan la distancia media del error: al redistribuir masa
> predictiva hacia las clases minoritarias, el modelo acierta mejor la estructura
> ordinal global pero se aleja más, en promedio, de la categoría exacta. La
> selección privilegia el kappa cuadrático por ser la métrica principal del
> diseño; esta compensación se documenta explícitamente.

## A.13 Reformulación predictiva vs. causal

El marco conceptual ya es correcto (§2.6, §2.7, Resumen). Lo que falta es el
barrido léxico del capítulo 5 y de las figuras. Cambios concretos:

| Ubicación | Actual | Propuesto |
|---|---|---|
| H3 (§1.2) | «aparecerán como **determinantes explicativos dominantes**» | «concentrarán la mayor **contribución predictiva** atribuida por SHAP» |
| §5.5.1 | «las variables que **explican** la satisfacción» | «las variables cuya contribución a la predicción del modelo es mayor» |
| Figuras 5.5–5.10, eje Y | «Efecto ALE sobre satisfacción democrática» | «Cambio en la predicción del modelo (ALE)» — **ya corregido en el código**; al reejecutar el NB04 las figuras salen con la etiqueta nueva |
| §5.5.2 | «la confianza en el Congreso conserva un **efecto positivo**» | «desplaza la predicción por encima del promedio del modelo» |
| §5.5.3 | «la situación económica del país **domina** las explicaciones» | «es la variable con mayor peso en las aproximaciones locales de LIME» |
| Tabla 5.11 y §5.6 | «determinantes SHAP» | «patrones predictivos» |
| Cap. 6, OE4/OE5 | «los determinantes identificados» | «los predictores con mayor contribución» |

**Añadir un párrafo de apertura en §5.5:**

> Las tres técnicas de esta sección describen el comportamiento del modelo, no
> mecanismos causales. Una contribución SHAP elevada indica que la variable
> aporta información predictiva sobre la categoría de satisfacción declarada; no
> implica que modificarla alteraría esa satisfacción. La advertencia es
> especialmente pertinente con predictores correlacionados: los valores de
> Shapley reparten entre variables próximas una señal compartida, de modo que la
> importancia individual de cada una no es identificable de forma unívoca (Aas,
> Jullum y Løland, 2021; Kumar et al., 2020). Por esa razón la lectura por bloque
> temático se privilegia sobre la lectura por variable individual, y la sección
> 5.5.2 cuantifica esa incertidumbre.

**Y en la definición de la Tabla 5.11 (§5.6), añadir:**

> La convergencia entre un patrón predictivo y una teoría indica compatibilidad
> descriptiva; no valida el mecanismo causal que la teoría postula.

---

# PARTE B — Material nuevo que el código ya genera

Cada apartado indica **qué tabla o figura insertar, en qué sección y con qué
texto**. Eso se puede maquetar ya.

Los valores numéricos que aparecen aquí se calcularon sobre los artefactos de la
corrida del 31 de agosto de 2026 y están marcados **[run anterior]**: sirven
para dimensionar cada tabla y comprobar que el texto propuesto tiene sentido,
pero **hay que reemplazarlos** por los de la corrida nueva leyendo los archivos
que indica el Anexo I. Los valores marcados **[estructural]** no cambian.

## B.1 Nueva Tabla 4.5b — Trazabilidad del pipeline

**Insertar al final de §4.3.4**, después de la Tabla 4.5. Archivos fuente:
`results/tables/conjuntos_resumen_antes_exclusiones.csv`,
`conjuntos_resumen_despues_exclusiones.csv`, `conjuntos_efecto_exclusiones.csv`.

```latex
\begin{table}[htbp]
\centering
\caption{Trazabilidad del conjunto de datos, desde la carga hasta los conjuntos finales.}
\label{tab:trazabilidad}
\begin{tabular}{lrr}
\hline
\textbf{Etapa} & \textbf{Registros} & \textbf{Pérdida} \\
\hline
Carga inicial del Latinobarómetro (24 olas)      & 489.771 & --- \\
Eliminación de registros sin país o año          & 457.499 & $-32.272$ \\
Fusión contextual con V-Dem (cobertura 100\,\%)  & 457.499 & 0 \\
Exclusión por target NS/NR (\texttt{A\_003\_031})& 436.463 & $-21.036$ \\
Exclusión de Venezuela posterior a 2017          & 431.756 & $-4.707$ \\
Olas fuera del rango del split                   & 431.756 & 0 \\
Nicaragua fuera del conjunto de validación       & 430.895 & $-861$ \\
\hline
Entrenamiento (1995--2018, 21 olas, 18 países)   & 378.592 & \\
Validación (2020, 1 ola, 16 países)              & 17.219  & \\
Prueba (2023--2024, 2 olas, 16 países)           & 35.084  & \\
\hline
\textbf{Total modelado}                          & \textbf{430.895} & \\
\hline
\end{tabular}
\end{table}
```

**Texto de acompañamiento:**

> La Tabla 4.5b documenta la transición desde los 489.771 registros iniciales
> hasta los 430.895 efectivamente modelados. La exclusión de mayor magnitud
> corresponde a los 21.036 registros sin respuesta válida en la variable objetivo
> (4,6 % del conjunto tras la depuración geográfica), seguida por los 4.707
> registros de Venezuela posteriores a 2017 y los 861 de Nicaragua retirados de
> la validación por consistencia de dominio con la prueba. La fusión con V-Dem no
> produce pérdidas: los 18 países y las 24 olas encuentran correspondencia
> completa en V-Dem v16. El *missingness* global del conjunto resultante es del
> 6,9 %.

## B.2 Nueva Tabla 5.3a — Selección en validación (responde a la observación central)

**Insertar en §5.3.1, antes de la Tabla 5.3.** Fuente:
`results/tables/metricas_kappa_pivot_val.csv` y `mejor_estrategia_por_modelo.csv`.

Kappa cuadrático en **validación (2020)** **[run anterior]**:

| Modelo | Sin balanceo | Pesos de clase | SMOTE-NC | Mejor (val) |
|---|---|---|---|---|
| OLO | 0,3985 | **0,4722** | 0,4690 | pesos de clase |
| XGBoost | 0,4578 | **0,4829** | 0,4641 | pesos de clase |
| CatBoost | 0,4583 | **0,4883** | 0,4840 | pesos de clase |
| LightGBM | 0,4610 | **0,4828** | 0,4599 | pesos de clase |
| TabNet | 0,4642 | 0,4642 | **0,4874** | SMOTE-NC |

```latex
\begin{table}[htbp]
\centering
\caption{Kappa cuadrático en validación (2020) por modelo y estrategia de balanceo.
Los valores en negrita indican la estrategia seleccionada para cada modelo.}
\label{tab:kappa-val}
\begin{tabular}{lcccc}
\hline
\textbf{Modelo} & \textbf{Sin balanceo} & \textbf{Pesos de clase} & \textbf{SMOTE-NC} & \textbf{Selección} \\
\hline
OLO (línea base) & 0,3985 & \textbf{0,4722} & 0,4690 & Pesos de clase \\
XGBoost  & 0,4578 & \textbf{0,4829} & 0,4641 & Pesos de clase \\
CatBoost & 0,4583 & \textbf{0,4883} & 0,4840 & Pesos de clase \\
LightGBM & 0,4610 & \textbf{0,4828} & 0,4599 & Pesos de clase \\
TabNet   & 0,4642 & 0,4642 & \textbf{0,4874} & SMOTE-NC \\
\hline
\end{tabular}
\end{table}
```

**Texto de acompañamiento (es el argumento clave frente al revisor):**

> La configuración principal se elige sobre la validación de 2020, sin observar
> el conjunto de prueba. El máximo corresponde a CatBoost con pesos de clase
> (κw = 0,4883), seguido de TabNet con SMOTE-NC (0,4874) y XGBoost con pesos de
> clase (0,4829). Promediando por estrategia, los pesos de clase (0,4781)
> superan a SMOTE-NC (0,4729) y a la ausencia de balanceo (0,4480).
>
> La comparación con la Tabla 5.3 permite verificar que la separación entre
> selección y evaluación no altera los resultados sustantivos: los cinco modelos
> conservan la misma mejor estrategia en validación y en prueba, y la
> configuración global seleccionada —CatBoost con pesos de clase— es también la
> de mayor kappa en prueba. Además, los valores de validación (0,3985–0,4883) son
> sistemáticamente inferiores a los de prueba (0,4475–0,5418), lo que indica que
> el conjunto de prueba no resulta más exigente que el de calibración y que no
> hay indicios de sobreajuste a la validación.

## B.3 Nueva Tabla 5.3b — Intervalos de confianza por bootstrap de clústeres

**Insertar en §5.3.1, después de la Tabla 5.4.** Fuente:
`results/tables/bootstrap_ic_modelos.csv`.

> ⚠ **Los IC calculados en local cubren 9 de las 15 configuraciones**: los
> pipelines de XGBoost y TabNet no se pudieron deserializar fuera del servidor
> (versión de XGBoost distinta y TabNet serializado en GPU). Al reejecutar el
> NB03 en el servidor saldrán las 15. Los valores de abajo son definitivos para
> las nueve configuraciones que sí se reconstruyeron.

Kappa cuadrático en prueba con IC 95 % (bootstrap de **16 clústeres país**,
B = 1000) **[run anterior]**:

| Modelo | Estrategia | κw | IC 95 % | EE bootstrap |
|---|---|---|---|---|
| CatBoost | pesos de clase | 0,5418 | [0,4857; 0,5841] | 0,0255 |
| CatBoost | SMOTE-NC | 0,5386 | [0,4785; 0,5826] | 0,0276 |
| LightGBM | pesos de clase | 0,5315 | [0,4795; 0,5723] | 0,0240 |
| OLO | pesos de clase | 0,5198 | [0,4675; 0,5599] | 0,0231 |
| OLO | SMOTE-NC | 0,5159 | [0,4632; 0,5554] | 0,0230 |
| CatBoost | sin balanceo | 0,5070 | [0,4490; 0,5492] | 0,0266 |
| LightGBM | sin balanceo | 0,5062 | [0,4508; 0,5469] | 0,0258 |
| LightGBM | SMOTE-NC | 0,5051 | [0,4419; 0,5477] | 0,0269 |
| OLO | sin balanceo | 0,4475 | [0,3973; 0,4804] | 0,0216 |

## B.4 Nueva Tabla 5.3c — Comparación pareada contra la configuración principal

**Insertar inmediatamente después.** Fuente:
`results/tables/bootstrap_pareado_vs_principal.csv`.

Diferencia de kappa cuadrático frente a CatBoost con pesos de clase, evaluando
ambos modelos sobre **las mismas réplicas** (B = 1000) **[run anterior]**:

| Comparado | Δκw | IC 95 % de Δ | P(Δ>0) | Conclusión |
|---|---|---|---|---|
| OLO [sin balanceo] | +0,0943 | [+0,0768; +0,1109] | 1,000 | distinguible |
| LightGBM [SMOTE-NC] | +0,0367 | [+0,0206; +0,0537] | 1,000 | distinguible |
| LightGBM [sin balanceo] | +0,0356 | [+0,0239; +0,0465] | 1,000 | distinguible |
| CatBoost [sin balanceo] | +0,0348 | [+0,0235; +0,0452] | 1,000 | distinguible |
| OLO [SMOTE-NC] | +0,0259 | [+0,0103; +0,0437] | 1,000 | distinguible |
| **OLO [pesos de clase]** | **+0,0220** | **[+0,0078; +0,0395]** | **0,999** | **distinguible** |
| LightGBM [pesos de clase] | +0,0103 | [+0,0043; +0,0159] | 1,000 | distinguible |
| CatBoost [SMOTE-NC] | +0,0032 | [−0,0069; +0,0146] | 0,771 | **no distinguible** |

**Texto de acompañamiento (responde directamente al revisor):**

> Para evitar atribuir superioridad a diferencias puramente descriptivas, las
> comparaciones entre configuraciones se acompañan de intervalos de confianza
> obtenidos por remuestreo de clústeres. Los registros de un mismo país-año
> comparten los indicadores de V-Dem y provienen de la misma muestra nacional, de
> modo que no son independientes; remuestrear registros individuales subestimaría
> el error estándar, por lo que se remuestrean clústeres completos con reemplazo
> (Cameron y Miller, 2015; Efron y Tibshirani, 1993). La comparación entre pares
> de modelos es pareada: en cada réplica se evalúan ambos sobre exactamente las
> mismas observaciones, lo que elimina la variabilidad común del remuestreo y
> permite que el intervalo describa la diferencia.
>
> El resultado matiza la lectura de la Tabla 5.3 en una dirección precisa. La
> ventaja de CatBoost con pesos de clase sobre la línea base con la misma
> estrategia es de 0,0220 puntos de kappa, con un intervalo de [+0,0078; +0,0395]
> que **excluye el cero**: la diferencia es distinguible del ruido de muestreo, y
> el respaldo de H1 no depende únicamente de la comparación puntual. En cambio,
> la diferencia frente a CatBoost con SMOTE-NC es de 0,0032 puntos con un
> intervalo de [−0,0069; +0,0146] que **incluye el cero**: ambas estrategias son
> indistinguibles para este modelo y la elección entre ellas no puede sostenerse
> en el rendimiento predictivo. El intervalo de la propia estimación de κw
> (±0,025 aproximadamente, Tabla 5.3b) es del mismo orden de magnitud que varias
> de las diferencias entre modelos de *gradient boosting*, lo que confirma que la
> jerarquía interna de esa familia no queda determinada por el tamaño efectivo de
> muestra disponible: 16 clústeres país en el conjunto de prueba.

## B.5 Nueva Tabla 5.4b y figuras — Métricas por categoría del modelo principal

**Insertar como nueva §5.3.3** (o al final de §5.3.1). Fuentes:
`results/tables/metricas_por_clase_CatBoost_pesos_clase_ordinal_4clases_test.csv`,
`matriz_confusion_*.csv`, figuras `03_metricas_por_clase.png` y
`03_confusion_modelo_principal.png`.

Precision, recall y F1 por categoría, CatBoost con pesos de clase, prueba
2023–2024 **[run anterior]**:

| Clase | Precision | Recall | F1 | Soporte | % |
|---|---|---|---|---|---|
| 0 — Para nada satisfecho | 0,3959 | 0,4805 | 0,4341 | 3.969 | 11,31 |
| 1 — No muy satisfecho | 0,4191 | 0,4572 | 0,4374 | 7.414 | 21,13 |
| 2 — Más bien satisfecho | 0,5749 | 0,3918 | 0,4660 | 14.746 | 42,03 |
| 3 — Muy satisfecho | 0,5060 | 0,6854 | 0,5822 | 8.955 | 25,52 |
| Promedio macro | 0,4740 | 0,5037 | 0,4799 | 35.084 | 100 |
| Promedio ponderado | 0,5041 | 0,4906 | 0,4860 | 35.084 | 100 |

Matriz de confusión, porcentaje por clase real **[run anterior]**:

| Real \ Predicho | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 — Para nada satisfecho | **48,0** | 28,1 | 14,2 | 9,7 |
| 1 — No muy satisfecho | 21,3 | **45,7** | 23,0 | 10,0 |
| 2 — Más bien satisfecho | 6,9 | 20,9 | **39,2** | 33,0 |
| 3 — Muy satisfecho | 3,5 | 5,5 | 22,4 | **68,5** |

Conteos absolutos: 1907/1115/562/385 · 1577/3390/1706/741 ·
1016/3087/5777/4866 · 317/496/2004/6138.

**Texto de acompañamiento:**

> Las métricas agregadas resumen el rendimiento en una sola cifra por modelo y
> ocultan el comportamiento diferencial entre categorías. El desglose muestra tres
> patrones. Primero, la categoría mejor recuperada es «Muy satisfecho»
> (F1 = 0,5822, recall = 68,5 %) y la más débil es «Para nada satisfecho»
> (F1 = 0,4341), con una brecha de 0,1481 puntos. Segundo, la clase mayoritaria
> «Más bien satisfecho» presenta la mayor precisión (0,5749) pero el **recall más
> bajo de las cuatro** (0,3918): la ponderación por pesos de clase desplaza masa
> predictiva desde la categoría más frecuente hacia las minoritarias, de modo que
> el modelo deja de sobrepredecir la moda a costa de recuperar menos casos de esa
> misma categoría. Tercero, los errores se concentran entre categorías contiguas:
> el 48,0 % de los insatisfechos extremos se clasifica correctamente y el 28,1 %
> se desplaza una sola categoría, mientras que solo el 9,7 % cae en el extremo
> opuesto. En conjunto, los errores con distancia ordinal mayor o igual a dos
> categorías son 3.517, el 10,02 % del conjunto de prueba, y los de distancia
> máxima, 702, el 2,00 %.
>
> Este desglose es la evidencia directa para H2: con pesos de clase, la clase
> minoritaria 0 alcanza un F1 de 0,4341 y un recall de 0,4805, sustancialmente
> por encima de lo que produciría un clasificador concentrado en la moda, que
> obtendría recall nulo en esa categoría.

**Figuras a incluir:** `03_metricas_por_clase.png` y
`03_confusion_modelo_principal.png` (dos paneles: conteos y % por fila, con los
errores de distancia ≥ 2 resaltados).

## B.6 Ampliar la Tabla 5.8 con intervalos de confianza y estabilidad del rango

**Sustituir la Tabla 5.8 por su versión con incertidumbre**, o añadirla como
Tabla 5.8b. Fuente: `results/tables/shap_importancia_ic_CatBoost_pesos_clase.csv`.

Bootstrap de 16 clústeres país, B = 1000, sobre los valores SHAP reales
**[run anterior]**:

| # | Variable | \|SHAP\| | IC 95 % | Rango (IC) | % top-5 |
|---|---|---|---|---|---|
| 1 | Apoyo a la democracia | 0,2775 | [0,2700; 0,2852] | 1 (1–1) | 100 |
| 2 | Situación económica país | 0,2334 | [0,2233; 0,2438] | 2 (2–2) | 100 |
| 3 | Aprobación gobierno | 0,1824 | [0,1734; 0,1916] | 3 (3–4) | 100 |
| 4 | Confianza Gobierno | 0,1679 | [0,1618; 0,1743] | 4 (4–5) | 100 |
| 5 | Distribución ingreso justa | 0,1487 | [0,1439; 0,1537] | 5 (5–6) | 83,4 |
| 6 | País para todos / poderosos | 0,1364 | [0,1344; 0,1386] | 6 (6–7) | 0 |
| 7 | Componente igualitario | 0,1285 | **[0,0959; 0,1740]** | 7 **(4–8)** | 16,6 |
| 8 | Economía país vs. año anterior | 0,0943 | [0,0893; 0,0989] | 8 (7–9) | 0 |
| 9 | Progreso contra corrupción | 0,0923 | [0,0855; 0,0980] | 9 (8–9) | 0 |
| 10 | Confianza Congreso | 0,0867 | [0,0832; 0,0902] | 10 (10–10) | 0 |
| 11 | Confianza Partidos Políticos | 0,0712 | [0,0696; 0,0729] | 11 (11–12) | 0 |
| 12 | Igualdad ante la ley | 0,0620 | **[0,0438; 0,0830]** | 12 **(11–15)** | 0 |
| 13 | Expectativa económica país | 0,0599 | [0,0553; 0,0646] | 13 (12–15) | 0 |
| 14 | Confianza Poder Judicial | 0,0571 | [0,0552; 0,0591] | 14 (12–15) | 0 |
| 15 | Integridad institucional (corrupción) | 0,0560 | **[0,0467; 0,0660]** | 15 **(12–15)** | 0 |

**Texto de acompañamiento (responde a la primera observación del revisor):**

> El ranking de importancia es una estimación puntual y su orden no es
> equivalente en todos los tramos. Para cuantificar esa incertidumbre se
> remuestrean con reemplazo los 16 clústeres país del conjunto de prueba
> (B = 1000 réplicas) y se recalcula tanto el valor `|SHAP|` medio como el rango
> de cada variable.
>
> El resultado distingue tres regímenes. Las cuatro primeras posiciones son
> estables: apoyo a la democracia y situación económica del país conservan las
> posiciones 1 y 2 en el 100 % de las réplicas, y aprobación del Gobierno y
> confianza en el Gobierno permanecen en el top-5 también en el 100 %. Un segundo
> grupo ocupa posiciones intercambiables: los intervalos de importancia de trece
> pares consecutivos del top-20 se solapan, de modo que el orden relativo dentro
> de esos pares no queda identificado por los datos. El tercer régimen es el de
> los indicadores contextuales de V-Dem, que son los más inestables: el componente
> igualitario tiene un intervalo de rango de la 4.ª a la 8.ª posición y permanece
> en el top-5 solo en el 16,6 % de las réplicas, mientras que igualdad ante la ley
> e integridad institucional oscilan entre las posiciones 11 y 15. La razón es
> estructural: los índices de V-Dem toman un único valor por país-año, de manera
> que al remuestrear países su contribución varía mucho más que la de las
> variables individuales, cuya variación se promedia sobre miles de encuestados.
>
> En consecuencia, la afirmación de que el componente igualitario es el indicador
> contextual de mayor importancia se sostiene —su intervalo no se solapa con el de
> los otros índices de V-Dem—, pero su posición exacta dentro del ranking global
> no debe interpretarse como un ordinal preciso.

## B.7 Nueva tabla — Importancia por bloque con IC (más estable que la individual)

Fuente: `results/tables/shap_bloques_ic_CatBoost_pesos_clase.csv` **[run anterior]**.

| # | Bloque | Variables | \|SHAP\| total | IC 95 % |
|---|---|---|---|---|
| 1 | Percepción política | 4 | 0,6290 | [0,6205; 0,6394] |
| 2 | Evaluación económica | 5 | 0,5613 | [0,5425; 0,5788] |
| 3 | Confianza institucional | 7 | 0,4697 | [0,4579; 0,4817] |
| 4 | Contexto democrático | 4 | 0,2655 | [0,2204; 0,3237] |
| 5 | Corrupción y seguridad | 3 | 0,1110 | [0,1036; 0,1173] |
| 6 | Características sociodemográficas | 5 | 0,0061 | [0,0058; 0,0063] |

**Texto:**

> A diferencia del ranking por variable, la jerarquía por bloque es completamente
> estable: **ningún par de bloques consecutivos presenta intervalos solapados**.
> El orden —percepción política, evaluación económica, confianza institucional,
> contexto democrático, corrupción y seguridad, características
> sociodemográficas— es por tanto robusto al remuestreo, mientras que el orden
> interno entre variables próximas no lo es. Esto respalda la decisión
> metodológica de §4.11.1 de reportar la importancia en dos niveles y de basar la
> discusión sustantiva en la agregación por bloque.

## B.8 Nueva tabla — Concordancia entre modelos empatados

Fuentes: `results/tables/shap_concordancia_modelos.csv`, `shap_top5_por_modelo.csv`.

> ⚠ Calculado con CatBoost y LightGBM; **falta XGBoost** porque su pipeline no se
> pudo deserializar en local. Al reejecutar el NB04 en el servidor se añadirá.

**[run anterior]** ρ de Spearman entre rankings = **0,9449**; W de Kendall =
**0,9711** (χ² = 52,44; p = 2,4 × 10⁻³; n = 28 variables).

Top-5 por modelo:

| # | CatBoost | LightGBM |
|---|---|---|
| 1 | Apoyo a la democracia | Apoyo a la democracia |
| 2 | Situación económica país | Situación económica país |
| 3 | Aprobación gobierno | Confianza Gobierno |
| 4 | Confianza Gobierno | Distribución ingreso justa |
| 5 | Distribución ingreso justa | Aprobación gobierno |

**Texto:**

> Como CatBoost y LightGBM tienen rendimientos cuya diferencia es de una
> centésima de kappa, cabe preguntarse si el ranking de importancia describe los
> datos o la elección de algoritmo. La correlación de Spearman entre ambos
> rankings es de 0,9449 y la W de Kendall de 0,9711, lo que indica un acuerdo
> alto. El conjunto de las cinco variables principales es **idéntico** en los dos
> modelos, aunque su orden interno difiere en las posiciones 3 a 5. La lectura
> sustantiva —el predominio de la percepción política y de la evaluación económica
> sobre la confianza institucional— no depende, por tanto, del algoritmo elegido;
> el orden preciso dentro del top-5, sí.

## B.9 §5.5.2 — Las curvas ALE cubrirán los tres bloques

Al reejecutar el NB04, la selección de variables ALE cambia: antes el bucle se
interrumpía tras el primer bloque y las seis curvas correspondían todas a
confianza institucional **[verificado en la salida ejecutada]**, en contradicción
con §4.11.2, que describe un recorrido por tres bloques. El código corregido toma
**las dos variables más importantes de cada bloque**: confianza institucional,
evaluación económica y percepción política.

**Consecuencia:** las Figuras 5.5–5.10 se sustituyen. Las nuevas serán, según el
ranking SHAP actual: Confianza Gobierno y Confianza Congreso; Situación económica
país y Distribución ingreso justa; Apoyo a la democracia y Aprobación gobierno.
El texto de §5.5.2 debe reescribirse sobre las curvas nuevas y §4.11.2 puede
mantenerse tal como está (ahora el código sí hace lo que describe).

## B.10 §5.5.3 — LIME pasa de 30 a 200 casos

La metodología declara 200 casos (100 representativos + 50 de error máximo + 50
discordantes) y la selección efectivamente los produce, pero el código explicaba
solo **los 10 primeros de cada grupo** **[verificado: la salida dice
«Representativo: 10 casos», «Error maximo: 10 casos», «Discordante: 10 casos»]**.
Ya está corregido a 100 por grupo.

**Consecuencia:** al reejecutar, la Tabla 5.10 y la afirmación de que «la
situación económica del país es la variable dominante» se sostendrán sobre 200
casos en lugar de 30. Conviene añadir el número de casos explicado en el pie de
la Tabla 5.10 y en §4.11.3.

---

# PARTE C — Incongruencias que requieren una decisión tuya

## C.1 CRÍTICO — La línea base «OLO» de la corrida anterior no era ordinal

> **Resuelto en el código.** Lo que hay que redactar está en **D.1**. Este
> apartado se conserva porque documenta el diagnóstico, que conviene tener a
> mano si el tribunal pregunta por la corrida anterior.


`utils/models.py` intenta `import mord` y, si falla, sustituye
`mord.LogisticIT` por una `LogisticRegression` multinomial de scikit-learn con
`C = 1/alpha`. El registro de hiperparámetros del run del servidor lo confirma
**[run anterior]**: los cuatro archivos `models/hp_OLO_*.json` contienen

```json
"hp_fijos": {"C": 134.06, "solver": "lbfgs", "max_iter": 500,
             "random_state": 42, "implementacion": "sklearn.LogisticRegression"}
```

Además, `mord` **no figuraba en `requirements.txt`** (ya lo añadí), de modo que
cualquier reproducción en un entorno limpio caería en el mismo fallback.

Esto afecta a: Resumen, Abstract, §2.3.1 (regresión logística ordinal,
McCullagh, supuesto de odds proporcionales), §4.9.1, H1, H2, §5.3.1, Tabla 5.3 y
Cap. 6. En todos esos lugares se describe y se contrasta un modelo ordinal que no
se entrenó; lo que se entrenó es una regresión logística **multinomial**, que es
precisamente el modelo cuya limitación describe §2.3.1 al final («trata los
códigos de clase como etiquetas y no incorpora que las respuestas forman una
secuencia»).

**Opción 1 (recomendada) — reentrenar solo OLO.** Es el modelo más rápido del
conjunto: 3 estrategias × 50 ensayos con un solo hiperparámetro. Añadí `mord>=0.7`
a `requirements.txt` y el NB02 imprime ahora el backend efectivo:

```
Backend OLO    : mord.LogisticIT (ordinal)
```

Verifica que ese mensaje aparezca antes de dar por bueno el run. Con esto, todo
el texto actual queda correcto y solo cambian las cifras de OLO.

**Opción 2 — documentar la línea base como multinomial.** Exige reescribir §2.3.1
(la línea base pasa a ser la multinomial de §2.3.1, no la ordinal de §2.3.1),
H1 y H2 («superan a la regresión logística **multinomial**»), el Resumen, el
Abstract y las conclusiones. Además debilita el argumento del capítulo 2, que
justifica el modelo ordinal por preservar el orden de la escala Likert.

**Nota adicional para cualquiera de las dos opciones:** aun con `mord` instalado,
`LogisticIT` minimiza la pérdida *immediate-threshold*, no la verosimilitud del
modelo de odds proporcionales de McCullagh (1980). Conserva umbrales ordenados y
un vector β compartido, así que la descripción conceptual de §2.3.1 es
defendible, pero conviene una nota al pie precisando el estimador:

> La implementación empleada (`mord.LogisticIT`) estima el modelo minimizando la
> pérdida de umbral inmediato en lugar de la verosimilitud del modelo de odds
> proporcionales; conserva la estructura de umbrales ordenados y el vector de
> coeficientes compartido, con regularización L2 sobre este último.

## C.2 Tabla 5.7 — La asignación de subregiones cambió

La Tabla 5.7 ubica a Perú en Cono Sur y deja la Región Andina con tres países,
pero `utils/config.py` ahora tiene Perú en la Región Andina. Al reejecutar, los
MAE por subregión de la Tabla 5.7 cambiarán, y también la muestra estratificada
de LIME y el contraste de H4/H5 del NB05. Además el README listaba la asignación
antigua (ya lo actualicé a la de `config.py`).

**Decide** cuál es la asignación definitiva y, si es la nueva, regenera la
Tabla 5.7 y el texto de §5.4 tras reejecutar NB03 y NB05.

## C.3 §5.3.1 — El brazo SMOTE-NC de los árboles no es *ceteris paribus*

> **Documentado en el código.** El NB02 lleva ahora esta advertencia en el
> encabezado de la estrategia C, con el texto que se propone abajo. Falta
> llevarla al documento.


§4.8 declara que los modelos de árboles conservan los NaN nativos. Eso es cierto
en los brazos «sin balanceo» y «pesos de clase», pero en el brazo SMOTE-NC el
código aplica `smote.fit_resample(X_tr_imp, y_tr)`, es decir, remuestrea sobre la
matriz **ya imputada con MICE** (SMOTE-NC no admite NaN). Por tanto, para
XGBoost, CatBoost y LightGBM la comparación entre estrategias mezcla dos
factores: remuestreo e imputación.

**Opción 1 (sin coste):** documentarlo. Añadir a §4.9.3:

> Como SMOTE-NC no admite valores faltantes, la estrategia C se aplica sobre la
> matriz de entrenamiento ya imputada. En consecuencia, para los modelos de
> árboles el brazo SMOTE-NC combina dos factores —sobremuestreo sintético e
> imputación previa— frente a los brazos A y B, que reciben los valores faltantes
> nativos. Las diferencias observadas en esa columna no deben atribuirse
> exclusivamente al remuestreo.

y a §6.2 como séptima limitación.

**Opción 2 (con coste de reejecución):** añadir un cuarto brazo de control
(imputado sin SMOTE) para separar ambos efectos.

## C.4 §5.2 y Cap. 6 — «24 transformaciones de escala, seis recodificaciones»

Estas cifras no son verificables contra el código. Lo que hay es: armonización de
tres variables económicas comparativas (`D_001_021`, `D_001_041`, `D_001_091`),
recodificación binaria de tres variables efectivas (`B_006_061`, `B_001_101`,
`S_001`; la cuarta del diccionario, `H_001_011`, se excluye antes en
`VARS_EXCLUIR_LB`), armonización longitudinal de la victimización `I_001_001`
sobre tres esquemas de codificación y el caso especial de `G_002_011` en la ola
2013. **Sustituye los conteos por esa enumeración**, que es auditable.

## C.5 Los pesos de clase se calculan sobre el conjunto de entrenamiento

> **Resuelto en el código.** El texto que hay que escribir está en **D.3**. El
> diagnóstico original se conserva a continuación.


`PESOS_CLASE` se computa en el NB02 justo tras recodificar el target, usando el
dataset completo (41.437 / 112.974 / 187.030 / 90.315 sobre 431.756 registros)
**[run anterior]**, es decir incorporando las frecuencias de clase de validación y
prueba. Es una filtración menor —solo marginales de clase, no observaciones— pero
contradice la «regla de oro anti-data leakage» que enuncia el propio notebook.

**Decide:** si quieres corregirlo (una línea: calcular las frecuencias solo sobre
`SPLIT["train"]`), entra en el bloque 3 porque cambia todos los resultados de E1.
Si no, documéntalo en §4.9.3:

> Los pesos inversos de clase se calculan sobre la distribución del conjunto
> integrado y no solo del entrenamiento. La información utilizada se limita a las
> frecuencias marginales de la variable objetivo y no a observaciones concretas de
> validación o prueba, pero conviene señalarlo explícitamente.

## C.6 §5 — El test de Friedman ya no se calcula

El NB03 calculaba un test de Friedman con cinco modelos y tres bloques (las
estrategias de balanceo) que el capítulo 5 no reportaba. **Se eliminó del
código**, por tres razones que conviene dejar dichas en el documento:

1. Con n = 3 bloques la potencia del contraste es prácticamente nula.
2. Los bloques no son conjuntos de datos independientes —el uso para el que
   Demšar (2006) propone el procedimiento— sino tres tratamientos del mismo
   conjunto de datos.
3. El contraste ignora la variabilidad muestral dentro del conjunto de prueba,
   que es la fuente de incertidumbre relevante aquí.

**Qué cambiar:** si el documento o el README mencionan Friedman o Nemenyi,
sustituirlo por el bootstrap pareado de clústeres (B.3 y B.4), que sí incorpora
esa variabilidad y además cuantifica el tamaño del efecto, no solo su
significancia. La §10 del NB03 contiene ahora esa justificación redactada y
puede reutilizarse casi literalmente.

---

# PARTE D — Pasajes que hay que reescribir tras la reejecución

Todo lo de esta parte ya está implementado en el código. Lo que queda es
redactar el documento con los resultados de la corrida nueva.

## D.1 §4.9 y Tabla 4.8 — La línea base es ordinal

`entrenar_olo()` usa `mord.LogisticIT`: un **logit acumulativo con umbrales de
umbral inmediato**, que estima un único vector de coeficientes β compartido por
todas las categorías y K−1 umbrales θ. El registro
`models/hp_OLO_pesos_clase_ordinal_4clases.json` guarda en
`config_entrenamiento` los campos `n_coeficientes` y `umbrales_theta` como
evidencia verificable de que el modelo es ordinal y no multinomial.

**Qué cambiar:**

- Describir la línea base como logit ordinal acumulativo, no como regresión
  logística multinomial ni como «regresión logística ordinal» sin más.
- El hiperparámetro optimizado es `alpha` (regularización L2), con 20 ensayos de
  Optuna. Justificación del presupuesto menor: `mord` resuelve el problema con
  L-BFGS-B sobre un objetivo implementado en Python puro, así que cada ensayo
  cuesta bastante más que uno de un modelo arbóreo.
- Reemplazar todas las métricas de OLO de las Tablas 5.1–5.4 y de la Tabla 5.5:
  cambian por completo respecto del run anterior.
- Añadir en §4.9 la nota de que la ausencia de `mord` **interrumpe** la
  ejecución en lugar de degradar silenciosamente a un modelo no ordinal.

## D.2 §4.9.4 y §5.3.1 — TabNet: el balanceo se aplica en la pérdida

En el run anterior los tres brazos de TabNet daban κw = 0.5127 idéntico porque
`fit(..., weights=1)` activaba el muestreo ponderado por clase en las tres
estrategias, con lo que las tres eran, en realidad, la misma configuración.

El código actual desactiva ese muestreo (`weights=0`) en las tres estrategias y
aplica el peso de clase donde corresponde: en la **función de pérdida**, con una
entropía cruzada ponderada por la frecuencia inversa de clase estimada sobre el
conjunto de entrenamiento.

**Qué cambiar:**

- En §4.9.4, describir el mecanismo: TabNet no admite `sample_weight` por
  registro, así que el desbalance se corrige en la pérdida y el muestreo se
  mantiene uniforme. Conviene señalar la consecuencia: **el factor de expansión
  muestral `X_020` no interviene en el ajuste de TabNet**, a diferencia del
  resto de los modelos. Es una asimetría del diseño y debe declararse en §6.2.
- En §5.3.1, eliminar la explicación del valor repetido (ya no se repite) y
  comentar la diferencia real entre los tres brazos.
- Verificación tras la corrida: en
  `results/tables/hiperparametros_modelos.csv`, la fila de TabNet con
  `pesos_clase` debe mostrar en `config_entrenamiento` el campo
  `loss_fn: CrossEntropyLoss ponderada por frecuencia inversa de clase` y el
  vector `pesos_clase`; la de `sin_balanceo`, `cross_entropy (por defecto)`.

## D.3 §4.9.3 — Los pesos de clase se estiman sobre el conjunto de entrenamiento

**Qué cambiar:** en la fórmula del peso de clase, sustituir el total del
conjunto integrado por el del conjunto de entrenamiento:

> Los pesos de clase se estiman exclusivamente sobre las olas de
> entrenamiento (1995–2018), como
> $w_c = n_{\text{train}} / (K \cdot n_{c,\text{train}})$. Calcularlos sobre el
> conjunto integrado introduciría información de las olas de validación y
> prueba en el ajuste del modelo.

El NB02 imprime el número de registros usados para estimarlos, de modo que la
cifra es verificable en la salida.

## D.4 Nueva sección — Estabilidad temporal del rendimiento

La §19 del NB02 replica el esquema de validación en tres pliegues históricos
con ventana de entrenamiento expansiva y origen fijo en 1995:

| Pliegue | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| 1 | 1995–2007 (12 olas) | 2008 | 2009–2010 |
| 2 | 1995–2010 (15 olas) | 2011 | 2013–2015 |
| 3 | 1995–2015 (18 olas) | 2016 | 2017–2018 |
| Definitivo | 1995–2018 (21 olas) | 2020 | 2023–2024 |

Cada pliegue reajusta imputación, escalado, pesos de clase e hiperparámetros
solo con su propia ventana de entrenamiento, y aplica las mismas reglas de
exclusión de países. Los conjuntos no se solapan y la prueba es siempre
posterior a la validación.

**Qué insertar** (sugerencia: nueva §5.3.4, «Estabilidad temporal del
rendimiento»), con los valores de
`results/tables/validacion_temporal_resumen.csv`:

| Modelo | κw medio ± sd | κw mín–máx | MAE ordinal medio ± sd |
|---|---|---|---|
| … | … ± … | …–… | … ± … |

Texto propuesto (ajustar las cifras tras la corrida):

> El diseño descansa en un único corte temporal, de modo que la métrica
> reportada no lleva asociada una medida de su variabilidad entre períodos. Para
> acotarla se replicó el mismo esquema en tres pliegues históricos con ventana
> de entrenamiento expansiva (Tabla X). El kappa cuadrático del modelo
> principal presenta una desviación estándar de [SD] entre los cuatro cortes,
> con un rango de [MÍN] a [MÁX]. [Interpretar: si la sd es pequeña respecto de
> las diferencias entre modelos de la Tabla 5.3c, el ordenamiento de modelos es
> estable entre períodos; si es del mismo orden, la comparación entre familias
> depende del período elegido y debe reportarse así.]

**Alcance que hay que declarar explícitamente**, porque limita la lectura:

- Se corre una sola estrategia de balanceo (`pesos_clase`), no las tres: el
  objetivo es medir estabilidad temporal, no repetir el experimento E1.
- El presupuesto de Optuna en los pliegues es de 15 ensayos frente a 50 en el
  corte definitivo, así que sus valores absolutos son ligeramente pesimistas.
  **Lo interpretable es la dispersión, no el nivel.**
- Los modelos de los pliegues no sustituyen a los reportados: sus
  hiperparámetros se registran con el sufijo `_foldN` y no se persisten
  pipelines.

Esto responde a la observación del revisor sobre la validación temporal sin
reclamar más de lo que el diseño permite.

## D.5 §5.3 — El bootstrap pasa a clústeres país-año

El NB02 persiste `año` y `peso_muestral` en los parquets, así que el bootstrap
del NB03 usa **32 clústeres país-año** en lugar de 16 clústeres país. Los
intervalos de confianza de la Tabla 5.3b y de la comparación pareada de la
Tabla 5.3c se estrechan respecto de las cifras [run anterior] de la Parte B.

**Qué cambiar:** en el pie de ambas tablas, indicar «bootstrap de 1.000
repeticiones sobre 32 clústeres país-año»; y revisar si alguna comparación que
antes no era distinguible pasa a serlo, porque eso altera la redacción de
§5.3.1.

## D.6 §2.4 y §5.3 — Métricas ponderadas por el factor de expansión

Con `REPORTAR_PONDERADAS = True`, el NB03 reporta cada métrica también en su
versión ponderada por `X_020`. Esto responde a la observación sobre el
estimando: las métricas sin ponderar describen el rendimiento sobre la muestra
efectivamente encuestada; las ponderadas, sobre la población que la muestra
representa.

**Qué insertar:** un párrafo en §2.4 explicando la distinción y, en §5.3, la
tabla `results/tables/metricas_ponderadas_vs_no_*.csv`. Si ambas lecturas
coinciden, decirlo: es un resultado de robustez.

## D.7 Pendiente no implementado — Estabilidad multi-semilla del ranking

Reentrenar el modelo principal con cinco semillas y añadir la matriz ρ
semilla × semilla a §5.5.2 **no está implementado**. Queda como limitación
declarada en §6.2:

> La incertidumbre del ranking de importancias se cuantifica por remuestreo del
> conjunto de prueba (bootstrap de clústeres país-año), lo que captura la
> variabilidad muestral. No se cuantifica la variabilidad debida a la
> inicialización aleatoria del entrenamiento, que requeriría reentrenar el
> modelo con varias semillas.

Si prefieres implementarlo, es una extensión acotada del bucle de la §19 del
NB02 y hay que decirlo antes de lanzar la corrida, porque conviene hacerlo en la
misma ejecución.

---

# PARTE E — Entradas BibTeX nuevas

Estas referencias se citan en los textos propuestos y **no están** en la
bibliografía actual:

```bibtex
@inproceedings{bergstra2011tpe,
  author    = {Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  title     = {Algorithms for Hyper-Parameter Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {24},
  year      = {2011}
}

@inproceedings{akiba2019optuna,
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {2623--2631},
  year      = {2019}
}

@book{efron1993bootstrap,
  author    = {Efron, Bradley and Tibshirani, Robert J.},
  title     = {An Introduction to the Bootstrap},
  publisher = {Chapman \& Hall},
  address   = {New York, NY, USA},
  year      = {1993}
}

@article{cameron2008bootstrap,
  author  = {Cameron, A. Colin and Gelbach, Jonah B. and Miller, Douglas L.},
  title   = {Bootstrap-Based Improvements for Inference with Clustered Errors},
  journal = {The Review of Economics and Statistics},
  volume  = {90}, number = {3}, pages = {414--427}, year = {2008}
}

@article{demsar2006statistical,
  author  = {Dem{\v{s}}ar, Janez},
  title   = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  journal = {Journal of Machine Learning Research},
  volume  = {7}, pages = {1--30}, year = {2006}
}

@article{vanbuuren2011mice,
  author  = {van Buuren, Stef and Groothuis-Oudshoorn, Karin},
  title   = {mice: Multivariate Imputation by Chained Equations in R},
  journal = {Journal of Statistical Software},
  volume  = {45}, number = {3}, pages = {1--67}, year = {2011}
}

@article{aas2021shapley,
  author  = {Aas, Kjersti and Jullum, Martin and L{\o}land, Anders},
  title   = {Explaining Individual Predictions When Features Are Dependent:
             More Accurate Approximations to Shapley Values},
  journal = {Artificial Intelligence},
  volume  = {298}, pages = {103502}, year = {2021}
}

@inproceedings{kumar2020shapley,
  author    = {Kumar, I. Elizabeth and Venkatasubramanian, Suresh and Scheidegger, Carlos and Friedler, Sorelle},
  title     = {Problems with Shapley-value-based Explanations as Feature Importance Measures},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  year      = {2020}
}

@incollection{molnar2022pitfalls,
  author    = {Molnar, Christoph and K{\"o}nig, Gunnar and Herbinger, Julia and Freiesleben, Timo
               and Dandl, Susanne and Scholbeck, Christian A. and Casalicchio, Giuseppe
               and Grosse-Wentrup, Moritz and Bischl, Bernd},
  title     = {General Pitfalls of Model-Agnostic Interpretation Methods for Machine Learning Models},
  booktitle = {xxAI --- Beyond Explainable AI},
  series    = {Lecture Notes in Computer Science},
  volume    = {13200}, pages = {39--68}, publisher = {Springer}, year = {2022}
}

@article{hooker2021permutation,
  author  = {Hooker, Giles and Mentch, Lucas and Zhou, Siyu},
  title   = {Unrestricted Permutation Forces Extrapolation: Variable Importance
             Requires at Least One More Model, or There Is No Free Variable Importance},
  journal = {Statistics and Computing},
  volume  = {31}, number = {6}, pages = {82}, year = {2021}
}
```

---

# PARTE F — Checklist

## FASE 1 — Antes de la reejecución

### Preparación (Parte 0)

- [ ] Copiar `models/hp_OLO_*.json` y `notebooks/output/` fuera del repositorio
- [ ] `git status` limpio en `results/`
- [ ] Prueba de humo completa sin errores: `MODO_EJECUCION=humo bash run_all.sh`
- [ ] `MODO_EJECUCION = "real"` en `utils/config.py` (o en el entorno)
- [ ] El banner de la primera celda dice `MODO CORRIDA REAL`
- [ ] La celda §2 del NB02 imprime `Backend OLO: mord.LogisticIT (ordinal)`
- [ ] La celda §2 del NB02 imprime `GPU solicitada / disponible : True / True`

### Redacción independiente de las cifras (Parte A)

- [ ] A.1 Corregir 465.000 → 489.771 / 430.895 en Resumen, Abstract, §4.3.1 y Cap. 6
- [ ] A.2 §4.8 imputación simple, no múltiple + limitación en §6.2
- [ ] A.3 §4.9.1 Optuna como optimización bayesiana + 20 ensayos en TabNet + no hay CV
- [ ] A.4 §4.9.2 «50 ensayos, excepto TabNet con 20»
- [ ] A.5 Reemplazar Tabla 4.11 (LightGBM)
- [ ] A.6 Tablas 4.9 y 4.10: paso 100
- [ ] A.7 §4.10 criterio de selección en validación
- [ ] A.8 Título y pie de la Tabla 5.5
- [ ] A.9 §2.4.6 ocho métricas calculadas, cinco reportadas
- [ ] A.10 §4.6 umbral de Spearman (elegir opción)
- [ ] A.11 §5.3.1 y §6.2: TabNet y el valor repetido
- [ ] A.12 §5.3.1 matizar la afirmación sobre el MAE
- [ ] A.13 Barrido predictivo vs. causal + párrafo de apertura de §5.5

### Decisiones de redacción (Parte C)

- [ ] C.1 **OLO ordinal vs. multinomial** ← la más importante (ver también D.1)
- [ ] C.2 Asignación de subregiones (Perú)
- [ ] C.3 SMOTE-NC e imputación (el NB02 ya lo documenta en el código)
- [ ] C.4 Enumerar transformaciones en lugar de contarlas
- [ ] C.5 Pesos de clase desde train (ver también D.3)
- [ ] C.6 Friedman: el NB03 ya no lo calcula; §5.3 remite al bootstrap

### Bibliografía (Parte E)

- [ ] Añadir las entradas BibTeX nuevas

---

## FASE 2 — Después de la reejecución

### Maquetar ahora, rellenar cifras después (Parte B)

- [ ] B.1 Insertar Tabla 4.5b de trazabilidad
- [ ] B.2 Insertar Tabla 5.3a (validación) + texto de coincidencia val/test
- [ ] B.3 Insertar Tabla 5.3b (IC por bootstrap)
- [ ] B.4 Insertar Tabla 5.3c (comparación pareada) + texto
- [ ] B.5 Nueva §5.3.3 con métricas por clase, matriz de confusión y 2 figuras
- [ ] B.6 Ampliar Tabla 5.8 con IC y rango + texto de los tres regímenes
- [ ] B.7 Insertar tabla de bloques con IC
- [ ] B.8 Insertar concordancia entre modelos
- [ ] B.9 Reescribir §5.5.2 tras regenerar las curvas ALE
- [ ] B.10 Actualizar §4.11.3 y el pie de la Tabla 5.10 con el número de casos LIME

### Pasajes que la corrida nueva obliga a reescribir (Parte D)

- [ ] D.1 §4.9 / Tabla 4.8: línea base ordinal + reemplazar todas las métricas de OLO
- [ ] D.2 §4.9.4 / §5.3.1: TabNet con pérdida ponderada; quitar la nota del valor repetido
- [ ] D.3 §4.9.3: fórmula del peso de clase sobre train
- [ ] D.4 Nueva §5.3.4: estabilidad temporal en pliegues históricos + su alcance
- [ ] D.5 §5.3: pies de tabla con «32 clústeres país-año»; revisar qué comparaciones cambian
- [ ] D.6 §2.4 / §5.3: métricas ponderadas por `X_020`
- [ ] D.7 §6.2: limitación sobre la estabilidad multi-semilla (no implementada)

### Revisión final de coherencia

- [ ] Ninguna cifra del documento marcada [run anterior] quedó sin reemplazar
- [ ] El número de casos LIME del texto coincide con `CASOS_LIME_*` de `config.py`
- [ ] El número de ensayos de Optuna del texto coincide con `N_TRIALS_*`
- [ ] Las referencias a secciones de los notebooks apuntan a la numeración final
- [ ] Todos los artefactos leídos traen `modo_ejecucion = real`, no `humo`

---

## Anexo I — De dónde se lee cada cifra

Todos los archivos son relativos a la raíz del repositorio y los regenera la
corrida completa. La columna «Depende de la corrida» indica si el contenido
cambia con la reejecución o es estructural.

### Tablas

| Archivo | Contenido | Sección del documento | Depende de la corrida |
|---|---|---|---|
| `results/tables/conjuntos_*.csv` | Trazabilidad de los conjuntos y de las exclusiones | Tabla 4.5b | No (estructural) |
| `results/tables/hiperparametros_modelos.csv` | Registro completo de hiperparámetros por modelo × estrategia × variante × pliegue | Tablas 4.9–4.12 y 5.5 | Sí |
| `results/resultados_modelos.csv` | Métricas de los 8 indicadores en validación y prueba | Tablas 5.1–5.4 | Sí |
| `results/tables/metricas_completas_val.csv` | Todas las métricas en validación | §5.3.1 | Sí |
| `results/tables/metricas_kappa_pivot_val.csv` | κw en validación, modelo × estrategia | Tabla 5.3a | Sí |
| `results/tables/mejor_estrategia_por_modelo.csv` | Coincidencia entre la mejor estrategia en validación y en prueba | §5.3.1 | Sí |
| `results/tables/bootstrap_ic_modelos.csv` | IC 95 % de cuatro métricas por configuración | Tabla 5.3b | Sí |
| `results/tables/bootstrap_pareado_vs_principal.csv` | Δ con IC y P(Δ>0) frente a la configuración principal | Tabla 5.3c | Sí |
| `results/tables/bootstrap_ic_principal_ponderado.csv` | Los mismos IC ponderados por `X_020` | §5.3.1 y D.6 | Sí |
| `results/tables/metricas_por_clase_*.csv` | Precision, recall y F1 por categoría | Tabla 5.4b | Sí |
| `results/tables/matriz_confusion_*.csv` | Conteos y porcentajes por fila | §5.3.3 | Sí |
| `results/tables/metricas_ponderadas_vs_no_*.csv` | Comparación de la lectura muestral y la poblacional | §5.3.3 y D.6 | Sí |
| **`results/tables/validacion_temporal_folds.csv`** | κw, MAE y F1 de cada modelo en cada pliegue | Nueva §5.3.4 (D.4) | Sí |
| **`results/tables/validacion_temporal_resumen.csv`** | Media, sd, mínimo y máximo entre pliegues | Nueva §5.3.4 (D.4) | Sí |
| `results/tables/shap_importancia_ic_*.csv` | \|SHAP\| con IC, IC del rango y % en top-k | Tabla 5.8 ampliada | Sí |
| `results/tables/shap_bloques_ic_*.csv` | Importancia agregada por bloque con IC | §5.5.1 | Sí |
| `results/tables/shap_concordancia_modelos.csv` | ρ de Spearman entre los rankings de los modelos empatados | §5.5.1 | Sí |
| `results/tables/shap_top5_por_modelo.csv` | Top-5 de cada modelo, comparado | §5.5.1 | Sí |
| `results/tables/lime_*.csv` | Pesos LIME por caso y variable, con grupo y distancia ordinal | Tabla 5.10 | Sí |
| `results/tables/mae_subregiones.csv` | MAE, κw y accuracy por subregión | Tabla 5.7 | Sí |
| `results/tables/mae_por_pais_todos.csv` | MAE por país y estrategia | §5.4 | Sí |
| `results/tables/contraste_teorico_*.csv` | % de convergencia con el bloque dominante de cada teoría | Cap. 6 | Sí |
| `results/tables/tabla_convergencias_*.csv` | Convergencia o divergencia por variable | Cap. 6 | Sí |
| `results/tables/tabla_maestra_xai_*.csv` | Importancias con bloque y rangos teóricos | Cap. 6 | Sí |
| `results/tables/correlacion_año_target.csv` | ρ de Spearman año vs. target | §4.3.3 | No (estructural) |

### Figuras

| Archivo | Contenido | Sección del documento |
|---|---|---|
| `results/figures/03_metricas_comparativas.png` | Comparativa de métricas entre modelos | §5.3.1 |
| `results/figures/03_metricas_por_clase.png` | Barras de precision, recall y F1 por categoría | §5.3.3 |
| `results/figures/03_confusion_modelo_principal.png` | Matriz de confusión del modelo principal, dos paneles | §5.3.3 |
| `results/figures/03_rendimiento_por_pais.png` | MAE ordinal por país | §5.4 |
| `results/figures/04_shap_bloques.png` | Importancia SHAP agrupada por bloque temático | §5.5.1 |
| `results/figures/04_shap_importancia_ic_*.png` | Importancia con barras de error del bootstrap | §5.5.1 |
| `results/figures/04_shap_concordancia_modelos.png` | Mapa de calor de concordancia entre modelos | §5.5.1 |
| `results/figures/04_shap_beeswarm.png` | Dirección e intensidad del efecto por variable | §5.5.1 |
| `results/figures/04_ale_*.png` | Curvas ALE, dos variables por bloque sustantivo | §5.5.2 |
| `results/figures/05_shap_subregiones.png` | Importancia SHAP por subregión | §5.6 |
| `results/figures/05_spearman_estabilidad.png` | ρ entre rankings SHAP de subregiones | §5.6 |
| `results/figures/06_convergencias_teoricas_*.png` | Mapa de calor bloque × teoría | Cap. 6 |
| `results/figures/06_tabla_convergencias_*.png` | Tabla de convergencias como figura | Cap. 6 |
| `results/figures/06_ranking_empirico_vs_teorico_*.png` | Ranking empírico frente a las predicciones teóricas | Cap. 6 |

### Registros de trazabilidad (no van al documento, sirven para verificar)

| Archivo | Para qué |
|---|---|
| `models/hp_OLO_*_ordinal_4clases.json` | `hp_fijos.implementacion` debe decir `mord.LogisticIT`, y `config_entrenamiento` debe traer `n_coeficientes` y `umbrales_theta` (D.1) |
| `models/hp_TabNet_pesos_clase_*.json` | `config_entrenamiento.loss_fn` debe indicar la entropía cruzada ponderada y `weights` debe ser 0 (D.2) |
| `models/hp_*_fold[123].json` | Confirman que los pliegues no sobrescribieron los hiperparámetros del corte definitivo (D.4) |
| `data/processed/nan_audit.json` | Variables con 100 % de ausentes en prueba, para la limitación de §6.2 |
| `notebooks/output/*.ipynb` | Salida completa de la corrida, con las cifras impresas en cada celda |

---

## Anexo II — Dónde se configura cada parámetro

Todos los parámetros de ejecución están en `PARAMETERS`, dentro de
`utils/config.py`, que es la unión de `_PARAMETERS_COMUNES` con el perfil que
seleccione `MODO_EJECUCION`. **Los valores de la tabla son los del perfil
`"real"`**; el perfil `"humo"` usa presupuestos mínimos y sus cifras no son
publicables. Los valores que el documento menciona explícitamente deben
coincidir con los del perfil real:

| Afirmación del documento | Clave de `PARAMETERS` | Valor |
|---|---|---|
| Semilla de reproducibilidad | `SEED` | 42 |
| Ensayos de Optuna en los árboles de gradiente | `N_TRIALS_OPTUNA` | 50 |
| Ensayos de Optuna en la línea base ordinal | `N_TRIALS_OLO` | 20 |
| Ensayos de Optuna en TabNet | `N_TRIALS_TABNET` | 20 |
| Épocas máximas de TabNet | `EPOCAS_TABNET` | 200 |
| Paciencia del early stopping de TabNet | `PACIENCIA_TABNET` | 20 |
| Épocas por ensayo de la búsqueda de TabNet | `EPOCAS_TABNET_OPTUNA` | 100 |
| Métrica de selección | `METRICA_PRINCIPAL` | `kappa_cuadratico` |
| Conjunto donde se selecciona | `CONJUNTO_SELECCION` | `val` |
| Repeticiones del bootstrap | `N_BOOTSTRAP` | 1000 |
| Nivel de confianza | `NIVEL_CONFIANZA` | 0.95 |
| Unidad de remuestreo | `NIVEL_CLUSTER` | `pais_anio` |
| Casos LIME representativos | `CASOS_LIME_REPRESENTATIVOS` | 100 |
| Casos LIME de error máximo | `CASOS_LIME_ERRORES` | 50 |
| Casos LIME discordantes | `CASOS_LIME_DISCORDANTES` | 50 |
| Variables con curva ALE por bloque | `VARS_ALE_POR_BLOQUE` | 2 |
| Variables en los gráficos de importancia | `TOP_N_SHAP` | 20 |
| Repeticiones del bootstrap del ranking | `N_BOOTSTRAP_SHAP` | 1000 |
| Estrategia de los pliegues temporales | `ESTRATEGIA_FOLDS` | `pesos_clase` |
| Ensayos de Optuna en los pliegues | `N_TRIALS_FOLDS` | 15 |

Las olas de cada conjunto están en `SPLIT` y los pliegues históricos en
`SPLITS_TEMPORALES`, en el mismo archivo.
