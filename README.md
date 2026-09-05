# Explainable Tabular Deep Learning and Gradient Boosting Models for Predicting Satisfaction with Democracy in Latin America

Trabajo de titulación — Magíster en Inteligencia Artificial  
Universidad de Investigación de Tecnología Experimental Yachay · Julio 2026  
**Autor:** Mario Patricio Porras Martínez | **Tutor:** Ph.D. Erick Eduardo Cuenca Pauta

---

## Descripción

Este repositorio implementa un marco comparativo, explicable y reproducible para predecir la **satisfacción con la democracia** en América Latina a partir de datos tabulares del Latinobarómetro y del V-Dem durante el periodo 1995–2024.

Se comparan cinco familias de modelos: Regresión Logística Ordinal (OLO), XGBoost, LightGBM, CatBoost y TabNet, bajo dos experimentos secuenciales: el primero evalúa estrategias de manejo del desbalance de clases; el segundo compara las formulaciones de la variable objetivo. La explicabilidad se trabaja con SHAP (TreeSHAP/KernelExplainer), LIME y gráficos de efectos locales acumulados (ALE).

**Variable objetivo:** satisfacción con la democracia (A_003_031), 4 clases ordinales:

| Clase | Etiqueta |
|---|---|
| 0 | Para nada satisfecho |
| 1 | No muy satisfecho |
| 2 | Más bien satisfecho |
| 3 | Muy satisfecho |

**Métrica primaria:** Kappa cuadrático (Cohen's kappa con pesos cuadráticos), que penaliza los errores de predicción proporcionalmente a la distancia ordinal entre clases.

---

## Fuentes de datos

| Fuente | Descripción | Cobertura |
|---|---|---|
| **Latinobarómetro** | Encuesta regional de opinión pública | 24 olas entre 1995 y 2024; 489,771 registros |
| **V-Dem Core v16** | Indicadores institucionales y democráticos | 18 países · 1995–2024; 540 registros |

Los archivos originales están **versionados en el repositorio** como `.zip` en `data/raw_zip/`. Los notebooks los descomprimen automáticamente en sus carpetas de trabajo.

> Para descarga directa: [Latinobarómetro](https://www.latinobarometro.org/latContents.jsp) · [V-Dem](https://www.v-dem.net/data/the-v-dem-dataset/)

---

## Estructura del proyecto

```
DemocraticSatisfactionLatam/
├── data/
│   ├── raw_zip/                  ← archivos originales versionados (.zip)
│   ├── raw_latinobarometro/      ← olas .dta descomprimidas (generado por NB01)
│   ├── raw_v-dem/                ← V-Dem-CY-Core-v16.csv descomprimido (generado por NB01)
│   ├── base/                     ← datasets consolidados (generado por NB01)
│   ├── processed/                ← datasets listos para ML (generado por NB02)
│   └── variables/
│       ├── latinobarometro_variable_mapping.csv
│       └── variables_selection.csv
├── logs/                         ← logs de ejecución (generado por run_all.sh)
├── models/                       ← pipelines entrenados e hiperparámetros (generado por NB02)
├── notebooks/
│   ├── 01_carga_datos.ipynb
│   ├── 02_preprocesamiento_entrenamiento.ipynb
│   ├── 03_evaluacion_comparativa.ipynb
│   ├── 04_explicabilidad_xai.ipynb
│   ├── 05_estabilidad_temporal_regional.ipynb
│   ├── 06_contraste_teorico.ipynb
│   └── output/                   ← notebooks ejecutados por papermill (generado por run_all.sh)
├── results/
│   ├── figures/                  ← visualizaciones PNG
│   ├── metrics/                  ← resúmenes estadísticos
│   ├── shap/                     ← matrices de valores SHAP (Parquet)
│   └── tables/                   ← tablas exportadas (CSV)
├── utils/
│   ├── config.py
│   ├── io.py
│   ├── metrics.py
│   ├── models.py
│   ├── plots.py
│   └── preprocessing.py
├── .gitignore
├── README.md
├── requirements.txt
└── run_all.sh
```

---

## Descripción de carpetas

| Carpeta | Contenido |
|---|---|
| `data/raw_zip/` | Archivos `.zip` originales de Latinobarómetro y V-Dem, versionados en el repositorio como fuente de verdad. |
| `data/raw_latinobarometro/` | 24 archivos Stata (`.dta`) descomprimidos automáticamente por NB01, uno por ola encuestada (1995–2024). |
| `data/raw_v-dem/` | Dataset CSV de V-Dem Core v16 descomprimido automáticamente por NB01. |
| `data/base/` | Datasets consolidados generados por NB01: Latinobarómetro armonizado (~490 K registros), V-Dem filtrado, tabla de frecuencias y muestra estratificada. |
| `data/processed/` | Splits listos para ML en formato Parquet (`train.parquet`, `val.parquet`, `test.parquet` y pesos de entrenamiento), generados por NB02. |
| `data/variables/` | Diccionario de variables: mapeo de códigos por ola (`latinobarometro_variable_mapping.csv`) y selección de 40 variables con etiquetas (`variables_selection.csv`). |
| `logs/` | Logs de ejecución generados por `run_all.sh` al ejecutar los notebooks con Papermill. |
| `models/` | Pipelines serializados (`.pkl`) y registros completos de hiperparámetros (`.json`) de los 15 modelos de E1 (5 algoritmos × 3 estrategias de balanceo) más los 5 de E2 (variante binaria), generados por NB02. |
| `notebooks/` | Pipeline de análisis compuesto por 6 notebooks numerados que deben ejecutarse en orden. |
| `notebooks/output/` | Copias ejecutadas de los notebooks generadas por `run_all.sh` vía Papermill. |
| `results/figures/` | Visualizaciones PNG generadas por NB03–NB06 (métricas comparativas, matrices de confusión, SHAP, ALE, estabilidad regional, contraste teórico). |
| `results/metrics/` | Resúmenes estadísticos y métricas de evaluación. |
| `results/shap/` | Matrices de valores SHAP por modelo y estrategia (formato Parquet), generadas por NB04. |
| `results/tables/` | Tablas exportadas en CSV: métricas por modelo/país, importancias SHAP, correlaciones de estabilidad y convergencias teóricas. |
| `utils/` | Módulos Python reutilizables compartidos por todos los notebooks. |

---

## Notebooks

Los notebooks deben ejecutarse **en orden secuencial**; cada uno genera los archivos que consume el siguiente.

### NB01 — `01_carga_datos.ipynb`

Carga y armoniza las 24 olas de Latinobarómetro con los indicadores de V-Dem. Descomprime los archivos `.zip`, lee los 24 archivos Stata (`.dta`) detectando encoding, extrae y estandariza los nombres de columna heterogéneos entre olas, y consolida todo en un único DataFrame longitudinal de 489,771 registros × 43 columnas. Mapea los códigos de país (IDENPA) a nombres e ISO3. Carga y filtra V-Dem a los 18 países del estudio. Genera una tabla de frecuencias por ola y una muestra estratificada de aproximadamente 8,500 registros para inspección rápida.

**Genera:** `data/base/latinobarometro.csv`, `data/base/v-dem.csv`, `data/base/lb_frecuencia_valores_por_ola.csv`, `data/base/latinobarometro_muestra.csv`, `data/raw_latinobarometro/`, `data/raw_v-dem/`

---

### NB02 — `02_preprocesamiento_entrenamiento.ipynb`

Preprocesa los datos consolidados y ejecuta los dos experimentos del proyecto. Une Latinobarómetro y V-Dem por (pais_iso3, año). Limpia códigos NS/NR (-1 a -8). Armoniza escalas económicas entre olas (p. ej., escala de 3 puntos pre-2001 → equivalente de 5 puntos post-2001). Colapsa la victimización en variable binaria. Aplica exclusión de Venezuela y Nicaragua (en val y test). Imputa valores faltantes con MICE (IterativeImputer + BayesianRidge, 10 iteraciones; ajuste solo en train). Normaliza con min-max. Construye el split temporal único.

**Tablas descriptivas de los conjuntos** (secciones 6 y 13): resumen por conjunto —olas, número de olas, registros y países— y registros por país en cada conjunto, calculadas dos veces (antes de las exclusiones y después de todas ellas), más una tabla del efecto acumulado de las exclusiones.

**EDA** (secciones 10 y 12): distribución del target, missingness por variable y conjunto, correlaciones de Spearman de cada feature con el target y tres matrices de correlación entre features —Latinobarómetro (nivel individual), V-Dem (nivel país-año) y dataset fusionado— para documentar la redundancia informativa.

**Experimento E1:** entrena 5 algoritmos × 3 estrategias de balanceo = 15 modelos. Cada modelo se optimiza con Optuna (TPE, maximizando el kappa cuadrático en validación): 50 ensayos para los árboles de gradiente y 20 para la línea base ordinal y para TabNet, cuyo costo por ensayo es mucho mayor. La línea base es una regresión logística ordinal acumulativa (`mord.LogisticIT`); si `mord` no está instalado la ejecución se detiene, en lugar de sustituirla por un modelo multinomial.

**Experimento E2:** fija la mejor estrategia de balanceo de E1 y entrena los 5 algoritmos bajo 2 formulaciones de la variable objetivo (ordinal de 4 clases y binaria).

**Validación temporal en pliegues históricos** (sección 19): replica el esquema de validación en tres cortes hacia atrás con ventana de entrenamiento expansiva (train 1995–2007 / val 2008 / test 2009–2010; train 1995–2010 / val 2011 / test 2013–2015; train 1995–2015 / val 2016 / test 2017–2018), definidos en `SPLITS_TEMPORALES`. Cada pliegue reajusta imputación, escalado, pesos de clase e hiperparámetros solo con su propia ventana de entrenamiento y aplica las mismas reglas de exclusión de países, de modo que la dispersión del kappa cuadrático y del MAE ordinal entre cortes acota la variabilidad temporal del rendimiento. Los pliegues no sustituyen a los modelos reportados: sus hiperparámetros se registran con el sufijo `_foldN` y no se persisten pipelines. Se controla con `PARAMETERS["EJECUTAR_FOLDS_TEMPORALES"]`.

**Genera:** `data/processed/{train,val,test}.parquet`, `models/pipeline_*.pkl`, `models/hp_*.json`, `results/tables/conjuntos_*.csv`, `results/tables/correlaciones_matriz_*.csv`, `results/tables/hiperparametros_modelos.csv`, `results/tables/validacion_temporal_folds.csv`, `results/tables/validacion_temporal_resumen.csv`, `results/resultados_modelos.{csv,parquet}`

---

### NB03 — `03_evaluacion_comparativa.ipynb`

Responde la pregunta de investigación PI1: ¿qué familia de modelos ofrece el mejor equilibrio entre rendimiento predictivo e interpretabilidad? Calcula 8 métricas agregadas en validación y en prueba para los 15 modelos de E1 y genera sus matrices de confusión normalizadas (% por clase real).

**Separación entre selección y evaluación:** la configuración principal (modelo × estrategia) se elige maximizando el kappa cuadrático en el conjunto de **validación** (2020); el conjunto de prueba (2023–2024) se usa solo para reportar. La selección se escribe en `results/modelo_xai_seleccionado.json`, que es la fuente única para los notebooks 04, 05 y 06.

**Modelo principal:** reporte detallado con la matriz de confusión en conteos y en porcentajes por clase real —con los errores ordinales graves (distancia ≥ 2 clases) resaltados— y el desglose de precision, recall y F1 por categoría del target con su soporte.

**Incertidumbre:** bootstrap de clústeres país-año (1.000 repeticiones) para el intervalo de confianza de cada métrica, y bootstrap pareado para la diferencia entre cada configuración y la principal. Es la única inferencia que el notebook hace sobre diferencias entre modelos: el test de Friedman con las estrategias como bloques se descartó porque con n = 3 bloques su potencia es nula, los bloques no son conjuntos de datos independientes y el contraste ignora la variabilidad muestral del conjunto de prueba.

**Métricas ponderadas:** cada métrica se reporta también ponderada por el factor de expansión muestral `X_020`, para distinguir el rendimiento sobre la muestra encuestada del rendimiento sobre la población que representa.

Analiza además el MAE ordinal por país y subregión y evalúa las formulaciones de E2.

**Genera:** `results/tables/metricas_*.csv`, `results/tables/metricas_por_clase_*.csv`, `results/tables/matriz_confusion_*.csv`, `results/tables/bootstrap_ic_modelos.csv`, `results/tables/bootstrap_pareado_vs_principal.csv`, `results/tables/mae_por_pais_test.csv`, `results/figures/03_*.png`, `results/modelo_xai_seleccionado.json`

---

### NB04 — `04_explicabilidad_xai.ipynb`

Responde PI2 y OE4: ¿qué variables explican la satisfacción con la democracia y cuáles son sus efectos no lineales? Carga el mejor modelo seleccionado en NB03. Calcula valores SHAP globales (importancia por bloque temático) y locales (beeswarm por observación) usando TreeExplainer para modelos de árbol y KernelExplainer para OLO. Genera gráficos ALE para detectar efectos no lineales y umbrales. Cuantifica la incertidumbre del ranking de importancias con un bootstrap de clústeres país-año: intervalo del valor |SHAP|, intervalo del rango de cada variable, porcentaje de réplicas en que entra en el top-k, y concordancia (ρ de Spearman y W de Kendall) entre los rankings de los modelos cuyo rendimiento el NB03 no distingue entre sí. Aplica LIME sobre 200 casos: 100 representativos (estratificados por clase × subregión), 50 de mayor error ordinal y 50 con discordancia institucional (alta poliarquía + baja satisfacción predicha); las tres cuotas se fijan en `PARAMETERS["CASOS_LIME_*"]`. Para TabNet incluye análisis de pesos de atención nativos. Documenta errores graves (distancia ordinal ≥ 2).

**Genera:** `results/shap/*.parquet`, `results/tables/shap_importancias_*.csv`, `results/tables/lime_*.csv`, `results/tables/errores_graves_*.csv`, `results/figures/04_*.png`

---

### NB05 — `05_estabilidad_temporal_regional.ipynb`

Responde PI3 y OE3: ¿son robustos los determinantes identificados a través de subregiones geográficas? Evalúa la **estabilidad regional** comparando los rankings SHAP dentro del conjunto de prueba entre las 5 subregiones. Calcula correlaciones de Spearman entre pares de subregiones → prueba H5 (r ≥ 0.7 = determinantes robustos). Analiza la varianza entre bloques temáticos por región → prueba H4 (confianza/corrupción varían más que sociodemográficos). Incluye el MAE ordinal por país y estrategia de balanceo. La estabilidad **temporal** del rendimiento se estima por separado en la sección 19 del NB02.

**Genera:** `results/tables/spearman_subregiones.csv`, `results/tables/mae_subregiones.csv`, `results/tables/mae_por_pais_todos.csv`, `results/figures/05_*.png`

---

### NB06 — `06_contraste_teorico.ipynb`

Responde OE5: ¿coinciden los patrones explicativos algorítmicos con las predicciones de la teoría democrática? Codifica cuatro marcos teóricos — Easton (1975), Norris (2011), Lewis-Beck & Stegmaier (2000) y Devine (2024) — según los bloques temáticos que priorizan. Cuantifica la convergencia como el porcentaje de variables top-N del SHAP que caen en el bloque predicho por cada teoría. Genera un heatmap de convergencia (bloque × teoría), una tabla de clasificación variable a variable (converge / parcial / diverge) y análisis de divergencias (variables importantes algorítmicamente pero no predichas por ninguna teoría). Prueba H3 (confianza + corrupción + economía ≥ 60% del top-15). Exporta tablas para el capítulo de discusión de la tesis.

**Genera:** `results/tables/contraste_teorico_*.csv`, `results/tables/tabla_convergencias_*.csv`, `results/tables/tabla_maestra_xai_*.csv`, `results/figures/06_*.png`

---

## Módulos Python (`utils/`)

### `utils/config.py`

Punto único de control del proyecto. `MODO_EJECUCION` selecciona el perfil de la corrida (`"real"` o `"humo"`, ver el Paso 2 de la sección de ejecución), y `PARAMETERS` —la unión de `_PARAMETERS_COMUNES` con el perfil activo de `PERFILES_EJECUCION`— reúne **todas** las variables que gobiernan la ejecución —uso de GPU y paralelismo, presupuesto de Optuna por familia de modelo, activación de los pliegues temporales, métrica y conjunto de selección, repeticiones y nivel de clúster del bootstrap, y los parámetros de SHAP, LIME y ALE—, cada una comentada con su valor sugerido para la corrida definitiva. Los notebooks leen de ahí y no definen banderas propias, de modo que reconfigurar un experimento se hace en un solo archivo.

Define además las rutas (`PATHS`), el corte temporal definitivo (`SPLIT`), los pliegues históricos (`SPLITS_TEMPORALES`), las 5 subregiones (`SUBREGIONES`), los 6 bloques temáticos de features (`BLOQUES`), las paletas por modelo y clase (`THEME`) y las etiquetas en español (`ETIQUETAS`, `ETIQUETAS_FEATURES`). Incluye las listas de variables excluidas (`VARS_EXCLUIR_LB`, `VARS_EXCLUIR_VDEM`), el año de corte de Venezuela (`AÑO_CORTE_VEN=2017`) y los países excluidos de validación y prueba (`PAISES_EXCLUIR_EVAL`).

Funciones exportadas: `resumen_modo()`, que imprime el perfil activo al inicio de cada notebook y avisa de forma llamativa cuando la corrida es una prueba de humo; `es_prueba_de_humo()`; `hw_cfg()`, que arma la configuración de hardware y presupuesto que reciben las funciones de entrenamiento y degrada a CPU si no hay GPU visible; `setup_plots()`; `bloque_de(var)`; y `clean_process_folders()`.

### `utils/io.py`

Funciones de entrada/salida para los artefactos del proyecto. Carga y deserializa pipelines (`cargar_pipeline`), conjuntos Parquet (`cargar_split_parquet`), métricas (`cargar_resultados`) y matrices SHAP (`cargar_shap_values`, `guardar_shap_values`, `ruta_shap`), y lista lo disponible con `listar_pipelines_disponibles()` y `listar_shap_disponibles()`. Incluye el respaldo a CPU para pipelines TabNet entrenados en GPU.

`modelo_xai_seleccionado()` resuelve el modelo principal y su estrategia leyendo la selección que escribe el NB03: como la estrategia forma parte del nombre de los archivos SHAP, los notebooks 04, 05 y 06 deben partir de la misma fuente para no buscar archivos inexistentes.

`normalizar_shap_2d()` lleva cualquier salida de `shap` a una matriz (n_muestras × n_features) identificando los ejes por su longitud, de modo que el código es estable ante los cambios de forma entre versiones de la librería y entre tipos de modelo.

`shap_vigente()` compara la fecha de entrenamiento del pipeline con la de modificación del Parquet de valores SHAP, para que un reentrenamiento invalide el caché en lugar de producir explicaciones de un modelo que ya no es el que se reporta.

### `utils/metrics.py`

Función central `evaluar()` que calcula 8 métricas agregadas sobre cualquier par (y_true, y_pred): `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `kappa_lineal`, `kappa_cuadratico` (métrica primaria del proyecto), `mae_ordinal` y `auroc_macro`. Soporta salidas probabilísticas (`y_prob`) para AUROC y pesos de clase. Devuelve un diccionario con métricas y metadatos (modelo, estrategia, variante, split).

Para el modelo principal el análisis desciende al nivel de categoría: `metricas_por_clase()` devuelve precision, recall, F1 y soporte de cada clase más los promedios macro y ponderado; `matriz_confusion_df()` devuelve la matriz de confusión etiquetada en conteos o en porcentajes (por fila, por columna o sobre el total); y `reporte_detallado()` combina las tres salidas, las imprime y las exporta a `results/tables/`.

### `utils/preprocessing.py`

Transformaciones de datos previas al entrenamiento. `limpiar_nsnr()` convierte códigos NS/NR a NaN. `construir_split()` crea los conjuntos de entrenamiento, validación y prueba con las exclusiones de Venezuela y Nicaragua y calcula los pesos muestrales compuestos; acepta el parámetro `split`, de modo que los pliegues históricos de `SPLITS_TEMPORALES` se construyen con las mismas reglas que el corte definitivo. `imputar()` aplica MICE (BayesianRidge) para numéricas e imputación por moda para la variable categórica S_200, ajustando siempre solo sobre train. `normalizar()` aplica min-max (por defecto) o estandarización. `resumen_split()` imprime estadísticas de tamaño, distribución de clases y missingness. `resumen_conjuntos()`, `conjuntos_por_pais()` y `tablas_conjuntos()` producen las tablas descriptivas de los conjuntos (olas, número de olas, registros y países; registros por país y conjunto), con la opción de aplicar o no las exclusiones de Venezuela y Nicaragua para comparar la composición antes y después de ellas.

`preparar_features_modelo()` devuelve una matriz numérica al formato exacto que espera un modelo ya entrenado: texto con la categoría `-999` para CatBoost, `pandas.Categorical` con las categorías del propio booster para LightGBM, y sin cambios para los demás. Es lo que permite que LIME y ALE, que solo manejan números, puedan llamar a `predict_proba`.

`predecir_conjunto()` reconstruye las predicciones de un pipeline sobre un conjunto completo aplicando la misma cadena de preprocesamiento del entrenamiento —imputación y escalado en OLO y TabNet, valores ausentes nativos y codificación categórica propia en los árboles—. Es la única implementación de esa cadena: los notebooks 03, 04 y 05 la usan en lugar de repetirla.

### `utils/models.py`

Orquesta el entrenamiento de cada algoritmo. Una función por modelo (`entrenar_olo`, `entrenar_xgboost`, `entrenar_catboost`, `entrenar_lightgbm`, `entrenar_tabnet`), todas con su propio bucle Optuna (TPE, maximizando el kappa cuadrático en validación).

`entrenar_olo` ajusta un logit ordinal acumulativo (`mord.LogisticIT`): un único vector de coeficientes compartido por todas las categorías y K−1 umbrales. El registro de hiperparámetros guarda el número de coeficientes y los umbrales estimados como evidencia de que el modelo es ordinal.

Las tres estrategias de balanceo se traducen a cada familia de modelos según lo que admite: OLO y los árboles de gradiente reciben `sample_weight` con el producto del factor de expansión muestral por el peso de clase; TabNet, que no admite pesos por registro, recibe el peso de clase en su **función de pérdida** (entropía cruzada ponderada) y muestrea de forma uniforme en las tres estrategias. Como consecuencia, el factor de expansión muestral no interviene en el ajuste de TabNet, lo que queda documentado como limitación del diseño.

Cada función guarda en `models/hp_{modelo}_{estrategia}_{variante}[_foldN].json` el registro **completo** de su configuración, no solo los parámetros que busca Optuna: `hp_optimizados` (los de Optuna), `hp_fijos` (objective, número de clases, semilla, device, n_jobs, verbosidad), `hp_completos` (la unión de ambos, que es lo que recibe el constructor), `espacio_busqueda` (tipo y rango de cada parámetro optimizado), `config_entrenamiento` (early stopping, épocas, batch size, eval_set, uso de sample_weight, iteración final) y `params_efectivos_modelo` (el `get_params()` del estimador entrenado, que incluye los valores por defecto de cada librería). Se consulta con `cargar_hiperparametros()` y se resume con `tabla_hiperparametros()`.

### `utils/plots.py`

Biblioteca de visualización organizada por notebook. Funciones de apoyo comunes: `model_color()`, `save_figure()`. Para NB02: `plot_matriz_correlacion()`, que dibuja la matriz de correlación (Spearman por defecto) con barra lateral de bloques temáticos y reporte de los pares con multicolinealidad. Para NB03: `plot_metricas_comparativas()`, `plot_matrices_confusion()`, `plot_matriz_confusion_modelo()`, `plot_metricas_por_clase()`, `plot_rendimiento_por_pais()`. Para NB04: `plot_shap_bar_bloques()`, `plot_importancia_con_ic()`, `plot_shap_beeswarm()`, `plot_ale()`, `plot_spearman_estabilidad()`. Para NB05: `plot_shap_por_subregion()`, `plot_spearman_estabilidad()`. Para NB06: `plot_convergencias_teoricas()`, `plot_tabla_convergencias()`.

---

## Instalación y ejecución

### Paso 1 — Instalar dependencias

Requiere **Python ≥ 3.10**.

```bash
# Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2 — Elegir el modo de ejecución

Hay un único interruptor, `MODO_EJECUCION` en `utils/config.py`, que ajusta de
una vez todos los parámetros que dependen del tamaño de la corrida:

| Modo | Para qué | Datos | Optuna | Épocas TabNet | Bootstrap | Duración |
|---|---|---|---|---|---|---|
| `"real"` | corrida definitiva, la que produce las cifras del documento | olas completas | 50 / 20 / 20 ensayos | 200 | 1.000 réplicas | horas, en GPU |
| `"humo"` | verificar que el flujo completo corre sin errores antes de lanzar la real | muestra reducida | 2 ensayos | 12 | 50 réplicas | minutos (NB02–NB06) |

El recorte de épocas de TabNet no es un detalle menor: con las 200 épocas del
perfil real, el ajuste de la red domina el tiempo y una prueba de humo pasaría
de minutos a horas. Con 12 épocas la red no converge —no es el objetivo— pero se
ejercitan igual el bucle de entrenamiento, el early stopping y la pérdida
ponderada por clase.

El NB01 lee y armoniza las olas de Latinobarómetro en los dos modos, así que su
duración no cambia; el ahorro está en los notebooks 02 a 06.

El perfil de humo **recorta el volumen de cada etapa, nunca desactiva una
etapa**: los pliegues temporales, el bootstrap, SHAP, LIME y ALE se ejecutan
igual, de modo que la prueba recorre las mismas rutas de código que la corrida
real. Sus cifras, en cambio, no son publicables, y los notebooks lo advierten
con un banner al inicio.

Se cambia editando una línea de `utils/config.py`:

```python
MODO_EJECUCION = "real"   # o "humo"
```

o sin tocar el archivo, con una variable de entorno:

```bash
MODO_EJECUCION=humo bash run_all.sh
```

Los parámetros que dependen del modo están en `PERFILES_EJECUCION`, y los que
no, en `_PARAMETERS_COMUNES`. Ambos perfiles deben declarar exactamente las
mismas claves: si falta una, `utils/config.py` falla al importarse en lugar de
dejar que la corrida se rompa a mitad de camino.

Para no confundir los artefactos de una prueba con los de la corrida
definitiva, el modo queda estampado en `results/resultados_modelos.csv`
(columna `modo_ejecucion`) y en cada `models/hp_*.json`.

### Paso 3 — Ejecutar el pipeline

#### Opción A — Ejecución automática (recomendada)

Ejecuta los 6 notebooks en orden usando Papermill. Los notebooks ejecutados se guardan en `notebooks/output/` y los logs en `logs/`.

```bash
# Primero una prueba de humo, para verificar que todo corre
MODO_EJECUCION=humo bash run_all.sh

# Y después la corrida definitiva
MODO_EJECUCION=real bash run_all.sh
```

**Nota**: Para ejecutar el proyecto en un servidor linux se recomienda usar la herramienta "tmux". Este proyecto fue ejecutado en un servidor con las siguientes características, y duró al rededor de 6 horas en promedio para completar la ejecución.

| Componente | Especificación |
|---|---|
| Sistema operativo | Ubuntu 24.04.3 LTS |
| Kernel | 7.0.0-28-generic |
| CPU | AMD Ryzen 9 9900X3D 12-Core Processor |
| Hilos lógicos | 24 |
| Memoria RAM | 123Gi |
| GPU | NVIDIA GeForce RTX 4090 |
| Memoria GPU | 24564 MiB |
| Driver NVIDIA | 595.71.05 |
| Python | Python 3.12.3 |

#### Opción B — Ejecución manual en orden

Si se prefiere ejecutar notebook a notebook (por ejemplo, en Jupyter Lab o VS Code), respetar estrictamente el orden siguiente, porque cada uno consume los artefactos del anterior:

| Orden | Notebook | Descripción breve |
|---|---|---|
| 1 | `01_carga_datos.ipynb` | Carga y armoniza todas las olas de LB con V-Dem |
| 2 | `02_preprocesamiento_entrenamiento.ipynb` | Preprocesa, entrena E1 y E2, serializa modelos |
| 3 | `03_evaluacion_comparativa.ipynb` | Compara métricas, selecciona mejor modelo para XAI |
| 4 | `04_explicabilidad_xai.ipynb` | SHAP global/local, ALE y LIME |
| 5 | `05_estabilidad_temporal_regional.ipynb` | Estabilidad de rankings SHAP por subregión |
| 6 | `06_contraste_teorico.ipynb` | Contrasta patrones algorítmicos con teoría democrática |

**El modo de ejecución se fija editando una línea de `utils/config.py`**, porque
al ejecutar a mano no hay variable de entorno que lo controle:

```python
MODO = "humo"   # o "real"
```

Tres cosas que hay que tener presentes al ejecutar notebook a notebook:

1. **Hay que reiniciar el kernel después de editar `MODO`.** Python conserva en
   memoria el módulo `utils.config` ya importado, así que un notebook con el
   kernel vivo seguirá usando el modo anterior. En Jupyter Lab: *Kernel →
   Restart Kernel*; en VS Code: *Restart* en la barra del notebook.

2. **El banner de la primera celda confirma el modo activo.** Antes de dar por
   buena una cifra, comprobar que dice `MODO CORRIDA REAL`; en modo de humo el
   aviso es imposible de pasar por alto.

3. **El NB01 hay que reejecutarlo al cambiar de modo.** Es el que genera la
   muestra reducida (`data/base/latinobarometro_muestra.csv`) con las cuotas
   `MIN/MAX_NUMBER_RECORDS` del perfil activo, y es la que carga el NB02 cuando
   `LOAD_SAMPLE=True`. Si la muestra en disco viene del otro perfil, el NB02 se
   detiene con un mensaje explícito en lugar de continuar con olas de validación
   demasiado pequeñas para representar las cuatro clases.

> **Nota:** NB02 es el notebook de mayor duración (15 modelos con HPO Optuna, más
> los pliegues temporales). Se recomienda ejecutarlo en un entorno con GPU o con
> al menos 16 GB de RAM.

---

## Diseño experimental

El proyecto ejecuta dos experimentos secuenciales sobre el mismo split temporal único.

### Split temporal

| Conjunto | Olas | Descripción |
|---|---|---|
| Train | 1995–2018 (21 olas) | Entrenamiento; Venezuela incluida hasta 2017 |
| Val | 2020 (1 ola) | Calibración Optuna; KS test p=0.787 vs. distribución del test |
| Test | 2023–2024 (2 olas) | Evaluación final; sin Venezuela ni Nicaragua |

**Casos especiales:** Venezuela se excluye a partir de 2018 por sesgo de respuesta documentado en regímenes autoritarios (KS test p<0.001 entre 2018 y el patrón histórico). Nicaragua se excluye de val/test por falta de cobertura en los años de prueba.

---

### Experimento E1 — Estrategias de balanceo de clases

Compara los **5 algoritmos** bajo **3 estrategias de manejo del desbalance de clases**, produciendo 15 modelos entrenados. Identifica qué combinación maximiza el Kappa cuadrático en el conjunto de test.

| Estrategia | Descripción |
|---|---|
| `sin_balanceo` | Sin ajuste; línea base sobre datos desbalanceados |
| `pesos_clase` | Pesos inversamente proporcionales a la frecuencia de cada clase |
| `smotenc` | Sobremuestreo sintético de clases minoritarias (SMOTE-NC para variables mixtas) |

**Genera:** `models/pipeline_{modelo}_{estrategia}.pkl`, `models/hp_{modelo}_{estrategia}_{variante}.json`, `results/resultados_modelos.parquet`

---

### Experimento E2 — Formulaciones de la variable objetivo

Fija la mejor estrategia de balanceo encontrada en E1 y evalúa los 5 algoritmos bajo **2 formulaciones distintas de la variable objetivo**, verificando si la codificación ordinal de 4 clases es óptima o si una alternativa más simple ofrece mejor rendimiento.

| Formulación | Descripción |
|---|---|
| `ordinal_4clases` | 4 clases ordinales; formulación principal — reutiliza los modelos de E1 |
| `binario` | 2 clases: {0,1}→Insatisfecho, {2,3}→Satisfecho |

> La formulación de regresión sobre la escala Likert continua se descartó del diseño experimental: la métrica principal (Kappa cuadrático) ya penaliza el error proporcionalmente a la distancia ordinal, por lo que la comparación relevante es ordinal vs. binaria.

**Genera:** `results/tables/metricas_e2_variantes_target.csv`

---

## Hipótesis

| ID | Enunciado | Notebook de contraste |
| --- | --- | --- |
| H1 | Los modelos de gradient boosting superan a la regresión logística ordinal en Kappa cuadrático | NB03 |
| H2 | Las estrategias de balanceo mejoran el F1 de la clase minoritaria (clase 0) respecto a la línea base sin balanceo | NB03 |
| H3 | Los bloques de confianza institucional, corrupción y evaluación económica concentran ≥ 60% de las variables del top-15 SHAP | NB06 |
| H4 | Los determinantes de corrupción y confianza presentan mayor variación regional que los factores sociodemográficos | NB05 |
| H5 | La correlación de Spearman entre rankings SHAP de distintas subregiones es ≥ 0.7, indicando determinantes robustos en toda América Latina | NB05 |

---

## Bloques de features

| Bloque | Variables |
|---|---|
| Confianza institucional (7) | Congreso, Gobierno, Poder Judicial, Policía, Televisión, FF.AA., Partidos Políticos |
| Evaluación económica (5) | Situación económica país, economía vs. año anterior, expectativa económica país, expectativa personal, distribución ingreso justa |
| Percepción política (4) | Apoyo a la democracia, interés en política, país para todos/poderosos, aprobación gobierno |
| Corrupción y seguridad (3) | Conocimiento de caso de corrupción, progreso contra corrupción, victimización delictiva |
| Características sociodemográficas (5) | Sexo, edad, nivel educativo, situación ocupacional, nivel socioeconómico |
| Contexto democrático — V-Dem (4) | Democracia electoral (poliarquía), componente igualitario, integridad institucional (corrupción), igualdad ante la ley |

**Países (18) en 5 subregiones:**

- **Cono Sur:** Argentina, Chile, Uruguay, Paraguay
- **Región Andina:** Bolivia, Colombia, Ecuador, Perú
- **Brasil:** Brasil
- **Centroamérica:** Costa Rica, El Salvador, Guatemala, Honduras, Panamá
- **México y Caribe:** México, República Dominicana

---

## Dependencias principales

| Paquete | Uso |
|---|---|
| `xgboost`, `lightgbm`, `catboost` | Modelos de gradient boosting |
| `pytorch-tabnet` | Modelo de deep learning tabular |
| `scikit-learn` | Regresión logística ordinal, métricas, imputación MICE, pipelines |
| `shap` | Explicabilidad global y local (TreeSHAP, KernelExplainer) |
| `lime` | Explicaciones locales por instancia |
| `alibi` | Gráficos ALE (Accumulated Local Effects) |
| `optuna` | Optimización de hiperparámetros (TPE sampler) |
| `pyreadstat` | Lectura de archivos `.dta` (Stata) |
| `imbalanced-learn` | Manejo de desbalance de clases (SMOTE-NC) |
| `papermill` | Ejecución parametrizada de notebooks desde `run_all.sh` |
| `pyarrow` | Lectura y escritura de archivos Parquet |
