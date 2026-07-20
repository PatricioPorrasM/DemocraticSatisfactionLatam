# Explainable Tabular Deep Learning and Gradient Boosting Models for Predicting Satisfaction with Democracy in Latin America

Trabajo de titulación — Magíster en Inteligencia Artificial  
Universidad de Investigación de Tecnología Experimental Yachay · Julio 2026  
**Autor:** Mario Patricio Porras Martínez | **Tutor:** Ph.D. Erick Eduardo Cuenca Pauta

---

## Descripción

Este repositorio implementa un marco comparativo, explicable y reproducible para predecir la **satisfacción con la democracia** en América Latina a partir de datos tabulares de Latinobarómetro y V-Dem durante el periodo 1995–2024.

Se comparan cinco familias de modelos — Regresión Logística Ordinal (OLO), XGBoost, LightGBM, CatBoost y TabNet — bajo dos experimentos secuenciales: el primero evalúa estrategias de manejo del desbalance de clases; el segundo compara formulaciones de la variable objetivo. La explicabilidad se trabaja con SHAP (TreeSHAP/KernelExplainer), LIME y gráficos de efectos locales acumulados (ALE).

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
| **Latinobarómetro** | Encuesta regional de opinión pública | 24 olas: 1995–2024 (sin 1999 ni 2012); 489,771 registros |
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
│   ├── raw_v-dem/                ← V-Dem-CY-Core-v16.csv descomprimido
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
| `models/` | Pipelines serializados (`.pkl`) e hiperparámetros (`.json`) de los 15 modelos entrenados (5 algoritmos × 3 estrategias de balanceo), generados por NB02. |
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

Carga y armoniza las 24 olas de Latinobarómetro con los indicadores de V-Dem. Descomprime los archivos `.zip`, lee los 24 archivos Stata (`.dta`) detectando encoding, extrae y estandariza los nombres de columna heterogéneos entre olas, y consolida todo en un único DataFrame longitudinal de 489,771 registros × 43 columnas. Mapea los códigos de país (IDENPA) a nombres e ISO3. Carga y filtra V-Dem a los 18 países del estudio. Genera una tabla de frecuencias por ola y una muestra estratificada de 8,921 registros para inspección rápida.

**Genera:** `data/base/latinobarometro.csv`, `data/base/v-dem.csv`, `data/base/lb_frecuencia_valores_por_ola.csv`, `data/base/latinobarometro_muestra.csv`, `data/raw_latinobarometro/`, `data/raw_v-dem/`

---

### NB02 — `02_preprocesamiento_entrenamiento.ipynb`

Preprocesa los datos consolidados y ejecuta los dos experimentos del proyecto. Une Latinobarómetro × V-Dem por (pais_iso3, año). Limpia códigos NS/NR (-1 a -8). Armoniza escalas económicas entre olas (p. ej., escala de 3 puntos pre-2001 → equivalente de 5 puntos post-2001). Colapsa la victimización en variable binaria. Aplica exclusión de Venezuela (a partir de 2018) y Nicaragua (en val/test). Imputa valores faltantes con MICE (IterativeImputer + BayesianRidge, 10 iteraciones; ajuste solo en train). Normaliza con min-max. Construye el split temporal único.

**Experimento E1:** entrena 5 algoritmos × 3 estrategias de balanceo = 15 modelos. Cada modelo se optimiza con Optuna (TPE, 20–50 trials, objetivo: Kappa cuadrático en val).

**Experimento E2:** fija la mejor estrategia de balanceo de E1 y entrena los 5 algoritmos bajo 3 formulaciones del target (ordinal de 4 clases, binario, Likert continuo).

**Genera:** `data/processed/{train,val,test}.parquet`, `models/pipeline_*.pkl`, `models/hp_*.json`, `results/resultados_modelos.{csv,parquet}`

---

### NB03 — `03_evaluacion_comparativa.ipynb`

Responde la pregunta de investigación PI1: ¿qué familia de modelos ofrece el mejor equilibrio entre rendimiento predictivo, estabilidad temporal e interpretabilidad? Calcula 8 métricas en el conjunto de test para los 15 modelos de E1. Genera matrices de confusión normalizadas (% por clase real). Analiza MAE ordinal por país. Aplica prueba estadística de Friedman + post-hoc Nemenyi. Evalúa las 3 formulaciones de E2 por subregión. Selecciona el mejor modelo para la fase XAI y escribe su identificador en `results/modelo_xai_seleccionado.json`.

**Genera:** `results/tables/metricas_*.csv`, `results/tables/mae_por_pais_test.csv`, `results/figures/03_*.png`, `results/modelo_xai_seleccionado.json`

---

### NB04 — `04_explicabilidad_xai.ipynb`

Responde PI2 y OE4: ¿qué variables explican la satisfacción con la democracia y cuáles son sus efectos no lineales? Carga el mejor modelo seleccionado en NB03. Calcula valores SHAP globales (importancia por bloque temático) y locales (beeswarm por observación) usando TreeExplainer para modelos de árbol y KernelExplainer para OLO. Genera gráficos ALE para detectar efectos no lineales y umbrales. Aplica LIME sobre 200 casos: 100 representativos (estratificados por clase × subregión), 50 de mayor error ordinal y 50 con discordancia institucional (alta poliarquía + baja satisfacción predicha). Para TabNet incluye análisis de pesos de atención nativos. Documenta errores graves (distancia ordinal ≥ 2).

**Genera:** `results/shap/*.parquet`, `results/tables/shap_importancias_*.csv`, `results/tables/lime_*.csv`, `results/tables/errores_graves_*.csv`, `results/figures/04_*.png`

---

### NB05 — `05_estabilidad_temporal_regional.ipynb`

Responde PI3 y OE3: ¿son robustos los determinantes identificados a través de subregiones geográficas? Con el diseño de split único no es posible evaluar estabilidad temporal entre periodos; se evalúa la **estabilidad regional** comparando los rankings SHAP dentro del conjunto de test entre las 5 subregiones. Calcula correlaciones de Spearman entre pares de subregiones → prueba H5 (r ≥ 0.7 = determinantes robustos). Analiza la varianza entre bloques temáticos por región → prueba H4 (confianza/corrupción varían más que sociodemográficos). Genera heatmaps de estabilidad, bump charts de rankings y análisis de MAE ordinal por país y estrategia.

**Genera:** `results/tables/spearman_subregiones.csv`, `results/tables/mae_subregiones.csv`, `results/tables/mae_por_pais_todos.csv`, `results/figures/05_*.png`

---

### NB06 — `06_contraste_teorico.ipynb`

Responde OE5: ¿coinciden los patrones explicativos algorítmicos con las predicciones de la teoría democrática? Codifica cuatro marcos teóricos — Easton (1975), Norris (2011), Lewis-Beck & Stegmaier (2000) y Devine (2024) — según los bloques temáticos que priorizan. Cuantifica la convergencia como el porcentaje de variables top-N del SHAP que caen en el bloque predicho por cada teoría. Genera un heatmap de convergencia (bloque × teoría), una tabla de clasificación variable a variable (converge / parcial / diverge) y análisis de divergencias (variables importantes algorítmicamente pero no predichas por ninguna teoría). Prueba H3 (confianza + corrupción + economía ≥ 60% del top-15). Exporta tablas para el capítulo de discusión de la tesis.

**Genera:** `results/tables/contraste_teorico_*.csv`, `results/tables/tabla_convergencias_*.csv`, `results/tables/tabla_maestra_xai_*.csv`, `results/figures/06_*.png`

---

## Módulos Python (`utils/`)

### `utils/config.py`

Concentra toda la configuración del proyecto. Define rutas (`PATHS`), parámetros globales (`SEED=42`, `YEAR_START/END`), el split temporal (`SPLIT`), las 5 subregiones geográficas (`SUBREGIONES`), los 6 bloques temáticos de features (`BLOQUES`), las paletas de color por modelo y clase (`THEME`), y las etiquetas en español para variables y clases objetivo (`ETIQUETAS`, `ETIQUETAS_FEATURES`). Incluye también las listas de variables excluidas (`VARS_EXCLUIR_LB`, `VARS_EXCLUIR_VDEM`), el año de corte de Venezuela (`AÑO_CORTE_VEN=2017`) y los países excluidos de validación/test (`PAISES_EXCLUIR_EVAL`). Funciones exportadas: `setup_plots()`, `bloque_de(var)`, `clean_process_folders()`.

### `utils/io.py`

Funciones de entrada/salida para artefactos del proyecto. Carga y deserializa pipelines (`cargar_pipeline`), splits Parquet (`cargar_split_parquet`), métricas (`cargar_resultados`), matrices SHAP (`cargar_shap_values`, `guardar_shap_values`). Permite consultar el mejor modelo por métrica (`cargar_mejor_modelo`) y listar todos los pipelines disponibles (`listar_pipelines_disponibles`). Incluye manejo de fallback CPU para pipelines TabNet entrenados en GPU.

### `utils/metrics.py`

Función central `evaluar()` que calcula 8 métricas sobre cualquier par (y_true, y_pred): `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `kappa_lineal`, `kappa_cuadratico` (métrica primaria del proyecto), `mae_ordinal` y `auroc_macro`. Soporta salidas probabilísticas (`y_prob`) para AUROC y pesos de clase. Devuelve un diccionario con métricas y metadatos (modelo, estrategia, variante, split).

### `utils/preprocessing.py`

Transformaciones de datos previas al entrenamiento. `limpiar_nsnr()` convierte códigos NS/NR a NaN. `aplicar_transformaciones_deterministas()` armoniza escalas económicas entre olas y colapsa la victimización en binario. `construir_split()` crea los conjuntos train/val/test con las exclusiones de Venezuela/Nicaragua y calcula pesos muestrales compuestos. `imputar()` aplica MICE (BayesianRidge) para numéricas e imputación por moda para la variable categórica S_200, ajustando siempre solo sobre train. `normalizar()` aplica min-max (por defecto) o estandarización. `resumen_split()` imprime estadísticas de tamaño, distribución de clases y missingness.

### `utils/models.py`

Orquesta el entrenamiento de cada algoritmo. Una función por modelo (`entrenar_olo`, `entrenar_xgboost`, `entrenar_catboost`, `entrenar_lightgbm`, `entrenar_tabnet`, `entrenar_ridge`), todas con su propio bucle Optuna (TPE, maximizando Kappa cuadrático en val). Cada función serializa el pipeline completo (`.pkl`) y los hiperparámetros encontrados (`.json`). `predecir()` carga un pipeline existente, aplica las transformaciones necesarias y devuelve clase predicha, etiqueta y probabilidades por clase.

### `utils/plots.py`

Biblioteca de visualización con 80+ funciones organizadas por notebook. Funciones de apoyo comunes: `model_color()`, `save_figure()`. Para NB03: `plot_metricas_comparativas()`, `plot_matrices_confusion()`, `plot_rendimiento_por_pais()`. Para NB04: `plot_shap_bar_bloques()`, `plot_shap_beeswarm()`, `plot_ale()`. Para NB05: `plot_heatmap_estabilidad()`, `plot_bump_chart()`, `plot_spearman_estabilidad()`, `plot_shap_por_subregion()`. Para NB06: `plot_convergencias_teoricas()`, `plot_tabla_convergencias()`.

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

### Paso 2 — Ejecutar el pipeline

#### Opción A — Ejecución automática (recomendada)

Ejecuta los 6 notebooks en orden usando Papermill. Los notebooks ejecutados se guardan en `notebooks/output/` y los logs en `logs/`.

```bash
bash run_all.sh
```

#### Opción B — Ejecución manual en orden

Si se prefiere ejecutar notebook a notebook (por ejemplo, en Jupyter Lab o VS Code), respetar estrictamente el orden siguiente:

| Orden | Notebook | Descripción breve |
|---|---|---|
| 1 | `01_carga_datos.ipynb` | Carga y armoniza todas las olas de LB con V-Dem |
| 2 | `02_preprocesamiento_entrenamiento.ipynb` | Preprocesa, entrena E1 y E2, serializa modelos |
| 3 | `03_evaluacion_comparativa.ipynb` | Compara métricas, selecciona mejor modelo para XAI |
| 4 | `04_explicabilidad_xai.ipynb` | SHAP global/local, ALE y LIME |
| 5 | `05_estabilidad_temporal_regional.ipynb` | Estabilidad de rankings SHAP por subregión |
| 6 | `06_contraste_teorico.ipynb` | Contrasta patrones algorítmicos con teoría democrática |

> **Nota:** NB02 es el notebook de mayor duración (15 modelos con HPO Optuna). Se recomienda ejecutarlo en un entorno con GPU o con al menos 16 GB de RAM.

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

**Genera:** `models/pipeline_{modelo}_{estrategia}.pkl`, `models/hp_{modelo}_{estrategia}.json`, `results/resultados_modelos.parquet`

---

### Experimento E2 — Formulaciones del target

Fija la mejor estrategia de balanceo encontrada en E1 y evalúa los 5 algoritmos bajo **3 formulaciones distintas de la variable objetivo**, verificando si la codificación ordinal de 4 clases es óptima o si alternativas más simples o continuas ofrecen mejor rendimiento.

| Formulación | Descripción |
|---|---|
| `ordinal_4clases` | 4 clases ordinales; formulación principal — reutiliza los modelos de E1 |
| `binario` | 2 clases: {0,1}→Insatisfecho, {2,3}→Satisfecho |
| `likert_continuo` | Variable continua 0.0–3.0; Ridge regression con redondeo post-hoc |

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

- **Cono Sur:** Argentina, Chile, Uruguay, Paraguay, Perú
- **Región Andina:** Bolivia, Colombia, Ecuador
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
| `scikit-posthocs` | Pruebas post-hoc estadísticas (Nemenyi) |
