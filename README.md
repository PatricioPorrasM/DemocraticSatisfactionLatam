# Explainable Tabular Deep Learning and Gradient Boosting Models for Predicting Satisfaction with Democracy in Latin America

Trabajo de titulación — Magíster en Inteligencia Artificial  
Universidad de Investigación de Tecnología Experimental Yachay · Julio 2026  
**Autor:** Mario Patricio Porras Martínez | **Tutor:** Ph.D. Erick Eduardo Cuenca Pauta

---

## Descripción

Este repositorio implementa un marco comparativo, explicable y reproducible para predecir la **satisfacción con la democracia** en América Latina a partir de datos tabulares de Latinobarómetro y V-Dem durante el periodo 1995–2024.

Se comparan cinco familias de modelos — Regresión Logística Ordinal (OLO), XGBoost, LightGBM, CatBoost y TabNet — bajo un protocolo común de **validación temporal walk-forward** (expanding window) en tres subperiodos históricos. La explicabilidad se trabaja con SHAP (TreeSHAP), LIME y gráficos de efectos locales acumulados (ALE).

**Variable objetivo:** satisfacción con la democracia, 4 clases ordinales:
| Clase | Etiqueta |
|---|---|
| 0 | Para nada satisfecho |
| 1 | No muy satisfecho |
| 2 | Más bien satisfecho |
| 3 | Muy satisfecho |

---

## Fuentes de datos

| Fuente | Descripción | Cobertura |
|---|---|---|
| **Latinobarómetro** | Encuesta regional de opinión pública | 24 olas: 1995–2024 (sin 1999 ni 2012) |
| **V-Dem Core v16** | Indicadores institucionales y democráticos | 18 países · 1995–2024 |

Los archivos originales están **versionados en el repositorio** como `.zip` en `data/raw_zip/`. Los notebooks los descomprimen automáticamente en sus carpetas de trabajo.

> Para descarga directa: [Latinobarómetro](https://www.latinobarometro.org/latContents.jsp) · [V-Dem](https://www.v-dem.net/data/the-v-dem-dataset/)

---

## Estructura del proyecto

```
DemocraticSatisfactionLatam/
├── data/
│   ├── raw_zip/                  ← archivos originales versionados (.zip)
│   ├── raw_latinobarometro/      ← olas .dta descomprimidas (generado por nb 01)
│   ├── raw_v-dem/                ← V-Dem-CY-Core-v16.csv descomprimido
│   ├── base/                     ← datasets consolidados (generado por nb 01)
│   ├── processed/                ← datasets listos para ML (generado por nb 02)
│   └── variables/
│       ├── latinobarometro_variable_mapping.csv
│       └── variables_selection.csv
├── models/                       ← modelos entrenados (generado por nb 02)
├── notebooks/
│   ├── 01_carga_datos.ipynb
│   ├── 02_preprocesamiento_entrenamiento.ipynb
│   ├── 03_evaluacion_comparativa.ipynb
│   ├── 04_explicabilidad_xai.ipynb
│   ├── 05_estabilidad_temporal_regional.ipynb
│   └── 06_contraste_teorico.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── shap/
│   └── tables/
├── utils/
│   ├── config.py       ← paths, subperiodos, bloques temáticos, paletas
│   ├── io.py
│   ├── metrics.py
│   ├── models.py
│   ├── plots.py
│   └── preprocessing.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación

Requiere **Python ≥ 3.10**.

Para usar un entorno virtual:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Cómo ejecutar

Los notebooks deben ejecutarse **en orden**; cada uno genera los archivos que consume el siguiente.

| Notebook | Descripción | Genera |
|---|---|---|
| `01_carga_datos` | Carga y armoniza todas las olas de LB con V-Dem | `data/base/` |
| `02_preprocesamiento_entrenamiento` | Preprocesa, entrena y serializa los 5 modelos | `data/processed/`, `models/` |
| `03_evaluacion_comparativa` | Compara métricas entre modelos y subperiodos | `results/metrics/`, `results/tables/` |
| `04_explicabilidad_xai` | Análisis SHAP global/local, LIME y ALE | `results/shap/`, `results/figures/` |
| `05_estabilidad_temporal_regional` | Estabilidad de rankings SHAP por subperiodo y subregión | `results/figures/` |
| `06_contraste_teorico` | Contrasta patrones explicativos con teoría democrática | `results/figures/` |

La configuración de rutas, subperiodos y bloques temáticos se centraliza en [utils/config.py](utils/config.py).

---

## Diseño experimental

**Subperiodos — Expanding Window Walk-Forward**

| Subperiodo | Entrenamiento | Validación | Test | Descripción |
|---|---|---|---|---|
| SP1 | 1995–2005 | 2006 | 2007 | Consolidación democrática |
| SP2 | 1995–2016 | 2017 | 2018 | Crisis y reconfiguración |
| SP3 | 1995–2020 | 2023 | 2024 | Pandemia y recuperación |

**Países:** 18 países en 5 subregiones (Cono Sur, Región Andina, Brasil, Centroamérica, México y Caribe).

**Bloques de features**

| Bloque | Variables representativas |
|---|---|
| Confianza institucional | Confianza en Gobierno, Congreso, Poder Judicial, Policía, FF.AA., Partidos |
| Evaluación económica | Situación económica país, expectativas, desempleo, distribución del ingreso |
| Percepción política | Apoyo a la democracia, aprobación de gobierno, ideología |
| Corrupción y seguridad | Conocimiento de corrupción, progreso contra corrupción, victimización |
| Sociodemográfico | Sexo, edad, educación, situación ocupacional, nivel socioeconómico |
| Contexto democrático (V-Dem) | Índices de poliarquía, democracia liberal, Estado de derecho, corrupción institucional |

---

## Dependencias principales

| Paquete | Uso |
|---|---|
| `xgboost`, `lightgbm`, `catboost` | Modelos de gradient boosting |
| `pytorch-tabnet` | Modelo de deep learning tabular |
| `scikit-learn` | Regresión logística ordinal, métricas, pipelines |
| `shap` | Explicabilidad global y local (TreeSHAP) |
| `lime` | Explicaciones locales |
| `alibi` | Gráficos ALE |
| `optuna` | Optimización de hiperparámetros |
| `pyreadstat` | Lectura de archivos `.dta` (Stata) |
| `imbalanced-learn` | Manejo de desbalance de clases |
