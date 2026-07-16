#!/bin/bash

set -e

source .venv/bin/activate

papermill notebooks/01_carga_datos.ipynb \
           notebooks/output/01_carga_datos.ipynb

papermill notebooks/02_preprocesamiento_entrenamiento.ipynb \
           notebooks/output/02_preprocesamiento_entrenamiento.ipynb

papermill notebooks/03_evaluacion_comparativa.ipynb \
           notebooks/output/03_evaluacion_comparativa.ipynb

papermill notebooks/04_explicabilidad_xai.ipynb \
           notebooks/output/04_explicabilidad_xai.ipynb

papermill notebooks/05_estabilidad_temporal_regional.ipynb \
           notebooks/output/05_estabilidad_temporal_regional.ipynb

papermill notebooks/06_contraste_teorico.ipynb \
           notebooks/output/06_contraste_teorico.ipynb
           