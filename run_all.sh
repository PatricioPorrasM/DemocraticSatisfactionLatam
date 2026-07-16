#!/bin/bash

set -e

# Ir a la raíz del proyecto
cd "$(dirname "$0")"

# Activar el entorno virtual
source .venv/bin/activate

mkdir -p logs

# Ejecutar los notebooks desde la carpeta notebooks
cd notebooks

mkdir -p output

papermill 01_carga_datos.ipynb \
           output/01_carga_datos.ipynb

papermill 02_preprocesamiento_entrenamiento.ipynb \
           output/02_preprocesamiento_entrenamiento.ipynb

papermill 03_evaluacion_comparativa.ipynb \
           output/03_evaluacion_comparativa.ipynb

papermill 04_explicabilidad_xai.ipynb \
           output/04_explicabilidad_xai.ipynb

papermill 05_estabilidad_temporal_regional.ipynb \
           output/05_estabilidad_temporal_regional.ipynb

papermill 06_contraste_teorico.ipynb \
           output/06_contraste_teorico.ipynb
           