#!/bin/bash

set -e

# Modo de ejecución: "real" (corrida definitiva) o "humo" (prueba rápida).
#
#   bash run_all.sh              -> usa el modo fijado en utils/config.py
#   MODO_EJECUCION=humo bash run_all.sh   -> prueba de humo
#   MODO_EJECUCION=real bash run_all.sh   -> corrida definitiva
#
# La variable se exporta para que la vean los kernels que lanza papermill.
if [ -n "$MODO_EJECUCION" ]; then
  export MODO_EJECUCION
  echo "MODO_EJECUCION=$MODO_EJECUCION"
fi

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
           