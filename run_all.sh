#!/bin/bash

set -e

source .venv/bin/activate

papermill notebooks/01_carga.ipynb \
           notebooks/output/01_carga.ipynb

papermill notebooks/02_preprocesamiento_entrenamiento.ipynb \
           notebooks/output/02_preprocesamiento_entrenamiento.ipynb

papermill notebooks/03_xxx.ipynb \
           notebooks/output/03_xxx.ipynb

papermill notebooks/04_xxx.ipynb \
           notebooks/output/04_xxx.ipynb

papermill notebooks/05_xxx.ipynb \
           notebooks/output/05_xxx.ipynb

papermill notebooks/06_xxx.ipynb \
           notebooks/output/06_xxx.ipynb
           