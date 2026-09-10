#!/bin/bash

# set -e aborta en el primer fallo. pipefail es imprescindible al enviar la
# salida por una tubería: sin él el estado de salida sería el de `tee` —casi
# siempre 0— y un notebook que falla no detendría la corrida, de modo que los
# siguientes se ejecutarían sobre artefactos incompletos.
set -e
set -o pipefail

# Modo de ejecución: "real" (corrida definitiva) o "humo" (prueba rápida).
#
#   bash run_all.sh                       -> usa el modo fijado en utils/config.py
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

# Carpeta de logs propia de cada corrida: una ejecución nueva no sobrescribe el
# registro de la anterior, que es la traza de las cifras ya citadas en el
# documento.
STAMP="$(date +%Y%m%d_%H%M%S)"
DIR_LOGS="logs/$STAMP"
mkdir -p "$DIR_LOGS"
echo "Logs de esta corrida: $DIR_LOGS"

# Ejecutar los notebooks desde la carpeta notebooks
cd notebooks

mkdir -p output

# --log-output envía la salida de cada celda al logger de papermill, que es lo
# que captura el `tee`: sin esa bandera el archivo solo contendría la barra de
# progreso. --no-progress-bar la suprime para que el log quede legible.
ejecutar() {
  local nb="$1"
  local log="../$DIR_LOGS/$nb.log"
  {
    echo "=============================================================="
    echo "  $nb"
    echo "  inicio : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  modo   : ${MODO_EJECUCION:-<el fijado en utils/config.py>}"
    echo "=============================================================="
  } | tee "$log"
  papermill "$nb.ipynb" "output/$nb.ipynb" \
    --log-output --no-progress-bar \
    2>&1 | tee -a "$log"
  echo "  fin    : $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
}

ejecutar 01_carga_datos
ejecutar 02_preprocesamiento_entrenamiento
ejecutar 03_evaluacion_comparativa
ejecutar 04_explicabilidad_xai
ejecutar 05_estabilidad_temporal_regional
ejecutar 06_contraste_teorico

echo ""
echo "Corrida completa. Logs en $DIR_LOGS"
