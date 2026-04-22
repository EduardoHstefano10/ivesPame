#!/usr/bin/env bash
# Entry-point del contenedor:
#   1. Si no hay dataset unificado, lo genera midiendo tiempo.
#   2. Lanza Streamlit.
set -euo pipefail

cd /app

CSV="data/processed/endes_2024_unificado.csv"
GZ="data/processed/endes_2024_unificado.csv.gz"

if [[ "${UNIFICAR:-auto}" == "force" ]] || [[ ! -f "$CSV" && ! -f "$GZ" ]]; then
    echo ">> [entrypoint] Generando dataset unificado..."
    START=$(date +%s)
    python scripts/unificar_datos.py
    END=$(date +%s)
    echo ">> [entrypoint] Unificacion completada en $((END - START)) s."
else
    echo ">> [entrypoint] Dataset unificado ya existe, se omite la unificacion."
    echo ">> [entrypoint] (usa UNIFICAR=force para regenerarlo)"
fi

echo ">> [entrypoint] Iniciando Streamlit en :${STREAMLIT_SERVER_PORT:-8501}..."
exec streamlit run app_streamlit.py
