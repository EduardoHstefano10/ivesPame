FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Dependencias del sistema mínimas (para compilación de libs numéricas si hiciera falta).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1) Instala dependencias (capa cacheable).
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) Copia el código del proyecto.
COPY . .

RUN sed -i 's/\r$//' docker/*.sh && chmod +x docker/entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/app/docker/entrypoint.sh"]
