FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependency layer - rebuilt only when requirements.txt changes
COPY inference/requirements.txt ./requirements.txt
RUN pip install --timeout=120 --retries=20 -r requirements.txt \
 && useradd --system --uid 10001 --no-create-home logiedge

# Runtime configuration - switch model variant without rebuilding
ENV MODEL_PATH=cold_chain_mlp_fp32.keras \
    TRUCK_ID=truck_001 \
    MQTT_BROKER=host.docker.internal \
    MQTT_PORT=1883

# Application code and normalisation statistics
COPY inference/inference_service.py /app/inference/inference_service.py
COPY data_pipeline/training_stats.npy /app/data_pipeline/training_stats.npy

# Model layer LAST: an OTA model update invalidates only this layer
COPY cold_chain_mlp_fp32.keras cold_chain_mlp_ptq_int8.tflite cold_chain_mlp_pruned_int8.tflite /app/inference/

USER logiedge
WORKDIR /app/inference
CMD ["python", "inference_service.py"]
