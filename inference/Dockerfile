FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY inference/requirements.txt ./requirements.txt
RUN pip install --timeout=120 --retries=20 -r requirements.txt \
 && useradd --system --uid 10001 --no-create-home logiedge \
 && mkdir -p /var/lib/logibridge && chown logiedge /var/lib/logibridge

ENV MODEL_PATH=cold_chain_mlp_pruned_int8.tflite \
    TRUCK_ID=truck_001 \
    MQTT_BROKER=host.docker.internal \
    MQTT_PORT=1883 \
    ALERT_LOG_PATH=/var/lib/logibridge/alerts.jsonl

WORKDIR /app/inference
COPY inference/inference_service.py inference/temperature_rule.py inference/alerting.py /app/inference/
COPY data_pipeline/training_stats.npy /app/data_pipeline/training_stats.npy

COPY training/models/cold_chain_mlp_fp32.keras training/models/cold_chain_mlp_ptq_int8.tflite training/models/cold_chain_mlp_pruned_int8.tflite /app/inference/

USER logiedge
CMD ["python", "inference_service.py"]