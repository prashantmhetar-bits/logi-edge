# Base Image specification
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# REQUIREMENT: All pip install layers must appear BEFORE copying the model
COPY inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout=120 --retries=20 -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Environment variable to switch model variants dynamically without rebuilding code
ENV MODEL_PATH=cold_chain_mlp_fp32.keras
ENV TRUCK_ID=truck_001
ENV MQTT_BROKER=host.docker.internal
ENV MQTT_PORT=1883

# Copy application source, the root-level model artifact, and training statistics last
COPY inference/ /app/inference/
COPY cold_chain_mlp_fp32.keras /app/inference/cold_chain_mlp_fp32.keras
COPY data_pipeline/training_stats.npy /app/data_pipeline/training_stats.npy

# Command to run the inference service
WORKDIR /app/inference
CMD ["python", "inference_service.py"]
