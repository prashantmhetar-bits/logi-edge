# Base Image specification
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# REQUIREMENT: All pip install layers must appear BEFORE copying the model
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variable to switch model variants dynamically without rebuilding code
ENV MODEL_PATH=cold_chain_mlp_fp32.keras
ENV TRUCK_ID=truck_001
ENV MQTT_BROKER=host.docker.internal
ENV MQTT_PORT=1883

# Copy application source code and model artifacts last
COPY . /app/

# Command to run the inference service
CMD ["python", "inference_service.py"]
