# logi-edge
Intelligent Edge-AI platform for Cold-chain logistics
# LogiBridge: IoT Edge-to-Cloud Cold-Chain Monitoring Pipeline

This repository contains the end-to-end implementation for the M.Tech cold-chain monitoring project. The pipeline covers dataset generation, model training (M1 FP32), Post-Training Quantization (M2 INT8), Structured Pruning + PTQ (M3), and containerized edge deployment with OTA layer caching (Task D2).

## 🛠️ Complete Command Reference

Follow these steps in order within your project root virtual environment (`.venv`) to reproduce the entire pipeline.

### 1. Environment Setup & Dependencies

Install all required libraries for TensorFlow, Model Optimization, Pandas, and MQTT:

```
.\.venv\Scripts\python.exe -m pip install tensorflow tensorflow-model-optimization pandas numpy scikit-learn paho-mqtt

```

### 2. Dataset Generation (Task D1)

Generate the synthetic sensor data streams representing Normal, Warning, and Critical operating conditions:

```
.\.venv\Scripts\python.exe training\generate_dataset.py

```

*Outputs:* `dataset_normal.csv`, `dataset_warning.csv`, `dataset_critical.csv`

### 3. Model Variant Training & Optimization

* **Variant M1 — FP32 Baseline Training:**
  Train the 2-hidden-layer MLP (32 and 16 units, ReLU activation) and verify it exceeds the 88% validation accuracy threshold:

```
  .\.venv\Scripts\python.exe training\train_model.py
  
```

  *Output:* `cold_chain_mlp_fp32.keras`

* **Variant M2 — Post-Training Quantization (PTQ INT8):**
  Apply Full INT8 quantization using calibration samples via `tf.lite.TFLiteConverter`:

```
  .\.venv\Scripts\python.exe training\convert_ptq.py
  
```

  *Output:* `cold_chain_mlp_ptq_int8.tflite`

* **Variant M3 — Structured Pruning + PTQ INT8:**
  Apply 35% structured filter pruning with a PolynomialDecay schedule followed by INT8 PTQ:

```
  .\.venv\Scripts\python.exe training\prune_quantise.py
  
```

  *Output:* `cold_chain_mlp_pruned_int8.tflite`

### 4. Docker Containerization & OTA Demo (Task D2)

* **Ensure Docker Engine is Running:**
  Start Docker Desktop on Windows:

```
  Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  
```

  Verify daemon health:

```
  docker ps
  
```

* **Build the Edge Container Image:**
  The image ships all three variants (`cold_chain_mlp_fp32.keras`, `cold_chain_mlp_ptq_int8.tflite`, `cold_chain_mlp_pruned_int8.tflite`), so run steps 3 (M1, M2, M3) first. From the project root directory, run the build command targeting the inference Dockerfile:

```
  docker build -f inference/Dockerfile -t logi-edge:v1 .
  
```

* **Run the Container (with Environment-Based Model Switching):**
  Execute the container dynamically switching model variants without a rebuild:

```
  docker run -e MODEL_PATH=cold_chain_mlp_ptq_int8.tflite -e TRUCK_ID=truck_001 logi-edge:v1
  
```

### 5. PSI Drift Monitoring (Task E1)

Monitors the distribution of the model's P(Normal) output (published as `probabilities[0]` on `logibridge/trucks/{truck_id}/inference`) with a 4-bin PSI.

Build the reference from 300 clean inferences (simulator running with `--anomaly none`, ~50 min):

```
.\.venv\Scripts\python.exe monitoring\drift_monitor.py --mode reference
```

Monitor the last 100 inferences, PSI printed every 60 s, `[LOGIBRIDGE DRIFT ALERT]` when PSI > 0.25:

```
.\.venv\Scripts\python.exe monitoring\drift_monitor.py --mode monitor --log monitoring\psi_log.csv
```

### 6. Ansible Deployment (Task E2)

`deployment/logibridge_deploy.yml` deploys the inference container to the edge node in exactly 7 idempotent tasks; a second run with no changes reports `changed=0`. Ansible does not run natively on Windows, so use WSL2 with **Ubuntu 24.04** (Ansible 14 requires Python 3.12+) and Docker Desktop's WSL integration enabled:

```
python3 -m venv ~/ansible-venv && ~/ansible-venv/bin/pip install ansible==14.5.0 docker==7.2.0
docker run -d --name registry -p 5000:5000 registry:2
docker tag logi-edge:v1 localhost:5000/logi-edge:v1 && docker push localhost:5000/logi-edge:v1
~/ansible-venv/bin/ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml -K
~/ansible-venv/bin/ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml -K
```

### 7. CI

`.github/workflows/ci.yml` runs on every push: sanity + unit tests → M1 training (gates: val accuracy > 88%, Critical recall > 95%) → M2 / M3 conversion (gated on INT8 accuracy) → image build → MQTT integration test of all three variants from the same image via `MODEL_PATH` → Ansible deployment run twice (second run must report `changed=0`).
