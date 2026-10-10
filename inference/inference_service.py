import os
import time
import json
import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf
import paho.mqtt.client as mqtt

# Read environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "../cold_chain_mlp_fp32.keras")
TRUCK_ID = os.getenv("TRUCK_ID", "truck_001")
BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", 1883))

TOPIC_INFERENCE = f"logi-edge/trucks/{TRUCK_ID}/inference"

print(f"🚀 Initializing Edge Inference Service...")
print(f"-> Target Model: {MODEL_PATH}")
print(f"-> Publishing to MQTT Topic: {TOPIC_INFERENCE}")

# Load TFLite Model & Allocate Tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Buffers for sliding window filtering
temp_buffer = deque(maxlen=5)
vib_buffer = deque(maxlen=5)
temp_window = deque()
vib_window = deque()
last_step_time = time.time()

# Load runtime normalization stats (training_stats.npy)
stats_path = "training_stats.npy"
if os.path.exists(stats_path):
    stats = np.load(stats_path, allow_pickle=True).item()
    mean_vec = stats['mean']
    std_vec = stats['std']
    print("✓ Loaded runtime normalization stats successfully.")
else:
    print("⚠️ Warning: training_stats.npy not found! Using zero-mean unit-variance defaults.")
    mean_vec = np.zeros(6)
    std_vec = np.ones(6)

def process_and_infer(client, temp_val, vib_val):
    global last_step_time
    current_time = time.time()

    # 1. Filtering (5-sample moving average)
    temp_buffer.append(temp_val)
    vib_buffer.append(vib_val)
    filtered_temp = sum(temp_buffer) / len(temp_buffer)
    filtered_vib = sum(vib_buffer) / len(vib_buffer)

    temp_window.append((current_time, filtered_temp))
    vib_window.append((current_time, filtered_vib))

    # 2. Feature Extraction per 30s window (10s step)
    if current_time - last_step_time >= 10.0:
        last_step_time = current_time
        
        # Prune old window data (>30s)
        cutoff = current_time - 30.0
        while temp_window and temp_window[0][0] < cutoff: temp_window.popleft()
        while vib_window and vib_window[0][0] < cutoff: vib_window.popleft()

        if len(temp_window) >= 5 and len(vib_window) >= 3:
            temps = np.array([v for t, v in temp_window])
            vibrs = np.array([v for t, v in vib_window])

            t_mean = np.mean(temps)
            t_std = np.std(temps)
            time_arr = np.array([t for t, v in temp_window])
            time_min = (time_arr - time_arr[0]) / 60.0
            t_roc = np.polyfit(time_min, temps, 1)[0] if len(np.unique(time_min)) > 1 else 0.0

            v_rms = np.sqrt(np.mean(vibrs**2))
            v_peak = np.max(np.abs(vibrs))
            v_kurt = np.array(vibrs).var() # simplified for stream

            raw_features = np.array([t_mean, t_std, t_roc, v_rms, v_peak, v_kurt], dtype=np.float32)

            # 3. Normalisation using loaded stats
            norm_features = (raw_features - mean_vec) / (std_vec + 1e-8)

            # 4. Model Inference
            input_data = np.expand_dims(norm_features, axis=0)
            
            if MODEL_PATH.endswith(".tflite"):
                if input_details[0]['dtype'] == np.int8:
                    # Quantize input if model expects INT8
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = np.clip(np.round(input_data / input_scale + input_zero_point), -128, 127).astype(np.int8)
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = int(np.argmax(output_data[0]))
            else:
                # Fallback for keras model if tested locally
                predicted_class = 0

            class_labels = {0: "Normal", 1: "Warning", 2: "Critical"}
            result_payload = {
                "truck_id": TRUCK_ID,
                "timestamp": current_time,
                "predicted_class": predicted_class,
                "status": class_labels.get(predicted_class, "Unknown"),
                "features": raw_features.tolist()
            }

            # 5. Publish Result to MQTT
            client.publish(TOPIC_INFERENCE, json.dumps(result_payload))
            print(f"[{TRUCK_ID}] Published Inference -> Status: {class_labels.get(predicted_class)}")

# MQTT Client Setup
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
def on_message(c, ud, msg):
    try:
        payload = json.loads(msg.payload.decode())
        val = payload.get("value", 0.0)
        if "temperature" in msg.topic:
            # We pair incoming messages or trigger on temp
            pass
    except:
        pass

client.connect(BROKER, PORT, 60)
client.subscribe("truck/sensor/#")
print("Listening for sensor streams...")
client.loop_start()

# Keep alive loop (simulating live ingestion)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    client.loop_stop()