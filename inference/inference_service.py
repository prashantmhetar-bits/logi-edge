import json
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt
import tensorflow as tf
import numpy as np
import os

# Configuration from environment variables
TRUCK_ID = os.environ.get("TRUCK_ID", "truck_001")
MODEL_PATH = os.environ.get("MODEL_PATH", "cold_chain_mlp_fp32.keras")
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))

TOPIC_TELEMETRY = f"truck/sensor/#"
TOPIC_INFERENCE = f"logibridge/trucks/{TRUCK_ID}/inference"
WINDOW_SIZE_SEC = 30.0
STEP_SIZE_SEC = 10.0

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATS_PATH = PROJECT_DIR / "data_pipeline" / "training_stats.npy"

print("=" * 50)
print("🚀 Initializing Edge Inference Service...")
print(f"-> Target Truck ID: {TRUCK_ID}")
print(f"-> Target Model File: {MODEL_PATH}")
print(f"-> MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
print(f"-> Subscribing to: {TOPIC_TELEMETRY}")
print(f"-> Publishing to: {TOPIC_INFERENCE}")
print(f"-> Training statistics: {STATS_PATH}")
print("=" * 50)

if not STATS_PATH.is_file():
    raise FileNotFoundError(
        f"Training statistics not found at {STATS_PATH}. "
        "Generate data_pipeline/training_stats.npy before starting inference."
    )

stats = np.load(STATS_PATH, allow_pickle=True).item()
if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
    raise ValueError(f"{STATS_PATH} must contain 'mean' and 'std' arrays.")

train_mean = np.asarray(stats["mean"], dtype=np.float32)
train_std = np.asarray(stats["std"], dtype=np.float32)
if train_mean.shape != (6,) or train_std.shape != (6,):
    raise ValueError(
        f"{STATS_PATH} must contain six means and standard deviations; "
        f"got shapes {train_mean.shape} and {train_std.shape}."
    )
if not np.all(np.isfinite(train_mean)) or not np.all(np.isfinite(train_std)):
    raise ValueError(f"{STATS_PATH} contains non-finite values.")
if np.any(train_std < 0):
    raise ValueError(f"{STATS_PATH} contains negative standard deviations.")

train_std = train_std.copy()
train_std[2] = max(train_std[2], 0.2)
train_std[5] = max(train_std[5], 1.0)
safe_std = np.maximum(train_std, 0.1)

temp_buffer = deque(maxlen=5)
vib_buffer = deque(maxlen=5)
temp_window = deque()
vib_window = deque()
last_window_eval = time.time()

# Load Model dynamically based on extension (.tflite vs .keras)
if MODEL_PATH.endswith(".tflite"):
    print(f"-> Loading TFLite Model runtime...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    is_tflite = True
else:
    print(f"-> Loading Keras Model: {MODEL_PATH}...")
    keras_model = tf.keras.models.load_model(MODEL_PATH)
    is_tflite = False
print("-> Model loaded successfully!")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("[MQTT] Connected successfully to MQTT Broker!")
        client.subscribe(TOPIC_TELEMETRY)
    else:
        print(f"[MQTT] Failed to connect, return code {reason_code}")

def on_connect_fail(client, userdata):
    print(f"[MQTT] Could not connect to broker at {MQTT_BROKER}:{MQTT_PORT}; retrying...")

def extract_features(current_time):
    cutoff = current_time - WINDOW_SIZE_SEC
    while temp_window and temp_window[0][0] < cutoff:
        temp_window.popleft()
    while vib_window and vib_window[0][0] < cutoff:
        vib_window.popleft()

    if len(temp_window) < 5 or len(vib_window) < 3:
        return None

    temps = np.asarray([value for _, value in temp_window], dtype=np.float32)
    vibrs = np.asarray([value for _, value in vib_window], dtype=np.float32)

    temp_times = np.asarray([timestamp for timestamp, _ in temp_window])
    elapsed_minutes = (temp_times - temp_times[0]) / 60.0
    if len(np.unique(elapsed_minutes)) > 1:
        temp_rate, _ = np.polyfit(elapsed_minutes, temps, 1)
    else:
        temp_rate = 0.0

    vib_std = np.std(vibrs)
    if len(vibrs) > 3 and vib_std > 1e-5:
        centered_vibration = vibrs - np.mean(vibrs)
        vib_kurtosis = np.mean((centered_vibration / vib_std) ** 4) - 3.0
    else:
        vib_kurtosis = 0.0

    return np.array(
        [
            np.mean(temps),
            np.std(temps),
            temp_rate,
            np.sqrt(np.mean(vibrs**2)),
            np.max(np.abs(vibrs)),
            vib_kurtosis,
        ],
        dtype=np.float32,
    )

def on_message(client, userdata, msg):
    global last_window_eval
    try:
        payload = json.loads(msg.payload.decode())
        print(f"[Data In] Received telemetry from simulator on topic: {msg.topic}")

        value = payload.get("value")
        if value is None:
            return
        value = float(value)
        if not np.isfinite(value):
            return

        current_time = time.time()
        if msg.topic == "truck/sensor/temperature":
            temp_buffer.append(value)
            filtered_value = sum(temp_buffer) / len(temp_buffer)
            temp_window.append((current_time, filtered_value))
        elif msg.topic == "truck/sensor/vibration":
            vib_buffer.append(value)
            filtered_value = sum(vib_buffer) / len(vib_buffer)
            vib_window.append((current_time, filtered_value))
        else:
            return

        if current_time - last_window_eval < STEP_SIZE_SEC:
            return
        last_window_eval = current_time

        features = extract_features(current_time)
        if features is None:
            return

        normalized_features = np.clip(
            (features - train_mean) / safe_std, -10.0, 10.0
        ).reshape(1, 6)
        print(f"[Inference] Running prediction using {MODEL_PATH}...")
        if is_tflite:
            inp, out = input_details[0], output_details[0]
            model_input = normalized_features.astype(np.float32)
            if inp["dtype"] == np.int8:
                in_scale, in_zero_point = inp["quantization"]
                model_input = np.clip(
                    np.round(model_input / in_scale + in_zero_point), -128, 127
                ).astype(np.int8)
            interpreter.set_tensor(inp["index"], model_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(out["index"])
            if out["dtype"] == np.int8:
                out_scale, out_zero_point = out["quantization"]
                output_data = (output_data.astype(np.float32) - out_zero_point) * out_scale
        else:
            output_data = keras_model.predict(normalized_features, verbose=0)

        predicted_class = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))
        status_map = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}
        status = status_map.get(predicted_class, "UNKNOWN")

        result_payload = {
            "truck_id": TRUCK_ID,
            "status": status,
            "confidence": confidence,
            "probabilities": [float(p) for p in output_data[0]],
            "timestamp": time.time(),
        }
        client.publish(TOPIC_INFERENCE, json.dumps(result_payload))
        print(
            f"[Publish Success] Sent result -> Status: {status} "
            f"(Confidence: {confidence:.2f})"
        )
    except Exception as e:
        print(f"[Error] Failed to process incoming message: {e}")

# Setup MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_connect_fail = on_connect_fail
client.on_message = on_message

print(f"[MQTT] Connecting to broker at {MQTT_BROKER}:{MQTT_PORT}...", flush=True)
client.connect_async(MQTT_BROKER, MQTT_PORT, 60)

# Start network loop to continuously listen for simulator data
client.loop_start()

print("-> Edge service started; waiting for MQTT broker connection... (Press Ctrl+C to exit)", flush=True)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n-> Shutting down Edge Inference Service gracefully...")
    client.loop_stop()
    client.disconnect()
