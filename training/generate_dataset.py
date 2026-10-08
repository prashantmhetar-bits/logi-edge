#!/usr/bin/env python3
import time
import subprocess
import json
import sys
from collections import deque
import numpy as np
from scipy.stats import kurtosis
import paho.mqtt.client as mqtt

# Configuration for Dataset Generation Phases matching Task D1 requirements
PHASES = [
    {
        "name": "Normal Class",
        "label": 0,
        "anomaly_mode": "none",
        "output_file": "dataset_normal.csv",
        "duration_sec": 20 * 60,  # 20 minutes -> ~120 windows
        "target_samples": 120
    },
    {
        "name": "Warning Class (Temp Drift)",
        "label": 1,
        "anomaly_mode": "temp_drift",
        "output_file": "dataset_warning.csv",
        "duration_sec": 15 * 60,  # 15 minutes -> ~90 windows
        "target_samples": 90
    },
    {
        "name": "Critical Class (Combined)",
        "label": 2,
        "anomaly_mode": "combined",
        "output_file": "dataset_critical.csv",
        "duration_sec": 15 * 60,  # 15 minutes -> ~90 windows
        "target_samples": 90
    }
]

WINDOW_SIZE_SEC = 30.0
STEP_SIZE_SEC = 10.0
BROKER = "localhost"
PORT = 1883

class AutomatedFeatureExtractor:
    def __init__(self, label, output_file):
        self.label = label
        self.output_file = output_file
        self.temp_buffer = deque(maxlen=5)
        self.vib_buffer = deque(maxlen=5)
        self.temp_window = deque()
        self.vib_window = deque()
        self.last_step_time = time.time()
        self.samples_collected = 0

    def process_message(self, topic, payload_bytes):
        try:
            payload = json.loads(payload_bytes.decode())
            current_time = time.time()
            val = payload.get("value")
            
            if val is None:
                return

            if topic == "truck/sensor/temperature":
                self.temp_buffer.append(val)
                filtered = sum(self.temp_buffer) / len(self.temp_buffer)
                self.temp_window.append((current_time, filtered))
            elif topic == "truck/sensor/vibration":
                self.vib_buffer.append(val)
                filtered = sum(self.vib_buffer) / len(self.vib_buffer)
                self.vib_window.append((current_time, filtered))

            # Evaluate window features every 10 seconds step size
            if current_time - self.last_step_time >= STEP_SIZE_SEC:
                self.last_step_time = current_time
                feat = self.extract_features(current_time)
                if feat is not None:
                    self.save_sample(feat)

        except Exception as e:
            # Prevents background thread crash
            pass

    def extract_features(self, current_time):
        cutoff = current_time - WINDOW_SIZE_SEC
        while self.temp_window and self.temp_window[0][0] < cutoff:
            self.temp_window.popleft()
        while self.vib_window and self.vib_window[0][0] < cutoff:
            self.vib_window.popleft()

        if len(self.temp_window) < 5 or len(self.vib_window) < 3:
            return None

        temps = np.array([v for t, v in self.temp_window])
        vibrs = np.array([v for t, v in self.vib_window])

        # Feature 1: Temperature Mean
        t_mean = np.mean(temps)
        # Feature 2: Temperature Standard Deviation
        t_std = np.std(temps)
        # Feature 3: Temperature Rate-of-Change (°C/min) using linear regression slope
        time_arr = np.array([t for t, v in self.temp_window])
        time_min = (time_arr - time_arr[0]) / 60.0
        if len(np.unique(time_min)) > 1:
            slope, _ = np.polyfit(time_min, temps, 1)
            t_roc = slope
        else:
            t_roc = 0.0

        # Feature 4: Vibration RMS
        v_rms = np.sqrt(np.mean(vibrs**2))
        # Feature 5: Vibration Peak
        v_peak = np.max(np.abs(vibrs))
        # Feature 6: Vibration Kurtosis
        v_kurt = kurtosis(vibrs, fisher=True) if len(vibrs) > 3 and np.std(vibrs) > 1e-5 else 0.0

        return np.array([t_mean, t_std, t_roc, v_rms, v_peak, v_kurt])

    def save_sample(self, features):
        self.samples_collected += 1
        row = list(features) + [self.label]
        with open(self.output_file, "a") as f:
            f.write(",".join(map(str, row)) + "\n")
        print(f"[{self.samples_collected} samples] Logged to {self.output_file} | Features: {np.round(features, 3)}")

def run_phase(phase):
    print(f"\n==================================================")
    print(f"▶ STARTING PHASE: {phase['name']} (Label: {phase['label']})")
    print(f"Mode: --anomaly {phase['anomaly_mode']} | Target File: {phase['output_file']}")
    print(f"Target Samples: ~{phase['target_samples']} ({phase['duration_sec']//60} mins)")
    print(f"==================================================")

    # Initialize / Clear the specific output file for this phase
    open(phase["output_file"], "w").close()

    # Launch simulator subprocess using active python executable
    sim_process = subprocess.Popen([sys.executable, "data_pipeline/simulator.py", "--anomaly", phase["anomaly_mode"]])
    time.sleep(3) # Give broker/simulator a moment to start

    extractor = AutomatedFeatureExtractor(phase["label"], phase["output_file"])

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_message = lambda c, ud, msg: extractor.process_message(msg.topic, msg.payload)
    
    try:
        client.connect(BROKER, PORT, 60)
        client.subscribe("truck/sensor/#")
        client.loop_start()

        start_time = time.time()
        while time.time() - start_time < phase["duration_sec"]:
            elapsed = int(time.time() - start_time)
            remaining = phase["duration_sec"] - elapsed
            print(f"⏳ Progress: {elapsed//60}m {elapsed%60}s elapsed | {remaining//60}m remaining | Samples: {extractor.samples_collected}/{phase['target_samples']}", end="\r")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⚠️ Phase interrupted manually by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        sim_process.terminate()
        sim_process.wait()
        print(f"\n✓ Completed {phase['name']}. Total samples saved to {phase['output_file']}: {extractor.samples_collected}")

if __name__ == "__main__":
    print("🚀 Starting Automated Task D1 Cold-Chain Dataset Generator...")

    for phase in PHASES:
        run_phase(phase)
        if phase != PHASES[-1]:
            print("\nPausing 5 seconds before next phase...")
            time.sleep(5)

    print("\n🎉 All Task D1 dataset generation phases completed successfully!")
    print("Generated files:")
    for phase in PHASES:
        print(f" - {phase['output_file']}")