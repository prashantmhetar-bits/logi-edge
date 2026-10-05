#!/usr/bin/env python3
import argparse
import json
import time
from collections import deque
import numpy as np
from scipy.stats import kurtosis
import paho.mqtt.client as mqtt

# Window configuration
WINDOW_SIZE_SEC = 30.0
STEP_SIZE_SEC = 10.0
STATS_FILE = "data_pipeline/training_stats.npy"

class PreprocessingPipeline:
    def __init__(self, shift_sigma=False):
        self.shift_sigma = shift_sigma
        
        # 5-sample moving average deques
        self.temp_buffer = deque(maxlen=5)
        self.vib_buffer = deque(maxlen=5)
        
        # Sliding window data buffers (stores tuples of (timestamp, value))
        self.temp_window = deque()
        self.vib_window = deque()
        
        # Load training statistics
        self.load_stats()
        
        # Timing trackers for sliding window steps
        self.last_window_eval = time.time()

    def load_stats(self):
        try:
            stats = np.load(STATS_FILE, allow_pickle=True).item()
            self.train_mean = stats["mean"]
            self.train_std = stats["std"]
            
            # Prevent tiny standard deviation floors from blowing up volatile features like Kurtosis (index 5)
            # Kurtosis naturally fluctuates widely on small windows, so give it a realistic minimum std baseline
            self.train_std[2] = max(self.train_std[2], 0.2)
            self.train_std[5] = max(self.train_std[5], 1.0)
            
            if self.shift_sigma:
                # Apply mandatory +3 sigma shift experiment
                self.train_mean += 3.0 * self.train_std
                print("⚠️ [EXPERIMENT] Stats shifted by +3σ loaded successfully!")
            else:
                print("✓ Correct training stats loaded successfully.")
        except FileNotFoundError:
            print(f"⚠️ Warning: {STATS_FILE} not found. Run generation mode first if normalising live data.")
            self.train_mean = np.zeros(6)
            self.train_std = np.ones(6)

    def apply_moving_average(self, stream_type, value):
        if stream_type == "temperature":
            self.temp_buffer.append(value)
            return sum(self.temp_buffer) / len(self.temp_buffer)
        elif stream_type == "vibration":
            self.vib_buffer.append(value)
            return sum(self.vib_buffer) / len(self.vib_buffer)
        return value

    def extract_features(self, current_time):
        # Remove data older than 30 seconds
        cutoff = current_time - WINDOW_SIZE_SEC
        while self.temp_window and self.temp_window[0][0] < cutoff:
            self.temp_window.popleft()
        while self.vib_window and self.vib_window[0][0] < cutoff:
            self.vib_window.popleft()

        if len(self.temp_window) < 5 or len(self.vib_window) < 3:
            return None # Not enough data for a robust window

        temps = np.array([v for t, v in self.temp_window])
        vibrs = np.array([v for t, v in self.vib_window])

        # 1. Temperature Mean
        t_mean = np.mean(temps)
        # 2. Temperature Standard Deviation
        t_std = np.std(temps)
        # 3. Temperature Rate-of-Change (°C/min) using robust linear regression slope
        time_elapsed_arr = np.array([t for t, v in self.temp_window])
        time_elapsed_min = (time_elapsed_arr - time_elapsed_arr[0]) / 60.0
        if len(np.unique(time_elapsed_min)) > 1:
            # Fit a line (y = mx + b), where m is slope (°C per minute)
            slope, _ = np.polyfit(time_elapsed_min, temps, 1)
            t_roc = slope
        else:
            t_roc = 0.0

        # 4. Vibration RMS
        v_rms = np.sqrt(np.mean(vibrs**2))
        # 5. Vibration Peak
        v_peak = np.max(np.abs(vibrs))
        # 6. Vibration Kurtosis
        v_kurt = kurtosis(vibrs, fisher=True) if len(vibrs) > 3 else 0.0

        feature_vector = np.array([t_mean, t_std, t_roc, v_rms, v_peak, v_kurt])
        return feature_vector

    def normalize(self, features):
        # Use a safe minimum standard deviation floor to prevent division explosions
        safe_std = np.maximum(self.train_std, 0.1)
        
        # Compute standard Z-score
        normalized = (features - self.train_mean) / safe_std
        
        # Clip extreme outliers (vital for volatile features like kurtosis on small windows)
        return np.clip(normalized, -10.0, 10.0)
    

def generate_training_stats_mode(broker, port):
    print("Collecting 10 minutes of clean Normal-class data to build training_stats.npy...")
    print("Please run your simulator with `--anomaly none` in another terminal now!")
    
    collected_features = []
    pipeline = PreprocessingPipeline(shift_sigma=False)
    
    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            current_time = time.time()
            val = payload.get("value")
            
            if msg.topic == "truck/sensor/temperature":
                filtered = pipeline.apply_moving_average("temperature", val)
                pipeline.temp_window.append((current_time, filtered))
            elif msg.topic == "truck/sensor/vibration":
                filtered = pipeline.apply_moving_average("vibration", val)
                pipeline.vib_buffer.append((current_time, filtered))
                pipeline.vib_window.append((current_time, filtered))
                
                # Every time vibration arrives, check if we can extract a feature vector
                feat = pipeline.extract_features(current_time)
                if feat is not None:
                    collected_features.append(feat)
                    if len(collected_features) % 10 == 0:
                        print(f"Collected {len(collected_features)} feature windows...")
        except Exception as e:
            pass

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker, port, 60)
    client.subscribe("truck/sensor/#")
    client.loop_start()

    start_time = time.time()
    # Collect for up to 10 minutes (or press Ctrl+C when enough is gathered)
    try:
        while time.time() - start_time < 600 and len(collected_features) < 300:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping collection early...")
    
    client.loop_stop()
    client.disconnect()

    if collected_features:
        feat_matrix = np.array(collected_features)
        mean_vec = np.mean(feat_matrix, axis=0)
        std_vec = np.std(feat_matrix, axis=0)
        
        stats = {"mean": mean_vec, "std": std_vec}
        np.save(STATS_FILE, stats)
        print(f"✓ Saved training statistics to {STATS_FILE}")
        print(f"Mean Vector: {mean_vec}")
        print(f"Std Vector: {std_vec}")
    else:
        print("❌ No features collected. Make sure simulator is running with `--anomaly none`.")

def run_live_pipeline(broker, port, shift_sigma):
    pipeline = PreprocessingPipeline(shift_sigma=shift_sigma)
    last_step_time = time.time()

    def on_message(client, userdata, msg):
        nonlocal last_step_time
        try:
            payload = json.loads(msg.payload.decode())
            current_time = time.time()
            val = payload.get("value")
            
            if msg.topic == "truck/sensor/temperature":
                filtered = pipeline.apply_moving_average("temperature", val)
                pipeline.temp_window.append((current_time, filtered))
            elif msg.topic == "truck/sensor/vibration":
                filtered = pipeline.apply_moving_average("vibration", val)
                pipeline.vib_window.append((current_time, filtered))
            
            # Evaluate window features every 10 seconds step size
            if current_time - last_step_time >= STEP_SIZE_SEC:
                features = pipeline.extract_features(current_time)
                if features is not None:
                    normalized_features = pipeline.normalize(features)
                    
                    print(f"\n--- Sliding Window Feature Vector ({'Shifted +3σ' if shift_sigma else 'Correct Stats'}) ---")
                    print(f"Raw Features       : {np.round(features, 4)}")
                    print(f"Normalized Features: {np.round(normalized_features, 4)}")
                    
                    # Simple threshold-based inference check (e.g., anomaly if normalized value > 3.0)
                    is_anomaly = np.any(np.abs(normalized_features) > 3.0)
                    print(f"Inference Result   : {'ANOMALY DETECTED' if is_anomaly else 'NORMAL'}")
                
                last_step_time = current_time

        except Exception as e:
            print(f"Error processing message: {e}")

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker, port, 60)
    client.subscribe("truck/sensor/#")
    
    print(f"Starting Preprocessing Pipeline (Shift +3σ Stats = {shift_sigma}). Listening to broker...")
    client.loop_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold-Chain Preprocessing Pipeline")
    parser.add_argument("--mode", choices=["generate_stats", "run"], default="run", help="Pipeline execution mode")
    parser.add_argument("--shift-stats", action="store_true", help="Mandatory experiment: shift stats by +3 sigma")
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    
    args = parser.parse_args()

    if args.mode == "generate_stats":
        generate_training_stats_mode(args.broker, args.port)
    else:
        run_live_pipeline(args.broker, args.port, args.shift_stats)