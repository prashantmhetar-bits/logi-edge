#!/usr/bin/env python3
import argparse
import json
import random
import time
import paho.mqtt.client as mqtt

def parse_args():
    parser = argparse.ArgumentParser(description="Cold-Chain Truck Sensor Simulator")
    parser.add_argument(
        "--anomaly",
        choices=["none", "temp_drift", "vibration", "combined"],
        default="none",
        help="Anomaly mode to inject into sensor streams"
    )
    parser.add_argument("--broker", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize MQTT client
    client = mqtt.Client()
    try:
        client.connect(args.broker, args.port, 60)
        client.loop_start()
        print(f" Connected to MQTT broker at {args.broker}:{args.port}")
        print(f" Simulation running with anomaly mode: {args.anomaly.upper()}")
    except Exception as e:
        print(f"❌ Failed to connect to MQTT broker: {e}")
        return

    # Anomaly flags
    is_temp_anomaly = args.anomaly in ["temp_drift", "combined"]
    is_vib_anomaly = args.anomaly in ["vibration", "combined"]

    # State variables
    temp_drift_accumulator = 0.0
    door_state = "CLOSE"
    next_door_interval = random.randint(20, 40) # initial seconds before first door event

    # Timing trackers (in seconds)
    last_temp_time = 0.0
    last_vib_time = 0.0
    last_door_time = 0.0

    try:
        while True:
            current_time = time.time()

            # 1. Temperature Stream (1 Hz -> every 1.0 second)
            if current_time - last_temp_time >= 1.0:
                base_temp = random.gauss(4.0, 0.3)
                if is_temp_anomaly:
                    temp_drift_accumulator += 0.08  # Linear drift per reading
                
                current_temp = base_temp + temp_drift_accumulator
                
                payload = {
                    "sensor": "temperature",
                    "timestamp": int(current_time * 1000),
                    "value": round(current_temp, 2),
                    "unit": "°C",
                    "setpoint": 4.0,
                    "anomaly": is_temp_anomaly
                }
                client.publish("truck/sensor/temperature", json.dumps(payload))
                print(f"[TEMP] {payload}")
                last_temp_time = current_time

            # 2. Vibration RMS Stream (0.5 Hz -> every 2.0 seconds)
            if current_time - last_vib_time >= 2.0:
                if is_vib_anomaly:
                    current_vib = random.gauss(1.2, 0.15)  # Bearing wear anomaly
                else:
                    current_vib = random.gauss(0.45, 0.05) # Normal compressor RMS
                
                payload = {
                    "sensor": "vibration_rms",
                    "timestamp": int(current_time * 1000),
                    "value": round(current_vib, 3),
                    "unit": "g",
                    "anomaly": is_vib_anomaly
                }
                client.publish("truck/sensor/vibration", json.dumps(payload))
                print(f"[VIB] {payload}")
                last_vib_time = current_time

            # 3. Door Event Stream (Discrete periodic events)
            if current_time - last_door_time >= next_door_interval:
                door_state = "OPEN" if door_state == "CLOSE" else "CLOSE"
                payload = {
                    "sensor": "door_event",
                    "timestamp": int(current_time * 1000),
                    "event": door_state
                }
                client.publish("truck/sensor/door", json.dumps(payload))
                print(f"[DOOR] {payload}")
                last_door_time = current_time
                
                # Keep door open for a short duration (15s), then close for longer (30-60s)
                next_door_interval = 15 if door_state == "OPEN" else random.randint(30, 60)

            # Prevent high CPU utilization
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping simulator...")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Disconnected cleanly.")

if __name__ == "__main__":
    main()