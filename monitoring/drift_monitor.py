#!/usr/bin/env python3
"""LogiEdge - Task E1: PSI drift monitoring on the model's output confidence.

Runs on the truck's edge node as a sidecar to the inference container: it only
subscribes to the local MQTT inference topic, so it needs no access to the model,
sensors or cloud, and keeps working through cellular gaps.

Monitored score (--score):
  normal_prob (default)  P(Normal) = probabilities[0] from the inference payload.
                         ~1.0 on clean cargo, collapses towards 0 under a fault,
                         so its distribution shifts when the input data drifts.
  max_conf               max softmax confidence (the payload's "confidence").
                         Stays high under the combined anomaly because the model
                         is *confidently* CRITICAL, so it barely shifts. Kept for
                         the comparison experiment in the report.

Modes:
  reference  collect N inferences on clean Normal data (simulator --anomaly none)
             and write the 4-bin reference distribution to reference_dist.json
  monitor    keep the last W inferences; every I seconds compute PSI against the
             reference, print it, and alert when PSI > threshold
"""
import argparse
import csv
import json
import math
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
BIN_LABELS = ["[0.00,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"]
SMOOTHING = 0.5  # Jeffreys prior: +0.5 per bin, so an empty bin never forces ln(0)
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "reference_dist.json"


def bin_index(score):
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score {score} outside [0, 1]")
    for i in range(len(BIN_EDGES) - 2):
        if score < BIN_EDGES[i + 1]:
            return i
    return len(BIN_EDGES) - 2  # last bin is closed: [0.75, 1.0]


def distribution(scores):
    counts = [0] * (len(BIN_EDGES) - 1)
    for s in scores:
        counts[bin_index(s)] += 1
    n = len(scores)
    return counts, [c / n for c in counts]


def smoothed(counts, alpha=SMOOTHING):
    total = sum(counts) + alpha * len(counts)
    return [(c + alpha) / total for c in counts]


def psi(expected_counts, actual_counts):
    """Population Stability Index on additively smoothed bin proportions."""
    e_p, a_p = smoothed(expected_counts), smoothed(actual_counts)
    return sum((a - e) * math.log(a / e) for e, a in zip(e_p, a_p))


def extract_score(payload, score):
    if score == "normal_prob":
        probs = payload.get("probabilities")
        if not isinstance(probs, list) or len(probs) != 3:
            raise KeyError("payload has no 3-class 'probabilities' list "
                           "(inference_service.py must publish it)")
        return float(probs[0])
    return float(payload["confidence"])


class InferenceStream:
    """Subscribes to logibridge/trucks/<truck_id>/inference and feeds scores to a callback."""

    def __init__(self, broker, port, truck_id, score, on_score):
        self.topic = f"logibridge/trucks/{truck_id}/inference"
        self.score = score
        self.on_score = on_score
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.broker, self.port = broker, port

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(self.topic, qos=1)
            print(f"[MQTT] connected to {self.broker}:{self.port}, subscribed to {self.topic}", flush=True)
        else:
            print(f"[MQTT] connection refused: {reason_code}", flush=True)

    def _on_message(self, client, userdata, msg):
        try:
            self.on_score(extract_score(json.loads(msg.payload), self.score))
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[WARN] skipped malformed inference payload: {exc}", flush=True)

    def start(self):
        self.client.connect_async(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


def run_reference(args):
    scores, done = [], threading.Event()
    lock = threading.Lock()

    def on_score(s):
        with lock:
            if len(scores) < args.reference_size:
                scores.append(s)
                if len(scores) % 10 == 0 or len(scores) == args.reference_size:
                    print(f"[REFERENCE] {len(scores)}/{args.reference_size} inferences collected", flush=True)
                if len(scores) == args.reference_size:
                    done.set()

    stream = InferenceStream(args.broker, args.port, args.truck_id, args.score, on_score)
    stream.start()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    try:
        while not done.wait(1.0):  # 1 s ticks keep Ctrl+C responsive on Windows
            pass
    except KeyboardInterrupt:
        pass
    stream.stop()

    if len(scores) < args.reference_size:
        print(f"[REFERENCE] aborted with {len(scores)}/{args.reference_size} inferences; nothing written",
              file=sys.stderr)
        return 1

    counts, proportions = distribution(scores)
    reference = {
        "score": args.score,
        "bins": BIN_LABELS,
        "bin_edges": BIN_EDGES,
        "counts": counts,
        "proportions": proportions,
        "n": len(scores),
        "truck_id": args.truck_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.reference.write_text(json.dumps(reference, indent=2))
    print(f"[REFERENCE] wrote {args.reference}")
    for label, c, p in zip(BIN_LABELS, counts, proportions):
        print(f"  {label}  count={c:<4} p={p:.3f}")
    return 0


def run_monitor(args):
    reference = json.loads(args.reference.read_text())
    if reference.get("score") != args.score:
        print(f"[ERROR] reference was built with score={reference.get('score')!r}, "
              f"monitor started with --score {args.score!r}", file=sys.stderr)
        return 1
    expected = reference["counts"]
    print(f"[MONITOR] reference n={reference['n']} score={args.score} "
          f"p={[round(p, 3) for p in reference['proportions']]}")
    print(f"[MONITOR] window={args.window} inferences, PSI every {args.interval}s, "
          f"alert > {args.threshold}", flush=True)

    window = deque(maxlen=args.window)
    lock = threading.Lock()

    def on_score(s):
        with lock:
            window.append(s)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    log_file = open(args.log, "a", newline="") if args.log else None
    writer = csv.writer(log_file) if log_file else None
    if writer and log_file.tell() == 0:
        writer.writerow(["utc", "window_n", "psi", "alert"] + [f"p{b}" for b in BIN_LABELS])

    stream = InferenceStream(args.broker, args.port, args.truck_id, args.score, on_score)
    stream.start()
    next_report = time.monotonic() + args.interval
    try:
        while not stop.wait(1.0):  # 1 s ticks keep Ctrl+C responsive on Windows
            if time.monotonic() < next_report:
                continue
            next_report += args.interval
            with lock:
                snapshot = list(window)
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            if len(snapshot) < args.min_samples:
                print(f"{ts} [MONITOR] warming up: {len(snapshot)}/{args.min_samples} inferences", flush=True)
                continue
            counts, actual = distribution(snapshot)
            value = psi(expected, counts)
            alert = value > args.threshold
            print(f"{ts} [MONITOR] n={len(snapshot):<3} PSI={value:.3f} "
                  f"dist={[round(p, 2) for p in actual]}", flush=True)
            if alert:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)
            if writer:
                writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 len(snapshot), f"{value:.4f}", int(alert)] + [f"{p:.4f}" for p in actual])
                log_file.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        if log_file:
            log_file.close()
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="LogiEdge PSI drift monitor (Task E1)")
    p.add_argument("--mode", choices=["reference", "monitor"], required=True)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--truck-id", default="truck_001")
    p.add_argument("--score", choices=["normal_prob", "max_conf"], default="normal_prob")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--reference-size", type=int, default=300)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--min-samples", type=int, default=20,
                   help="inferences required in the window before PSI is reported")
    p.add_argument("--log", type=Path, default=None, help="append PSI timeline to this CSV")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(run_reference(a) if a.mode == "reference" else run_monitor(a))
