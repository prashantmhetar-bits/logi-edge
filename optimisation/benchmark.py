"""F2 — Five-metric benchmark of M1, M2 and M3 (Lab 2 methodology).

Every variant is measured on the same held-out validation split used in training:
  1. Mean inference latency (ms) - 200 timed runs after 10 warm-up runs (excluded)
  2. p95 inference latency (ms)
  3. Model file size (KB)
  4. Classification accuracy (%) on the validation split (+ per-class recall)
  5. Energy per inference (mJ) - E = P x t, with P = the laptop TDP (Lab 2 Step 3: peak
     power, a conservative upper bound); psutil CPU % is measured and reported alongside

One inference = one window of normalised features in, class probabilities out, the
same per-window work inference/inference_service.py does (batch size 1; the INT8
variants include input quantisation and output dequantisation).

Writes results/benchmark_results.json and prints a Markdown table; run pareto.py
afterwards for the CSV and the Pareto chart.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import keras
import numpy as np
import psutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from dataset_utils import CLASS_NAMES, MODELS_DIR, RESULTS_DIR, load_splits  # noqa: E402
from model_eval import LABELS, Int8Classifier  # noqa: E402

WARMUP_RUNS = 10
TIMED_RUNS = 200
CPU_WINDOW_SEC = 3.0     # well above the OS CPU-time tick (~15.6 ms on Windows) psutil relies on
DEFAULT_TDP_W = 15.0     # Intel Core i7-1255U processor base power
RESULTS_PATH = RESULTS_DIR / "benchmark_results.json"
SAVEDMODEL_DIR = MODELS_DIR / "cold_chain_mlp_fp32_savedmodel"

VARIANTS = {
    "M1_FP32": MODELS_DIR / "cold_chain_mlp_fp32.keras",
    "M2_PTQ_INT8": MODELS_DIR / "cold_chain_mlp_ptq_int8.tflite",
    "M3_PRUNED_INT8": MODELS_DIR / "cold_chain_mlp_pruned_int8.tflite",
}


class KerasRunner:
    def __init__(self, path):
        self.model = keras.models.load_model(path)

    def infer(self, row):
        return self.model(row.reshape(1, -1), training=False).numpy()[0]

    def infer_predict_api(self, row):
        """The call inference_service.py makes; adds Keras' per-call data-pipeline overhead."""
        return self.model.predict(row.reshape(1, -1), verbose=0)[0]

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)

    def memory_kb(self):
        """FP32 weights + every layer's activation tensor (batch size 1)."""
        weights = sum(w.nbytes for w in self.model.get_weights())
        activations = 4 * (self.model.input_shape[-1] + sum(layer.units for layer in self.model.layers))
        return (weights + activations) / 1024


class TFLiteRunner(Int8Classifier):
    def infer(self, row):
        return self.predict_one(row)

    def memory_kb(self):
        """All interpreter tensors: INT8 weights, INT32 biases and activation buffers."""
        return sum(int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
                   for d in self.interpreter.get_tensor_details()) / 1024


def time_latency(infer, X):
    """Latencies (ms) of TIMED_RUNS single-sample inferences after WARMUP_RUNS excluded ones."""
    for i in range(WARMUP_RUNS):
        infer(X[i % len(X)])
    times_ms = np.empty(TIMED_RUNS)
    for i in range(TIMED_RUNS):
        row = X[i % len(X)]
        start = time.perf_counter()
        infer(row)
        times_ms[i] = (time.perf_counter() - start) * 1000
    return times_ms


def cpu_utilisation(infer, X):
    """(process CPU %, 100 = one busy logical CPU as in Lab 2; system-wide CPU %) during back-to-back inference."""
    proc = psutil.Process()
    proc.cpu_percent(None)
    psutil.cpu_percent(None)
    start, i = time.perf_counter(), 0
    while time.perf_counter() - start < CPU_WINDOW_SEC:
        infer(X[i % len(X)])
        i += 1
    return proc.cpu_percent(None), psutil.cpu_percent(None)


def benchmark(name, runner, path, X_val, y_val, tdp_w):
    times_ms = time_latency(runner.infer, X_val)
    cpu_pct, system_cpu_pct = cpu_utilisation(runner.infer, X_val)
    mean_ms = float(times_ms.mean())
    power_w = tdp_w  # Lab 2 Step 3: peak power for every variant; CPU % is reported, not multiplied in

    y_pred = np.argmax(runner.predict_proba(X_val), axis=1)
    recalls = recall_score(y_val, y_pred, labels=LABELS, average=None, zero_division=0)
    critical = y_val == 2

    result = {
        "model_file": path.name,
        "mean_latency_ms": mean_ms,
        "p95_latency_ms": float(np.percentile(times_ms, 95)),
        "median_latency_ms": float(np.median(times_ms)),
        "max_latency_ms": float(times_ms.max()),
        "file_size_kb": path.stat().st_size / 1024,
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "recall_per_class": {cls: float(r) for cls, r in zip(CLASS_NAMES, recalls)},
        "critical_recall": float(recalls[2]),
        "critical_detected": f"{int((y_pred[critical] == 2).sum())}/{int(critical.sum())}",
        "process_cpu_percent": cpu_pct,
        "system_cpu_percent": system_cpu_pct,
        "power_w": power_w,
        "energy_mj": power_w * mean_ms,
        "model_ram_kb": runner.memory_kb(),
    }
    if isinstance(runner, KerasRunner):
        predict_ms = time_latency(runner.infer_predict_api, X_val)
        result["keras_predict_api_mean_ms"] = float(predict_ms.mean())
        result["keras_predict_api_p95_ms"] = float(np.percentile(predict_ms, 95))
        if SAVEDMODEL_DIR.is_dir():
            result["savedmodel_size_kb"] = sum(
                f.stat().st_size for f in SAVEDMODEL_DIR.rglob("*") if f.is_file()) / 1024

    print(f"[{name}] mean {mean_ms:.4f} ms | p95 {result['p95_latency_ms']:.4f} ms | "
          f"{result['file_size_kb']:.2f} KB | acc {result['val_accuracy']:.2%} | "
          f"CPU {cpu_pct:.1f}% | {power_w:.1f} W -> {result['energy_mj']:.5f} mJ | "
          f"Critical {result['critical_detected']}")
    return result


def markdown_table(results):
    lines = [
        "| Variant | Mean latency (ms) | p95 latency (ms) | File size (KB) | Accuracy (%) "
        "| Energy / inference (mJ) | Critical recall (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['mean_latency_ms']:.4f} | {r['p95_latency_ms']:.4f} | {r['file_size_kb']:.2f} "
            f"| {r['val_accuracy'] * 100:.2f} | {r['energy_mj']:.5f} | {r['critical_recall'] * 100:.2f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="F2 five-metric benchmark of M1/M2/M3")
    parser.add_argument("--tdp-w", type=float, default=DEFAULT_TDP_W,
                        help=f"CPU TDP in watts for E = P x t (default {DEFAULT_TDP_W}, i7-1255U)")
    args = parser.parse_args()

    missing = [p.name for p in VARIANTS.values() if not p.is_file()]
    if missing:
        print(f"Missing model files: {', '.join(missing)}. Run train_model.py, convert_ptq.py "
              "and prune_quantise.py first.", file=sys.stderr)
        return 1

    battery = psutil.sensors_battery()
    on_ac_power = None if battery is None else bool(battery.power_plugged)
    if on_ac_power is False:
        print("WARNING: running on battery - the OS may throttle the CPU; plug in for "
              "representative latency and energy figures.", file=sys.stderr)

    _, X_val, _, y_val = load_splits()
    print(f"Validation samples: {len(X_val)} | class counts {np.bincount(y_val).tolist()} | "
          f"{WARMUP_RUNS} warm-up + {TIMED_RUNS} timed runs | TDP {args.tdp_w} W")

    results = {}
    for name, path in VARIANTS.items():
        runner = KerasRunner(path) if path.suffix == ".keras" else TFLiteRunner(path)
        results[name] = benchmark(name, runner, path, X_val, y_val, args.tdp_w)

    report = {
        "methodology": {
            "warmup_runs_excluded": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "batch_size": 1,
            "energy_formula": "E[mJ] = TDP[W] x mean latency[ms] (Lab 2 Step 3; process CPU % reported, not multiplied in)",
            "cpu_window_sec": CPU_WINDOW_SEC,
            "tdp_w": args.tdp_w,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "on_ac_power": on_ac_power,
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
        },
        "n_val": int(len(y_val)),
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print("\n" + markdown_table(results))
    print(f"\nSaved {RESULTS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
