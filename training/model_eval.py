"""Shared evaluation, quality gates, INT8 conversion and TFLite inference for M1/M2/M3."""
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

from dataset_utils import CLASS_NAMES, RESULTS_DIR, SEED

MIN_VAL_ACCURACY = 0.88
MIN_CRITICAL_RECALL = 0.95
LABELS = [0, 1, 2]


def evaluate(variant, y_true, y_pred, extra=None):
    """Print report, write metrics_<variant>.json, return (metrics, passed)."""
    val_accuracy = float(accuracy_score(y_true, y_pred))
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(f"\n[{variant}] Classification report (validation):")
    print(classification_report(y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, zero_division=0))
    print(f"[{variant}] Confusion matrix (rows=true, cols=pred):\n{cm}")

    metrics = {
        "variant": variant,
        "tensorflow": tf.__version__,
        "n_val": int(len(y_true)),
        "val_accuracy": val_accuracy,
        "recall_per_class": {name: float(r) for name, r in zip(CLASS_NAMES, recalls)},
        "critical_recall": float(recalls[2]),
        "confusion_matrix": cm.tolist(),
        "gates": {"min_val_accuracy": MIN_VAL_ACCURACY, "min_critical_recall": MIN_CRITICAL_RECALL},
    }
    if extra:
        metrics.update(extra)

    failures = []
    if not val_accuracy > MIN_VAL_ACCURACY:
        failures.append("validation accuracy")
    if not metrics["critical_recall"] > MIN_CRITICAL_RECALL:
        failures.append("Critical recall")
    metrics["passed"] = not failures

    print(f"[{variant}] Validation accuracy : {val_accuracy:.2%} (gate > {MIN_VAL_ACCURACY:.0%})")
    print(f"[{variant}] Critical recall     : {metrics['critical_recall']:.2%} (gate > {MIN_CRITICAL_RECALL:.0%})")
    if failures:
        print(f"[{variant}] FAILED quality gate: {', '.join(failures)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"metrics_{variant.lower()}.json").write_text(json.dumps(metrics, indent=2))
    return metrics, not failures


def calibration_samples(X_train, minimum=200):
    """Shuffled normalised training samples for full-INT8 calibration (validation data never used)."""
    if len(X_train) < minimum:
        raise ValueError(f"Need >= {minimum} calibration samples, have {len(X_train)}")
    rng = np.random.default_rng(SEED)
    return X_train[rng.permutation(len(X_train))].astype(np.float32)


def convert_full_int8(model, X_calib, output_path):
    """Full-integer PTQ: int8 weights, activations, input and output."""
    def representative_dataset():
        for row in X_calib:
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())
    return output_path


class Int8Classifier:
    """Runs a full-INT8 TFLite model on float (normalised) inputs; returns float probabilities."""

    def __init__(self, path):
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]
        if self.inp["dtype"] != np.int8 or self.out["dtype"] != np.int8:
            raise TypeError(f"{path}: expected int8 input/output, got {self.inp['dtype']}/{self.out['dtype']}")

    def predict_one(self, row):
        q = np.clip(np.round(row / self.in_scale + self.in_zp), -128, 127).astype(np.int8)
        self.interpreter.set_tensor(self.inp["index"], q.reshape(self.inp["shape"]))
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.out["index"])[0].astype(np.float32)
        return (raw - self.out_zp) * self.out_scale

    def predict_proba(self, X):
        return np.stack([self.predict_one(row) for row in X])

    def describe(self, path):
        lo = (-128 - self.in_zp) * self.in_scale
        hi = (127 - self.in_zp) * self.in_scale
        return {
            "file_size_kb": round(os.path.getsize(path) / 1024, 2),
            "input_quantization": {"scale": float(self.in_scale), "zero_point": int(self.in_zp),
                                   "representable_range": [round(float(lo), 3), round(float(hi), 3)]},
            "output_quantization": {"scale": float(self.out_scale), "zero_point": int(self.out_zp)},
        }
