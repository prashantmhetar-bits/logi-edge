"""M1 — FP32 baseline: 2-hidden-layer MLP (32 -> 16, ReLU).

Quality gates (Task D1 / F3), enforced with a non-zero exit code:
  - validation accuracy must exceed 88%
  - Class 2 (Critical) recall must exceed 95%
"""

import json
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

from dataset_utils import CLASS_NAMES, REPO_ROOT, SEED, load_splits

MIN_VAL_ACCURACY = 0.88
MIN_CRITICAL_RECALL = 0.95
EPOCHS = 10
BATCH_SIZE = 16
MODEL_PATH = REPO_ROOT / "cold_chain_mlp_fp32.keras"
METRICS_PATH = REPO_ROOT / "metrics_m1.json"


def build_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(6,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )


def main():
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    X_train, X_val, y_train, y_val = load_splits()
    print(
        f"Samples: train={len(X_train)} val={len(X_val)} | class counts val={np.bincount(y_val).tolist()}"
    )

    model = build_model()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_accuracy = float(accuracy_score(y_val, y_pred))
    recalls = recall_score(
        y_val, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    critical_recall = float(recalls[2])

    print("\nClassification report (validation):")
    print(
        classification_report(
            y_val, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1, 2]))

    metrics = {
        "variant": "M1_FP32",
        "tensorflow": tf.__version__,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "val_accuracy": val_accuracy,
        "recall_per_class": {name: float(r) for name, r in zip(CLASS_NAMES, recalls)},
        "critical_recall": critical_recall,
        "confusion_matrix": confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist(),
        "gates": {
            "min_val_accuracy": MIN_VAL_ACCURACY,
            "min_critical_recall": MIN_CRITICAL_RECALL,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nValidation accuracy : {val_accuracy:.2%} (gate > {MIN_VAL_ACCURACY:.0%})")
    print(
        f"Critical recall     : {critical_recall:.2%} (gate > {MIN_CRITICAL_RECALL:.0%})"
    )

    failures = []
    if not val_accuracy > MIN_VAL_ACCURACY:
        failures.append("validation accuracy")
    if not critical_recall > MIN_CRITICAL_RECALL:
        failures.append("Critical recall")
    if failures:
        print(
            f"FAILED quality gate: {', '.join(failures)}. Model NOT saved.",
            file=sys.stderr,
        )
        return 1

    model.save(MODEL_PATH)
    print(f"PASSED quality gates. Saved {MODEL_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
