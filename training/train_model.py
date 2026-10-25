"""M1 — FP32 baseline: 2-hidden-layer MLP (32 -> 16, ReLU).

Quality gates (exit code 1 on failure, model not saved):
  - validation accuracy > 88%
  - Class 2 (Critical) recall > 95%
"""
import sys

import keras
import numpy as np
import tensorflow as tf

from dataset_utils import REPO_ROOT, SEED, load_splits
from model_eval import evaluate

EPOCHS = 10
BATCH_SIZE = 16
MODEL_PATH = REPO_ROOT / "cold_chain_mlp_fp32.keras"


def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])


def main():
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    X_train, X_val, y_train, y_val = load_splits()
    print(f"Samples: train={len(X_train)} val={len(X_val)} | class counts val={np.bincount(y_val).tolist()}")

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), verbose=2)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    _, passed = evaluate("M1_FP32", y_val, y_pred, extra={
        "n_train": int(len(X_train)),
        "params": int(model.count_params()),
    })
    if not passed:
        print("Model NOT saved.", file=sys.stderr)
        return 1

    model.save(MODEL_PATH)
    print(f"PASSED quality gates. Saved {MODEL_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
