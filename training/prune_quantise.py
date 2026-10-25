"""M3 — 35% structured (neuron-level) pruning of M1 + full INT8 PTQ.

1. Start from the trained M1 weights.
2. Fine-tune while removing whole hidden neurons (lowest L2 norm of incoming
   weights) following tfmot's PolynomialDecay schedule, 0% -> 35%.
3. Physically rebuild a smaller network from the surviving neurons and verify it
   is numerically identical to the masked network.
4. Full-INT8 PTQ (same calibration as M2), evaluate on the shared validation split.
Exit code 1 if the INT8 model fails the quality gates.
"""
import json
import math
import sys

import keras
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot  # used for its PolynomialDecay schedule only

from dataset_utils import REPO_ROOT, SEED, load_splits
from model_eval import Int8Classifier, calibration_samples, convert_full_int8, evaluate

M1_PATH = REPO_ROOT / "cold_chain_mlp_fp32.keras"
M3_KERAS_PATH = REPO_ROOT / "cold_chain_mlp_pruned.keras"
M3_TFLITE_PATH = REPO_ROOT / "cold_chain_mlp_pruned_int8.tflite"

FINAL_SPARSITY = 0.35
EPOCHS = 15
PRUNE_EPOCHS = 10          # schedule reaches 35% here; remaining epochs recover at fixed sparsity
BATCH_SIZE = 16
LEARNING_RATE = 5e-4


class NeuronPruning(keras.callbacks.Callback):
    """Masks whole neurons of the hidden Dense layers according to a sparsity schedule."""

    def __init__(self, layers, schedule):
        super().__init__()
        self.layers = layers
        self.schedule = schedule
        self.masks = [np.ones(layer.units, dtype=bool) for layer in layers]
        self.step = 0

    def _update_masks(self):
        _, sparsity = self.schedule(tf.constant(self.step, dtype=tf.int64))
        sparsity = float(sparsity)
        for i, layer in enumerate(self.layers):
            target_pruned = int(round(sparsity * layer.units))
            already_pruned = int((~self.masks[i]).sum())
            if target_pruned <= already_pruned:
                continue
            kernel, _ = layer.get_weights()
            norms = np.linalg.norm(kernel, axis=0)
            norms[~self.masks[i]] = np.inf
            for j in np.argsort(norms)[: target_pruned - already_pruned]:
                self.masks[i][j] = False

    def apply_masks(self):
        for layer, mask in zip(self.layers, self.masks):
            kernel, bias = layer.get_weights()
            kernel[:, ~mask] = 0.0
            bias[~mask] = 0.0
            layer.set_weights([kernel, bias])

    def on_train_batch_begin(self, batch, logs=None):
        self._update_masks()
        self.apply_masks()

    def on_train_batch_end(self, batch, logs=None):
        self.apply_masks()
        self.step += 1


def build_compact(masked_model, masks):
    """Rebuild a physically smaller MLP containing only the surviving neurons."""
    (k0, b0), (k1, b1), (k2, b2) = [layer.get_weights() for layer in masked_model.layers]
    keep0, keep1 = np.flatnonzero(masks[0]), np.flatnonzero(masks[1])
    compact = keras.Sequential([
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(len(keep0), activation="relu"),
        keras.layers.Dense(len(keep1), activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])
    compact.layers[0].set_weights([k0[:, keep0], b0[keep0]])
    compact.layers[1].set_weights([k1[keep0][:, keep1], b1[keep1]])
    compact.layers[2].set_weights([k2[keep1], b2])
    return compact


def main():
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    X_train, X_val, y_train, y_val = load_splits()
    m1 = keras.models.load_model(M1_PATH)
    m1_params = int(m1.count_params())

    model = keras.models.clone_model(m1)
    model.set_weights(m1.get_weights())
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)
    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=FINAL_SPARSITY,
        begin_step=0, end_step=steps_per_epoch * PRUNE_EPOCHS, frequency=1,
    )
    pruning = NeuronPruning(model.layers[:2], schedule)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), callbacks=[pruning], verbose=2)
    pruning.apply_masks()

    compact = build_compact(model, pruning.masks)
    max_diff = float(np.max(np.abs(model.predict(X_val, verbose=0) - compact.predict(X_val, verbose=0))))
    if max_diff > 1e-5:
        raise RuntimeError(f"Compact model diverges from masked model (max |diff| = {max_diff:.2e})")

    structure = {
        "hidden_units_before": [layer.units for layer in m1.layers[:2]],
        "hidden_units_after": [int(m.sum()) for m in pruning.masks],
        "neurons_removed_fraction": float(1 - sum(int(m.sum()) for m in pruning.masks)
                                          / sum(layer.units for layer in m1.layers[:2])),
        "params_before": m1_params,
        "params_after": int(compact.count_params()),
        "masked_vs_compact_max_abs_diff": max_diff,
    }
    print(json.dumps(structure, indent=2))
    compact.save(M3_KERAS_PATH)

    X_calib = calibration_samples(X_train)
    convert_full_int8(compact, X_calib, M3_TFLITE_PATH)
    clf = Int8Classifier(M3_TFLITE_PATH)
    details = clf.describe(M3_TFLITE_PATH)
    print(json.dumps(details, indent=2))

    fp32_pred = np.argmax(compact.predict(X_val, verbose=0), axis=1)
    int8_pred = np.argmax(clf.predict_proba(X_val), axis=1)
    agreement = float(np.mean(fp32_pred == int8_pred))
    print(f"Pruned FP32 vs INT8 prediction agreement on validation: {agreement:.2%}")

    _, passed = evaluate("M3_PRUNED_INT8", y_val, int8_pred, extra={
        "n_calibration": int(len(X_calib)),
        "fp32_int8_agreement": agreement,
        "structure": structure,
        **details,
    })
    if not passed:
        return 1
    print(f"PASSED quality gates. Saved {M3_TFLITE_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
