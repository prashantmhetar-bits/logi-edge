"""M2 — Post-Training Quantisation, full INT8.

Calibrates on >= 200 shuffled, normalised TRAINING samples (same normalisation as
the inference service), then evaluates the INT8 model on the shared validation split.
Exit code 1 if the INT8 model fails the quality gates.
"""
import json
import sys

import keras
import numpy as np

from dataset_utils import REPO_ROOT, load_splits
from model_eval import Int8Classifier, calibration_samples, convert_full_int8, evaluate

M1_PATH = REPO_ROOT / "cold_chain_mlp_fp32.keras"
M2_PATH = REPO_ROOT / "cold_chain_mlp_ptq_int8.tflite"


def main():
    X_train, X_val, y_train, y_val = load_splits()
    model = keras.models.load_model(M1_PATH)

    X_calib = calibration_samples(X_train)
    convert_full_int8(model, X_calib, M2_PATH)

    clf = Int8Classifier(M2_PATH)
    details = clf.describe(M2_PATH)
    print(json.dumps(details, indent=2))

    fp32_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    int8_pred = np.argmax(clf.predict_proba(X_val), axis=1)
    agreement = float(np.mean(fp32_pred == int8_pred))
    print(f"FP32 vs INT8 prediction agreement on validation: {agreement:.2%}")

    _, passed = evaluate("M2_PTQ_INT8", y_val, int8_pred, extra={
        "n_calibration": int(len(X_calib)),
        "fp32_int8_agreement": agreement,
        **details,
    })
    if not passed:
        return 1
    print(f"PASSED quality gates. Saved {M2_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
