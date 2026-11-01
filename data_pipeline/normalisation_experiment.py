#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))

import keras
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from dataset_utils import (
    CLASS_NAMES,
    MODELS_DIR,
    SEED,
    VAL_FRACTION,
    load_raw_dataset,
    load_stats,
    normalise,
)
from model_eval import LABELS, MIN_VAL_ACCURACY, Int8Classifier

RESULTS_PATH = REPO / "data_pipeline" / "results" / "normalisation_experiment.json"
SHIFTS = {"correct": 0.0, "+3 sigma": 3.0, "-3 sigma": -3.0}
VARIANTS = {
    "M1_FP32": MODELS_DIR / "cold_chain_mlp_fp32.keras",
    "M2_PTQ_INT8": MODELS_DIR / "cold_chain_mlp_ptq_int8.tflite",
    "M3_PRUNED_INT8": MODELS_DIR / "cold_chain_mlp_pruned_int8.tflite",
}


def predictor(path):
    if path.suffix == ".keras":
        model = keras.models.load_model(path)
        return lambda X: np.argmax(model.predict(X, verbose=0), axis=1)
    clf = Int8Classifier(path)
    return lambda X: np.argmax(clf.predict_proba(X), axis=1)


def score(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_per_class": {n: float(r) for n, r in zip(CLASS_NAMES, recalls)},
        "critical_recall": float(recalls[2]),
        "predicted_class_counts": np.bincount(y_pred, minlength=3).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main():
    missing = [p.name for p in VARIANTS.values() if not p.is_file()]
    if missing:
        print(
            f"Missing models: {missing}. Run the training scripts first.",
            file=sys.stderr,
        )
        return 1

    X_raw, y = load_raw_dataset()
    _, X_val_raw, _, y_val = train_test_split(
        X_raw, y, test_size=VAL_FRACTION, random_state=SEED, stratify=y
    )
    mean, sigma = load_stats()
    print(
        f"Validation samples: {len(y_val)} | true class counts {np.bincount(y_val).tolist()}"
    )
    print(f"Training stats mean : {np.round(mean, 4).tolist()}")
    print(
        f"Training stats sigma: {np.round(sigma, 4).tolist()} (with inference floors)"
    )

    results = {}
    for name, path in VARIANTS.items():
        predict = predictor(path)
        results[name] = {}
        for label, k in SHIFTS.items():
            y_pred = predict(normalise(X_val_raw, mean + k * sigma, sigma))
            results[name][label] = score(y_val, y_pred)

    lines = [
        "| Variant | Stats | Accuracy (%) | Δ vs correct (pp) | Recall N / W / C (%) | Predicted N / W / C |",
        "|---|---|---:|---:|---|---|",
    ]
    for name, by_shift in results.items():
        base = by_shift["correct"]["accuracy"]
        for label, r in by_shift.items():
            rec = " / ".join(f"{v * 100:.0f}" for v in r["recall_per_class"].values())
            pred = " / ".join(str(c) for c in r["predicted_class_counts"])
            delta = (
                "—" if label == "correct" else f"{(r['accuracy'] - base) * 100:+.1f}"
            )
            lines.append(
                f"| {name} | {label} | {r['accuracy'] * 100:.1f} | {delta} | {rec} | {pred} |"
            )
    table = "\n".join(lines)

    print("\n" + table)
    report = {
        "n_val": int(len(y_val)),
        "true_class_counts": np.bincount(y_val).tolist(),
        "shift_definition": "mean' = mean + k * sigma, k in {0, +3, -3}; normalised features move by -k",
        "stats_mean": mean.tolist(),
        "stats_sigma_used": sigma.tolist(),
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {RESULTS_PATH.relative_to(REPO)}")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(
                "### C2 normalisation experiment: correct stats vs stats shifted by 3σ\n\n"
            )
            f.write(
                f"Held-out validation split, n = {len(y_val)} "
                f"(true N / W / C = {' / '.join(map(str, np.bincount(y_val)))}). "
                "Shifting the stored mean by +3σ moves every normalised feature by −3.\n\n"
            )
            f.write(table + "\n\n")

    failed = [
        n for n in VARIANTS if not results[n]["correct"]["accuracy"] > MIN_VAL_ACCURACY
    ]
    if failed:
        print(
            f"Control run (correct stats) below the {MIN_VAL_ACCURACY:.0%} gate: {failed}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
