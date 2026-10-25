"""Shared dataset loading, normalisation and splitting for M1/M2/M3.

Every training and conversion script imports from here so all three model
variants see identical inputs, identical normalisation (matching
inference/inference_service.py) and the identical held-out validation split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = REPO_ROOT / "data_pipeline" / "training_stats.npy"
DATASET_FILES = {
    0: REPO_ROOT / "dataset_normal.csv",
    1: REPO_ROOT / "dataset_warning.csv",
    2: REPO_ROOT / "dataset_critical.csv",
}
CLASS_NAMES = ["Normal (0)", "Warning (1)", "Critical (2)"]
N_FEATURES = 6
SEED = 42
VAL_FRACTION = 0.2

# Must stay identical to inference/inference_service.py
STD_FLOORS = {2: 0.2, 5: 1.0}
MIN_STD = 0.1
CLIP = 10.0


def load_raw_dataset():
    """Return raw features X (n, 6) float32 and labels y (n,) int32."""
    frames = []
    for label, path in DATASET_FILES.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"{path} is missing or empty. Run training/generate_dataset.py first."
            )
        df = pd.read_csv(path, header=None)
        if df.shape[1] != N_FEATURES + 1:
            raise ValueError(
                f"{path}: expected {N_FEATURES + 1} columns, got {df.shape[1]}"
            )
        if not (df.iloc[:, -1] == label).all():
            raise ValueError(f"{path}: every row must have label {label}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.int32)
    if not np.all(np.isfinite(X)):
        raise ValueError("Dataset contains non-finite feature values.")
    return X, y


def load_stats():
    """Return (mean, safe_std) with the same floors the inference service applies."""
    if not STATS_PATH.is_file():
        raise FileNotFoundError(
            f"{STATS_PATH} is missing. Run data_pipeline/preprocessing.py --mode generate_stats first."
        )
    stats = np.load(STATS_PATH, allow_pickle=True).item()
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ValueError(f"{STATS_PATH} must contain 'mean' and 'std' arrays.")
    mean = np.asarray(stats["mean"], dtype=np.float32).copy()
    std = np.asarray(stats["std"], dtype=np.float32).copy()
    if mean.shape != (N_FEATURES,) or std.shape != (N_FEATURES,):
        raise ValueError(
            f"{STATS_PATH}: expected shape ({N_FEATURES},), got {mean.shape} and {std.shape}"
        )
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))) or np.any(std < 0):
        raise ValueError(f"{STATS_PATH} contains invalid values.")
    for idx, floor in STD_FLOORS.items():
        std[idx] = max(std[idx], floor)
    return mean, np.maximum(std, MIN_STD)


def normalise(X, mean, safe_std):
    return np.clip((X - mean) / safe_std, -CLIP, CLIP).astype(np.float32)


def load_splits():
    """Normalised, stratified train/validation split shared by every variant."""
    X, y = load_raw_dataset()
    mean, safe_std = load_stats()
    Xn = normalise(X, mean, safe_std)
    X_train, X_val, y_train, y_val = train_test_split(
        Xn, y, test_size=VAL_FRACTION, random_state=SEED, stratify=y
    )
    return X_train, X_val, y_train, y_val
