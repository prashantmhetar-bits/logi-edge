import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

print("🚀 Loading class datasets for M1 FP32 Baseline Training...")
dataset_files = ["dataset_normal.csv", "dataset_warning.csv", "dataset_critical.csv"]
dataframes = []

for file in dataset_files:
    if os.path.exists(file) and os.path.getsize(file) > 0:
        dataframes.append(pd.read_csv(file, header=None))
        print(f"✓ Loaded {file}")
    else:
        print(f"❌ Error: {file} is missing or empty. Run generate_dataset.py first.")
        exit(1)

df = pd.concat(dataframes, ignore_index=True)

# Features are columns 0-5, Label is the last column
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.int32)

# Normalize using the clean-data statistics used by the live preprocessing pipeline.
stats_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_pipeline",
    "training_stats.npy",
)
if not os.path.exists(stats_file):
    raise FileNotFoundError(
        f"{stats_file} is missing. Run data_pipeline/preprocessing.py "
        "in generate_stats mode first."
    )

stats = np.load(stats_file, allow_pickle=True).item()
if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
    raise ValueError(f"{stats_file} must contain 'mean' and 'std' arrays.")

train_mean = np.asarray(stats["mean"], dtype=np.float32)
train_std = np.asarray(stats["std"], dtype=np.float32)
if train_mean.shape != (X.shape[1],) or train_std.shape != (X.shape[1],):
    raise ValueError(
        f"Expected six training means and standard deviations in {stats_file}; "
        f"got shapes {train_mean.shape} and {train_std.shape}."
    )
if not np.all(np.isfinite(train_mean)) or not np.all(np.isfinite(train_std)):
    raise ValueError(f"{stats_file} contains non-finite values.")
if np.any(train_std < 0):
    raise ValueError(f"{stats_file} contains negative standard deviations.")

# Match the standard-deviation floors and clipping used during live preprocessing.
train_std[2] = max(train_std[2], 0.2)
train_std[5] = max(train_std[5], 1.0)
safe_std = np.maximum(train_std, 0.1)
X = np.clip((X - train_mean) / safe_std, -10.0, 10.0)
print(f"✓ Normalized features using {stats_file}")

print(f"\nTotal combined dataset samples: {len(X)}")

# Stratified 20% validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build recommended 2-hidden-layer MLP (32 and 16 units, ReLU activation)
print("\nBuilding MLP Architecture: 32 -> 16 units (ReLU)...")
model = Sequential([
    Input(shape=(6,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax') # 3 classes: Normal (0), Warning (1), Critical (2)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate on held-out validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"\n==========================================")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"==========================================")

if val_accuracy >= 0.88:
    print("✓ Success! Accuracy exceeds the mandatory 88% requirement.")
    model.save("cold_chain_mlp_fp32.keras")
    print("✓ M1 FP32 Baseline model saved as 'cold_chain_mlp_fp32.keras'")
else:
    print("⚠️ Warning: Validation accuracy is below 88%. Check feature extraction or increase training duration.")

# Detailed Classification Report
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["Normal (0)", "Warning (1)", "Critical (2)"]))