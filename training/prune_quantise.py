import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load datasets
dataframes = [pd.read_csv(f, header=None) for f in ["dataset_normal.csv", "dataset_warning.csv", "dataset_critical.csv"]]
df = pd.concat(dataframes, ignore_index=True)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.int32)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define Pruning Parameters (35% structured pruning)
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
end_step = np.ceil(len(X_train) / 16).astype(int) * 15 # 15 epochs

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.35, begin_step=0, end_step=end_step)
}

# Build Pruned Model Architecture
base_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_to_prune = prune_low_magnitude(base_model, **pruning_params)
model_to_prune.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

# Fine-tune with pruning
model_to_prune.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val), callbacks=callbacks)

# Strip pruning wrappers to prepare for conversion
pruned_model = tfmot.sparsity.keras.strip_pruning(model_to_prune)
pruned_model.save("cold_chain_mlp_pruned.keras")

# Apply PTQ INT8 on the pruned model
def representative_dataset_gen():
    for i in range(min(250, len(X))):
        yield [X[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_m3 = converter.convert()

with open("cold_chain_mlp_pruned_int8.tflite", "wb") as f:
    f.write(tflite_model_m3)

print("✓ M3 Structured Pruning + PTQ INT8 model successfully generated!")