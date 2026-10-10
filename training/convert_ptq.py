import tensorflow as tf
import numpy as np
import pandas as pd

# Load validation data for calibration representative dataset
dataframes = [pd.read_csv(f, header=None) for f in ["dataset_normal.csv", "dataset_warning.csv", "dataset_critical.csv"]]
df = pd.concat(dataframes, ignore_index=True)
X = df.iloc[:, :-1].values.astype(np.float32)

# Load trained FP32 Keras model
model = tf.keras.models.load_model("cold_chain_mlp_fp32.keras")

# Define representative dataset generator (>= 200 calibration samples)
def representative_dataset_gen():
    for i in range(min(250, len(X))):
        # Yield a 2D array matching input tensor shape
        yield [X[i:i+1]]

# Configure TFLite Converter for Full INT8 PTQ
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Enforce strict INT8 for input/output and internal tensors
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("cold_chain_mlp_ptq_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("✓ M2 PTQ INT8 TFLite model successfully generated and saved!")