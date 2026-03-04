import tensorflow as tf

# 1. Load your saved Keras model
print("Loading Keras model...")
model = tf.keras.models.load_model('best.keras')

# 2. Initialize the standard TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# NOTE: We are intentionally NOT adding converter.optimizations here.
# By leaving it out, TensorFlow defaults to preserving the FP32 weights.

# 3. Convert the model
print("Converting to FP32 TFLite...")
tflite_fp32_model = converter.convert()

# 4. Save the file
output_path = 'best_fp32.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_fp32_model)

print(f"✅ FP32 Conversion Complete! Saved as '{output_path}'")