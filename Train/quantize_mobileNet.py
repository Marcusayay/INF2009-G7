import tensorflow as tf
import numpy as np
import cv2 
import os 

DIR_TO_CONFIG="material-and-object-classifier-1"  # Change this to your validation folder
# 1. Load your high-accuracy model
model = tf.keras.models.load_model('best.keras')

# 2. Setup the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Enable INT8 Quantization (The magic for Section 10.3)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Mandatory: Representative Dataset
# This "tells" the converter the typical range of your trash images
def representative_data_gen():
    base_dir = os.path.join(DIR_TO_CONFIG, "valid")
    all_image_paths = []
    
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path):
            # Filter for common image extensions only
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                      if f.lower().endswith(valid_exts)]
            all_image_paths.extend(images[:20]) 

    np.random.shuffle(all_image_paths)

    count = 0
    for img_path in all_image_paths:
        if count >= 100: break # Stop once we hit our target
        
        img = cv2.imread(img_path)
        
        # --- THE FIX: Skip if the image failed to load ---
        if img is None:
            print(f"⚠️ Warning: Could not read {img_path}. Skipping...")
            continue
            
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        count += 1
        yield [np.expand_dims(img, axis=0)]

converter.representative_dataset = representative_data_gen

# 5. Ensure the model is strictly INT8 (No float fallback)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# 6. Convert and Save
tflite_model = converter.convert()

with open('best_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Quantization Complete! Move 'best_quantized.tflite' to your Pi 5.")