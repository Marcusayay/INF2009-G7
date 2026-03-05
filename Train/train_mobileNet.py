import os
from dotenv import load_dotenv
from roboflow import Roboflow 
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import cv2
import numpy as np 
from utils import get_latest_number 

#! download 
file_to_save_in = "material-and-object-classifier-2"
train_dir = f"{file_to_save_in}/train"
test_dir  = f"{file_to_save_in}/test"
val_dir   = f"{file_to_save_in}/valid"
#! load your API KEY
#! Get it from https://roboflow.com/account
#! ================================ 
load_dotenv(".env")  # Load API key from .env file
api_key = os.getenv("ROBOFLOW_API_KEY") 
#! ================================ 

def download(): 
    if not os.path.exists(file_to_save_in):
        print ("📥 DOWNLOADING DATASET FROM ROBOFLOW..." )
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("zfcrow").project("material-and-object-classifer")
        version = project.version(2)
        dataset = version.download("folder", location=file_to_save_in)       
    else: 
        print("✅ DATASET ALREADY DOWNLOADED.")
# ====================================================== 


def mobilenet_preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)  # scales to what MobileNetV2 expects
    return x, y


#! TRAINING
def train(): 
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE,
        label_mode="int"   # integers 0..8
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE,
        label_mode="int"
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir, shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE,
        label_mode="int"
    )
    
    counts = {}
    # Look inside the train directory
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        # Only count if it is an actual folder
        if os.path.isdir(class_path): 
            counts[class_name] = len(os.listdir(class_path))

    # save the names to label.txt
    with open("labels.txt", "w") as f:
        for class_name in sorted(counts.keys()):
            f.write(f"{class_name}\n") 

    print("Actual Training Images per Class:", counts)

    num_classes = len(counts)
    total = sum(counts.values())
    class_weight = {
        i: total / (num_classes * counts[name])
        for i, name in enumerate(train_dataset.class_names)
    }




    train_dataset = train_dataset.map(
        mobilenet_preprocess, num_parallel_calls=AUTOTUNE
    ).cache("cache_train").prefetch(AUTOTUNE)

    validation_dataset = validation_dataset.map(
        mobilenet_preprocess, num_parallel_calls=AUTOTUNE
    ).cache("cache_val").prefetch(AUTOTUNE)

    test_dataset = test_dataset.map(
        mobilenet_preprocess, num_parallel_calls=AUTOTUNE
    ).cache("cache_test").prefetch(AUTOTUNE)


    #! phase 1 

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False


    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])


    model = models.Sequential([
        data_augmentation, 
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')  # ✅ 9-way classifier
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',  # ✅ for integer labels
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2, monitor="val_loss"),
    ]


    model.summary()


    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=8,
        callbacks=callbacks,
        class_weight=class_weight
    )

    #! Phase 2 (fine-tune last layers)

    base_model.trainable = True

    # Fine-tune only the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # IMPORTANT: Keep BatchNormalization layers frozen
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]
    )


    model.summary()

    history_ft = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=callbacks,
        class_weight=class_weight
    ) 


    # --- Evaluate on Test Set ---
    test_loss, test_acc, test_top3 = model.evaluate(test_dataset)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Top-3 Accuracy: {test_top3:.4f}")

    return model, history, history_ft 

# --- Plot Training History ---
def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    
    plt.figure(figsize=(8, 4))
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(x=len(h1.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

def convert_and_quantize(model):
    # =========================================================================
    # 🚀 AUTOMATED TFLITE CONVERSIONS (FP32 & UINT8)
    # =========================================================================
    print("\n=======================================================")
    print("🚀 STARTING AUTOMATED TFLITE CONVERSIONS")
    print("=======================================================\n")

    # --- 1. FP32 CONVERSION ---
    print("Converting to FP32 TFLite (No Quantization)...")
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32_model = converter_fp32.convert()

    with open('best_fp32.tflite', 'wb') as f:
        f.write(tflite_fp32_model)
    print("✅ FP32 Conversion Complete! Saved as 'best_fp32.tflite'")


    # --- 2. UINT8/INT8 QUANTIZATION ---
    print("\nConverting to UINT8 TFLite (Full Integer Quantization)...")
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        # We can reuse the val_dir variable you defined at the top of the script!
        base_dir = val_dir 
        all_image_paths = []
        
        for class_folder in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_folder)
            if os.path.isdir(class_path):
                valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
                images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                        if f.lower().endswith(valid_exts)]
                all_image_paths.extend(images[:20]) 

        np.random.shuffle(all_image_paths)
        
        count = 0
        for img_path in all_image_paths:
            if count >= 100: break 
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 🚨 THE FIX: Use exact same preprocessing as the training loop
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img) 
            
            count += 1
            yield [tf.expand_dims(img, axis=0)]

    converter_quant.representative_dataset = representative_data_gen
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.inference_input_type = tf.uint8
    converter_quant.inference_output_type = tf.uint8

    tflite_quant_model = converter_quant.convert()

    with open('best_quantized.tflite', 'wb') as f:
        f.write(tflite_quant_model)
    print("✅ Quantization Complete! Saved as 'best_quantized.tflite'")

    print("\n🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY! Move your .tflite files and labels.txt to the Pi 5.")


    #! look for all 3 files and label.txt and move them to 
    #! mobnet_models/v{latest_version}/
    latest_version = get_latest_number("mobnet_models") 
    dest_dir = f"mobnet_models/v{latest_version}" 
    os.makedirs(dest_dir, exist_ok=True)

    for filename in ["best.keras","best_fp32.tflite", "best_quantized.tflite", "labels.txt"]:
        src_path = filename
        dest_path = os.path.join(dest_dir, filename)
        os.rename(src_path, dest_path)
        print(f"✅ Moved '{filename}' to '{dest_dir}'")



# plot_history(history, history_ft)


if __name__ == "__main__":
    download() 
    # model, h1,h2 = train() 
    # convert_and_quantize(model) 
    
    # #! plot the training history (optional, but nice to see the curves) 
    #plot_history(h1, h2) 