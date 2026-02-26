import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt



train_dir = 'MobileNet_Classification_Data/train'
test_dir  = 'MobileNet_Classification_Data/test'
val_dir   = 'MobileNet_Classification_Data/valid'

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

num_classes = len(train_dataset.class_names) 

counts = {
    "bottle": 26162,
    "can": 7471,
    "cardboard": 9439,
    "carton_tetrapack": 9189,
    "cup_mug": 7248,
    "disposable_cup": 2186,
    "paper_sheet": 2389,
    "plastic-wrapper": 7628,
    "wine_glass": 2023,
}
missing = [n for n in train_dataset.class_names if n not in counts]
extra = [n for n in counts if n not in train_dataset.class_names]
print("Missing in counts:", missing)
print("Extra in counts:", extra)


total = sum(counts.values())
class_weight = {
    i: total / (num_classes * counts[name])
    for i, name in enumerate(train_dataset.class_names)
}

def mobilenet_preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)  # scales to what MobileNetV2 expects
    return x, y


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

plot_history(history, history_ft)
