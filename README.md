# INF2009-G7 — Edge AI Waste Classification System

> **Demo Session Notice:** Poster, Video, and Peer Evaluation are due **Monday, 30th March, 0900 hrs**.
> The demo is a live showcase of the system's functionality — not a video presentation.
> Please include the GitHub repo link in the poster. All work, experiments, and iterations must be documented in the repo as this directly influences grading.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [ML Pipeline — Detailed](#ml-pipeline--detailed)
   - [Dataset Acquisition](#1-dataset-acquisition)
   - [Dataset Preparation for MobileNet](#2-dataset-preparation-for-mobilenet)
   - [MobileNetV2 Model Training](#3-mobilenetv2-model-training)
   - [TFLite Conversion (FP32)](#4-tflite-conversion-fp32)
   - [INT8 Quantization](#5-int8-quantization)
   - [YOLO Object Detection Training](#6-yolo-object-detection-training)
   - [YOLO Export & Format Benchmarking](#7-yolo-export--format-benchmarking)
   - [Accuracy Benchmarking Across Formats](#8-accuracy-benchmarking-across-formats)
4. [Inference Pipeline (Edge Deployment)](#inference-pipeline-edge-deployment)
   - [Model Loading & Runtime](#model-loading--runtime)
   - [run_mob.py — Full Benchmark Run](#run_mobpy--full-benchmark-run)
   - [run.py — Lightweight Detection Run](#runpy--lightweight-detection-run)
5. [Hardware Monitoring](#hardware-monitoring)
6. [Model Versioning System](#model-versioning-system)
7. [Results Analysis](#results-analysis)
8. [Model Iterations & Dataset Evolution](#model-iterations--dataset-evolution)
9. [File Reference](#file-reference)

---

## Project Overview

This project builds an **edge AI waste classification system** designed to run on a **Raspberry Pi 5**. Two ML approaches are implemented and benchmarked:

- **MobileNetV2** — a lightweight image classification model (crops → classify), optimised for low-power inference via INT8 quantization into TFLite.
- **YOLOv8/v26** — a full object detection model, exported to NCNN (FP16 & FP32) and TFLite (INT8) for comparative benchmarking.

The system captures live video, crops the centre region of interest, runs inference, and uses a **temporal buffer** to confirm detections before triggering a save/alert — minimising false positives under noisy real-world conditions.

---

## System Architecture

```
Camera Feed (640x480)
        │
        ▼
  Centre Crop (320x320)
        │
        ▼
  Resize to 224x224
        │
        ▼
 ┌──────────────────────┐
 │  MobileNetV2 TFLite  │  ← INT8 Quantized (primary)
 │  (or FP32 fallback)  │    or FP32 TFLite
 └──────────────────────┘
        │
        ▼
  Temporal Buffer (10 frames)
        │
        ▼
  Majority Vote + Confidence Threshold
        │
        ▼
  Detection Alert + Image Save
```

---

## ML Pipeline — Detailed

### 1. Dataset Acquisition

**Script:** `train_yolo.py`, `train_mobileNet.py`

Two separate Roboflow datasets were used:

| Dataset | Purpose | Format |
|---|---|---|
| `recycle_object_detection_3` (Roboflow workspace: `zfcrow`) | YOLO bounding box detection | YOLOv8 / yolo26 |
| `waste_classifier-e14pp` (Roboflow workspace: `cloudcomputing-s-workspace`, version 2) | MobileNet classification | Folder format (class-per-folder) |

- API keys are loaded securely from a `.env` file using `python-dotenv`. No hardcoded credentials.
- Dataset download is **idempotent** — if the folder already exists, download is skipped. This avoids re-downloading during re-runs.
- YOLO dataset path is resolved to an absolute path to avoid working-directory issues during training.

---

### 2. Dataset Preparation for MobileNet

**Scripts:** `mobnet_dataset.py`, `extract.py`

Since MobileNet is a **classifier** (not a detector), bounding-box datasets must be pre-processed into cropped per-class images.

#### `mobnet_dataset.py` — YOLO Dataset → Classification Crops

This script converts the YOLO-format dataset (images + `.txt` label files) into a classification-ready folder structure:

```
MobileNet_Classification_Data/
├── train/
│   ├── plastic_bottle/
│   ├── cardboard/
│   └── ...
├── valid/
└── test/
```

**Process:**
1. Loads class names from `data.yaml`.
2. Iterates over all images in `train`, `valid`, and `test` splits.
3. For each label file, reads YOLO-normalised coordinates `(class_id, x_center, y_center, w, h)`.
4. **De-normalises** to pixel coordinates: `x_min = (x_center - w/2) * img_width`, etc.
5. Clamps coordinates to image boundaries to avoid out-of-bounds crops.
6. Saves each crop as `{img_name}_crop_{i}.jpg` into the correct class subfolder.
7. Skips zero-size crops gracefully.

#### `extract.py` — COCO Annotation → Cropped Images

An alternative extractor for COCO-format datasets (e.g., the `recycle_object_detection_3.coco` split):

1. Loads `_annotations.coco.json`.
2. Finds the target class by name (e.g., `disposable_cup`) — case-insensitive.
3. Maps annotation `image_id` → filename via COCO's `images` array.
4. Extracts crops using COCO bbox format `[x_min, y_min, width, height]`.
5. Applies a **critical fix**: values are cast via `int(float(v))` to handle string-encoded floats in some COCO exports.

---

### 3. MobileNetV2 Model Training

**Script:** `train_mobileNet.py`

Training uses **transfer learning from ImageNet weights** in two phases. The training pipeline was iterated multiple times — model versions are tracked under `mobnet_models/v{n}/`.

#### Architecture

```
Input (224x224x3)
    │
    ▼
Data Augmentation (RandomFlip, RandomRotation±5%, RandomZoom±10%, RandomContrast±10%)
    │
    ▼
MobileNetV2 backbone (pre-trained, ImageNet)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dropout (0.2)
    │
    ▼
Dense(num_classes, activation='softmax')
```

#### Preprocessing

- Uses `MobileNetV2`'s native `preprocess_input()` which scales pixel values to `[-1, 1]` — **not** a simple `/255.0` normalisation. This was an important correctness fix; mismatched preprocessing between training and inference causes silent accuracy degradation.
- Applied via `tf.data.Dataset.map()` with `AUTOTUNE` parallelism.
- Datasets are `.cache()`d to disk and `.prefetch()`d to eliminate I/O bottlenecks during training.

#### Class Imbalance Handling

Real-world waste datasets are heavily imbalanced (e.g., many plastic bottles, few glass jars). **Inverse-frequency class weights** are computed and passed to `model.fit()`:

```python
class_weight = {
    i: total / (num_classes * counts[name])
    for i, name in enumerate(train_dataset.class_names)
}
```

This penalises the model more for misclassifying rare classes.

#### Phase 1 — Classifier Head Training (Frozen Backbone)

- `base_model.trainable = False` — only the new Dense head is trained.
- Optimizer: `Adam(lr=1e-4)`
- Loss: `sparse_categorical_crossentropy` (integer labels, not one-hot)
- **Up to 8 epochs**, with:
  - `ModelCheckpoint` — saves `best.keras` only when `val_accuracy` improves
  - `EarlyStopping(patience=3)` — stops if validation accuracy plateaus
  - `ReduceLROnPlateau(patience=2, factor=0.2)` — halves the LR on val_loss plateau

#### Phase 2 — Fine-tuning (Partial Unfreeze)

- `base_model.trainable = True`, but **only the last 30 layers** of MobileNetV2 are unfrozen.
- All `BatchNormalization` layers are **kept frozen** — critical to preserve running statistics from ImageNet and avoid training instability at low LR.
- Optimizer: `Adam(lr=1e-5)` — 10× lower than Phase 1 to prevent catastrophic forgetting.
- Adds `SparseTopKCategoricalAccuracy(k=3)` metric — tracks whether the correct class is in the top-3 predictions.
- **Up to 10 epochs**, same callbacks as Phase 1.
- Final evaluation on the held-out test set reports both top-1 and top-3 accuracy.

#### `train_with_custom_base()` — Incremental Fine-tuning

A second training mode was developed for **iterative improvement**: instead of starting from ImageNet, it loads a previously saved `.keras` model and re-trains its backbone on a new dataset version. This enabled reuse of learned trash-specific features across dataset versions without full retraining.

- Extracts the MobileNetV2 backbone (`old_model.layers[1]`) from the old Sequential model.
- Rebuilds the Sequential stack with a fresh Dense head sized for the new number of classes.
- Runs the same Phase 1 → Phase 2 training loop.

#### Labels & Config Persistence

- Class names are saved to `labels.txt` (alphabetically sorted) — this file is read at inference time to map class IDs back to human-readable names.
- Training hyperparameters are saved to `training_config.txt` for reproducibility tracking.

---

### 4. TFLite Conversion (FP32)

**Scripts:** `convertToTFLitefromKeras_mobileNet.py`, `train_mobileNet.py` (`convert_and_quantize()`)

After training, the `.keras` model is converted to TFLite FP32 for deployment:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No optimizations — preserves FP32 weights exactly
tflite_fp32_model = converter.convert()
```

- No `converter.optimizations` are set — this is intentional to preserve full float precision as the baseline.
- Output: `best_fp32.tflite` — larger file, slower inference, but highest accuracy ceiling.

---

### 5. INT8 Quantization

**Scripts:** `quantize_mobileNet.py`, `train_mobileNet.py` (`convert_and_quantize()`)

Quantization reduces model size ~4× and significantly accelerates inference on hardware lacking FP32 acceleration (e.g., Raspberry Pi 5 without a dedicated NPU).

#### Representative Dataset

Full integer quantization requires a **calibration dataset** to determine the range of activations at every layer:

```python
def representative_data_gen():
    # Samples up to 100 images from the validation set
    # Applies identical preprocessing to training: resize → RGB → preprocess_input()
    yield [tf.expand_dims(img, axis=0)]
```

Key details:
- Samples at most 20 images per class folder, shuffled, capped at 100 total.
- **Crucially uses the same `preprocess_input()` preprocessing** as training — an earlier version of this function used `/255.0` normalisation which caused a significant accuracy drop post-quantization.
- Images that fail to load (`cv2.imread()` returns `None`) are skipped with a warning — hardened against corrupted dataset files.

#### Quantization Config

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```

- `TFLITE_BUILTINS_INT8` enforces **strict INT8** — no float fallback ops. If any layer cannot be quantized, conversion fails rather than silently falling back.
- Both input and output are `uint8` — inference code must handle dequantization of the output using the stored `(scale, zero_point)` metadata.
- Output: `best_quantized.tflite`

---

### 6. YOLO Object Detection Training

**Script:** `train_yolo.py`

A parallel detection approach using Ultralytics YOLO was developed and benchmarked against the MobileNet classifier.

#### Training Setup

- Base model: configurable (`yolo26n.pt` in latest run — a YOLOv8-family nano model).
- Dataset: `taco_data_v7_without_coco` — downloaded from Roboflow in `yolo26` format.
- GPU training on RTX 4070 (CUDA device 0).

#### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `epochs` | 100 | Upper bound; early stopping via `patience=20` |
| `imgsz` | 640 | Standard YOLO input resolution |
| `batch` | -1 | Auto-sized by Ultralytics to fit VRAM |
| `workers` | 8 | Parallel data loading |
| `cache` | `'disk'` | Caches preprocessed images to disk |
| `optimizer` | `'auto'` | Ultralytics auto-selects (SGD/Adam) |
| `patience` | 20 | Early stopping on mAP plateau |
| `close_mosaic` | 10 | Disables mosaic augmentation for final 10 epochs (improves detection of single objects) |

---

### 7. YOLO Export & Format Benchmarking

**Script:** `train_yolo.py`

After training, the `best.pt` checkpoint was exported to multiple formats to benchmark speed vs. accuracy trade-offs on the Raspberry Pi 5:

| Format | Command | Purpose |
|---|---|---|
| NCNN FP32 | `model.export(format="ncnn")` | Default NCNN — baseline speed |
| NCNN FP16 | `model.export(format="ncnn", half=True)` | Half-precision — faster inference, slight accuracy drop |
| TFLite INT8 | `model.export(format="tflite", int8=True, data=data_path)` | Full integer quantization — smallest model, fastest on CPU |

- NCNN FP16 was identified as the **"Max Speed"** winner on the Pi 5.
- TFLite INT8 was identified as the **"Best Efficiency"** winner (power/accuracy trade-off).

---

### 8. Accuracy Benchmarking Across Formats

**Script:** `accuracy.py`

All four exported YOLO model formats were evaluated on the validation split using Ultralytics' built-in `.val()`:

```python
r = m.val(data=dataset_loc, split="val", imgsz=640, conf=0.25, iou=0.7)
```

Metrics collected per model: Precision, Recall, mAP@0.5, mAP@0.5:0.95.

Models benchmarked:
- `best_ncnn_model_fp32` (NCNN FP32)
- `best_ncnn_model_fp16` (NCNN FP16)
- `best_integer_quant.tflite` (TFLite INT8)
- `best_full_integer_quant.tflite` (TFLite Full INT8)

Results were aggregated into a `defaultdict` and printed in tabular form for direct comparison.

---

## Inference Pipeline (Edge Deployment)

### Model Loading & Runtime

**Script:** `mobileNet_helpers.py`

The Pi 5 does not have a standard TFLite runtime install. A compatibility shim is used:

```python
import ai_edge_litert
sys.modules["tflite_runtime"] = ai_edge_litert
sys.modules["tflite_runtime.interpreter"] = ai_edge_litert.interpreter
import tflite_runtime.interpreter as tflite
```

This aliases Google's `ai-edge-litert` package (the successor to the standalone `tflite-runtime`) to the standard import path, so inference code is portable.

#### `load_model()`

- Accepts `quantized=True/False` and optional `model_path` to support the versioning system.
- Allocates tensors and returns `(interpreter, input_details, output_details, model_path)`.

#### `run_inference()`

Handles both quantized and FP32 paths:

**Quantized (INT8/UINT8):**
- Input is cast to `np.uint8` (no normalisation — the quantized model handles scale internally).
- Output is dequantized using stored `(scale, zero_point)` metadata: `probabilities = scale * (output - zero_point)`.

**FP32:**
- Input is normalized to `[0, 1]` as `np.float32 / 255.0`.
- Output is already softmax probabilities.

Returns: `(class_id, confidence, inference_time_ms, full_probabilities_array)`

The full probabilities array enables downstream visualisation of all class scores, not just the top-1.

---

### `run_mob.py` — Full Benchmark Run

The full edge deployment script with hardware monitoring, image saving, and live video display.

#### Temporal Buffer / Smoothing Logic

A **sliding window buffer** of the last 10 frames is maintained to avoid triggering on single-frame noise:

```
Buffer (10 frames): [(class_id, confidence), ...]
    │
    ▼
Majority vote: most frequent class_id
    │
    ▼
Frequency check: ≥ 7/10 frames must agree
    │
    ▼
Confidence check: average confidence ≥ 0.85
    │
    ▼
DETECTION CONFIRMED → save image
```

#### Cooldown System

After a confirmed detection, a **3-second cooldown** prevents duplicate saves of the same object. During cooldown:
- The buffer is cleared on every frame (stale data rejected).
- A visual countdown is overlaid on the video feed.
- After cooldown expires, the buffer re-accumulates from scratch.

#### White Balance Correction

A grey-world white balance algorithm is applied to each frame before inference:

```python
overall_mean = (r_mean + g_mean + b_mean) / 3
img_wb[:,:,0] *= (overall_mean / r_mean)  # scale each channel
```

This compensates for different lighting conditions (fluorescent, warm indoor, outdoor), improving real-world generalisation without retraining.

#### Capture Box Strategy

Rather than cropping to exactly `224×224` (the model input size), a larger `320×320` region is cropped from the centre of the frame and then **resized** to `224×224`. This gives the model more context about the object's surroundings, which was found to improve detection accuracy in practice.

#### Live Statistics Overlay

- FPS, CPU temperature, power consumption (top-left)
- RAM usage (below)
- All class probabilities with confidence percentages (right panel)
- Detection alert box (red, 15-frame countdown)

#### Benchmark Metrics Recorded

After the session, the following are printed:
- System FPS mean, System latency mean
- Model-only inference mean
- Python overhead (system latency − model inference)
- Peak RAM, peak CPU temperature, peak power draw

---

### `run.py` — Lightweight Detection Run

A stripped-down version of `run_mob.py` for quick testing:

- No image saving, no video recording, no hardware monitoring overhead.
- Same buffer/confidence logic.
- Adds version-specific confidence threshold overrides (e.g., v14 uses 0.89, v15 uses 0.50) — these were determined empirically during testing iterations.
- Prints a full buffer dump on detection including per-frame confidence margin (top-1 − top-2 score) to diagnose borderline cases.
- Handles `KeyboardInterrupt` gracefully — reports buffer state at time of interruption.

---

## Hardware Monitoring

**Script:** `monitoring.py`

Three hardware metrics are tracked in real time on the Raspberry Pi 5:

| Metric | Method |
|---|---|
| RAM Usage (%) | `psutil.virtual_memory().percent` |
| CPU Temperature (°C) | `psutil.sensors_temperatures()["cpu_thermal"][0].current` |
| Board Power Draw (W) | `vcgencmd pmic_read_adc` — parses PMIC ADC output |

#### Power Measurement Detail

Power is estimated by reading voltage and current from the Pi 5's PMIC via `vcgencmd pmic_read_adc`. Four rails are summed:

| Rail | Index (current, volt) | Purpose |
|---|---|---|
| VDD_CORE | current7, volt15 | CPU core power |
| 1V1_SYS | current5, volt13 | System 1.1V rail |
| 0V8_SW | current6, volt14 | Switching regulator |
| 1V8_SYS | current2, volt10 | System 1.8V rail |

`Total Power = VDD_CORE + 1V1_SYS + 0V8_SW + 1V8_SYS`

These indices were determined empirically by reading and labelling the full `vcgencmd pmic_read_adc` output.

---

## Model Versioning System

**Scripts:** `utils.py`, `pick_model.py`

Models are stored under `mobnet_models/v{n}/` with auto-incrementing version numbers.

#### `utils.get_latest_number()`

Scans the `mobnet_models/` directory, extracts version numbers from folder names matching `v{n}`, and returns `max + 1` for the next version.

#### `pick_model.return_latest_version_path()`

Given a model type (`"mobilenet"` or `"yolo"`), returns the path to the latest version folder. Used by inference scripts to automatically load the most recent trained model without hardcoding paths.

#### Version Contents

Each version folder (`mobnet_models/v{n}/`) contains:
- `best.keras` — full Keras model (for further fine-tuning)
- `best_fp32.tflite` — FP32 TFLite model
- `best_quantized.tflite` — INT8 quantized TFLite model
- `labels.txt` — class name list (one per line, alphabetically sorted)
- `training_config.txt` — hyperparameter record

---

## Results Analysis

**Script:** `analysis.py`

Benchmark results (CSV format) are loaded with `pandas` and displayed as a formatted table using `tabulate` with ANSI colour/bold formatting. Results are sorted by `system_fps_mean` descending to rank models by throughput.

The `family` column is dropped before display. The script handles both local and subdirectory CSV paths gracefully with a try/except fallback.

---

## Model Iterations & Dataset Evolution

The model underwent multiple iterations, primarily driven by dataset refinement rather than architectural changes.

Initially, a default MobileNetV2 model was trained using a standard dataset. However, early results showed poor generalisation when deployed in the actual system environment. This highlighted that the training data distribution did not match real-world conditions.

To address this, additional datasets were sourced and merged to improve class diversity and coverage. While this improved baseline performance, inconsistencies remained when testing on the physical bin setup.

A key turning point was the collection of custom images directly from the deployed system. These images captured the actual camera perspective, object placement, and environmental conditions. In particular, the presence of UV lighting introduced a strong colour bias that was not represented in earlier datasets.

To mitigate this, two approaches were explored:

* Incorporating real bin-captured images into the training dataset to improve domain alignment
* Applying frame-level white balancing during preprocessing to counteract the UV lighting effects

Empirically, incorporating real-world images had the most significant impact on model performance, as it allowed the model to learn the true distribution of features under deployment conditions. White balancing was tested as an additional correction step but was less effective than directly training on representative data.

Through these iterations, the model progressively improved in robustness and accuracy within the target environment, demonstrating the importance of dataset alignment over purely architectural changes.

---
## File Reference

| File | Purpose |
|---|---|
| `train_mobileNet.py` | Full MobileNetV2 training pipeline (download → train → convert → quantize → version) |
| `train_yolo.py` | YOLO training, export to NCNN/TFLite |
| `mobnet_dataset.py` | Convert YOLO dataset → per-class image crops for MobileNet |
| `extract.py` | Extract crops from COCO-format dataset |
| `convertToTFLitefromKeras_mobileNet.py` | Standalone FP32 TFLite conversion |
| `quantize_mobileNet.py` | Standalone INT8 quantization |
| `accuracy.py` | YOLO model format accuracy benchmarking |
| `mobileNet_helpers.py` | TFLite model loading + inference (handles LiteRT alias) |
| `run_mob.py` | Full edge inference benchmark with monitoring, video, image saving |
| `run.py` | Lightweight inference script for quick testing |
| `monitoring.py` | RAM / temperature / power monitoring via psutil + vcgencmd |
| `pick_model.py` | Auto-select latest versioned model path |
| `utils.py` | Version number utilities |
| `analysis.py` | Benchmark CSV display and ranking |
| `test.py` | Verify GPU availability for training machine |

---

---

# System Integration — `main.py` & Iteration History

This section documents the full embedded system pipeline (`main.py`) and the complete evolution from early prototypes to the final production code.

---

## Table of Contents (System Integration)

1. [Why YOLO Was Dropped](#why-yolo-was-dropped)
2. [Final System Architecture](#final-system-architecture)
3. [Hardware Stack](#hardware-stack)
4. [Camera Management](#camera-management)
5. [Servo & ArUco Tracking Daemon](#servo--aruco-tracking-daemon)
6. [Weight Sensor Pipeline](#weight-sensor-pipeline)
7. [Spectrometer Pipeline (AS7343)](#spectrometer-pipeline-as7343)
8. [Sensor Fusion Logic](#sensor-fusion-logic)
9. [MQTT Publishing](#mqtt-publishing)
10. [Startup Calibration Sequence](#startup-calibration-sequence)
11. [Main Pipeline Loop](#main-pipeline-loop)
12. [Iteration History — What Changed and Why](#iteration-history--what-changed-and-why)
13. [Calibration Data](#calibration-data)

---

## Why YOLO Was Dropped

YOLO (trained and benchmarked in `train_yolo.py` and `accuracy.py`) was the first approach considered for object classification on the edge. After benchmarking all four export formats (NCNN FP32, NCNN FP16, TFLite INT8, TFLite Full INT8), YOLO was rejected for the embedded pipeline for the following reasons:

- **Latency:** Even the smallest YOLO nano model at NCNN FP16 had inference latency that was too high to run alongside the servo tracking daemon and HX711 polling on the Pi 5's 4 cores simultaneously without contention.
- **Architecture mismatch:** YOLO is a full detection model (bounding box + class). The system does not need localisation — the object is placed on a fixed platform in front of the camera, so a classifier is sufficient and far more efficient.
- **Memory footprint:** YOLO's NCNN model is significantly larger than a MobileNetV2 TFLite INT8 model, putting more pressure on the Pi 5's RAM when running alongside multiple sensor threads.
- **Complexity:** Integrating YOLO's output format (bounding boxes, NMS) into the pipeline added unnecessary post-processing overhead versus MobileNetV2's direct softmax class probabilities.

**Decision:** MobileNetV2 INT8 TFLite was selected as the vision backbone. It runs at well under 100 ms per inference on the Pi 5 and integrates cleanly into the `run_inference()` helper.

---

## Final System Architecture

```
Object placed on platform
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    WEIGHT DAEMON (background thread)         │
│  HX711 polls every 150ms                                     │
│  Debounce: 4 consecutive readings > 3g threshold            │
│  Fires _weight_triggered Event on confirmed object          │
└─────────────────────────────────────────────────────────────┘
        │ (event fires)
        ▼
┌─────────────────────────────────────────────────────────────┐
│              PARALLEL DETECTION (ThreadPoolExecutor)         │
│                                                              │
│  PATH 1 — Vision (MobileNetV2 INT8 TFLite)                  │
│    640×480 cap → centre 320×320 crop → 224×224 → inference  │
│    Temporal buffer: 10 frames, 7/10 majority, ≥70% conf     │
│    Timeout: 1.0s → best-guess fallback                      │
│                                                              │
│  PATH 2 — Material Detection                                 │
│    SN04-N inductive → METAL (100% conf)                     │
│    else AS7343 spectrometer → PLASTIC / GLASS               │
│      SAM (70%) + Chi-squared (30%) ensemble                 │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    SENSOR FUSION (fuse_results)              │
│  1. Tetrapack → METAL compartment                           │
│  2. Metal (inductive) → METAL (weight check for oversize)   │
│  3. Beam sensor → contamination gate (liquid detected)      │
│  4. Disposable cup → GENERAL_WASTE                          │
│  5. Vision/Spec agree → use agreed material                 │
│  6. Vision/Spec disagree → weight tiebreaker               │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
MQTT publish → Servo arm rotates to compartment → Arm tilts → HOME
```

---

## Hardware Stack

| Component | Part | GPIO / Interface | Role |
|---|---|---|---|
| Weight sensor | HX711 + load cell | DOUT=15, SCK=14 | Object presence trigger + weight measurement |
| Inductive sensor | SN04-N | GPIO 16 (pull-up) | Metal detection (hardware, not ML) |
| Break beam sensor | IR emitter/receiver | GPIO 26 (pull-up) | Liquid contamination detection |
| Spectrometer | AS7343 (14-channel) | I2C | Plastic vs. glass spectral fingerprinting |
| Continuous servo | Pin 20 (PWM) | GPIO 20, 50Hz | Rotating arm — delivers to compartment |
| Position servo | Pin 21 (PWM) | GPIO 21, 50Hz | Arm tilt — drops item into chute |
| Tracking camera | UV-lit, 320×240 | v4l2 | ArUco marker tracking for arm angle |
| Vision camera | Standard, 640×480 | v4l2 | MobileNetV2 classification |

**Motor speed constants** (duty cycle values for 50Hz PWM):
- `SPEED_FWD = 6.0`, `SPEED_BWD = 9.0`, `SPEED_NEUTRAL = 7.5`, `SPEED_STOP = 0`
- These were tuned empirically — early versions used 6.5/8.5 (in `main_backup.py`), adjusted to 6.0/9.0 in later versions for faster traverse.

---

## Camera Management

**Problem encountered:** Early versions (`main_before_optimization.py`, `main_final.py`) hardcoded camera paths as `/dev/video0` and `/dev/video2`. This caused failures when the Pi enumerated cameras in a different order on reboot, or when swapping USB cameras between ports.

**Solution (`main_final3.py` onwards):** An interactive `select_cameras()` function was added that:
1. Runs `v4l2-ctl --list-devices` to enumerate all real capture devices.
2. Prints an indexed menu of detected device names and paths.
3. Prompts the operator to assign tracking and vision cameras by index, with duplicate-device guard.
4. Falls back to `/dev/video0` + `/dev/video2` if fewer than 2 devices are found.

After selection, per-role `v4l2-ctl` settings are applied via `subprocess.run()`:

| Camera | Key settings | Why |
|---|---|---|
| **Tracking** | `auto_exposure=1`, `exposure_time_absolute=25`, `gain=200`, `white_balance_temperature=6500` | UV-lit environment — manual exposure prevents bloom; fixed WB for consistent ArUco contrast |
| **Vision** | `auto_exposure=3`, `brightness=128`, `white_balance_automatic=1` | Natural light — auto settings produce better colour for MobileNet |

A per-channel **LUT-based colour correction** (`correct_frame()`) was implemented for the tracking camera to counteract the purple UV cast (scale factors: B×0.457, G×0.85, R×0.926). LUTs are computed once and cached globally so the correction runs in O(1) per frame via `cv2.LUT()`.

---

## Servo & ArUco Tracking Daemon

The continuous servo (pin 20) rotates the arm to the correct compartment. Its angle is tracked using a **4×4 ArUco marker** mounted on the arm, detected by the tracking camera.

### Detection Pipeline

```
320×240 frame → 2× upscale (640×480) → grayscale → CLAHE → ArucoDetector
```

- **2× upscale before detection:** The marker is small in the 320×240 frame. Upscaling before detection significantly improved detection rate under UV lighting.
- **CLAHE (Contrast Limited Adaptive Histogram Equalisation):** Applied with `clipLimit=2.0, tileGridSize=(8,8)` for local contrast boost — critical under UV which creates uneven illumination.
- **Tuned ArUco parameters:** `adaptiveThreshWinSizeMin=3`, `maxMarkerPerimeterRate=0.5`, `errorCorrectionRate=1.0`, `cornerRefinementMethod=CORNER_REFINE_SUBPIX` — these required manual tuning under the UV light conditions.
- Returned corner coordinates are divided back by `ARUCO_UPSCALE` to restore original frame space.

### Angle Calculation

Angle is derived purely from the marker's own orientation — no fixed rotation centre point is needed. The bottom→top midpoint vector of the marker gives the inward direction of the arm:

```python
dx = top_x - bottom_x
dy = top_y - bottom_y
angle = math.degrees(math.atan2(-dy, dx)) % 360
```

### Frame-Skip Optimisation

The ArUco detector runs only **every 2nd frame** (`DETECT_EVERY_N = 2`). On skip frames, the last known angle is held — the arm cannot jump meaningfully between adjacent frames. This halves the detector's CPU cost with negligible accuracy impact.

### Motor Control

- **Deceleration ramp:** `DECEL_START=60°`, `DECEL_NEAR=25°` — motor decelerates quadratically as it approaches the target, then crawls at `CRAWL_FACTOR=0.15` of full speed within 25°.
- **Overshoot correction:** Initial diff sign is captured when the target is set. If the diff sign flips (arm passed the target), the servo is briefly driven backwards at crawl speed to correct.
- **Direction-reversal guard:** Outbound direction (CW vs CCW) is determined at target-set time. General Waste is the only compartment reached CCW; all others go CW.
- **Marker loss watchdog:** If the marker is not detected for >300ms, the motor stops to allow reacquisition. If lost >500ms with a target set, a full-loss alert is printed.

### Arm Tilt

Pin 21 controls a position servo that tilts the arm to drop items into the chute:
- `set_angle_instant_21(angle)` converts angle to duty cycle: `duty = angle/18.0 + 2.0`
- At each compartment: tilt up to 100°, wait 1s, tilt back to 0°. A 1s `time.sleep()` allows the servo to reach position before reversing.

---

## Weight Sensor Pipeline

**Hardware:** HX711 24-bit ADC with a load cell. Raw ADC values are converted to grams:

```python
weight_g = (mean(raw_readings) - OFFSET) / RATIO
```

**RATIO calibration:** The ratio was recalibrated multiple times across versions as the physical load cell and mounting changed:
- `main_backup.py`: RATIO = 492.22
- `main_final.py` through `main_final4`: RATIO = 120.20
- `main.py` (final): RATIO = 112.36

**IQR outlier rejection:** `_iqr_clean()` removes readings outside `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR` before averaging. This handles mechanical vibration spikes from the servo motor.

**Debounce (added in final versions):** Early versions (`main_backup.py`) triggered immediately on a single reading above threshold. This caused false triggers from vibration when the servo returned home. The final version requires `WEIGHT_DEBOUNCE_COUNT = 4` consecutive readings above `WEIGHT_TRIGGER_THRESHOLD = 3.0g` before firing `_weight_triggered`. The weight daemon runs in a background thread with a `threading.Lock()` (`_hx711_lock`) to prevent concurrent HX711 access from the main pipeline reading weight for fusion.

**Weight thresholds:**
- `WEIGHT_TRIGGER_THRESHOLD = 3.0g` — minimum to register an object (avoids triggering on platform vibration).
- `METAL_CONTAMINATION_WEIGHT_LIMIT = 40.0g` — metal objects heavier than this are routed to General Waste as likely non-recyclable.
- `PLASTIC_GLASS_WEIGHT_THRESHOLD = 25.0g` — used as the tiebreaker when vision and spectrometer disagree: heavier → Glass, lighter → Plastic.

---

## Spectrometer Pipeline (AS7343)

The AS7343 is a 14-channel spectral sensor. It reads two banks of 7 channels each. `flicker` and `clear` channels are excluded, leaving 8 channels (`fz, fy, fxl, nir, vis_tl, vis_br, saturated, gain`) used for classification.

### Fingerprint Normalisation

Raw channel counts are normalised to a unit-sum vector:

```python
fingerprint = {ch: raw_count / total for ch in channels}
```

This makes the fingerprint invariant to absolute illumination intensity — critical because the LED brightness and item placement height both affect absolute counts.

### Calibration

Two calibration profiles are collected at startup (plastic, glass), each from `SPEC_CALIBRATION_SAMPLES = 12` readings. Outliers are rejected using the same Euclidean-distance sigma method as `_spec_reject_outliers()`. The mean and std of each channel are stored. Calibration is persisted to `calibration.json` and can be reloaded on subsequent runs without recalibrating.

Two separate calibration runs were done (evidenced by `calibration.json` and `calibration_copy.json`). The `vis_br` channel showed the largest shift between runs (plastic: 0.5324 → 0.3927, glass: 0.5630 → 0.4328), consistent with different item placement height or LED angle. This confirmed that placement consistency matters significantly and that the SAM metric (brightness-invariant) was the right choice.

### Distance Metric Evolution

**Original (`main_backup.py`, `main_before_optimization.py`):** Weighted Euclidean distance with per-channel `1/sigma` weighting:

```python
dist = sqrt(sum(((scan[k] - mean[k]) / (std[k] + 1e-6))^2 for k in channels))
```

Problems identified during testing:
- Treats all channels as independent — adjacent spectral bands are correlated, so this over-penalises correlated differences.
- Sensitive to absolute intensity shifts from placement variation, even after normalisation.
- Both profiles sometimes looked equally close under LED flicker conditions.

**Replacement (from `main_old_working_copy.py` / `main_final2.py` onwards):** SAM + Chi-squared ensemble.

**SAM (Spectral Angle Mapper) — 70% weight:**
```python
angle = arccos(dot(scan_vec, profile_vec) / (|scan_vec| × |profile_vec|))
```
Measures the **angle** between spectral vectors, not their magnitude difference. Completely invariant to uniform brightness scaling from LED drift or inconsistent placement height.

**Chi-squared distance — 30% weight:**
```python
chi = sum((scan[k] - mean[k])^2 / (scan[k] + mean[k] + 1e-9) for k in channels)
```
Naturally down-weights near-zero channels that carry low signal-to-noise ratio.

**Ensemble score** (lower = better match):
```python
score_p = 0.70 × (sam_p / (sam_p+sam_g)) + 0.30 × (chi_p / (chi_p+chi_g))
score_g = 0.70 × (sam_g / (sam_p+sam_g)) + 0.30 × (chi_g / (chi_p+chi_g))
```

**Confidence score** is computed from angular proximity to the winning profile and the decisiveness of the margin, relative to how far apart the two calibration profiles are. Low profile separation → low confidence → downstream fusion uses weight tiebreaker.

### Scan Sample Reduction

`SPEC_SCAN_SAMPLES` was reduced from **6** (in `main_backup.py`, `main_before_optimization.py`) to **2** (from `main_final2.py` onwards) to reduce classification latency. SAM is robust with fewer samples because it is angle-based rather than mean-dependent, and the outlier rejection step still runs on whatever samples are collected.

---

## Sensor Fusion Logic

`fuse_results()` takes all sensor inputs and produces a final `Material` decision plus MQTT material/shape strings. The function's signature evolved significantly:

**Early (`main_backup.py`):** `fuse_results(vision_label, material, weight_g) → Material` — no spec confidence, simpler contamination gate only.

**Final (`main.py`):** `fuse_results(vision_label, spec_material, spec_confidence, weight_g) → tuple[Material, str, str]` — full fusion with MQTT shape output.

### Decision Tree (Final)

```
1. Is it a carton_tetrapack?
   → Yes, weight OK → METAL / "tetra" / "carton"
   → Yes, too heavy → GENERAL_WASTE / "general" / "others"

2. Did the inductive sensor trigger? (material == METAL)
   → Weight ≤ 40g → METAL / "metal" / shape
   → Weight > 40g → GENERAL_WASTE / "general" / shape

3. Is the break beam broken? (liquid contamination)
   → GENERAL_WASTE regardless of material

4. Is it a disposable_cup?
   → GENERAL_WASTE / "general" / "others"

5. Do vision and spectrometer agree on plastic/glass?
   → Use agreed material

6. Vision vs spectrometer disagree?
   → Weight tiebreaker:
      ≥ 25g → GLASS
      < 25g → PLASTIC
```

The weight tiebreaker was added because glass bottles are consistently heavier than plastic bottles of the same shape, providing a robust physical signal when optical sensors disagree (e.g., a frosted plastic bottle reads spectrally similar to glass).

---

## MQTT Publishing

Each classification cycle publishes one JSON message to `pi/raw_transaction` on broker `10.254.93.107:1883`:

```json
{"material": "plastic", "type": "bottle", "weight": "42.3g"}
```

The MQTT client uses `CallbackAPIVersion.VERSION2` (Paho MQTT v2 API). An `on_disconnect` handler attempts reconnection in a retry loop with 5-second backoff — robust against temporary broker unavailability during the demo.

---

## Startup Calibration Sequence

On every boot, `main_pipeline()` runs a guided calibration sequence:

1. **Scale tare** — ensures platform is empty, collects 20 HX711 readings with IQR cleaning for the zero offset.
2. **Break beam alignment check** — if the beam is already broken at startup, the operator is prompted to realign before continuing.
3. **Servo tracking daemon start** — background thread begins ArUco tracking.
4. **Weight daemon start** — background polling thread starts.
5. **Compartment angle calibration** (can be skipped via comment-out) — operator fine-tunes servo angles for each compartment using keyboard input (D/A = ±1°, C/Z = ±5°) with live camera feedback.
6. **Spectrometer calibration** — operator places plastic, then glass on sensor. Saved to `calibration.json`; on subsequent runs, operator is offered to reuse saved profiles.

---

## Main Pipeline Loop

After calibration, the loop is:

```
wait for _weight_triggered Event (set by weight daemon)
    │
    ├── if arm still moving: ignore trigger, re-arm
    │
    ▼
pause servo tracking daemon (_tracking_pause.set())
    │
    ▼
ThreadPoolExecutor(max_workers=2):
    ├── PATH 1: MobileNetV2 vision inference
    └── PATH 2: inductive + spectrometer material detection
    │
    ▼
read final weight
    │
    ▼
fuse_results() → final_material, mqtt_mat, mqtt_shape
    │
    ▼
mqtt_publish_result()
    │
    ▼
_weight_triggered.clear()   ← re-arm for next object
_tracking_pause.clear()     ← resume servo daemon
_set_target(compartment_angle)
    │
    ▼
wait for target_angle_20 == None (arm has homed)
    │
    ▼
loop
```

The servo tracking daemon is **paused during detection** (`_tracking_pause.set()`) to free CPU cores for HX711 polling and TFLite inference. The daemon is a tight loop reading the camera — pausing it gives back ~1 full core to the detection threads.

---

## Iteration History — What Changed and Why

| File | Key Changes from Previous |
|---|---|
| `main_backup.py` | Earliest complete version. Single camera (hardcoded). No HX711 lock. No weight debounce. Weighted Euclidean spectrometer distance. `fuse_results` returns only `Material` (no MQTT shape). `path_2_material_detection` returns only `Material` (no confidence). `SPEC_SCAN_SAMPLES=6`. `VISION_TIMEOUT_S=15.0`. Compartment angles: METAL=37, GLASS=100, PLASTIC=222, GW=293. RATIO=492.22. Motor speeds: 6.5/8.5. |
| `main_before_optimization.py` | Added psutil profiling instrumentation (`_rss_mb()`). Camera still hardcoded to `/dev/video0` (tracking) only — tracking camera v4l2 settings applied globally. SAM + chi-squared spectrometer now present but still `SPEC_SCAN_SAMPLES=6`. Compartment angles changed: METAL=220, GLASS=290, PLASTIC=65, GW=120. RATIO remains 492.22. |
| `main_final.py` | HX711 threading lock introduced. `RATIO` recalibrated to 120.20 (physical load cell remount). Vision camera given separate v4l2 settings (`/dev/video2`). Tracking camera settings separately applied to `/dev/video0`. |
| `main_final2.py` | `SPEC_SCAN_SAMPLES` reduced **6→2** (latency optimisation after profiling). `VISION_TIMEOUT_S` reduced from 15.0 to **10.0**. Compartment angles settled: METAL=220, GLASS=65, PLASTIC=320, GW=120. `path_2_material_detection` now returns `tuple[Material, float]` (adds confidence). `fuse_results` updated signature with `spec_confidence` parameter and returns `tuple[Material, str, str]` with MQTT shape. Weight debounce not yet present. |
| `main_final3.py` | **`select_cameras()` introduced** — interactive camera assignment at boot, replacing hardcoded paths. v4l2 settings now applied to dynamically-assigned devices. |
| `main_final4_copy.py` / `main_final4_profiling.py` | Profiling instrumentation run. `cProfile` used to identify hotspots. Result: spectrometer wait (`time.sleep(0.30)` per bank read) and camera I/O were the dominant latencies, not inference. |
| `main_old_working_copy.py` / `main_old_working_copy_2.py` | Working copies with `VISION_LABEL_MAP` for MQTT shape routing. Weight debounce logic present. `PLASTIC_GLASS_WEIGHT_THRESHOLD` for tiebreaker. |
| **`main.py` (FINAL)** | `WEIGHT_DEBOUNCE_COUNT=4` debounce implemented to kill false triggers from servo vibration. `VISION_TIMEOUT_S` reduced to **1.0s** (inference is fast enough that a 1s timeout is sufficient after buffer fills). `RATIO` final recalibration to 112.36. Full `fuse_results()` with all 6-step decision tree. Complete `select_cameras()`, MQTT reconnection, and all cleanup in `finally` block. |

---

## Calibration Data

Two spectrometer calibration sessions were performed (`calibration_copy.json` = session 1, `calibration.json` = session 2 / final). The `vis_br` channel showed the largest shift between sessions:

| Channel | Plastic Session 1 | Plastic Session 2 | Glass Session 1 | Glass Session 2 |
|---|---|---|---|---|
| fz | 0.1456 | 0.1627 | 0.1366 | 0.1497 |
| fy | 0.0759 | 0.1243 | 0.0655 | 0.1213 |
| fxl | 0.0531 | 0.0874 | 0.0463 | 0.0854 |
| nir | 0.0285 | 0.0664 | 0.0213 | 0.0492 |
| vis_tl | 0.1036 | 0.1216 | 0.1029 | 0.1120 |
| vis_br | **0.5324** | **0.3927** | **0.5630** | **0.4328** |

The `vis_br` channel dominates Session 1 far more than Session 2. This is consistent with the material being placed slightly further from the sensor in Session 2, reducing near-IR saturation. The shift confirmed the need for the SAM metric (illumination-invariant) and motivated recalibration before each demo session.
