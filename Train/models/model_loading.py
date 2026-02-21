from ultralytics import YOLO
import os
import shutil 

# 1. Download the different sizes and generations
YOLO("base/yolo26s.pt")
YOLO("base/yolo26m.pt")
YOLO("base/yolo26l.pt") 
YOLO("base/yolo26n.pt")
YOLO("base/yolo11n.pt")
YOLO("base/yolov8n.pt")

YOLO("base/yolo11s.pt")
YOLO("base/yolo11m.pt") 
YOLO("base/yolo11l.pt") 

YOLO("base/yolov8s.pt")
YOLO("base/yolov8m.pt") 
YOLO("base/yolov8l.pt") 
# Point this to your dataset configuration so the INT8 exporter 
# can grab sample images for the calibration phase.
dataset_config = "coco8.yaml"


def force_rename_dir(src, dst):
    """Renames a directory, overwriting the destination if it already exists."""
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst) # Completely delete the old target folder
        os.rename(src, dst)    # Safely move the new folder into place


def extract_tflite_and_cleanup(saved_model_dir):
    """Moves .tflite files out of the saved_model folder and deletes the folder."""
    if os.path.exists(saved_model_dir):
        # Look at all the files inside the directory
        for file in os.listdir(saved_model_dir):
            if file.endswith(".tflite"):
                source = os.path.join(saved_model_dir, file)
                destination = os.path.join(".", file)
                # Move the .tflite file into the main directory
                shutil.move(source, destination)
        
        # Safely delete the remaining directory and TensorFlow junk files
        shutil.rmtree(saved_model_dir)

for model_file in os.listdir("base"):
    if model_file.endswith(".pt"):
        base_name = model_file.replace(".pt", "")
        
        print(f"\n=========================================")
        print(f"⏳ Processing {model_file}")
        print(f"=========================================\n")
        
        model = YOLO(model_file)
        
        # ---------------------------------------------------
        # 1. EXPORT FULL-SIZE (FP32)
        # ---------------------------------------------------
        print("--> Exporting Full-Size (FP32) versions...")
        model.export(format="onnx", simplify=True)
        model.export(format="tflite")
        model.export(format="ncnn")
        
        # --- RENAME FOLDERS ---
        ncnn_dir = f"{base_name}_ncnn_model"
        if os.path.exists(ncnn_dir):
            force_rename_dir(ncnn_dir, f"{base_name}_fp32_ncnn_model")
            
        # --- CLEAN UP TFLITE ---
        saved_model_dir = f"{base_name}_saved_model"
        extract_tflite_and_cleanup(saved_model_dir)
        print("--> Extracted FP32 TFLite files and cleaned up junk files!")

        # ---------------------------------------------------
        # 2. EXPORT QUANTIZED / COMPRESSED
        # ---------------------------------------------------
        print("\n--> Exporting Quantized / Compressed versions...")
        model.export(format="tflite", int8=True, data=dataset_config)
        # NCNN only supports FP16 (half) directly through the Ultralytics API
        model.export(format="ncnn", half=True)
        
        # --- RENAME FOLDERS ---
        if os.path.exists(ncnn_dir):
            force_rename_dir(ncnn_dir, f"{base_name}_fp16_ncnn_model")
            
        # --- CLEAN UP TFLITE ---
        extract_tflite_and_cleanup(saved_model_dir)
        print("--> Extracted INT8 TFLite files and cleaned up junk files!")

print("\n✅ All exports completed! Files extracted and separated cleanly.")