from ultralytics import YOLO
import os

# 1. Download the different sizes and generations
YOLO("yolo26s.pt")
YOLO("yolo26m.pt")
YOLO("yolo11n.pt")
YOLO("yolov8n.pt")

# 2. Export YOLO26 Nano to other edge-friendly formats

for model_file in os.listdir(".models"):
    if model_file.endswith(".pt"):
        print(f"⏳ Exporting {model_file} to ONNX, TFLite, and OpenVINO, ncnn...")
        model = YOLO(os.path.join(".models", model_file))
        model.export(format="onnx", simplify=True)
        model.export(format="tflite")
        model.export(format="openvino")
        model.export(format="ncnn")