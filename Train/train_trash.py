from ultralytics import YOLO
from roboflow import Roboflow
import torch
import os 


if __name__ == "__main__": 
    # 1. CHECK FOR YOUR RTX 4070
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ WARNING: GPU not found. Training will be slow!")

    # ======================================================
    # 2. PASTE YOUR ROBOFLOW CODE HERE (The one from your screenshot)
    # ======================================================

    if not os.path.exists("taco_data_v2"):
        print ("📥 DOWNLOADING DATASET FROM ROBOFLOW..." )
        rf = Roboflow(api_key="M9ALLX8G3YwIkloygU6P")
        project = rf.workspace("roboflow-universe-projects").project("taco-object-detection-kcxyn")
        version = project.version(2) #! originally was 5
        dataset = version.download("yolov8", location="taco_data_v2")
        data_path = f"{dataset.location}/data.yaml"

        
    else: 
        print("✅ DATASET ALREADY DOWNLOADED.")
        data_path = os.path.abspath("taco_data_v2/data.yaml") 
                    
    # ======================================================


    # 3. START TRAINING
    # We use 'yolov8n.pt' (Nano) because it is fast enough for Raspberry Pi
    print("🚀 STARTING TRAINING on RTX 4070...")
    model = YOLO("yolov8n.pt") 

    # Train for 50 epochs (Should take ~10-15 mins on your 4070)
    model.train(
        data=data_path, 
        epochs=50, 
        imgsz=640, 
        device=0  # This forces it to use your GPU
    )

    # 4. EXPORT FOR RASPBERRY PI
    print("💾 EXPORTING TO NCNN...")
    model.export(format="ncnn")

    print("🎉 DONE! Look for the 'best_ncnn_model' folder.")