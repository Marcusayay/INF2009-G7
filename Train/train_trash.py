from ultralytics import YOLO
from roboflow import Roboflow
import torch
import os 


file_to_save_in = "taco_data_v4_tetrapak"  
base_model = "yolov8n.pt"


if __name__ == "__main__": 
    # 1. CHECK FOR YOUR RTX 4070
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ WARNING: GPU not found. Training will be slow!")

    # ======================================================
    # 2. PASTE YOUR ROBOFLOW CODE HERE (The one from your screenshot)
    # ======================================================

    if not os.path.exists(file_to_save_in):
        print ("📥 DOWNLOADING DATASET FROM ROBOFLOW..." )
        rf = Roboflow(api_key="M9ALLX8G3YwIkloygU6P")
        project = rf.workspace("zfcrow").project("beverages_waste_detection")
        version = project.version(1)
        dataset = version.download("yolov8", location=file_to_save_in)
        data_path = f"{dataset.location}/data.yaml"

                
    else: 
        print("✅ DATASET ALREADY DOWNLOADED.")
        data_path = os.path.abspath(f"{file_to_save_in}/data.yaml") 
                    
    # ======================================================


    # 3. START TRAINING
    # We use 'yolov8n.pt' (Nano) because it is fast enough for Raspberry Pi
    print("🚀 STARTING TRAINING on RTX 4070...")
    #model = YOLO("yolov8n.pt")
    # use the base model you want 
    ##C:\code\INF2009-G7\runs\detect\train5\weights
    model = YOLO(base_model) 


    # Train for 50 epochs (Should take ~10-15 mins on your 4070)
    # model.train(
    #     data=data_path, 
    #     epochs=100, 
    #     imgsz=640, 
    #     device=0  # This forces it to use your GPU
    # )
    model.train(
            data=data_path, 
            epochs=100, 
            imgsz=640, 
            batch=64,     # Limits VRAM usage to fit your 8GB card
            workers=4,    # Uses your 40GB RAM to pre-load images
            cache=True,
            device=0,
            optimizer='auto'
        )
    # 4. EXPORT FOR RASPBERRY PI
    print("💾 EXPORTING TO NCNN...")
    model.export(format="ncnn")

    print("🎉 DONE! Look for the 'best_ncnn_model' folder.")