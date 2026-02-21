from ultralytics import YOLO
from roboflow import Roboflow
import torch
import os 

#!
# yolo export model=runs/detect/train11/weights/best.pt format=ncnn half=True
# yolo export model=runs/detect/train11/weights/best.pt format=tflite int8=True data=Train/taco_data_v5_paper/data.yaml
#!
file_to_save_in = "taco_data_v5_paper"  
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
        # project = rf.workspace("zfcrow").project("beverages_waste_detection")
        # version = project.version(1)
        project = rf.workspace("zfcrow").project("recyclable_object_detection")
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
            workers=4,   
            cache=False,
            device=0,
            optimizer='auto',
            patience=20,
            close_mosaic=10
        )
    # 4. EXPORT FOR RASPBERRY PI
    # print("💾 EXPORTING TO NCNN...")
    # model.export(format="ncnn")


    # Export 1: NCNN FP16 (Your "Max Speed" winner)
    # half=True forces it to use FP16 precision
    print("Exporting to NCNN (FP16)...")
    model.export(format="ncnn", half=True)

    # Export 2: TFLite INT8 (Your "Best Efficiency" winner)
    # int8=True forces integer quantization
    print("Exporting to TFLite (INT8)...")
    model.export(format="tflite", int8=True)

    print("🎉 DONE! Look for the 'best_ncnn_model' folder.")