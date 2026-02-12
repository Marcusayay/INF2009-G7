from ultralytics import YOLO

checkpoint_path = r"C:\code\Edge Computing\INF2009-G7\runs\detect\train7\weights\last.pt"
data_path = r"C:\code\Edge Computing\INF2009-G7\Train\taco_data_v4_tetrapak\data.yaml"
# 1. Load the LAST checkpoint (not best.pt)
model = YOLO(checkpoint_path) 

# 2. Resume training
# Since you have 16GB VRAM now, try batch=32 or 64 (previously likely 8 or 16)
model.train(
    resume=True, 
    batch=64,
    workers=4, 
    cache=True,
    device=0)