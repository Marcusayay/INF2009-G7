from ultralytics import YOLO
from collections import defaultdict 

models = [
    r'C:\code\Edge Computing\INF2009-G7\runs\detect\train13\weights\best_ncnn_model_fp32',
    r'C:\code\Edge Computing\INF2009-G7\runs\detect\train13\weights\best_ncnn_model_fp16',
    r'C:\code\Edge Computing\INF2009-G7\runs\detect\train13\weights\best_saved_model\best_integer_quant.tflite',
    r'C:\code\Edge Computing\INF2009-G7\runs\detect\train13\weights\best_saved_model\best_full_integer_quant.tflite'
]

dataset_loc = r"C:\code\Edge Computing\INF2009-G7\Train\taco_data_v7_without_coco\data.yaml"
specs = defaultdict(list)

for w in models:
    m = YOLO(w)
    r = m.val(data=dataset_loc, split="val", imgsz=640, conf=0.25, iou=0.7)
    print(w, "precision:", r.box.p, "recall:", r.box.r, "mAP50:", r.box.map50, "mAP50-95:", r.box.map)
    specs[w].append(r.box.p)
    specs[w].append(r.box.r) 
    specs[w].append(r.box.map50) 
    specs[w].append(r.box.map) 

for model, metrics in specs.items():
    print (metrics)
    print(f"Model: {model}")
    print ("-" * 30)
    
for model, metrics in specs.items(): 
    print(f"Precision: {metrics[0].mean():.4f}")
    print(f"Recall: {metrics[1].mean():.4f}")
    print(f"mAP@0.5: {metrics[2]:.4f}")
    print(f"mAP@0.5-0.95: {metrics[3]:.4f}")
    print("-" * 30)