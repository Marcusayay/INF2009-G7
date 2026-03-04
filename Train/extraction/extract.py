import json
import os
import cv2

# --- Config ---
BASE_DIR = "recycle_object_detection_3.coco/train"
JSON_PATH = os.path.join(BASE_DIR, '_annotations.coco.json')
IMAGE_DIR = BASE_DIR  # Change this if your images are in a subfolder like 'train' 
OUTPUT_DIR = 'extracted_disposable_cup'
TARGET_CLASS = 'disposable_cup'  # Change this if the dataset named it 'glass-cup' or 'glass'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load the COCO Data ---
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# --- 2. Find the Target Class ID ---
category_id = None
for cat in data['categories']:
    if cat['name'].lower() == TARGET_CLASS.lower():
        category_id = cat['id']
        break

if category_id is None:
    print(f"❌ Class '{TARGET_CLASS}' not found in categories!")
    print("Available classes in this dataset:", [c['name'] for c in data['categories']])
    exit()

# --- 3. Map Image IDs to Filenames ---
image_map = {img['id']: img['file_name'] for img in data['images']}

# --- 4. Crop and Save ---
count = 0
for ann in data['annotations']:
    if ann['category_id'] == category_id:
        img_filename = image_map[ann['image_id']]
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # COCO bbox format is [x_min, y_min, width, height]
            # 🚨 THE FIX: Convert to float first, then to integer
            x, y, w, h = [int(float(v)) for v in ann['bbox']]

            # Prevent out-of-bounds cropping errors
            img_h, img_w = img.shape[:2]
            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(img_w, x + w), min(img_h, y + h)
            
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            # Skip invalid/tiny crops
            if cropped_img.size == 0:
                continue
                
            save_path = os.path.join(OUTPUT_DIR, f"{TARGET_CLASS}_{count}.jpg")
            cv2.imwrite(save_path, cropped_img)
            count += 1

print(f"✅ Successfully extracted {count} tightly cropped images of '{TARGET_CLASS}'!")