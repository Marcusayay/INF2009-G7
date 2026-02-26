import os
import cv2
import yaml
from pathlib import Path

# ==========================================
# 1. SETUP PATHS
# ==========================================
# Point this to the folder you already downloaded
dataset_path = "taco_data_v7_without_coco" 
output_dir = "MobileNet_Classification_Data"

# Load class names from the YOLO data.yaml file
yaml_path = os.path.join(dataset_path, "data.yaml")
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)
    class_names = data['names']

print(f"Found classes: {class_names}")

# ==========================================
# 2. CREATE FOLDERS
# ==========================================
splits = ['train', 'valid', 'test']
for split in splits:
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# ==========================================
# 3. CROP AND SAVE
# ==========================================
def process_split(split_name):
    image_dir = os.path.join(dataset_path, split_name, "images")
    label_dir = os.path.join(dataset_path, split_name, "labels")
    
    if not os.path.exists(image_dir):
        return

    print(f"Processing {split_name} split...")
    
    for img_name in os.listdir(image_dir):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        # Read the image
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Find the matching label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        if not os.path.exists(label_path):
            continue
            
        # Parse the YOLO coordinates and crop
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            
            # De-normalize coordinates to exact pixels
            x_min = int((x_center - w / 2) * img_width)
            y_min = int((y_center - h / 2) * img_height)
            x_max = int((x_center + w / 2) * img_width)
            y_max = int((y_center + h / 2) * img_height)
            
            # Ensure coordinates are within image boundaries
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_width, x_max), min(img_height, y_max)
            
            # Crop the image
            cropped_img = img[y_min:y_max, x_min:x_max]
            
            # Skip invalid crops (e.g., width or height is 0)
            if cropped_img.size == 0:
                continue
                
            # Save the crop to the correct class folder
            class_name = class_names[class_id]
            save_name = f"{os.path.splitext(img_name)[0]}_crop_{i}.jpg"
            save_path = os.path.join(output_dir, split_name, class_name, save_name)
            
            cv2.imwrite(save_path, cropped_img)

# Run the extraction
for split in splits:
    process_split(split)

print(f"🎉 Done! Your classification dataset is ready in: {output_dir}")