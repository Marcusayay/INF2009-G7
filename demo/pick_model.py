import os 

def return_latest_version_path(model: str,) : 
    if model == "mobilenet":
        model_dir = "../Train/mobnet_models"
        existing_versions = [d for d in os.listdir(model_dir) if d.startswith("v")]
        if not existing_versions: 
            return None  # No existing models 
        latest_version = max(existing_versions, key=lambda x: int(x[1:]))  # Extract version number
        latest_model_path = os.path.join(model_dir, latest_version)
        print(f"✅ Latest MobileNet model found: {latest_model_path}") 
        return latest_model_path
        
    elif model == "yolo":
        model_dir = "yolo_models"
        existing_versions = [d for d in os.listdir(model_dir) if d.startswith("v")]
        if not existing_versions: 
            return None  # No existing models 
        latest_version = max(existing_versions, key=lambda x: int(x[1:]))  # Extract version number
        latest_model_path = os.path.join(model_dir, latest_version)
        print(f"✅ Latest YOLO model found: {latest_model_path}") 
        return latest_model_path 

if __name__ == "__main__":
    latest_mobnet = return_latest_version_path("yolo") 