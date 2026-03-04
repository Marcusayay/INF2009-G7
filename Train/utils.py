
import os


def get_latest_number(base_path="mobnet_models"):
    existing_versions = [d for d in os.listdir(base_path) if d.startswith("v")]
    if not existing_versions:
        return 1
    latest_version = max([int(v[1:]) for v in existing_versions])
    return latest_version + 1 

if __name__ == "__main__":
    print(f"Latest version number: {get_latest_number()}") 