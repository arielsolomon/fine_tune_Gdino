import os
import json
import random
import shutil
from pathlib import Path

# --- Configuration ---
SOURCE_ROOT = "/work/datasets/Chinase_env_arial_IR_fine_tune"
DEST_ROOT = "/work/datasets/Chinase_env_arial_IR_fine_tune_fewshots"
NUM_SHOTS = 20  # Number of images to sample from each split
SPLITS = ["train", "test", "val"]

def create_fewshot_split(split_name):
    print(f"Processing {split_name} split...")
    
    # 1. Define Paths
    img_src_dir = os.path.join(SOURCE_ROOT, f"images/{split_name}")
    json_src_file = os.path.join(SOURCE_ROOT, f"labels/{split_name}.json")
    
    img_dest_dir = os.path.join(DEST_ROOT, f"images/{split_name}")
    json_dest_file = os.path.join(DEST_ROOT, f"labels/{split_name}.json")
    
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_dest_file), exist_ok=True)

    # 2. Load Original JSON
    with open(json_src_file, 'r') as f:
        coco_data = json.load(f)

    # 3. Sample Images
    all_images = coco_data['images']
    sampled_images = random.sample(all_images, min(NUM_SHOTS, len(all_images)))
    sampled_image_ids = {img['id'] for img in sampled_images}

    # 4. Filter Annotations
    sampled_annotations = [
        ann for ann in coco_data['annotations'] 
        if ann['image_id'] in sampled_image_ids
    ]

    # 5. Copy Image Files
    for img in sampled_images:
        src_path = os.path.join(img_src_dir, img['file_name'])
        dest_path = os.path.join(img_dest_dir, img['file_name'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Warning: Image {img['file_name']} not found in {img_src_dir}")

    # 6. Save New JSON
    new_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": coco_data.get("categories", [])
    }

    with open(json_dest_file, 'w') as f:
        json.dump(new_coco, f, indent=4)
    
    print(f"Successfully created few-shot {split_name} with {len(sampled_images)} images.")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    for split in SPLITS:
        create_fewshot_split(split)
