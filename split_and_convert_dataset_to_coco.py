import os
import shutil
import random
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- CONFIGURATION ---
SRC_ROOT = '/home/user1/ariel/datasets/chinase_env_areal_IR/'
DST_ROOT = '/home/user1/ariel/datasets/Chinase_env_arial_IR_fine_tune/'

SPLIT_RATIO = {'train': 0.8, 'test': 0.1, 'val': 0.1}

def parse_xml_for_coco(xml_path):
    """
    Parses Pascal VOC XML. Since your objects are named '1', '2', etc.,
    this captures ALL objects and maps them to Category 0.
    """
    bboxes = []
    width, height = 640, 480 

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        if size is not None:
            width = int(float(size.find('width').text))
            height = int(float(size.find('height').text))
            
        for obj in root.findall('object'):
            # We ignore the specific name (1, 2, 3...) and treat all as Category 0
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            w = float(max(0.0, xmax - xmin))
            h = float(max(0.0, ymax - ymin))
            
            bboxes.append({
                "bbox": [xmin, ymin, w, h],
                "area": float(w * h),
                "category_id": 0 # All objects mapped to 'Vehicle'
            })
    except Exception as e:
        pass
        
    return bboxes, width, height

def main():
    # 1. Scan folders 1 to 87
    all_data_pairs = []
    print("Scanning source directories for folders 1 to 87...")

    for folder_id in range(1, 88):
        folder_name = str(folder_id)
        img_dir = os.path.join(SRC_ROOT, 'Images', folder_name)
        ann_dir = os.path.join(SRC_ROOT, 'Annotations', folder_name)
        
        if not os.path.exists(img_dir): 
            continue

        for i in range(1, 251):
            file_base = f"{i:03d}"
            img_file = f"{file_base}.bmp"
            xml_file = f"{file_base}.xml"
            
            img_path = os.path.join(img_dir, img_file)
            xml_path = os.path.join(ann_dir, xml_file)
            
            if os.path.exists(img_path) and os.path.exists(xml_path):
                all_data_pairs.append({
                    'img_src': img_path,
                    'xml_src': xml_path,
                    'new_name': f"{folder_name}_{i}"
                })

    if not all_data_pairs:
        print("Error: No valid image/XML pairs found.")
        return

    # 2. Shuffle and Split
    random.seed(42)
    random.shuffle(all_data_pairs)
    total = len(all_data_pairs)
    tr_idx = int(total * SPLIT_RATIO['train'])
    te_idx = tr_idx + int(total * SPLIT_RATIO['test'])
    splits = {'train': all_data_pairs[:tr_idx], 'test': all_data_pairs[tr_idx:te_idx], 'val': all_data_pairs[te_idx:]}

    # 3. Process Splits
    for split_name, items in splits.items():
        print(f"\nProcessing {split_name} split...")
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "Vehicle"}]
        }
        
        img_dest_dir = os.path.join(DST_ROOT, 'images', split_name)
        label_dest_dir = os.path.join(DST_ROOT, 'labels')
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(label_dest_dir, exist_ok=True)

        ann_id_counter = 1
        for img_id_counter, item in enumerate(tqdm(items), start=1):
            new_img_filename = f"{item['new_name']}.bmp"
            shutil.copy(item['img_src'], os.path.join(img_dest_dir, new_img_filename))
            
            boxes, w, h = parse_xml_for_coco(item['xml_src'])
            
            coco_output['images'].append({
                "id": img_id_counter,
                "file_name": new_img_filename,
                "width": w,
                "height": h
            })
            
            for b in boxes:
                # This creates the exact format you requested:
                # {"id": ..., "image_id": ..., "category_id": 0, "bbox": [...], "area": ..., "iscrowd": 0}
                coco_output['annotations'].append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": 0,
                    "bbox": b['bbox'],
                    "area": b['area'],
                    "iscrowd": 0
                })
                ann_id_counter += 1

        # Save the JSON file
        json_path = os.path.join(label_dest_dir, f"{split_name}.json")
        with open(json_path, 'w') as f:
            json.dump(coco_output, f)
        
        print(f"Split {split_name} complete. Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")

    print(f"\nSuccessfully created COCO dataset at: {DST_ROOT}")

if __name__ == "__main__":
    main()