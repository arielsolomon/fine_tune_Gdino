import json

def patch_coco_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Add the missing keys required by pycocotools
    if 'info' not in data:
        data['info'] = {"description": "Vehicle Dataset", "version": "1.0", "year": 2024}
    if 'licenses' not in data:
        data['licenses'] = []
    
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Patched {path}")


patch_coco_json('/home/user1/ariel/datasets/Chinase_env_arial_IR_fine_tune/labels/train.json')
patch_coco_json('/home/user1/ariel/datasets/Chinase_env_arial_IR_fine_tune/labels/test.json')
patch_coco_json('/home/user1/ariel/datasets/Chinase_env_arial_IR_fine_tune/labels/val.json')