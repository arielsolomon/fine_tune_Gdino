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


patch_coco_json('/home/user1/ariel/datasets/mini_chinese_dataset/labels/train.json')
patch_coco_json('/home/user1/ariel/datasets/mini_chinese_dataset/labels/test.json')