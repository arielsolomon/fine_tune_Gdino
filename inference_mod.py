from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import os

# 1. Setup paths
config_file = 'your_config_filename.py' # The name of the script you just ran
checkpoint_file = 'work_dirs/grounding_dino_finetune3/epoch_12.pth'
img_path = '/work/datasets/mini_chinese_dataset/images/test/YOUR_IMAGE_NAME.jpg' # Pick a test image
out_file = 'result_visualization.jpg'

# 2. Initialize the model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 3. Run inference
result = inference_detector(model, img_path)

# 4. Visualize and save
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=False,
    wait_time=0,
    out_file=out_file,
    pred_score_thr=0.3 # Only show detections with >30% confidence
)

print(f"Done! Check {out_file} to see the detections.")