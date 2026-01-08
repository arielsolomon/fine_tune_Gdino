from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import os
import nltk
import json
# Ensure required NLP resources are available for Grounding DINO's NER
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# 1. Setup paths
config_file = 'configs/custom/grounding_dino_finetune_skip.py'
checkpoint_file = 'weights/fine_tune_36.pth'

file_list = os.listdir('/work/datasets/Chinase_env_arial_IR_fine_tune/images/test/')

img_path = '/work/datasets/Chinase_env_arial_IR_fine_tune/images/test/1_42.bmp' # Pick a test image
out_file = '1_42_out.jpg'

# 2. Initialize the model
model = init_detector(config_file, checkpoint_file, device='cuda:0')
prompt = "Vehicle" 
# 3. Run inference
result = inference_detector(model, img_path, text_prompt=prompt)
#result = inference_detector(model, img_path)
score_thr = 0.3

# 2. Extract and filter data
# result.pred_instances contains: bboxes, labels, and scores
inst = result.pred_instances
# Find indices where score is above threshold
keep_indices = inst.scores >= score_thr

filtered_bboxes = inst.bboxes[keep_indices].tolist()
filtered_labels = inst.labels[keep_indices].tolist()
filtered_scores = inst.scores[keep_indices].tolist()

# 3. Create the clean dictionary
predictions = {
    'bboxes': filtered_bboxes,
    'labels': filtered_labels,
    'scores': filtered_scores
}

# 4. Save to file
with open('filtered_predictions.json', 'w') as f:
    json.dump(predictions, f, indent=4)

print("Prediction labels saved to predictions.json")
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