from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import os, glob
# import nltk
import json
# Ensure required NLP resources are available for Grounding DINO's NER
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

# 1. Setup paths
config_file = '/work/GroundingDINO/fine_tune_Gdino/configs/custom/grounding_dino_finetune_skip.py'
checkpoint_file = '/work/GroundingDINO/fine_tune_Gdino/weights/g_dino_pretrained.pth'

out_folder = '/work/GroundingDINO/fine_tune_Gdino/results_g_dino_pretrained/'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


# 2. Initialize the model
model = init_detector(config_file, checkpoint_file, device='cuda:0')
prompt = "Car" 
# 3. Run inference on all images
file_list = glob.glob(os.path.join('/work/datasets/Chinase_env_arial_IR_fine_tune/images/test/', '*.bmp'))
for file in file_list:
    out_file = os.path.join(out_folder,os.path.basename(file).split('.')[0] + '_out.jpg')
    result = inference_detector(model, file, text_prompt=prompt)
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
    with open(out_file.split('.')[0]+'.json', 'w') as f:
        json.dump(predictions, f, indent=4)

    print("Prediction labels saved to predictions.json")
# 4. Visualize and save
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    img = mmcv.imread(file)
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

    print(f"Done! Check {os.path.basename(file).split('.')[0]} to see the detections.")