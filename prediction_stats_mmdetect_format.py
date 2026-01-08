import json
import os
import torch
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix, classification_report

def load_mmdet_json(file_path):
    """Loads MMDet format JSON: {'bboxes': [], 'labels': [], 'scores': []}"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def evaluate_folders(gt_folder, pred_folder, iou_threshold=0.5, score_thr=0.3):
    y_true = []
    y_pred = []
    
    # 1. Get all JSON files using os.listdir
    if not os.path.exists(gt_folder):
        print(f"Error: GT folder not found at {gt_folder}")
        return

    files = os.listdir(gt_folder)
    gt_json_files = [f for f in files if f.endswith('.json')]
    
    for filename in gt_json_files:
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        
        gt_data = load_mmdet_json(gt_path)
        pred_data = load_mmdet_json(pred_path)
        
        # Prepare GT Boxes
        gt_boxes = torch.tensor(gt_data['bboxes']) if gt_data and gt_data['bboxes'] else torch.empty((0, 4))
        gt_labels = gt_data['labels'] if gt_data else []
        
        # Prepare and Filter Prediction Boxes
        if pred_data and pred_data['bboxes']:
            pr_all_boxes = torch.tensor(pred_data['bboxes'])
            pr_all_labels = torch.tensor(pred_data['labels'])
            pr_all_scores = torch.tensor(pred_data['scores'])
            
            # Filter by confidence
            keep = pr_all_scores >= score_thr
            pr_boxes = pr_all_boxes[keep]
            pr_labels = pr_all_labels[keep].tolist()
        else:
            pr_boxes = torch.empty((0, 4))
            pr_labels = []

        # 2. Matching Logic
        if len(gt_boxes) > 0 and len(pr_boxes) > 0:
            iou_matrix = box_iou(gt_boxes, pr_boxes)
            matched_pr_indices = set()
            
            for i in range(len(gt_boxes)):
                max_iou, argmax = iou_matrix[i].max(dim=0)
                if max_iou >= iou_threshold:
                    y_true.append(gt_labels[i])
                    y_pred.append(pr_labels[argmax])
                    matched_pr_indices.add(argmax.item())
                else:
                    y_true.append(gt_labels[i])
                    y_pred.append(-1) # Missed (False Negative)
            
            # Count False Positives (ghost detections)
            for j in range(len(pr_labels)):
                if j not in matched_pr_indices:
                    y_true.append(-1) # Background
                    y_pred.append(pr_labels[j])
        else:
            # Handle cases where one file is empty
            for label in gt_labels:
                y_true.append(label)
                y_pred.append(-1)
            for label in pr_labels:
                y_true.append(-1)
                y_pred.append(label)

    # 3. Final Reporting with Empty Check
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    if not unique_labels:
        print("\n--- Statistics ---")
        print("No valid data found to compare. Check your paths and score_thr.")
        return

    print(f"\n--- Statistics: Folder Comparison (IoU > {iou_threshold}) ---")
    
    # Map label indices to names
    target_names = [f"Class_{i}" if i != -1 else "Background" for i in unique_labels]
    
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=unique_labels))

# Set your actual folder paths here
gt_path = '/work/GroundingDINO/fine_tune_Gdino/results_g_dino_pretrained/'
pred_path = '/work/GroundingDINO/fine_tune_Gdino/results_finetune_asis/'

# Run the script
evaluate_folders(gt_path, pred_path, iou_threshold=0.5, score_thr=0.3)
