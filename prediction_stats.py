import json
import numpy as np
import torch
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix, classification_report

def coco_to_xyxy(bbox):
    """Converts COCO [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

def gd_original_to_xyxy(bbox, img_w, img_h):
    """Converts Normalized [cx, cy, w, h] to Absolute [xmin, ymin, xmax, ymax]"""
    cx, cy, w, h = bbox
    xmin = (cx - w / 2) * img_w
    ymin = (cy - h / 2) * img_h
    xmax = (cx + w / 2) * img_w
    ymax = (cy + h / 2) * img_h
    return [xmin, ymin, xmax, ymax]

def evaluate_predictions(gt_path, predictions, img_metadata, mode="original", iou_threshold=0.5):
    """
    Args:
        gt_path: Path to COCO json
        predictions: List of dicts/tensors from Grounding DINO
        img_metadata: Dict mapping img_id to {width, height}
        mode: "original" (normalized cxcywh) or "huggingface" (absolute xyxy)
    """
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Map image IDs to their ground truth annotations
    gt_map = {img['id']: [] for img in gt_data['images']}
    for ann in gt_data['annotations']:
        gt_map[ann['image_id']].append({
            'bbox': coco_to_xyxy(ann['bbox']),
            'category_id': ann['category_id']
        })

    y_true = []
    y_pred = []

    for img_id, pred_list in predictions.items():
        w, h = img_metadata[img_id]['width'], img_metadata[img_id]['height']
        gt_anns = gt_map.get(img_id, [])
        
        processed_preds = []
        for p in pred_list:
            if mode == "original":
                # Original returns normalized [cx, cy, w, h]
                box = gd_original_to_xyxy(p['bbox'], w, h)
            else:
                # Hugging Face returns absolute [xmin, ymin, xmax, ymax]
                box = p['bbox']
            processed_preds.append({'bbox': box, 'label': p['label']})

        if not gt_anns and not processed_preds:
            continue

        # Match predictions to GT using IoU
        if gt_anns and processed_preds:
            gt_boxes = torch.tensor([ann['bbox'] for ann in gt_anns])
            pr_boxes = torch.tensor([pr['bbox'] for pr in processed_preds])
            iou_matrix = box_iou(gt_boxes, pr_boxes)
            
            # Simple matching: find max IoU for each GT
            matched_pr_indices = set()
            for i, gt_ann in enumerate(gt_anns):
                max_iou, argmax = iou_matrix[i].max(dim=0)
                if max_iou >= iou_threshold:
                    y_true.append(gt_ann['category_id'])
                    y_pred.append(processed_preds[argmax]['label'])
                    matched_pr_indices.add(argmax.item())
                else:
                    y_true.append(gt_ann['category_id'])
                    y_pred.append(-1) # Background/Miss
            
            # Count False Positives (predictions with no GT match)
            for j, pr in enumerate(processed_preds):
                if j not in matched_pr_indices:
                    y_true.append(-1) # Background
                    y_pred.append(pr['label'])
        else:
            # Handle cases with only GT or only Preds
            for ann in gt_anns:
                y_true.append(ann['category_id'])
                y_pred.append(-1)
            for pr in processed_preds:
                y_true.append(-1)
                y_pred.append(pr['label'])

    # Filter out background index (-1) for cleaner reports if desired
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    print(f"--- Statistics ({mode.upper()} mode) ---")
    print(classification_report(y_true, y_pred, labels=unique_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=unique_labels))

# Example Usage
# metadata = {0: {'width': 640, 'height': 480}, ...} derived from test.json
# predictions = {0: [{'bbox': [0.5, 0.5, 0.1, 0.1], 'label': 1}], ...}
