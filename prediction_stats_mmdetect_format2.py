import json
import os
import torch

def get_folder_stats(folder_path, score_thr=0.3):
    """Calculates summary statistics for all JSONs in a folder."""
    if not os.path.exists(folder_path):
        return None

    total_images = 0
    total_boxes = 0
    all_scores = []
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for filename in files:
        total_images += 1
        with open(os.path.join(folder_path, filename), 'r') as f:
            data = json.load(f)
            
        scores = data.get('scores', [])
        # Filter boxes by the provided threshold
        filtered_scores = [s for s in scores if s >= score_thr]
        
        total_boxes += len(filtered_scores)
        all_scores.extend(filtered_scores)

    avg_scores = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_boxes_per_img = total_boxes / total_images if total_images > 0 else 0
    
    return {
        "Folder": os.path.basename(folder_path.strip('/')),
        "Images": total_images,
        "Total Detections": total_boxes,
        "Avg Boxes/Img": round(avg_boxes_per_img, 2),
        "Mean Confidence": round(avg_scores, 4)
    }

def print_comparison_table(folder1, folder2, score_thr=0.3):
    stats1 = get_folder_stats(folder1, score_thr)
    stats2 = get_folder_stats(folder2, score_thr)
    
    if not stats1 or not stats2:
        print("Error: One or both folders could not be found.")
        return

    header = f"{'Metric':<20} | {'Folder 1 (Pretrained)':<25} | {'Folder 2 (Finetuned)':<25}"
    separator = "-" * len(header)
    
    print(f"\nSummary Statistics (Score Threshold: {score_thr})")
    print(separator)
    print(header)
    print(separator)
    
    metrics = ["Images", "Total Detections", "Avg Boxes/Img", "Mean Confidence"]
    for m in metrics:
        print(f"{m:<20} | {str(stats1[m]):<25} | {str(stats2[m]):<25}")
    print(separator)

# --- EXECUTION ---
path_pretrained = '/work/GroundingDINO/fine_tune_Gdino/results_g_dino_pretrained/'
path_finetuned = '/work/GroundingDINO/fine_tune_Gdino/results_asis_with_W_averaging/'

print_comparison_table(path_pretrained, path_finetuned, score_thr=0.3)
