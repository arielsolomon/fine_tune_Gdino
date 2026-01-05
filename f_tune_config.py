# Use the file seen in your screenshot
_base_ = 'configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'

data_root = '/home/user1/ariel/GroundingDINO/mmdetection/mini_chinese_dataset/'

# --- MODEL MODIFICATIONS ---
model = dict(
    bbox_head=dict(
        num_classes=7  # Matches your 7 infrared categories
    )
)

# --- DATASET MODIFICATIONS ---
# IMPORTANT: Use the actual names of your 7 classes here!
# For Grounding DINO, these names ARE the text prompts.
class_names = ('car', 'truck', 'bus', 'van', 'other_vehicle', 'trailer', 'tanker')

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img='train/images/'),
        metainfo=dict(classes=class_names)
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test/test.json',
        data_prefix=dict(img='test/images/'),
        metainfo=dict(classes=class_names)
    )
)

# --- LOAD WEIGHTS ---
# Match the weights to your base config (Swin-T)
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'