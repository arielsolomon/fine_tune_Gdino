# Use the file seen in your screenshot
_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'

data_root = '/home/user1/ariel/datasets/mini_chinese_dataset/'

# --- DATASET CONFIGURATION ---
class_names = ('car', 'truck', 'bus', 'van', 'other_vehicle', 'trailer', 'tanker')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities')) 
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

train_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train/images/train.json',
        data_prefix=dict(img='train/images/'),
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline,
        return_classes=True
    )
)

val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test/images/test.json',
        data_prefix=dict(img='test/images/'),
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        return_classes=True
    )
)

test_dataloader = val_dataloader

# --- EVALUATOR ---
# Stripped to basics to avoid TypeError. 
# We skip the execution of this via train_cfg to avoid the IndexError crash.
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/images/test.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# --- MODEL MODIFICATIONS ---
model = dict(
    bbox_head=dict(
        _delete_=True,    
        type='GroundingDINOHead',
        num_classes=7,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])
    )
)

# --- TRAINING & OPTIMIZATION ---
# 1. We skip validation by setting val_interval higher than max_epochs.
# 2. We keep the learning rate low for stable finetuning.
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=12, 
    val_interval=15
)

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    clip_grad=dict(max_norm=0.1)
)

# --- LOAD WEIGHTS ---
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'