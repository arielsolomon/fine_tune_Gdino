_base_ = 'grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'

# --- 1. PATHS AND CLASSES ---
data_root = '/work/datasets/mini_chinese_dataset/'
# Use a trailing comma to ensure this is a tuple of length 1
class_names = ('Vehicle',) 

# --- 2. MODEL CONFIGURATION ---
model = dict(
    type='GroundingDINO',
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=len(class_names), # Should be 1
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)
    ),
    train_cfg=dict(
        _delete_=True, 
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(_delete_=True, max_per_img=300)
)

# --- 3. DATA PIPELINES ---
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
    dict(type='LoadAnnotations', with_bbox=True), # Added to help evaluator
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

# --- 4. DATALOADERS ---
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='labels/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dict(classes=class_names),
        filter_cfg=dict(filter_empty_gt=True, min_size=32), 
        pipeline=train_pipeline,
        return_classes=True 
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='labels/test.json',
        data_prefix=dict(img='images/test/'),
        metainfo=dict(classes=class_names),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True
    )
)

test_dataloader = val_dataloader

# --- 5. EVALUATORS ---
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'labels/test.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# --- 6. TRAINING STRATEGY ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True, 
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001), 
    clip_grad=dict(max_norm=0.1)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# --- 7. HOOKS AND LOGGING ---
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50)
)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'