"""
mmdetection YOLOX-X config matching the ByteTrack checkpoint architecture.

ByteTrack uses YOLOX-X trained at 800x1440 on DanceTrack (person class only).
For COCO-pretrained YOLOX-X (80 classes) keep num_classes=80.
Adjust num_classes and test input size to match your specific checkpoint.
"""
_base_ = 'mmdet::yolox/yolox_x_8xb8-300e_coco.py'

# ByteTrack DanceTrack model is a single-class (person) detector.
# Change to 80 if using a COCO-pretrained YOLOX-X checkpoint.
_num_classes = 1

_img_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]  # ImageNet mean (RGB order)
_img_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]    # ImageNet std  (RGB order)

model = dict(
    bbox_head=dict(
        num_classes=_num_classes,
    ),
    # ByteTrack legacy preprocessing: BGR→RGB, /255, ImageNet normalize.
    data_preprocessor=dict(
        mean=_img_mean,
        std=_img_std,
        bgr_to_rgb=True,
        batch_augments=[],
    ),
    test_cfg=dict(score_thr=0.0, nms=dict(type='nms', iou_threshold=0.7), max_per_img=300),
)

# Test-time resize matching ByteTrack DanceTrack inference size.
# Override if your checkpoint was trained at a different resolution.
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(800, 1440), keep_ratio=True),
    dict(
        type='Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_shape'),
    ),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
