import os
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Config:

    # Paths
    data_root: str = "data/sar-ship-detection-dataset"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"

    # Model
    num_classes: int = 2
    backbone: str = "resnet50"
    pretrained: bool = True

    # Anchors (SAR optimized)
    anchor_sizes = [(32,), (64,), (128,), (256,), (512,)]
    aspect_ratios = [(0.25, 0.5, 1.0, 2.0, 4.0)] * 5

    # Image
    target_size: int = 800
    max_size: int = 1333

    # Training
    batch_size: int = 4
    epochs: int = 50
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step: int = 20
    lr_gamma: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"

    # RPN
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 1000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 500

    # ROI
    roi_output_size: int = 7
    roi_sampling_ratio: int = 2
    box_detections_per_img: int = 300
    box_nms_thresh: float = 0.3
    box_score_thresh: float = 0.05

    # Validation
    val_interval: int = 5
    save_interval: int = 10

    # Inference
    conf_thresh: float = 0.5
    inference_overlap: int = 200

    def __post_init__(self):
        import os
        import torch

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


cfg = Config()