"""Faster R-CNN model for SAR Ship Detection."""

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


def build_backbone(pretrained=True):
    """Build ResNet-50 + FPN backbone."""
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone_base = resnet50(weights=weights)

    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3',
    }

    backbone = BackboneWithFPN(
        backbone_base,
        return_layers=return_layers,
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )

    return backbone


def build_model(num_classes=2, pretrained=True, config=None):
    """Build Faster R-CNN with FPN backbone for SAR ship detection."""

    backbone = build_backbone(pretrained)

    # SAR-optimized anchors (ships have varied aspect ratios)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),) * 5,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2,
    )

    # Build model with config overrides
    kwargs = {
        'backbone': backbone,
        'num_classes': num_classes,
        'rpn_anchor_generator': anchor_generator,
        'box_roi_pool': roi_pooler,
        'min_size': 800,
        'max_size': 1333,
        'box_detections_per_img': 300,
        'box_nms_thresh': 0.3,
        'box_score_thresh': 0.05,
    }

    if config:
        kwargs.update({
            'min_size': getattr(config, 'target_size', 800),
            'max_size': getattr(config, 'max_size', 1333),
            'box_detections_per_img': getattr(config, 'box_detections_per_img', 300),
            'box_nms_thresh': getattr(config, 'box_nms_thresh', 0.3),
            'box_score_thresh': getattr(config, 'box_score_thresh', 0.05),
        })

    model = FasterRCNN(**kwargs)
    return model