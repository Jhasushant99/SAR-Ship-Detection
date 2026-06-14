"""SAR Ship Dataset for Faster R-CNN training.

Supports SSDD (SAR Ship Detection Dataset) with COCO-format annotations.
Safe loading, SAR-specific preprocessing, albumentations augmentation.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SARShipDataset(Dataset):
    """SSDD dataset loader for Faster R-CNN."""

    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        self.base_dir = os.path.join(root_dir, 'SSDD')
        self.img_dir = os.path.join(self.base_dir, 'images', split)
        ann_file = os.path.join(self.base_dir, 'annotations', f'{split}.json')

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f'Annotation not found: {ann_file}')
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f'Image directory not found: {self.img_dir}')

        with open(ann_file, 'r') as f:
            coco = json.load(f)

        # Build lookup maps
        self.images_info = {img['id']: img for img in coco['images']}
        self.image_annotations = {}
        for ann in coco['annotations']:
            self.image_annotations.setdefault(ann['image_id'], []).append(ann)

        # Filter to images that actually exist on disk AND have annotations
        self.image_ids = []
        skipped = 0
        for img_id in self.images_info:
            img_name = self.images_info[img_id]['file_name']
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path) and img_id in self.image_annotations:
                self.image_ids.append(img_id)
            else:
                skipped += 1

        print(f'[{split}] Loaded: {len(self.image_ids)} images '
              f'({skipped} skipped) | Annotations: {len(coco["annotations"])}')

    def _load_image(self, path):
        """Load image and convert to 3-channel uint8."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # Handle grayscale SAR images
        if len(img.shape) == 2:
            # SAR feature engineering: raw + sqrt + smoothed
            img_f = img.astype(np.float32)
            img_norm = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX)
            ch1 = img_norm
            ch2 = np.sqrt(img_norm / 255.0) * 255.0
            ch3 = cv2.GaussianBlur(img_norm, (3, 3), 0)
            img = np.stack([ch1, ch2, ch3], axis=-1).astype(np.uint8)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # Ensure uint8
        if img.dtype != np.uint8:
            img = cv2.normalize(img.astype(np.float32), None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

        return img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image
        image = self._load_image(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        # Load annotations
        anns = self.image_annotations.get(img_id, [])
        boxes, labels = [], []

        h, w = image.shape[:2]
        for ann in anns:
            x, y, bw, bh = ann['bbox']  # COCO format: x, y, width, height
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            # Skip degenerate boxes
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # class 1 = ship

        # Apply augmentations
        if self.transforms and len(boxes) > 0:
            transformed = self.transforms(
                image=image, bboxes=boxes, labels=labels,
            )
            image = transformed['image']
            t_boxes = transformed['bboxes']
            t_labels = transformed['labels']

            if len(t_boxes) > 0:
                boxes = torch.tensor(t_boxes, dtype=torch.float32)
                labels = torch.tensor(t_labels, dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        elif self.transforms:
            transformed = self.transforms(image=image, bboxes=[], labels=[])
            image = transformed['image']
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        # Ensure image is float tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        # Ensure boxes is 2D
        if isinstance(boxes, torch.Tensor) and boxes.dim() == 1 and boxes.numel() > 0:
            boxes = boxes.unsqueeze(0)
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    if boxes.shape[0] > 0 else torch.zeros((0,)),
            'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        return image, target


def get_transforms(split='train', size=800):
    """Get albumentations transforms for train/val."""
    if split == 'train':
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['labels'],
            min_area=1, min_visibility=0.3,
        ))

    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc', label_fields=['labels'],
        min_area=1, min_visibility=0.3,
    ))


def collate_fn(batch):
    """Custom collate for variable-size targets."""
    return tuple(zip(*batch))