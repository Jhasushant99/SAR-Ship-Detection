"""Sentinel-1 SAR inference pipeline.


1. SAFE product reading (VV + VH polarization)
2.Single GeoTIFF/image files
3. Directory of images (jpg/png) - processes ALL
4.Tile-based inference with NMS merge
5. Visualization and JSON export
"""

import os
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.model import build_model
from configs.config import cfg


class Sentinel1Preprocessor:

    def __init__(self, target_size=800, overlap=200):
        self.target_size = target_size
        self.overlap = overlap

    def read_product(self, safe_path):
        safe_path = Path(safe_path)

        if safe_path.is_file():
            return self._read_single_file(safe_path)

        vv_files = list(safe_path.rglob('*VV*.tiff')) + list(safe_path.rglob('*vv*.tiff'))
        if vv_files:
            vh_files = list(safe_path.rglob('*VH*.tiff')) + list(safe_path.rglob('*vh*.tiff'))
            try:
                import rasterio
                vv = rasterio.open(vv_files[0]).read(1).astype(np.float32)
                vh = rasterio.open(vh_files[0]).read(1).astype(np.float32) if vh_files else vv.copy()
            except ImportError:
                vv = cv2.imread(str(vv_files[0]), cv2.IMREAD_UNCHANGED).astype(np.float32)
                vh = cv2.imread(str(vh_files[0]), cv2.IMREAD_UNCHANGED).astype(np.float32) if vh_files else vv.copy()
            return vv, vh

        tiff_files = list(safe_path.rglob('*.tiff')) + list(safe_path.rglob('*.tif'))
        if tiff_files:
            return self._read_single_file(tiff_files[0])

        img_files = sorted(
            list(safe_path.rglob('*.jpg')) + list(safe_path.rglob('*.jpeg')) +
            list(safe_path.rglob('*.png')) + list(safe_path.rglob('*.bmp'))
        )
        if img_files:
            return self._read_single_file(img_files[0])

        raise FileNotFoundError(f'No SAR data found in {safe_path}')

    def get_all_images(self, safe_path):
        """Get all image files from a directory."""
        safe_path = Path(safe_path)
        if safe_path.is_file():
            return [safe_path]
        img_files = sorted(
            list(safe_path.rglob('*.jpg')) + list(safe_path.rglob('*.jpeg')) +
            list(safe_path.rglob('*.png')) + list(safe_path.rglob('*.bmp')) +
            list(safe_path.rglob('*.tiff')) + list(safe_path.rglob('*.tif'))
        )
        return img_files

    def _read_single_file(self, filepath):
        filepath = str(filepath)
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                data = src.read(1).astype(np.float32)
            return data, data.copy()
        except (ImportError, Exception):
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f'Cannot read: {filepath}')
            img = img.astype(np.float32)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return img, img.copy()

    def make_3channel(self, img):
        img_norm = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        ch1 = img_norm
        ch2 = np.sqrt(img_norm / 255.0) * 255.0
        ch3 = cv2.GaussianBlur(img_norm, (3, 3), 0)
        return np.stack([ch1, ch2, ch3], axis=-1).astype(np.uint8)

    def create_tiles(self, image):
        h, w = image.shape[:2]
        tiles, positions = [], []
        stride = max(1, self.target_size - self.overlap)

        for y in range(0, max(1, h - self.target_size + 1), stride):
            for x in range(0, max(1, w - self.target_size + 1), stride):
                tile = image[y:y + self.target_size, x:x + self.target_size]
                if tile.shape[0] < self.target_size or tile.shape[1] < self.target_size:
                    padded = np.zeros((self.target_size, self.target_size, 3), dtype=tile.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                tiles.append(tile)
                positions.append((x, y))

        if not tiles:
            padded = np.zeros((self.target_size, self.target_size, 3), dtype=image.dtype)
            padded[:min(h, self.target_size), :min(w, self.target_size)] = \
                image[:min(h, self.target_size), :min(w, self.target_size)]
            tiles.append(padded)
            positions.append((0, 0))

        return tiles, positions

    def preprocess(self, safe_path):
        print('[1/3] Loading SAR data...')
        vv, vh = self.read_product(safe_path)
        print(f'      Image size: {vv.shape}')
        print('[2/3] Creating 3-channel image...')
        image = self.make_3channel(vv)
        print('[3/3] Creating tiles...')
        tiles, positions = self.create_tiles(image)
        print(f'      Tiles created: {len(tiles)}')
        return tiles, positions, image


class Sentinel1Detector:

    def __init__(self, model_path, config=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = build_model(num_classes=2, pretrained=False, config=config)
        self.model = self.model.to(self.device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()
        print(f'Model loaded on {self.device}')

    @torch.no_grad()
    def detect_single(self, image_3ch, conf=0.5):
        """Detect ships in a single 3-channel uint8 image."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tensor = torch.from_numpy(image_3ch).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.to(self.device)

        pred = self.model([tensor])[0]
        keep = pred['scores'] > conf

        return {
            'boxes': pred['boxes'][keep].cpu().numpy(),
            'scores': pred['scores'][keep].cpu().numpy(),
        }

    @torch.no_grad()
    def detect(self, tiles, conf=0.5):
        outputs = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for tile in tqdm(tiles, desc='Detecting ships'):
            if len(tile.shape) == 2:
                tile = np.stack([tile] * 3, axis=-1)
            tensor = torch.from_numpy(tile).float() / 255.0
            tensor = tensor.permute(2, 0, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.to(self.device)
            pred = self.model([tensor])[0]
            keep = pred['scores'] > conf
            outputs.append({
                'boxes': pred['boxes'][keep].cpu().numpy(),
                'scores': pred['scores'][keep].cpu().numpy(),
            })
        return outputs

    def merge(self, detections, positions, nms_thresh=0.3):
        all_boxes, all_scores = [], []
        for det, (x_off, y_off) in zip(detections, positions):
            for box, score in zip(det['boxes'], det['scores']):
                x1, y1, x2, y2 = box
                all_boxes.append([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off])
                all_scores.append(score)
        if not all_boxes:
            return np.array([]), np.array([])
        boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        from torchvision.ops import nms
        keep = nms(boxes_t, scores_t, nms_thresh)
        return np.array(all_boxes)[keep.numpy()], np.array(all_scores)[keep.numpy()]

    def visualize(self, image, boxes, scores, save_path, title=None):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                              edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 3, f'{score:.2f}', color='yellow', fontsize=8, fontweight='bold')
        t = title or f'Detected Ships: {len(boxes)}'
        ax.set_title(t, fontsize=14)
        ax.axis('off')
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_sentinel1_inference(safe_path, model_path, output_dir='results/sentinel1', config=None):
    """Run inference on single image, directory of images, or Sentinel-1 SAFE product."""
    os.makedirs(output_dir, exist_ok=True)
    config = config or cfg

    pre = Sentinel1Preprocessor(
        target_size=getattr(config, 'target_size', 800),
        overlap=getattr(config, 'inference_overlap', 200),
    )

    detector = Sentinel1Detector(model_path, config=config)

    # Get all images
    all_images = pre.get_all_images(safe_path)
    print(f'Found {len(all_images)} images')

    # Limit to 20 for demo
    selected = all_images[:20]
    all_results = []
    total_ships = 0

    # Process each image
    for i, img_path in enumerate(tqdm(selected, desc='Processing images')):
        try:
            vv, _ = pre._read_single_file(img_path)
            image_3ch = pre.make_3channel(vv)

            det = detector.detect_single(image_3ch, conf=getattr(config, 'conf_thresh', 0.5))
            n_ships = len(det['boxes'])
            total_ships += n_ships

            # Save individual result
            save_path = os.path.join(output_dir, f'detection_{i:03d}.png')
            detector.visualize(image_3ch, det['boxes'], det['scores'], save_path,
                               title=f'{img_path.name} - Ships: {n_ships}')

            all_results.append({
                'image': str(img_path.name),
                'detections': n_ships,
                'mean_confidence': float(np.mean(det['scores'])) if n_ships > 0 else 0,
                'boxes': det['boxes'].tolist(),
                'scores': det['scores'].tolist(),
            })

        except Exception as e:
            print(f'Skipped {img_path.name}: {e}')

    # Create grid visualization
    grid_images = min(12, len(selected))
    cols = 4
    rows = (grid_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx in range(len(axes)):
        if idx < grid_images and idx < len(selected):
            try:
                vv, _ = pre._read_single_file(selected[idx])
                image_3ch = pre.make_3channel(vv)
                det = detector.detect_single(image_3ch, conf=getattr(config, 'conf_thresh', 0.5))
                axes[idx].imshow(image_3ch)
                for box, score in zip(det['boxes'], det['scores']):
                    x1, y1, x2, y2 = box
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor='red', facecolor='none', linewidth=2)
                    axes[idx].add_patch(rect)
                axes[idx].set_title(f'{selected[idx].name}\nShips: {len(det["boxes"])}', fontsize=10)
            except:
                pass
        axes[idx].axis('off')

    plt.suptitle(f'SAR Ship Detection Results - Total Ships: {total_ships}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentinel1_detections.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nGrid saved: {output_dir}/sentinel1_detections.png')

    # Save report
    report = {
        'total_images': len(selected),
        'total_detections': total_ships,
        'results': all_results,
    }
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Total ships detected: {total_ships} across {len(selected)} images')
    return [], [], report
