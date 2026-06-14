"""Utility functions for SAR Ship Detection."""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_map):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_map': best_map,
    }, path)
    print(f'Checkpoint saved: {path}')


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        return 0, 0.0

    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    epoch = ckpt.get('epoch', 0)
    best_map = ckpt.get('best_map', 0.0)
    print(f'Loaded checkpoint: epoch {epoch}, mAP {best_map:.4f}')
    return epoch, best_map


def plot_training_history(history, save_dir):
    """Plot and save training curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    # mAP
    if history.get('val_map'):
        axes[1].plot(history['val_map'], 'g-', linewidth=2)
        axes[1].set_title('Validation mAP@0.5', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP')
        axes[1].grid(True, alpha=0.3)

    # Precision & Recall
    if history.get('val_precision') and history.get('val_recall'):
        axes[2].plot(history['val_precision'], 'r-', linewidth=2, label='Precision')
        axes[2].plot(history['val_recall'], 'm-', linewidth=2, label='Recall')
        axes[2].set_title('Precision & Recall', fontsize=14)
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {save_dir}/training_curves.png')


def visualize_predictions(image, preds, targets=None, score_thresh=0.5, save_path=None):
    """Visualize detection results with bounding boxes."""
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.dim() == 3:
            img = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
    else:
        img = np.array(image)
        if img.max() > 1:
            img = img / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)

    # Draw predictions (red)
    pred_count = 0
    if 'boxes' in preds and 'scores' in preds:
        boxes = preds['boxes'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        keep = scores > score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        pred_count = len(boxes)

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none',
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{score:.2f}', color='red',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))

    # Draw ground truth (green)
    gt_count = 0
    if targets is not None and 'boxes' in targets:
        gt_boxes = targets['boxes'].cpu().numpy()
        gt_count = len(gt_boxes)
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--',
            )
            ax.add_patch(rect)

    ax.set_title(f'Detections: {pred_count} (red) | Ground Truth: {gt_count} (green)',
                 fontsize=14)
    ax.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f'Saved: {save_path}')

    plt.close()


def denormalize_image(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize an ImageNet-normalized tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)