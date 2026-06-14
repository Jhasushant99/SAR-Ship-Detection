"""Evaluation metrics for SAR Ship Detection."""

import torch
import numpy as np
from torchvision.ops import box_iou


def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Match predicted boxes to ground truth using IoU."""
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    ious = box_iou(pred_boxes, gt_boxes).cpu().numpy()
    matched_gt = set()
    tp, fp = 0, 0

    for i in range(len(pred_boxes)):
        best_iou, best_j = 0, -1
        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            if ious[i, j] > best_iou:
                best_iou = ious[i, j]
                best_j = j

        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_metrics(preds, targets, iou_threshold=0.5):
    """Compute detection metrics (AP, Precision, Recall)."""
    tp_total, fp_total, fn_total = 0, 0, 0
    all_scores, all_matches = [], []

    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes'].cpu()
        pred_scores = pred['scores'].cpu()
        gt_boxes = target['boxes'].cpu()

        # Sort predictions by confidence
        sort_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

        tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        # For AP computation
        matched_gt = set()
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes).cpu().numpy()
            for i in range(len(pred_boxes)):
                all_scores.append(pred_scores[i].item())
                best_j = -1
                best_iou = 0
                for j in range(len(gt_boxes)):
                    if j not in matched_gt and ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_j = j
                if best_iou >= iou_threshold and best_j >= 0:
                    all_matches.append(1)
                    matched_gt.add(best_j)
                else:
                    all_matches.append(0)
        elif len(pred_boxes) > 0:
            for i in range(len(pred_boxes)):
                all_scores.append(pred_scores[i].item())
                all_matches.append(0)

    # Compute AP (11-point interpolation)
    ap = 0.0
    if len(all_scores) > 0:
        idx = np.argsort(all_scores)[::-1]
        matches = np.array(all_matches)[idx]
        tp_cum = np.cumsum(matches)
        fp_cum = np.cumsum(1 - matches)
        precision_curve = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall_curve = tp_cum / (tp_total + fn_total + 1e-8)

        for t in np.linspace(0, 1, 11):
            p = precision_curve[recall_curve >= t].max() if np.any(recall_curve >= t) else 0
            ap += p / 11

    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)

    return {'AP': float(ap), 'precision': float(precision), 'recall': float(recall)}


def evaluate_model(model, loader, device):
    """Run full evaluation on a data loader."""
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, targs in loader:
            images = [img.to(device) for img in images]

            # FasterRCNN in eval mode returns predictions directly
            outputs = model(images)

            for o, t in zip(outputs, targs):
                preds.append({
                    'boxes': o['boxes'].cpu(),
                    'scores': o['scores'].cpu(),
                    'labels': o['labels'].cpu(),
                })
                targets.append({
                    'boxes': t['boxes'].cpu(),
                    'labels': t['labels'].cpu(),
                })

    metrics = compute_metrics(preds, targets, iou_threshold=0.5)

    return {
        'mAP@0.5': metrics['AP'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
    }