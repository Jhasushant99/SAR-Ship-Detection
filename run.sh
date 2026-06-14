#!/bin/bash
# SAR Ship Detection - Run Script
# Usage:
#   bash run.sh train          # Train model
#   bash run.sh eval           # Evaluate model
#   bash run.sh visualize      # Visualize detections
#   bash run.sh infer --safe-path /path/to/sentinel1  # Sentinel-1 inference

set -e

echo "==========================================="
echo "  SAR Ship Detection - Faster R-CNN"
echo "==========================================="

MODE=${1:-train}
shift 2>/dev/null || true

case $MODE in
    train)
        echo "Mode: Training"
        python main.py --mode train \
            --data-root data/sar-ship-detection-dataset \
            --epochs 50 \
            --batch-size 4 \
            --lr 0.005 \
            "$@"
        ;;
    eval)
        echo "Mode: Evaluation"
        python main.py --mode eval \
            --data-root data/sar-ship-detection-dataset \
            --model-path checkpoints/best_model.pth \
            --split test \
            "$@"
        ;;
    visualize)
        echo "Mode: Visualization"
        python main.py --mode visualize \
            --data-root data/sar-ship-detection-dataset \
            --model-path checkpoints/best_model.pth \
            --output-dir results/visualizations \
            "$@"
        ;;
    infer)
        echo "Mode: Sentinel-1 Inference"
        python main.py --mode infer \
            --model-path checkpoints/best_model.pth \
            --output-dir results/sentinel1 \
            "$@"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run.sh {train|eval|visualize|infer}"
        exit 1
        ;;
esac

echo "==========================================="
echo "  Done!"
echo "==========================================="
