# Ship Detection in SAR Imagery Using Deep Learning

**Author:** Sushant Shekhar  
**Personal Technical Portfolio & Deep Learning Research Project**  

## Objective
Design, train, and evaluate a deep learning-based ship detection model using Synthetic Aperture Radar (SAR) imagery. The trained model performs inference on real-world SAR data acquired from Sentinel-1.

## Final Evaluation Results (SSDD Test Set)
- **mAP@0.5**: **0.9068 (90.68%)**
- **Precision**: **0.7309 (73.09%)**
- **Recall**: **0.9652 (96.52%)**





## Architecture

**Faster R-CNN** with:
- **Backbone**: ResNet-50 + Feature Pyramid Network (FPN)
- **Anchors**: SAR-optimized (5 scales x 5 aspect ratios at each FPN level)
- **Input**: 3-channel SAR fusion (raw intensity + sqrt enhancement + Gaussian smoothing)
- **Training**: SGD with StepLR, gradient clipping, ImageNet normalization

## Project Structure

```
SARDATA/
├── main.py                          # Entry point (train/eval/infer/visualize)
├── run.sh                           # Quick-run script
├── configs/
│   └── config.py                    # All hyperparameters
├── src/
│   ├── dataset.py                   # SSDD data loader + augmentations
│   ├── model.py                     # Faster R-CNN builder
│   ├── train.py                     # Training loop with tqdm
│   └── evaluate.py                  # mAP/Precision/Recall computation
├── inference/
│   └── sentinel1_inference.py       # Sentinel-1 SAFE product inference
├── utils/
│   └── utils.py                     # Checkpoints, visualization, plots
├── data/
│   └── sar-ship-detection-dataset/
│       └── SSDD/
│           ├── images/{train,test}/  # SAR images
│           └── annotations/         # COCO JSON annotations
├── checkpoints/                     # Saved models
├── results/                         # Plots, metrics, reports
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
python main.py --mode train --epochs 50 --batch-size 4
# or
bash run.sh train
```

### 3. Evaluate
```bash
python main.py --mode eval --model-path checkpoints/best_model.pth
# or
bash run.sh eval
```

### 4. Visualize Detections
```bash
python main.py --mode visualize --model-path checkpoints/best_model.pth
# or
bash run.sh visualize
```

### 5. Sentinel-1 Inference
```bash
python main.py --mode infer --model-path checkpoints/best_model.pth --safe-path /path/to/sentinel1.SAFE
# or
bash run.sh infer --safe-path /path/to/sentinel1.SAFE
```

### 6. Generate Technical Report
```bash
python scripts/generate_report.py
```

## Dataset

**SSDD (SAR Ship Detection Dataset)**
- 1,160 images (928 train / 233 test)
- 2,456 ship annotations in COCO format
- Single-channel SAR intensity images

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| Learning Rate | 0.005, StepLR (step=20, gamma=0.1) |
| Batch Size | 4 |
| Epochs | 50 |
| Input Size | 800x800 |
| Gradient Clipping | max_norm=1.0 |

## Implemented Features

- [x] Modular codebase for data preprocessing, model training, and evaluation
- [x] Technical analysis report (PDF) covering dataset, architecture, and results
- [x] Sentinel-1 inference pipeline with tile cropping and NMS box merging
- [x] Robust evaluation logging (mAP@0.5, Precision, Recall tracking)

## Device Support

Automatically detects: **CUDA GPU** > **Apple MPS** > **CPU**
