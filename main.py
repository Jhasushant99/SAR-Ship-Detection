"""SAR Ship Detection - Main Entry Point."""

import os
import sys
import argparse
import torch

from configs.config import cfg
from src.dataset import SARShipDataset, get_transforms, collate_fn
from src.model import build_model
from src.train import Trainer
from src.evaluate import evaluate_model
from utils.utils import visualize_predictions
from torch.utils.data import DataLoader
from inference.sentinel1_inference import run_sentinel1_inference


def parse_args():
    parser = argparse.ArgumentParser(description='SAR Ship Detection')
    parser.add_argument('--mode', default='train',
                        choices=['train', 'eval', 'infer', 'visualize'])
    parser.add_argument('--data-root', type=str,
                        default='data/sar-ship-detection-dataset')
    parser.add_argument('--save-dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--model-path', default='checkpoints/best_model.pth')
    parser.add_argument('--safe-path', default=None)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--split', default='test')
    return parser.parse_args()


def get_device(device_str='auto'):
    """Auto-detect best available device."""
    if device_str != 'auto':
        return device_str
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def setup_config(args):
    cfg.data_root = args.data_root
    cfg.save_dir = args.save_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.results_dir = args.output_dir
    cfg.device = get_device(args.device)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    return cfg


def load_trained_model(model_path, device, config):
    """Load a trained model from checkpoint."""
    model = build_model(config.num_classes, pretrained=False, config=config)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both full checkpoint and raw state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint (epoch {checkpoint.get("epoch", "?")}, '
              f'mAP: {checkpoint.get("best_map", 0):.4f})')
    else:
        model.load_state_dict(checkpoint)
        print(f'Loaded model weights from {model_path}')

    model.eval()
    return model


def train_mode(args):
    config = setup_config(args)
    Trainer(config).train()


def eval_mode(args):
    config = setup_config(args)
    device = torch.device(config.device)
    model = load_trained_model(args.model_path, device, config)

    dataset = SARShipDataset(config.data_root, args.split, get_transforms('val'))
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)

    metrics = evaluate_model(model, loader, device)

    print('\n' + '=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    print(f'mAP@0.5:   {metrics["mAP@0.5"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall:    {metrics["recall"]:.4f}')
    print('=' * 50)


def infer_mode(args):
    if not args.safe_path:
        print('ERROR: --safe-path required for inference mode')
        sys.exit(1)

    config = setup_config(args)
    run_sentinel1_inference(
        safe_path=args.safe_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        config=config,
    )


def visualize_mode(args):
    config = setup_config(args)
    device = torch.device(config.device)
    model = load_trained_model(args.model_path, device, config)

    dataset = SARShipDataset(config.data_root, args.split, get_transforms('val'))

    import random
    indices = random.sample(range(len(dataset)), min(5, len(dataset)))

    os.makedirs(args.output_dir, exist_ok=True)

    for idx in indices:
        image, target = dataset[idx]
        image = image.to(device)

        with torch.no_grad():
            outputs = model([image])[0]

        save_path = os.path.join(args.output_dir, f'vis_{idx}.png')
        visualize_predictions(image, outputs, target, score_thresh=0.5, save_path=save_path)
        print(f'Saved: {save_path}')


def main():
    args = parse_args()

    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'eval':
        eval_mode(args)
    elif args.mode == 'infer':
        infer_mode(args)
    elif args.mode == 'visualize':
        visualize_mode(args)


if __name__ == '__main__':
    main()
