"""Training pipeline for SAR Ship Detection using Faster R-CNN."""

import os
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from configs.config import Config, cfg
from src.dataset import SARShipDataset, get_transforms, collate_fn
from src.model import build_model
from src.evaluate import evaluate_model
from utils.utils import save_checkpoint, load_checkpoint, plot_training_history


class Trainer:

    def __init__(self, config):
        self.config = config

        # Device (Mac MPS / CUDA / CPU)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print('=' * 60)
        print(f'Device: {self.device}')
        print(f'Epochs: {self.config.epochs}')
        print(f'Batch Size: {self.config.batch_size}')
        print(f'Learning Rate: {self.config.lr}')
        print('=' * 60)

        # History
        self.history = {
            'train_loss': [], 'val_map': [],
            'val_precision': [], 'val_recall': [],
        }

        # Dataset
        self._setup_datasets()

        # Model
        self.model = build_model(
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            config=self.config,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model Parameters: {total_params:,} (trainable: {trainable:,})')

        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            params, lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step,
            gamma=self.config.lr_gamma,
        )

        self.best_map = 0.0
        self.start_time = time.time()

    def _setup_datasets(self):
        train_ds = SARShipDataset(
            self.config.data_root, split='train',
            transforms=get_transforms('train'),
        )
        val_ds = SARShipDataset(
            self.config.data_root, split='test',
            transforms=get_transforms('test'),
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0,
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch+1}/{self.config.epochs}',
                    leave=True, ncols=100)

        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [
                {
                    'boxes': t['boxes'].to(self.device) if torch.is_tensor(t['boxes'])
                             else torch.tensor(t['boxes'], dtype=torch.float32).to(self.device),
                    'labels': t['labels'].to(self.device) if torch.is_tensor(t['labels'])
                              else torch.tensor(t['labels'], dtype=torch.int64).to(self.device),
                }
                for t in targets
            ]

            # Skip batch if all targets have empty boxes
            if all(t['boxes'].shape[0] == 0 for t in targets):
                continue

            try:
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
            except Exception as e:
                print(f'\nWarning: batch skipped due to error: {e}')
                continue

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Free memory on MPS/CUDA to prevent crashes
            del images, targets, loss_dict, loss
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self):
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        return metrics

    def train(self):
        print(f'\nStarting training for {self.config.epochs} epochs...')
        print(f'Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}')

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)

            # Validate every val_interval epochs or last epoch
            if (epoch + 1) % self.config.val_interval == 0 or epoch == self.config.epochs - 1:
                metrics = self.validate()
                current_map = metrics['mAP@0.5']
                precision = metrics['precision']
                recall = metrics['recall']
            else:
                current_map = self.history['val_map'][-1] if self.history['val_map'] else 0
                precision = self.history['val_precision'][-1] if self.history['val_precision'] else 0
                recall = self.history['val_recall'][-1] if self.history['val_recall'] else 0

            self.history['train_loss'].append(train_loss)
            self.history['val_map'].append(current_map)
            self.history['val_precision'].append(precision)
            self.history['val_recall'].append(recall)

            elapsed = time.time() - self.start_time
            lr = self.optimizer.param_groups[0]['lr']

            print(f'\nEpoch {epoch+1}/{self.config.epochs} | '
                  f'Loss: {train_loss:.4f} | mAP: {current_map:.4f} | '
                  f'P: {precision:.4f} | R: {recall:.4f} | '
                  f'LR: {lr:.6f} | Time: {elapsed:.0f}s')

            # Save best model
            if current_map > self.best_map:
                self.best_map = current_map
                save_checkpoint(
                    os.path.join(self.config.save_dir, 'best_model.pth'),
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.best_map,
                )
                print(f'*** New Best Model! mAP: {self.best_map:.4f} ***')

            # Periodic save
            if (epoch + 1) % self.config.save_interval == 0:
                save_checkpoint(
                    os.path.join(self.config.save_dir, f'epoch_{epoch+1}.pth'),
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.best_map,
                )

            self.scheduler.step()

        # Final save
        save_checkpoint(
            os.path.join(self.config.save_dir, 'final_model.pth'),
            self.model, self.optimizer, self.scheduler,
            self.config.epochs - 1, self.best_map,
        )

        # Plot history
        plot_training_history(self.history, self.config.results_dir)

        # Save metrics JSON
        with open(os.path.join(self.config.results_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        total_time = time.time() - self.start_time
        print('\n' + '=' * 60)
        print(f'Training Complete!')
        print(f'Best mAP@0.5: {self.best_map:.4f}')
        print(f'Total Time: {total_time/60:.1f} minutes')
        print('=' * 60)


if __name__ == '__main__':
    trainer = Trainer(cfg)
    trainer.train()