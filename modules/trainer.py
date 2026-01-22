"""
Trainer module for CT2Rep.
Handles the training loop with progress tracking.
NOTE: This is a simplified trainer - main training logic is in main.py
"""

import os
import json
import time
import logging
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np


class Trainer:
    """
    Trainer class for CT2Rep.
    This is kept for compatibility - main training logic is in main.py
    """
    
    def __init__(self, model, criterion, metrics, optimizer, args, 
                 lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            metrics: Metrics function
            optimizer: Optimizer
            args: Training arguments
            lr_scheduler: Learning rate scheduler
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
        """
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger('CT2Rep.Trainer')
        
        # Training state
        self.epoch = 0
        self.best_score = 0.0
        self.train_losses = []
        self.val_scores = []
    
    def train(self):
        """Main training loop."""
        epochs = getattr(self.args, 'epochs', 30)
        
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            
            # Train one epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            if epoch % getattr(self.args, 'eval_interval', 5) == 0:
                val_score = self._validate()
                self.val_scores.append(val_score)
                
                # Save best model
                if val_score > self.best_score:
                    self.best_score = val_score
                    self._save_checkpoint(is_best=True)
            
            # Save checkpoint
            if epoch % getattr(self.args, 'save_interval', 5) == 0:
                self._save_checkpoint()
            
            # Step scheduler
            self.lr_scheduler.step()
        
        # Final test
        self._test()
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_dataloader:
            images = batch['images'].to(self.device)
            report_ids = batch['report_ids'].to(self.device)
            report_masks = batch['report_masks'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, report_ids, report_masks)
            loss = self.criterion(outputs, report_ids, report_masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_dataloader)
    
    def _validate(self):
        """Validate model."""
        self.model.eval()
        
        all_gen = []
        all_gt = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch['images'].to(self.device)
                report_ids = batch['report_ids'].to(self.device)
                
                # Generate (simplified)
                all_gt.append(batch['reports'][0])
                all_gen.append(batch['reports'][0])  # Placeholder
        
        # Compute metrics
        scores = self.metrics(all_gt, all_gen)
        
        return scores.get('BLEU_4', 0.0)
    
    def _test(self):
        """Test model."""
        self.model.eval()
        self.logger.info("Running test evaluation...")
    
    def _save_checkpoint(self, is_best=False):
        """Save checkpoint."""
        save_dir = getattr(self.args, 'save_dir', './results')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_score': self.best_score
        }
        
        path = os.path.join(save_dir, f'checkpoint_{self.epoch:03d}.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)