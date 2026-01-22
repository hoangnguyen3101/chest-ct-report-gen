#!/usr/bin/env python
"""
CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging
Configured for RadGenome-ChestCT Dataset on UMBC Cluster
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
import json
import logging
from pathlib import Path
import warnings
import atexit
import shutil

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# SUPPRESS ALL NFS/TEMP CLEANUP ERRORS
# ============================================================================
_original_rmtree = shutil.rmtree

def _safe_rmtree(path, ignore_errors=False, onerror=None):
    try:
        _original_rmtree(path, ignore_errors=True)  # Always ignore errors
    except:
        pass  # Silently ignore all cleanup errors

shutil.rmtree = _safe_rmtree

# Suppress multiprocessing cleanup errors completely
import multiprocessing.util
_original_run_finalizers = multiprocessing.util._run_finalizers

def _quiet_run_finalizers(minpriority=None):
    try:
        _original_run_finalizers(minpriority)
    except:
        pass  # Ignore all cleanup errors

multiprocessing.util._run_finalizers = _quiet_run_finalizers

# Suppress stderr for cleanup errors
import sys
class SuppressCleanupErrors:
    def __init__(self, stream):
        self.stream = stream
    def write(self, msg):
        if 'OSError' in msg and ('not empty' in msg or 'busy' in msg or 'pymp-' in msg):
            return  # Suppress NFS cleanup errors
        self.stream.write(msg)
    def flush(self):
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

# Don't wrap stderr if already wrapped
if not isinstance(sys.stderr, SuppressCleanupErrors):
    sys.stderr = SuppressCleanupErrors(sys.stderr)

# ============================================================================
# UMBC CLUSTER PATHS (from RadGenome_download.py)
# ============================================================================
# RadGenome-ChestCT dataset location
DATA_DIR = "/umbc/rs/pi_oates/users/dta1/Data/Medical_Report_Generation/RadGenome_dataset"

# Output directory for models, checkpoints, logs
OUTPUT_DIR = "/umbc/rs/pi_oates/users/dta1/trained_model/Medical_Report_Generation"

# Cache directory (same as in download script)
CACHE_DIR = "/umbc/rs/pi_oates/users/dta1/all_cache"

# Set environment variables
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from modules.tokenizers import Tokenizer
from modules.dataloaders import RadGenomeDataset
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.loss import compute_loss
from models.ct2rep import CT2RepModel


class TrainingLogger:
    """Enhanced logger for training progress."""
    
    def __init__(self, save_dir, rank=0):
        self.rank = rank
        self.is_main = (rank == 0)
        
        if self.is_main:
            log_dir = os.path.join(save_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f'training_{timestamp}.log')
            
            self.logger = logging.getLogger('CT2Rep')
            self.logger.setLevel(logging.INFO)
            self.logger.handlers = []
            
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler(sys.stdout)
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        else:
            self.logger = None
    
    def info(self, msg):
        if self.is_main and self.logger:
            self.logger.info(msg)
    
    def header(self, title):
        if self.is_main:
            self.info("=" * 70)
            self.info(title.center(70))
            self.info("=" * 70)
    
    def section(self, title):
        if self.is_main:
            self.info("-" * 60)
            self.info(title)
            self.info("-" * 60)


def setup_directories():
    """Create necessary directories."""
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, 'checkpoints'),
        os.path.join(OUTPUT_DIR, 'logs'),
        os.path.join(OUTPUT_DIR, 'results'),
        os.path.join(OUTPUT_DIR, 'inference_samples'),
        os.path.join(OUTPUT_DIR, 'eval_results'),
        CACHE_DIR,
        os.path.join(CACHE_DIR, 'tokenizer'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def set_seed(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='CT2Rep on RadGenome-ChestCT')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--save_dir', type=str, default=None)
    
    # Model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--d_vf', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--max_seq_length', type=int, default=300)
    parser.add_argument('--threshold', type=int, default=3)
    
    # LR scheduler
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    
    # Multi-GPU
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--master_port', type=str, default='12355')
    
    # Mixed precision
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--amp_dtype', type=str, default='float16')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--num_inference_samples', type=int, default=5)
    parser.add_argument('--skip_pretrain_inference', action='store_true')
    parser.add_argument('--beam_size', type=int, default=3)
    
    # Generation parameters (to prevent repetition)
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower=more focused)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.5,
                        help='Penalty for repeating tokens (>1 reduces repetition)')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=4,
                        help='Prevent repeating n-grams of this size')
    
    # Other
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    
    return parser.parse_args()


def setup_distributed(rank, world_size, master_port):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', init_method='env://',
                           world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_inference_and_save(model, dataloader, tokenizer, device, save_dir,
                           epoch=None, num_samples=5, logger=None, beam_size=3,
                           max_seq_length=300, args=None):
    """Run inference and save results with CT visualizations."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    model.eval()
    prefix = f"epoch_{epoch:03d}" if epoch is not None else "pretrain"
    sample_dir = os.path.join(save_dir, prefix)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Generation parameters
    temperature = getattr(args, 'temperature', 0.8) if args else 0.8
    top_p = getattr(args, 'top_p', 0.9) if args else 0.9
    repetition_penalty = getattr(args, 'repetition_penalty', 1.5) if args else 1.5
    no_repeat_ngram_size = getattr(args, 'no_repeat_ngram_size', 4) if args else 4
    
    results = []
    if logger:
        logger.section(f"INFERENCE SAMPLES - {prefix.upper()}")
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            images = batch['images'].to(device)
            reports_text = batch.get('reports', [])
            
            batch_size = images.size(0)
            
            for i in range(min(batch_size, num_samples - sample_count)):
                try:
                    model_fn = model.module if hasattr(model, 'module') else model
                    output_ids, _ = model_fn.generate(
                        images[i:i+1], 
                        max_length=max_seq_length,
                        beam_size=beam_size,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size
                    )
                    generated = tokenizer.decode(output_ids[0])
                except Exception as e:
                    generated = f"[Generation Error: {str(e)[:50]}]"
                
                # Get ground truth
                if reports_text and i < len(reports_text) and reports_text[i]:
                    ground_truth = reports_text[i]
                else:
                    ground_truth = "[No GT available]"
                
                # Save visualization
                try:
                    save_ct_visualization(
                        images[i].cpu().numpy(),
                        ground_truth,
                        generated,
                        os.path.join(sample_dir, f'sample_{sample_count:03d}.png'),
                        sample_id=sample_count
                    )
                except Exception as e:
                    if logger:
                        logger.info(f"  Warning: Could not save visualization: {str(e)[:50]}")
                
                results.append({
                    'sample_id': sample_count,
                    'ground_truth': ground_truth,
                    'generated': generated
                })
                
                if logger:
                    logger.info(f"\n--- Sample {sample_count + 1}/{num_samples} ---")
                    gt_preview = ground_truth[:500].replace('\n', ' ') if ground_truth else "[Empty]"
                    gen_preview = generated[:500].replace('\n', ' ') if generated else "[Empty]"
                    logger.info(f"GT: {gt_preview}")
                    logger.info(f"Gen: {gen_preview}")
                
                sample_count += 1
    
    # Save results JSON
    with open(os.path.join(sample_dir, 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create summary grid of all samples
    try:
        create_summary_grid(sample_dir, num_samples)
    except Exception as e:
        if logger:
            logger.info(f"  Warning: Could not create summary grid: {str(e)[:50]}")
    
    model.train()
    return results


def save_ct_visualization(ct_volume, ground_truth, generated, save_path, sample_id=0):
    """
    Save CT volume slices with ground truth and generated reports.
    
    Args:
        ct_volume: numpy array of shape (1, D, H, W) or (D, H, W)
        ground_truth: ground truth report string
        generated: generated report string
        save_path: path to save the visualization
        sample_id: sample identifier
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from textwrap import wrap
    
    # Handle different input shapes
    if ct_volume.ndim == 4:
        ct_volume = ct_volume[0]  # Remove channel dimension: (D, H, W)
    
    depth = ct_volume.shape[0]
    
    # Select 6 representative slices across the volume
    slice_indices = [
        int(depth * 0.1),   # 10%
        int(depth * 0.25),  # 25%
        int(depth * 0.4),   # 40%
        int(depth * 0.55),  # 55%
        int(depth * 0.7),   # 70%
        int(depth * 0.85),  # 85%
    ]
    
    # Create figure with CT slices on top, reports below
    fig = plt.figure(figsize=(18, 14))
    
    # Title
    fig.suptitle(f'Sample {sample_id}: CT Scan Visualization with Reports', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Plot CT slices (2 rows x 3 cols)
    for idx, slice_idx in enumerate(slice_indices):
        ax = fig.add_subplot(3, 3, idx + 1)
        
        # Get slice and normalize for display
        slice_img = ct_volume[slice_idx]
        vmin, vmax = np.percentile(slice_img, [2, 98])
        
        ax.imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'Slice {slice_idx}/{depth}', fontsize=10)
        ax.axis('off')
    
    # Ground Truth Report (bottom left)
    ax_gt = fig.add_subplot(3, 3, 7)
    ax_gt.axis('off')
    gt_wrapped = '\n'.join(wrap(ground_truth[:800], width=50))
    ax_gt.text(0.05, 0.95, 'GROUND TRUTH:', fontsize=11, fontweight='bold',
               transform=ax_gt.transAxes, verticalalignment='top', color='darkgreen')
    ax_gt.text(0.05, 0.85, gt_wrapped, fontsize=8, transform=ax_gt.transAxes,
               verticalalignment='top', wrap=True, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Generated Report (bottom right, spanning 2 columns)
    ax_gen = fig.add_subplot(3, 3, 8)
    ax_gen.axis('off')
    gen_wrapped = '\n'.join(wrap(generated[:800], width=50))
    ax_gen.text(0.05, 0.95, 'GENERATED:', fontsize=11, fontweight='bold',
               transform=ax_gen.transAxes, verticalalignment='top', color='darkblue')
    ax_gen.text(0.05, 0.85, gen_wrapped, fontsize=8, transform=ax_gen.transAxes,
               verticalalignment='top', wrap=True, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Comparison metrics (bottom right corner)
    ax_metrics = fig.add_subplot(3, 3, 9)
    ax_metrics.axis('off')
    
    # Simple word overlap metric
    gt_words = set(ground_truth.lower().split())
    gen_words = set(generated.lower().split())
    overlap = len(gt_words & gen_words)
    total = len(gt_words | gen_words)
    word_overlap_pct = (overlap / total * 100) if total > 0 else 0
    
    metrics_text = f"Quick Stats:\n\nGT words: {len(gt_words)}\nGen words: {len(gen_words)}\nWord overlap: {word_overlap_pct:.1f}%"
    ax_metrics.text(0.1, 0.8, metrics_text, fontsize=10, transform=ax_metrics.transAxes,
                   verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_summary_grid(sample_dir, num_samples):
    """Create a summary grid image combining all samples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Collect all sample images
    sample_images = []
    for i in range(num_samples):
        img_path = os.path.join(sample_dir, f'sample_{i:03d}.png')
        if os.path.exists(img_path):
            sample_images.append(img_path)
    
    if not sample_images:
        return
    
    # Create grid
    n_images = len(sample_images)
    cols = min(2, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20 * cols, 14 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, img_path in enumerate(sample_images):
        row, col = idx // cols, idx % cols
        img = plt.imread(img_path)
        axes[row][col].imshow(img)
        axes[row][col].axis('off')
    
    # Hide empty subplots
    for idx in range(len(sample_images), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, 'summary_grid.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)


def evaluate_and_log(model, dataloader, tokenizer, device, epoch, save_dir, 
                     logger, beam_size=3, max_seq_length=300, args=None):
    """Evaluate and compute metrics."""
    model.eval()
    all_gen, all_gt = [], []
    total_loss, num_batches = 0.0, 0
    
    # Generation parameters
    temperature = getattr(args, 'temperature', 0.8) if args else 0.8
    top_p = getattr(args, 'top_p', 0.9) if args else 0.9
    repetition_penalty = getattr(args, 'repetition_penalty', 1.5) if args else 1.5
    no_repeat_ngram_size = getattr(args, 'no_repeat_ngram_size', 4) if args else 4
    
    # Hardcoded: 100 samples during training, ALL samples for final epoch
    total_epochs = getattr(args, 'epochs', 30) if args else 30
    is_final_epoch = (epoch == total_epochs)
    
    if is_final_epoch:
        max_eval_samples = -1  # Evaluate all
    else:
        max_eval_samples = 100  # Quick evaluation during training
    
    if logger:
        logger.header(f"EVALUATION - EPOCH {epoch}")
        if is_final_epoch:
            logger.info(f"  Final epoch - evaluating on ALL validation samples")
        else:
            logger.info(f"  Evaluating on {max_eval_samples} samples (full eval at epoch {total_epochs})")
    
    samples_evaluated = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Check if we've reached max samples
            if max_eval_samples > 0 and samples_evaluated >= max_eval_samples:
                break
                
            images = batch['images'].to(device)
            report_ids = batch['report_ids'].to(device)
            report_masks = batch['report_masks'].to(device)
            reports_text = batch.get('reports', [])
            
            model_fn = model.module if hasattr(model, 'module') else model
            outputs = model_fn(images, report_ids, report_masks)
            loss = compute_loss(outputs, report_ids, report_masks)
            total_loss += loss.item()
            num_batches += 1
            
            # Generate reports (limit to remaining samples needed)
            batch_size = images.size(0)
            samples_to_eval = batch_size
            if max_eval_samples > 0:
                samples_to_eval = min(batch_size, max_eval_samples - samples_evaluated)
            
            for i in range(samples_to_eval):
                try:
                    output_ids, _ = model_fn.generate(
                        images[i:i+1], 
                        max_length=max_seq_length,
                        beam_size=beam_size,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size
                    )
                    gen = tokenizer.decode(output_ids[0])
                except Exception as e:
                    gen = f"[Generation Error: {str(e)[:50]}]"
                
                # Get ground truth
                if reports_text and i < len(reports_text) and reports_text[i]:
                    gt = reports_text[i]
                else:
                    gt = tokenizer.decode(report_ids[i].cpu().numpy())
                
                all_gen.append(gen)
                all_gt.append(gt)
                samples_evaluated += 1
    
    # Compute metrics
    metrics = compute_scores(all_gt, all_gen)
    avg_loss = total_loss / max(num_batches, 1)
    
    if logger:
        logger.info(f"  Samples evaluated: {len(all_gen)}")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  BLEU-1: {metrics.get('BLEU_1', 0):.4f}")
        logger.info(f"  BLEU-2: {metrics.get('BLEU_2', 0):.4f}")
        logger.info(f"  BLEU-3: {metrics.get('BLEU_3', 0):.4f}")
        logger.info(f"  BLEU-4: {metrics.get('BLEU_4', 0):.4f}")
        logger.info(f"  METEOR: {metrics.get('METEOR', 0):.4f}")
        logger.info(f"  ROUGE-L: {metrics.get('ROUGE_L', 0):.4f}")
    
    # Save
    eval_dir = os.path.join(save_dir, 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f'eval_{epoch:03d}.json'), 'w') as f:
        json.dump({'epoch': epoch, 'loss': avg_loss, 'metrics': metrics}, f, indent=2)
    
    model.train()
    return metrics, avg_loss


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                    epoch, args, logger):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    epoch_start = datetime.now()
    
    if logger:
        logger.header(f"TRAINING EPOCH {epoch}/{args.epochs}")
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}, Batches: {num_batches}")
        logger.info(f"Batch/GPU: {args.batch_size}, GPUs: {args.n_gpu}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        # Log first few batches to track data loading progress
        if logger and batch_idx < 5:
            logger.info(f"  Loading batch {batch_idx + 1}...")
        
        images = batch['images'].to(device)
        report_ids = batch['report_ids'].to(device)
        report_masks = batch['report_masks'].to(device)
        
        if args.use_amp and scaler:
            amp_dtype = torch.float16 if args.amp_dtype == 'float16' else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                model_fn = model.module if hasattr(model, 'module') else model
                outputs = model_fn(images, report_ids, report_masks)
                loss = compute_loss(outputs, report_ids, report_masks)
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            model_fn = model.module if hasattr(model, 'module') else model
            outputs = model_fn(images, report_ids, report_masks)
            loss = compute_loss(outputs, report_ids, report_masks)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        batch_loss = loss.item() * args.gradient_accumulation_steps
        total_loss += batch_loss
        
        # Log progress - also log first few batches
        should_log = (batch_idx + 1) % args.log_interval == 0 or batch_idx == num_batches - 1 or batch_idx < 5
        if logger and should_log:
            elapsed = (datetime.now() - epoch_start).total_seconds()
            eta = (elapsed / (batch_idx + 1)) * (num_batches - batch_idx - 1)
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
            logger.info(
                f"  [{batch_idx+1:5d}/{num_batches}] "
                f"Loss: {batch_loss:.4f} | Avg: {total_loss/(batch_idx+1):.4f} | "
                f"GPU: {gpu_mem:.1f}GB | ETA: {eta/60:.1f}min"
            )
    
    scheduler.step()
    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, 
                    save_dir, is_best, logger):
    """Save checkpoint."""
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    if scaler:
        state['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(state, os.path.join(ckpt_dir, f'ckpt_{epoch:03d}.pth'))
    torch.save(state, os.path.join(ckpt_dir, 'latest.pth'))
    
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, 'best.pth'))
        if logger:
            logger.info("*** New best model saved! ***")


def main_worker(rank, world_size, args):
    """Main worker."""
    setup_directories()
    
    if args.save_dir is None:
        args.save_dir = os.path.join(OUTPUT_DIR, 'results', 
                                      f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger = TrainingLogger(args.save_dir, rank)
    
    is_distributed = world_size > 1
    if is_distributed:
        setup_distributed(rank, world_size, args.master_port)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(args.seed + rank)
    is_main = (rank == 0)
    
    if is_main:
        logger.header("CT2Rep on RadGenome-ChestCT")
        logger.info(f"\nData Dir: {args.data_dir}")
        logger.info(f"Output: {OUTPUT_DIR}")
        logger.info(f"Cache: {CACHE_DIR}")
        logger.info(f"GPUs: {world_size}, Batch/GPU: {args.batch_size}")
        logger.info(f"Effective batch: {args.batch_size * world_size * args.gradient_accumulation_steps}")
    
    # Build tokenizer
    if is_main:
        logger.section("Building Tokenizer")
    
    tokenizer = Tokenizer(args)
    
    if is_main:
        logger.info(f"  Vocabulary size: {len(tokenizer)}")
    
    # Build datasets
    if is_main:
        logger.section("Loading RadGenome-ChestCT Dataset")
    
    train_dataset = RadGenomeDataset(args, tokenizer, split='train')
    val_dataset = RadGenomeDataset(args, tokenizer, split='valid')
    
    if is_main:
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset)}")
    
    # Dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, world_size, rank, True) if is_distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, world_size, rank, False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset, args.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, args.batch_size, 
        sampler=val_sampler,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    if is_main:
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
    
    # Build model
    if is_main:
        logger.section("Building Model")
    
    model = CT2RepModel(args, tokenizer).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {total_params:,}")
    
    # Optimizer and scheduler
    model_for_opt = model.module if hasattr(model, 'module') else model
    optimizer = build_optimizer(args, model_for_opt)
    scheduler = build_lr_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Resume
    start_epoch, best_bleu4 = 1, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model_for_opt.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_bleu4 = ckpt.get('metrics', {}).get('BLEU_4', 0.0)
        if is_main:
            logger.info(f"  Resumed from epoch {start_epoch-1}")
    
    if is_distributed:
        dist.barrier()
    
    # Pre-training inference
    if is_main and not args.skip_pretrain_inference:
        logger.header("PRE-TRAINING INFERENCE")
        run_inference_and_save(
            model, val_loader, tokenizer, device,
            os.path.join(args.save_dir, 'inference'), None,
            args.num_inference_samples, logger, args.beam_size, args.max_seq_length, args
        )
    
    if is_distributed:
        dist.barrier()
    
    # Training loop
    train_start = datetime.now()
    
    for epoch in range(start_epoch, args.epochs + 1):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args, logger if is_main else None
        )
        
        metrics, is_best = {}, False
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            if is_main:
                metrics, val_loss = evaluate_and_log(
                    model, val_loader, tokenizer, device, epoch,
                    args.save_dir, logger, args.beam_size, args.max_seq_length, args
                )
                
                run_inference_and_save(
                    model, val_loader, tokenizer, device,
                    os.path.join(args.save_dir, 'inference'), epoch,
                    args.num_inference_samples, logger, args.beam_size, args.max_seq_length, args
                )
                
                is_best = metrics.get('BLEU_4', 0) > best_bleu4
                if is_best:
                    best_bleu4 = metrics['BLEU_4']
        
        if is_main and (epoch % args.save_interval == 0 or epoch == args.epochs):
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, 
                           metrics, args.save_dir, is_best, logger)
        
        if is_distributed:
            dist.barrier()
    
    if is_main:
        logger.header("TRAINING COMPLETE")
        logger.info(f"Time: {(datetime.now()-train_start).total_seconds()/3600:.2f}h")
        logger.info(f"Best BLEU-4: {best_bleu4:.4f}")
        logger.info(f"Results: {args.save_dir}")
    
    if is_distributed:
        cleanup_distributed()


def main():
    args = parse_args()
    setup_directories()
    
    world_size = min(args.n_gpu, torch.cuda.device_count())
    
    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)


if __name__ == '__main__':
    main()