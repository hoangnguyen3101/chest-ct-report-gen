"""
Optimizers module for CT2Rep.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


def build_optimizer(args, model):
    """Build optimizer."""
    lr = getattr(args, 'lr', 5e-5)
    weight_decay = getattr(args, 'weight_decay', 5e-5)
    
    # Separate parameters
    no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    return AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def build_lr_scheduler(args, optimizer):
    """Build learning rate scheduler."""
    step_size = getattr(args, 'step_size', 1)
    gamma = getattr(args, 'gamma', 0.9)
    
    return StepLR(optimizer, step_size=step_size, gamma=gamma)