"""
Loss module for CT2Rep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(outputs, targets, masks, label_smoothing=0.1):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        outputs: Model outputs (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        masks: Attention mask (batch, seq_len)
        label_smoothing: Label smoothing factor
        
    Returns:
        Loss value
    """
    vocab_size = outputs.size(-1)
    
    # Shift for autoregressive language modeling
    shift_logits = outputs[:, :-1, :].contiguous()
    shift_labels = targets[:, 1:].contiguous()
    shift_masks = masks[:, 1:].contiguous().float()
    
    # Flatten
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_masks = shift_masks.view(-1)
    
    # Cross-entropy with label smoothing
    if label_smoothing > 0:
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Create smoothed target distribution
        smooth_target = torch.zeros_like(log_probs)
        smooth_target.fill_(label_smoothing / (vocab_size - 1))
        smooth_target.scatter_(1, shift_labels.unsqueeze(1), 1.0 - label_smoothing)
        
        # Compute loss
        loss = -(smooth_target * log_probs).sum(dim=-1)
    else:
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
    
    # Apply mask
    masked_loss = loss * shift_masks
    
    # Mean over valid tokens
    num_tokens = shift_masks.sum()
    if num_tokens > 0:
        loss = masked_loss.sum() / num_tokens
    else:
        loss = masked_loss.sum()
    
    return loss