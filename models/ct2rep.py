"""
CT2Rep Model for 3D Medical Image Report Generation.
Features:
- 3D Vision Feature Extractor for CT volumes
- Relational Memory for context modeling
- Transformer Decoder with MCLN for report generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv3DBlock(nn.Module):
    """3D Convolution block."""
    
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, stride, padding)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CTEncoder(nn.Module):
    """3D Vision Encoder for CT volumes."""
    
    def __init__(self, d_model=512, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Ensure num_heads divides d_model
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        # 3D CNN backbone
        self.conv_layers = nn.Sequential(
            Conv3DBlock(1, 64, kernel=7, stride=2, padding=3),
            Conv3DBlock(64, 128, kernel=3, stride=2, padding=1),
            Conv3DBlock(128, 256, kernel=3, stride=2, padding=1),
            Conv3DBlock(256, d_model, kernel=3, stride=1, padding=1),
        )
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 8, 8))
        
        self.num_patches = 4 * 8 * 8  # 256 patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: CT volume (batch, 1, D, H, W)
        Returns:
            Visual features (batch, num_patches, d_model)
        """
        # Ensure float32 for convolutions
        x = x.float()
        
        # 3D CNN
        x = self.conv_layers(x)
        
        # Pool to fixed size
        x = self.adaptive_pool(x)
        
        # Reshape to sequence
        batch_size = x.size(0)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_model)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class RelationalMemory(nn.Module):
    """Relational Memory module."""
    
    def __init__(self, num_slots=3, d_model=512, num_heads=8):
        super().__init__()
        
        self.d_model = d_model
        self.num_slots = num_slots
        
        self.memory = nn.Parameter(torch.zeros(1, num_slots, d_model))
        nn.init.xavier_uniform_(self.memory)
        
        # Ensure num_heads divides d_model
        num_heads = min(num_heads, d_model // 64)  # At least 64 dim per head
        num_heads = max(1, num_heads)
        while d_model % num_heads != 0:
            num_heads -= 1
        
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        memory = self.memory.expand(batch_size, -1, -1)
        
        mem_out, _ = self.attn(x, memory, memory)
        
        gate = self.gate(torch.cat([x, mem_out], dim=-1))
        output = gate * mem_out + (1 - gate) * x
        
        return self.norm(output)


class MCLN(nn.Module):
    """Memory-driven Conditional Layer Normalization."""
    
    def __init__(self, d_model, memory_dim=None):
        super().__init__()
        
        # Default memory_dim to d_model if not specified
        if memory_dim is None:
            memory_dim = d_model
        
        self.norm = nn.LayerNorm(d_model)
        
        self.gamma_gen = nn.Sequential(nn.Linear(memory_dim, d_model), nn.Tanh())
        self.beta_gen = nn.Sequential(nn.Linear(memory_dim, d_model), nn.Tanh())
    
    def forward(self, x, memory):
        x_norm = self.norm(x)
        
        gamma = self.gamma_gen(memory).unsqueeze(1)
        beta = self.beta_gen(memory).unsqueeze(1)
        
        return (1 + gamma) * x_norm + beta


class ReportDecoder(nn.Module):
    """Transformer decoder for report generation."""
    
    def __init__(self, vocab_size, d_model=512, num_layers=3, num_heads=8, 
                 d_ff=2048, max_seq_length=300, dropout=0.1, pad_token_id=0):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Ensure num_heads divides d_model
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Memory modules - ensure d_model is passed correctly
        self.rel_memory = RelationalMemory(num_slots=3, d_model=d_model, num_heads=min(num_heads, d_model // 64))
        self.mcln = MCLN(d_model, memory_dim=d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, encoder_output, tgt_ids, tgt_mask=None):
        """
        Args:
            encoder_output: Visual features (batch, num_patches, d_model)
            tgt_ids: Target token IDs (batch, seq_len)
            tgt_mask: Target attention mask (batch, seq_len)
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tgt_ids.shape
        device = tgt_ids.device
        
        # Embed tokens
        tgt_embed = self.embed(tgt_ids) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # Relational memory
        tgt_embed = self.rel_memory(tgt_embed)
        
        # MCLN conditioning
        memory_ctx = encoder_output.mean(dim=1)
        tgt_embed = self.mcln(tgt_embed, memory_ctx)
        
        # Causal mask
        causal_mask = self.generate_causal_mask(seq_len, device)
        
        # Padding mask
        if tgt_mask is not None:
            tgt_key_padding_mask = (tgt_mask == 0)
        else:
            tgt_key_padding_mask = None
        
        # Decode
        output = self.transformer(
            tgt_embed,
            encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.norm(output)
        logits = self.output_proj(output)
        
        return logits


class CT2RepModel(nn.Module):
    """
    CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging.
    """
    
    def __init__(self, args, tokenizer):
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer
        
        # Model dimensions
        d_model = getattr(args, 'd_model', 512)
        d_vf = getattr(args, 'd_vf', 512)
        num_layers = getattr(args, 'num_layers', 3)
        num_heads = getattr(args, 'num_heads', 8)
        d_ff = getattr(args, 'd_ff', 2048)
        dropout = getattr(args, 'dropout', 0.1)
        max_seq_length = getattr(args, 'max_seq_length', 300)
        
        vocab_size = len(tokenizer)
        
        # Encoder
        self.encoder = CTEncoder(
            d_model=d_vf,
            num_layers=4,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Visual projection
        if d_vf != d_model:
            self.visual_proj = nn.Linear(d_vf, d_model)
        else:
            self.visual_proj = nn.Identity()
        
        # Decoder
        self.decoder = ReportDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_token_id=tokenizer.pad_token_id
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, images, report_ids, report_masks=None):
        """
        Forward pass for training.
        """
        visual_features = self.encoder(images)
        visual_features = self.visual_proj(visual_features)
        
        logits = self.decoder(visual_features, report_ids, report_masks)
        
        return logits
    
    @torch.no_grad()
    def generate(self, images, max_length=300, beam_size=3, temperature=0.8,
                 top_p=0.9, repetition_penalty=1.5, no_repeat_ngram_size=4):
        """
        Generate report with improved decoding to prevent repetition.
        
        Args:
            images: CT volumes (batch, 1, D, H, W)
            max_length: Maximum sequence length
            beam_size: Beam size (1 = sampling)
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens (>1 = less repetition)
            no_repeat_ngram_size: Prevent repeating n-grams of this size
        """
        batch_size = images.size(0)
        device = images.device
        
        # Encode
        visual_features = self.encoder(images)
        visual_features = self.visual_proj(visual_features)
        
        sos_id = self.tokenizer.sos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        # Use sampling with repetition penalty
        return self._sample_decode(
            visual_features, max_length, sos_id, eos_id, pad_id,
            temperature, top_p, repetition_penalty, no_repeat_ngram_size
        )
    
    def _sample_decode(self, encoder_output, max_length, sos_id, eos_id, pad_id,
                       temperature=0.8, top_p=0.9, repetition_penalty=1.5, 
                       no_repeat_ngram_size=4):
        """Nucleus sampling with repetition penalty and n-gram blocking."""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with SOS token
        generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        scores = torch.zeros(batch_size, device=device)
        
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Track generated n-grams per sequence for blocking repeats
        generated_ngrams = [{} for _ in range(batch_size)]
        
        for step in range(max_length - 1):
            # Get logits for next token
            logits = self.decoder(encoder_output, generated)
            next_logits = logits[:, -1, :] / max(temperature, 0.1)
            
            # Apply repetition penalty to previously generated tokens
            for batch_idx in range(batch_size):
                if finished[batch_idx]:
                    continue
                    
                # Penalize tokens that have already been generated
                for prev_token in generated[batch_idx].tolist():
                    if prev_token in [pad_id, sos_id]:
                        continue
                    # Reduce probability of repeated tokens
                    if next_logits[batch_idx, prev_token] > 0:
                        next_logits[batch_idx, prev_token] /= repetition_penalty
                    else:
                        next_logits[batch_idx, prev_token] *= repetition_penalty
            
            # Block repeated n-grams
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
                for batch_idx in range(batch_size):
                    if finished[batch_idx]:
                        continue
                    
                    # Get the last (n-1) tokens as prefix
                    prefix_tokens = generated[batch_idx, -(no_repeat_ngram_size-1):].tolist()
                    ngram_prefix = tuple(prefix_tokens)
                    
                    # Ban any token that would complete a previously seen n-gram
                    if ngram_prefix in generated_ngrams[batch_idx]:
                        for banned_token in generated_ngrams[batch_idx][ngram_prefix]:
                            next_logits[batch_idx, banned_token] = float('-inf')
            
            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Create mask for tokens to remove (cumulative prob > top_p)
            sorted_mask = cumulative_probs > top_p
            # Keep at least one token
            sorted_mask[:, 0] = False
            # Shift mask right to keep first token above threshold
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            
            # Apply mask
            for batch_idx in range(batch_size):
                indices_to_remove = sorted_indices[batch_idx][sorted_mask[batch_idx]]
                next_logits[batch_idx, indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_logits, dim=-1)
            
            # Handle potential NaN/Inf issues
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs + 1e-10  # Prevent all zeros
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update n-gram tracking BEFORE adding new token
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size - 1:
                for batch_idx in range(batch_size):
                    if finished[batch_idx]:
                        continue
                    # Record this n-gram
                    prefix_tokens = generated[batch_idx, -(no_repeat_ngram_size-1):].tolist()
                    ngram_prefix = tuple(prefix_tokens)
                    if ngram_prefix not in generated_ngrams[batch_idx]:
                        generated_ngrams[batch_idx][ngram_prefix] = set()
                    generated_ngrams[batch_idx][ngram_prefix].add(next_token[batch_idx].item())
            
            # Force EOS for finished sequences, append next_token for others
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, eos_id),
                next_token
            )
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update finished status
            finished = finished | (next_token.squeeze(-1) == eos_id)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated, scores