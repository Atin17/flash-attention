# FlashAttention Usage Guide

Complete guide for using FlashAttention in your projects.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Integration Examples](#integration-examples)
3. [Performance Tuning](#performance-tuning)
4. [Troubleshooting](#troubleshooting)
5. [Best Practices](#best-practices)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/flash-attention.git
cd flash-attention

# Install dependencies
pip install -r requirements.txt

# Build CUDA extension
pip install -e .
```

### Basic Usage

```python
import torch
from flash_attention import FlashAttention

# Initialize
flash_attn = FlashAttention(causal=False)

# Create input tensors
batch, heads, seq_len, head_dim = 2, 8, 1024, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')

# Forward pass
output = flash_attn(q, k, v)
```

## Integration Examples

### 1. Transformer Block

```python
import torch.nn as nn
from flash_attention import MultiHeadFlashAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadFlashAttention(
            d_model=d_model,
            num_heads=num_heads,
            causal=False,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

### 2. GPT-style Decoder

```python
class GPTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size=50000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=2048,
        dropout=0.1
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

### 3. Vision Transformer (ViT)

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        d_model=768,
        num_heads=12,
        num_layers=12,
        dropout=0.1
    ):
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits
```

## Performance Tuning

### Memory Optimization

```python
# 1. Use gradient checkpointing for training
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x)
    
    def _forward(self, x):
        # ... transformer operations
        pass

# 2. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 3. Adjust batch size based on sequence length
def get_batch_size(seq_len):
    if seq_len <= 512:
        return 32
    elif seq_len <= 1024:
        return 16
    elif seq_len <= 2048:
        return 8
    else:
        return 4
```

### Speed Optimization

```python
# 1. Use torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')

# 2. Enable TF32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 3. Use efficient data loading
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```python
# Solution 1: Reduce batch size
batch_size = batch_size // 2

# Solution 2: Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Clear cache periodically
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

#### 2. Numerical Instability

```python
# Use mixed precision with appropriate loss scaling
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(init_scale=2.**16)

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 3. Slow Compilation

```bash
# Use ninja for faster compilation
pip install ninja

# Set environment variables
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"  # Your GPU architectures
```

## Best Practices

### 1. Model Design

```python
# ✓ Good: Use FlashAttention for long sequences
class LongSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FlashAttention()
        
# ✗ Bad: Using standard attention for long sequences
# Memory usage will be prohibitive
```

### 2. Training Configuration

```python
# ✓ Good: Balanced configuration
config = {
    'batch_size': 16,
    'seq_len': 1024,
    'gradient_accumulation': 2,
    'mixed_precision': True,
    'gradient_clipping': 1.0
}

# ✗ Bad: Unbalanced configuration
config = {
    'batch_size': 64,  # Too large
    'seq_len': 4096,   # Too long
    'gradient_accumulation': 1,
    'mixed_precision': False
}
```

### 3. Evaluation

```python
# ✓ Good: Disable dropout and use inference mode
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    
    for batch in dataloader:
        output = model(batch)
        # ... compute metrics
    
    model.train()

# ✗ Bad: Evaluating in training mode
# Will give inconsistent results due to dropout
```

### 4. Monitoring

```python
# Track memory usage
def log_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Track attention patterns (for debugging)
def visualize_attention(model, input_ids):
    # Hook to capture attention weights
    attention_weights = []
    
    def hook(module, input, output):
        attention_weights.append(output.detach())
    
    # Register hook
    handle = model.attention.register_forward_hook(hook)
    
    # Forward pass
    _ = model(input_ids)
    
    # Remove hook
    handle.remove()
    
    return attention_weights
```

## Advanced Usage

### Custom Attention Patterns

```python
from flash_attention import FlashAttention

class CustomAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.flash_attn = FlashAttention()
    
    def forward(self, q, k, v, mask=None):
        # Apply custom masking before attention
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        
        # FlashAttention
        output = self.flash_attn(q, k, v)
        
        return output
```

### Dynamic Sequence Length

```python
def process_variable_length_batch(model, sequences):
    """Handle variable length sequences efficiently."""
    
    # Sort by length (descending)
    lengths = [len(seq) for seq in sequences]
    sorted_indices = sorted(range(len(lengths)), 
                          key=lambda i: lengths[i], 
                          reverse=True)
    
    sorted_sequences = [sequences[i] for i in sorted_indices]
    
    # Pad to max length in batch
    max_len = lengths[sorted_indices[0]]
    padded = torch.nn.utils.rnn.pad_sequence(
        sorted_sequences, 
        batch_first=True
    )
    
    # Process
    output = model(padded)
    
    # Unpad and restore original order
    # ...
    
    return output
```

## Performance Checklist

- [ ] Using FlashAttention for sequences > 512 tokens
- [ ] Mixed precision training enabled
- [ ] Appropriate batch size for GPU memory
- [ ] Gradient checkpointing for deep models
- [ ] Efficient data loading (multiple workers, pin_memory)
- [ ] Gradient clipping to prevent instability
- [ ] Model compiled with torch.compile (PyTorch 2.0+)
- [ ] Regular memory cleanup in long training runs
- [ ] Monitoring GPU utilization and memory usage

## Additional Resources

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [PyTorch CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732)
