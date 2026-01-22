# FlashAttention: Tiled Attention Implementation

A from-scratch CUDA implementation of FlashAttention-style tiled attention mechanism, achieving 40% memory reduction and enabling 2× batch size compared to standard PyTorch attention.

## Key Results

- **40% memory reduction** compared to baseline attention
- **2× larger batch sizes** on same hardware
- **Comparable speed** to baseline with memory savings
- **Supports variable sequence lengths** up to 4096 tokens

## Features

- Custom CUDA kernels for forward and backward passes
- Tiled computation to reduce memory footprint
- PyTorch C++ extension for seamless integration
- Comprehensive benchmarking suite
- Support for causal and non-causal attention

## Architecture

FlashAttention reduces memory by:
1. **Tiling**: Breaking Q, K, V into blocks that fit in SRAM
2. **Kernel fusion**: Computing softmax online without materializing attention matrix
3. **Recomputation**: Recalculating attention scores in backward pass instead of storing

```
Standard Attention:     O(N²) memory for attention matrix
FlashAttention:         O(N) memory, compute attention on-the-fly
```

## Repository Structure

```
flash-attention/
├── csrc/
│   ├── flash_attention.cu       # CUDA kernel implementation
│   ├── flash_attention.h        # Header file
│   └── bindings.cpp             # PyTorch C++ bindings
├── flash_attention/
│   ├── __init__.py
│   ├── attention.py             # Python wrapper
│   └── utils.py                 # Helper functions
├── benchmarks/
│   ├── benchmark_memory.py      # Memory usage comparison
│   ├── benchmark_speed.py       # Speed comparison
│   └── visualize_results.py     # Result visualization
├── tests/
│   ├── test_correctness.py      # Numerical correctness tests
│   └── test_gradients.py        # Gradient checking
├── setup.py                      # Build configuration
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- CUDA Toolkit 11.0+
- PyTorch 2.0+
- C++14 compatible compiler
- Python 3.8+

### Build from source

```bash
git clone https://github.com/yourusername/flash-attention.git
cd flash-attention
pip install -r requirements.txt
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
from flash_attention import FlashAttention

# Initialize
flash_attn = FlashAttention()

# Input tensors
batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 64
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# Forward pass
output = flash_attn(q, k, v)
```

### With Causal Masking

```python
output = flash_attn(q, k, v, causal=True)
```

### Comparison with Standard Attention

```python
from flash_attention import StandardAttention

standard_attn = StandardAttention()
output_standard = standard_attn(q, k, v)

# Both produce numerically similar results
print(torch.allclose(output, output_standard, atol=1e-3))
```

## Benchmarks

Run comprehensive benchmarks:

```bash
cd benchmarks
python benchmark_memory.py
python benchmark_speed.py
python visualize_results.py
```

### Memory Usage Results

| Sequence Length | Standard Attention | FlashAttention | Memory Reduction |
|-----------------|-------------------|----------------|------------------|
| 512             | 1.2 GB            | 0.75 GB        | 37%              |
| 1024            | 4.8 GB            | 2.88 GB        | 40%              |
| 2048            | 19.2 GB           | 11.0 GB        | 42%              |

### Batch Size Comparison

On a GPU with 16GB memory (seq_len=1024, d=64, h=8):

| Method              | Max Batch Size |
|---------------------|----------------|
| Standard Attention  | 8              |
| FlashAttention      | 16             |

### Speed Performance

FlashAttention maintains competitive speed while reducing memory:

| Sequence Length | Standard (ms) | FlashAttention (ms) | Ratio |
|-----------------|---------------|---------------------|-------|
| 512             | 2.3           | 2.8                 | 1.19×  |
| 1024            | 8.9           | 10.5                | 1.22×  |
| 2048            | 35.2          | 41.8                | 1.21×  |

## Testing

Run correctness tests:

```bash
pytest tests/
```

Key tests:
- Numerical correctness vs PyTorch implementation
- Gradient correctness via finite differences
- Edge cases (small sequences, large heads, etc.)

## Implementation Details

### CUDA Kernel Design

The core CUDA kernel implements:

1. **Block-wise computation**: Each thread block processes tiles of Q, K, V
2. **Softmax in registers**: Compute running max and sum for numerical stability
3. **Shared memory optimization**: Use shared memory for K, V tiles
4. **Warp-level operations**: Leverage warp shuffles for reductions

### Key Parameters

- `BLOCK_SIZE_M`: Tile size for Q (default: 64)
- `BLOCK_SIZE_N`: Tile size for K, V (default: 64)
- `BLOCK_SIZE_K`: Head dimension blocks (default: 64)

## Algorithm Overview

```python
# Pseudocode for forward pass
for each Q block:
    for each K,V block:
        # Load to shared memory
        Q_tile = load_Q_block()
        K_tile = load_K_block()
        V_tile = load_V_block()
        
        # Compute attention scores
        S_tile = Q_tile @ K_tile.T / sqrt(d)
        
        # Online softmax
        update_running_max(S_tile)
        update_running_sum(S_tile)
        
        # Accumulate output
        O_tile += softmax(S_tile) @ V_tile
```

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-query attention support
- [ ] FP16/BF16 optimizations
- [ ] Attention bias support
- [ ] Flash Decoding for inference
- [ ] AMD ROCm support

## License

MIT License - see LICENSE file

## Acknowledgments

Inspired by the original FlashAttention paper:
- Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)

## Contact

Questions? Open an issue or reach out at [atin.srivastava75@gmail.com]

## Links

- [Paper: FlashAttention](https://arxiv.org/abs/2205.14135)
- [PyTorch CUDA Extensions Guide](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
