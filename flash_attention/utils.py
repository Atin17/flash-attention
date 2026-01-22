"""
Utility functions for FlashAttention.

Includes helper functions for benchmarking, testing, and debugging.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def create_causal_mask(seq_len: int, device: str = 'cuda') -> torch.Tensor:
    """
    Create a causal mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Boolean mask of shape [seq_len, seq_len]
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1
    )
    return mask


def create_random_mask(
    batch_size: int,
    seq_len: int, 
    mask_prob: float = 0.15,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create random attention mask (useful for testing).
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_prob: Probability of masking a position
        device: Device to create mask on
        
    Returns:
        Boolean mask of shape [batch_size, seq_len, seq_len]
    """
    mask = torch.rand(batch_size, seq_len, seq_len, device=device) < mask_prob
    return mask


def get_memory_usage(device: str = 'cuda') -> Tuple[float, float]:
    """
    Get current GPU memory usage.
    
    Args:
        device: CUDA device
        
    Returns:
        Tuple of (allocated_gb, reserved_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    
    return allocated, reserved


def print_memory_summary(device: str = 'cuda'):
    """Print detailed memory summary."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"\n{'='*60}")
    print(f"GPU Memory Summary - {torch.cuda.get_device_name(device)}")
    print(f"{'='*60}")
    
    allocated, reserved = get_memory_usage(device)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Total:     {total:.2f} GB")
    print(f"Free:      {total - reserved:.2f} GB")
    print(f"Usage:     {(reserved / total * 100):.1f}%")
    print(f"{'='*60}\n")


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    standard_attention: bool = True
) -> float:
    """
    Estimate memory usage for attention computation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        standard_attention: If True, estimates for standard attention,
                          otherwise for FlashAttention
    
    Returns:
        Estimated memory in GB
    """
    bytes_per_float = 4  # float32
    
    # Q, K, V tensors
    qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * bytes_per_float
    
    if standard_attention:
        # Attention matrix: [batch, heads, seq_len, seq_len]
        attn_memory = batch_size * num_heads * seq_len * seq_len * bytes_per_float
        total_memory = qkv_memory + attn_memory
    else:
        # FlashAttention: only small tile overhead
        tile_size = 64
        tile_memory = batch_size * num_heads * tile_size * tile_size * bytes_per_float
        total_memory = qkv_memory + tile_memory
    
    return total_memory / (1024 ** 3)  # Convert to GB


def calculate_theoretical_speedup(seq_len: int) -> float:
    """
    Calculate theoretical speedup based on complexity analysis.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Estimated speedup ratio
    """
    # FlashAttention reduces memory transfers but has similar FLOPs
    # Speedup comes from better memory hierarchy usage
    
    # Rough model: speedup diminishes for very small sequences
    if seq_len < 128:
        return 0.9  # Overhead dominates
    elif seq_len < 512:
        return 1.0  # Break-even
    elif seq_len < 2048:
        return 1.1  # Slight speedup
    else:
        return 1.2  # Better memory reuse


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str = "tensors",
    atol: float = 1e-3,
    rtol: float = 1e-3
) -> bool:
    """
    Compare two tensors and print detailed statistics.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name: Name for printing
        atol: Absolute tolerance
        rtol: Relative tolerance
        
    Returns:
        True if tensors are close
    """
    if tensor1.shape != tensor2.shape:
        print(f"❌ {name}: Shape mismatch {tensor1.shape} vs {tensor2.shape}")
        return False
    
    diff = (tensor1 - tensor2).abs()
    rel_diff = diff / (tensor2.abs() + 1e-8)
    
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    
    is_close = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    
    status = "✓" if is_close else "❌"
    print(f"{status} {name}:")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")
    print(f"  Max rel diff:  {max_rel_diff:.6e}")
    
    if not is_close:
        print(f"  Tolerance: atol={atol}, rtol={rtol}")
    
    return is_close


def profile_attention(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_iterations: int = 100,
    warmup: int = 10
) -> dict:
    """
    Profile attention function performance.
    
    Args:
        attention_fn: Attention function to profile
        q, k, v: Input tensors
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with profiling statistics
    """
    device = q.device
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = attention_fn(q, k, v)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile
    times = []
    for _ in range(num_iterations):
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                _ = attention_fn(q, k, v)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            import time
            start = time.time()
            with torch.no_grad():
                _ = attention_fn(q, k, v)
            end = time.time()
            times.append((end - start) * 1000)
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }


def generate_test_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random test inputs for attention.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        device: Device to create tensors on
        dtype: Data type
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (Q, K, V) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device=device, dtype=dtype)
    
    return q, k, v


def compute_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """
    Reference attention implementation using PyTorch operations.
    
    This is the ground truth for correctness testing.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        causal: Whether to apply causal masking
        dropout_p: Dropout probability
        
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    batch, heads, seq_len, head_dim = q.shape
    scale = head_dim ** -0.5
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax
    attn = F.softmax(scores, dim=-1)
    
    # Dropout
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    
    # Weighted sum
    output = torch.matmul(attn, v)
    
    return output


def check_cuda_availability():
    """Check CUDA availability and print device information."""
    print("\n" + "="*60)
    print("CUDA Availability Check")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"✓ Device count: {torch.cuda.device_count()}")
        print(f"✓ Current device: {torch.cuda.current_device()}")
        print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
        
        props = torch.cuda.get_device_properties(0)
        print(f"✓ Compute capability: {props.major}.{props.minor}")
        print(f"✓ Total memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("❌ CUDA not available")
        print("   FlashAttention will fall back to PyTorch implementation")
    
    print("="*60 + "\n")


def save_benchmark_results(results: dict, filename: str = "benchmark_results.json"):
    """Save benchmark results to JSON file."""
    import json
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to {filename}")


if __name__ == "__main__":
    # Run basic tests
    check_cuda_availability()
    
    if torch.cuda.is_available():
        print_memory_summary()
        
        # Test memory estimation
        print("\nMemory Estimation Examples:")
        configs = [
            (4, 512, 8, 64),
            (4, 1024, 8, 64),
            (4, 2048, 8, 64),
        ]
        
        for batch, seq_len, heads, head_dim in configs:
            std_mem = estimate_memory_usage(batch, seq_len, heads, head_dim, True)
            flash_mem = estimate_memory_usage(batch, seq_len, heads, head_dim, False)
            reduction = (std_mem - flash_mem) / std_mem * 100
            
            print(f"  seq_len={seq_len}: Standard={std_mem:.2f}GB, "
                  f"Flash={flash_mem:.2f}GB, Reduction={reduction:.1f}%")
