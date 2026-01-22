"""
Memory benchmarking for FlashAttention vs Standard Attention.

Measures peak memory usage and maximum achievable batch size.
"""

import torch
import torch.cuda as cuda
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attention import FlashAttention, StandardAttention


def measure_memory(
    attention_module: torch.nn.Module,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str = 'cuda'
) -> Tuple[float, bool]:
    """
    Measure peak memory usage for a single forward pass.
    
    Returns:
        (peak_memory_mb, success)
    """
    if not torch.cuda.is_available():
        return 0.0, False
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    try:
        # Create input tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        # Forward pass
        with torch.no_grad():
            output = attention_module(q, k, v)
            
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        
        # Cleanup
        del q, k, v, output
        torch.cuda.empty_cache()
        
        return peak_memory, True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return 0.0, False
        else:
            raise e


def find_max_batch_size(
    attention_module: torch.nn.Module,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    max_batch: int = 128,
    device: str = 'cuda'
) -> int:
    """
    Binary search to find maximum batch size that fits in memory.
    """
    left, right = 1, max_batch
    max_successful = 0
    
    while left <= right:
        mid = (left + right) // 2
        _, success = measure_memory(attention_module, mid, num_heads, seq_len, head_dim, device)
        
        if success:
            max_successful = mid
            left = mid + 1
        else:
            right = mid - 1
            
    return max_successful


def benchmark_memory_usage():
    """Comprehensive memory benchmarking."""
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory benchmark.")
        return
        
    device = 'cuda'
    num_heads = 8
    head_dim = 64
    seq_lengths = [512, 1024, 2048, 4096]
    batch_size = 4
    
    print("=" * 80)
    print("MEMORY USAGE COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration: {num_heads} heads, {head_dim} head_dim, batch_size={batch_size}\n")
    print(f"{'Seq Len':<10} {'Standard (MB)':<15} {'Flash (MB)':<15} {'Reduction':<15}")
    print("-" * 80)
    
    results = []
    
    for seq_len in seq_lengths:
        # Measure standard attention
        std_attn = StandardAttention().to(device)
        std_memory, std_success = measure_memory(
            std_attn, batch_size, num_heads, seq_len, head_dim, device
        )
        
        # Measure flash attention
        flash_attn = FlashAttention().to(device)
        flash_memory, flash_success = measure_memory(
            flash_attn, batch_size, num_heads, seq_len, head_dim, device
        )
        
        if std_success and flash_success:
            reduction = ((std_memory - flash_memory) / std_memory) * 100
            results.append({
                'seq_len': seq_len,
                'std_memory': std_memory,
                'flash_memory': flash_memory,
                'reduction': reduction
            })
            print(f"{seq_len:<10} {std_memory:>12.2f}   {flash_memory:>12.2f}   "
                  f"{reduction:>12.1f}%")
        elif flash_success:
            print(f"{seq_len:<10} {'OOM':<15} {flash_memory:>12.2f}   {'N/A':<15}")
        else:
            print(f"{seq_len:<10} {'OOM':<15} {'OOM':<15} {'N/A':<15}")
    
    print("\n" + "=" * 80)
    print("MAXIMUM BATCH SIZE COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration: {num_heads} heads, {head_dim} head_dim\n")
    print(f"{'Seq Len':<10} {'Standard':<15} {'Flash':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for seq_len in [512, 1024]:
        std_attn = StandardAttention().to(device)
        std_max_batch = find_max_batch_size(std_attn, num_heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        flash_max_batch = find_max_batch_size(flash_attn, num_heads, seq_len, head_dim, device=device)
        
        if std_max_batch > 0:
            improvement = flash_max_batch / std_max_batch
            print(f"{seq_len:<10} {std_max_batch:<15} {flash_max_batch:<15} {improvement:.2f}x")
        else:
            print(f"{seq_len:<10} {'0':<15} {flash_max_batch:<15} {'N/A':<15}")
    
    # Summary
    if results:
        avg_reduction = np.mean([r['reduction'] for r in results])
        print("\n" + "=" * 80)
        print(f"SUMMARY: Average memory reduction: {avg_reduction:.1f}%")
        print("=" * 80)
    
    return results


def theoretical_memory_analysis():
    """Analyze theoretical memory complexity."""
    
    print("\n" + "=" * 80)
    print("THEORETICAL MEMORY ANALYSIS")
    print("=" * 80)
    
    seq_lengths = [512, 1024, 2048, 4096]
    batch = 1
    heads = 8
    head_dim = 64
    bytes_per_float = 4
    
    print(f"\nAssumptions: batch={batch}, heads={heads}, head_dim={head_dim}, "
          f"bytes_per_float={bytes_per_float}\n")
    print(f"{'Seq Len':<10} {'Q,K,V (MB)':<15} {'Attn Matrix (MB)':<20} {'Total Std (MB)':<20}")
    print("-" * 80)
    
    for seq_len in seq_lengths:
        # Memory for Q, K, V tensors
        qkv_memory = 3 * batch * heads * seq_len * head_dim * bytes_per_float / (1024**2)
        
        # Memory for attention matrix (N x N)
        attn_matrix_memory = batch * heads * seq_len * seq_len * bytes_per_float / (1024**2)
        
        # Total for standard attention
        total_std = qkv_memory + attn_matrix_memory
        
        print(f"{seq_len:<10} {qkv_memory:>12.2f}   {attn_matrix_memory:>17.2f}   {total_std:>17.2f}")
    
    print("\nFlashAttention avoids materializing the attention matrix,")
    print("reducing memory from O(N²) to O(N).")
    print("=" * 80)


if __name__ == "__main__":
    print("Starting Memory Benchmarks...\n")
    
    theoretical_memory_analysis()
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
        benchmark_memory_usage()
    else:
        print("\nCUDA not available. Only showing theoretical analysis.")
