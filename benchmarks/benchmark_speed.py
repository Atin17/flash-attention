"""
Speed benchmarking for FlashAttention vs Standard Attention.

Measures forward pass latency and throughput.
"""

import torch
import time
import numpy as np
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attention import FlashAttention, StandardAttention


def benchmark_forward_pass(
    attention_module: torch.nn.Module,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark forward pass speed.
    
    Returns:
        Dictionary with timing statistics
    """
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = attention_module(q, k, v)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.no_grad():
                _ = attention_module(q, k, v)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.time()
            with torch.no_grad():
                _ = attention_module(q, k, v)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }


def benchmark_speed():
    """Comprehensive speed benchmarking."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Warning: Running on CPU. Results may not be representative.")
    
    num_heads = 8
    head_dim = 64
    batch_size = 4
    seq_lengths = [512, 1024, 2048, 4096]
    
    print("=" * 80)
    print("SPEED BENCHMARK - Forward Pass")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Configuration: batch={batch_size}, heads={num_heads}, head_dim={head_dim}\n")
    
    print(f"{'Seq Len':<10} {'Standard (ms)':<15} {'Flash (ms)':<15} {'Speedup':<15}")
    print("-" * 80)
    
    results = []
    
    for seq_len in seq_lengths:
        try:
            # Benchmark standard attention
            std_attn = StandardAttention().to(device)
            std_stats = benchmark_forward_pass(
                std_attn, batch_size, num_heads, seq_len, head_dim, device=device
            )
            
            # Benchmark flash attention
            flash_attn = FlashAttention().to(device)
            flash_stats = benchmark_forward_pass(
                flash_attn, batch_size, num_heads, seq_len, head_dim, device=device
            )
            
            speedup = std_stats['mean'] / flash_stats['mean']
            
            results.append({
                'seq_len': seq_len,
                'std_mean': std_stats['mean'],
                'flash_mean': flash_stats['mean'],
                'speedup': speedup
            })
            
            print(f"{seq_len:<10} {std_stats['mean']:>12.2f}   {flash_stats['mean']:>12.2f}   "
                  f"{speedup:>12.2f}x")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{seq_len:<10} {'OOM':<15} {'?':<15} {'N/A':<15}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    # Detailed statistics for selected sequence length
    if results:
        print("\n" + "=" * 80)
        print(f"DETAILED STATISTICS (seq_len={results[0]['seq_len']})")
        print("=" * 80)
        
        seq_len = results[0]['seq_len']
        std_attn = StandardAttention().to(device)
        std_stats = benchmark_forward_pass(
            std_attn, batch_size, num_heads, seq_len, head_dim, device=device
        )
        
        flash_attn = FlashAttention().to(device)
        flash_stats = benchmark_forward_pass(
            flash_attn, batch_size, num_heads, seq_len, head_dim, device=device
        )
        
        print("\nStandard Attention:")
        print(f"  Mean:   {std_stats['mean']:.2f} ms")
        print(f"  Median: {std_stats['median']:.2f} ms")
        print(f"  Std:    {std_stats['std']:.2f} ms")
        print(f"  Min:    {std_stats['min']:.2f} ms")
        print(f"  Max:    {std_stats['max']:.2f} ms")
        print(f"  P95:    {std_stats['p95']:.2f} ms")
        print(f"  P99:    {std_stats['p99']:.2f} ms")
        
        print("\nFlashAttention:")
        print(f"  Mean:   {flash_stats['mean']:.2f} ms")
        print(f"  Median: {flash_stats['median']:.2f} ms")
        print(f"  Std:    {flash_stats['std']:.2f} ms")
        print(f"  Min:    {flash_stats['min']:.2f} ms")
        print(f"  Max:    {flash_stats['max']:.2f} ms")
        print(f"  P95:    {flash_stats['p95']:.2f} ms")
        print(f"  P99:    {flash_stats['p99']:.2f} ms")
    
    # Throughput analysis
    print("\n" + "=" * 80)
    print("THROUGHPUT ANALYSIS")
    print("=" * 80)
    print(f"\n{'Seq Len':<10} {'Std Tokens/s':<20} {'Flash Tokens/s':<20} {'Improvement':<15}")
    print("-" * 80)
    
    for result in results:
        seq_len = result['seq_len']
        tokens_per_batch = batch_size * seq_len
        
        std_throughput = tokens_per_batch / (result['std_mean'] / 1000)
        flash_throughput = tokens_per_batch / (result['flash_mean'] / 1000)
        improvement = flash_throughput / std_throughput
        
        print(f"{seq_len:<10} {std_throughput:>17.0f}   {flash_throughput:>17.0f}   "
              f"{improvement:>12.2f}x")
    
    print("=" * 80)
    
    return results


def benchmark_scaling():
    """Benchmark how performance scales with batch size."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("\nSkipping scaling benchmark on CPU.")
        return
    
    print("\n" + "=" * 80)
    print("BATCH SIZE SCALING")
    print("=" * 80)
    
    num_heads = 8
    head_dim = 64
    seq_len = 1024
    batch_sizes = [1, 2, 4, 8, 16]
    
    print(f"\nConfiguration: seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}\n")
    print(f"{'Batch':<10} {'Standard (ms)':<15} {'Flash (ms)':<15} {'Speedup':<15}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        try:
            std_attn = StandardAttention().to(device)
            std_stats = benchmark_forward_pass(
                std_attn, batch_size, num_heads, seq_len, head_dim, 
                num_warmup=5, num_iterations=50, device=device
            )
            
            flash_attn = FlashAttention().to(device)
            flash_stats = benchmark_forward_pass(
                flash_attn, batch_size, num_heads, seq_len, head_dim,
                num_warmup=5, num_iterations=50, device=device
            )
            
            speedup = std_stats['mean'] / flash_stats['mean']
            print(f"{batch_size:<10} {std_stats['mean']:>12.2f}   {flash_stats['mean']:>12.2f}   "
                  f"{speedup:>12.2f}x")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:<10} {'OOM':<15} {'?':<15} {'N/A':<15}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    print("=" * 80)


if __name__ == "__main__":
    print("Starting Speed Benchmarks...\n")
    
    benchmark_speed()
    benchmark_scaling()
    
    print("\n✓ Benchmarking complete!")
