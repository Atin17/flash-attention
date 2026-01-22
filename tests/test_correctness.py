"""
Correctness tests for FlashAttention implementation.

Verifies numerical correctness against PyTorch's standard attention.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attention import FlashAttention, StandardAttention


class TestFlashAttentionCorrectness:
    """Test suite for FlashAttention correctness."""
    
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_small_sequence(self, device):
        """Test with small sequence length."""
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        std_attn = StandardAttention().to(device)
        
        output_flash = flash_attn(q, k, v)
        output_std = std_attn(q, k, v)
        
        assert torch.allclose(output_flash, output_std, atol=1e-3, rtol=1e-3), \
            f"Max diff: {(output_flash - output_std).abs().max()}"
    
    def test_medium_sequence(self, device):
        """Test with medium sequence length."""
        batch, heads, seq_len, head_dim = 2, 8, 512, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        std_attn = StandardAttention().to(device)
        
        output_flash = flash_attn(q, k, v)
        output_std = std_attn(q, k, v)
        
        assert torch.allclose(output_flash, output_std, atol=1e-3, rtol=1e-3)
    
    def test_causal_attention(self, device):
        """Test causal masking."""
        batch, heads, seq_len, head_dim = 1, 4, 64, 32
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention(causal=True).to(device)
        std_attn = StandardAttention(causal=True).to(device)
        
        output_flash = flash_attn(q, k, v)
        output_std = std_attn(q, k, v)
        
        assert torch.allclose(output_flash, output_std, atol=1e-3, rtol=1e-3)
    
    def test_single_head(self, device):
        """Test with single attention head."""
        batch, heads, seq_len, head_dim = 2, 1, 128, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        std_attn = StandardAttention().to(device)
        
        output_flash = flash_attn(q, k, v)
        output_std = std_attn(q, k, v)
        
        assert torch.allclose(output_flash, output_std, atol=1e-3, rtol=1e-3)
    
    def test_large_head_dim(self, device):
        """Test with larger head dimension."""
        batch, heads, seq_len, head_dim = 1, 4, 64, 128
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        std_attn = StandardAttention().to(device)
        
        output_flash = flash_attn(q, k, v)
        output_std = std_attn(q, k, v)
        
        assert torch.allclose(output_flash, output_std, atol=1e-3, rtol=1e-3)
    
    def test_batch_independence(self, device):
        """Test that batch elements are processed independently."""
        batch, heads, seq_len, head_dim = 4, 4, 64, 64
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        
        # Process all batches together
        output_batched = flash_attn(q, k, v)
        
        # Process each batch separately
        outputs_separate = []
        for i in range(batch):
            out = flash_attn(q[i:i+1], k[i:i+1], v[i:i+1])
            outputs_separate.append(out)
        output_separate = torch.cat(outputs_separate, dim=0)
        
        assert torch.allclose(output_batched, output_separate, atol=1e-5)
    
    def test_attention_pattern(self, device):
        """Test that attention focuses on relevant positions."""
        batch, heads, seq_len, head_dim = 1, 1, 10, 8
        
        # Create Q that should attend strongly to position 5
        q = torch.zeros(batch, heads, seq_len, head_dim, device=device)
        q[0, 0, 0, :] = 1.0
        
        # Create K with strong value at position 5
        k = torch.zeros(batch, heads, seq_len, head_dim, device=device)
        k[0, 0, 5, :] = 1.0
        
        # Create V with unique value at position 5
        v = torch.zeros(batch, heads, seq_len, head_dim, device=device)
        v[0, 0, 5, :] = 10.0
        
        flash_attn = FlashAttention().to(device)
        output = flash_attn(q, k, v)
        
        # Output should be close to V[5] due to strong attention
        expected_value = 10.0 / seq_len  # Softmax spreads attention
        assert output[0, 0, 0, 0] > 0.5, \
            "Attention should focus on position with matching key"
    
    def test_zero_input(self, device):
        """Test with zero inputs."""
        batch, heads, seq_len, head_dim = 1, 2, 32, 64
        
        q = torch.zeros(batch, heads, seq_len, head_dim, device=device)
        k = torch.zeros(batch, heads, seq_len, head_dim, device=device)
        v = torch.zeros(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        output = flash_attn(q, k, v)
        
        # Output should be close to zero (uniform attention on zero values)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)
    
    def test_numerical_stability(self, device):
        """Test numerical stability with extreme values."""
        batch, heads, seq_len, head_dim = 1, 2, 32, 64
        
        # Large positive values
        q = torch.ones(batch, heads, seq_len, head_dim, device=device) * 100
        k = torch.ones(batch, heads, seq_len, head_dim, device=device) * 100
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        output = flash_attn(q, k, v)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


def run_tests():
    """Run all tests manually without pytest."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on {device}...")
    
    test_suite = TestFlashAttentionCorrectness()
    
    tests = [
        ("Small sequence", test_suite.test_small_sequence),
        ("Medium sequence", test_suite.test_medium_sequence),
        ("Causal attention", test_suite.test_causal_attention),
        ("Single head", test_suite.test_single_head),
        ("Large head dim", test_suite.test_large_head_dim),
        ("Batch independence", test_suite.test_batch_independence),
        ("Attention pattern", test_suite.test_attention_pattern),
        ("Zero input", test_suite.test_zero_input),
        ("Numerical stability", test_suite.test_numerical_stability),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func(device)
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {str(e)}")
            failed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"{failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
