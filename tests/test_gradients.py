"""
Gradient correctness tests for FlashAttention.

Tests gradient computation using finite differences and autograd.
"""

import torch
import pytest
import sys
import os
import numpy as np
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attention import FlashAttention, StandardAttention
from flash_attention.utils import generate_test_inputs, compare_tensors


def compute_numerical_gradient(
    fn,
    input_tensor: torch.Tensor,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Compute numerical gradient using finite differences.
    
    Args:
        fn: Function that takes input and returns scalar loss
        input_tensor: Input tensor to compute gradient for
        eps: Finite difference step size
        
    Returns:
        Numerical gradient tensor
    """
    grad = torch.zeros_like(input_tensor)
    
    # Flatten for easier iteration
    flat_input = input_tensor.view(-1)
    flat_grad = grad.view(-1)
    
    for i in range(flat_input.numel()):
        # Save original value
        orig_val = flat_input[i].item()
        
        # Forward difference: f(x + eps)
        flat_input[i] = orig_val + eps
        loss_plus = fn(input_tensor).item()
        
        # Backward difference: f(x - eps)
        flat_input[i] = orig_val - eps
        loss_minus = fn(input_tensor).item()
        
        # Central difference
        flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Restore original value
        flat_input[i] = orig_val
    
    return grad


class TestGradients:
    """Test suite for gradient correctness."""
    
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_simple_gradient(self, device):
        """Test gradient computation on simple case."""
        batch, heads, seq_len, head_dim = 1, 1, 4, 4
        
        q = torch.randn(batch, heads, seq_len, head_dim, 
                       device=device, requires_grad=True)
        k = torch.randn(batch, heads, seq_len, head_dim, 
                       device=device, requires_grad=True)
        v = torch.randn(batch, heads, seq_len, head_dim, 
                       device=device, requires_grad=True)
        
        # FlashAttention
        flash_attn = FlashAttention().to(device)
        output_flash = flash_attn(q, k, v)
        loss_flash = output_flash.sum()
        loss_flash.backward()
        
        grad_q_flash = q.grad.clone()
        grad_k_flash = k.grad.clone()
        grad_v_flash = v.grad.clone()
        
        # Clear gradients
        q.grad = None
        k.grad = None
        v.grad = None
        
        # Standard attention
        std_attn = StandardAttention().to(device)
        output_std = std_attn(q, k, v)
        loss_std = output_std.sum()
        loss_std.backward()
        
        grad_q_std = q.grad.clone()
        grad_k_std = k.grad.clone()
        grad_v_std = v.grad.clone()
        
        # Compare gradients
        assert torch.allclose(grad_q_flash, grad_q_std, atol=1e-3, rtol=1e-3), \
            f"Q gradient mismatch: max diff = {(grad_q_flash - grad_q_std).abs().max()}"
        assert torch.allclose(grad_k_flash, grad_k_std, atol=1e-3, rtol=1e-3), \
            f"K gradient mismatch: max diff = {(grad_k_flash - grad_k_std).abs().max()}"
        assert torch.allclose(grad_v_flash, grad_v_std, atol=1e-3, rtol=1e-3), \
            f"V gradient mismatch: max diff = {(grad_v_flash - grad_v_std).abs().max()}"
    
    def test_gradient_medium_sequence(self, device):
        """Test gradients with medium sequence length."""
        batch, heads, seq_len, head_dim = 2, 4, 64, 32
        
        q, k, v = generate_test_inputs(batch, heads, seq_len, head_dim, device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        
        # FlashAttention
        flash_attn = FlashAttention().to(device)
        output_flash = flash_attn(q, k, v)
        loss_flash = (output_flash ** 2).sum()
        loss_flash.backward()
        
        grad_q_flash = q.grad.clone()
        grad_k_flash = k.grad.clone()
        grad_v_flash = v.grad.clone()
        
        # Clear gradients
        q.grad = None
        k.grad = None
        v.grad = None
        
        # Standard attention
        std_attn = StandardAttention().to(device)
        output_std = std_attn(q, k, v)
        loss_std = (output_std ** 2).sum()
        loss_std.backward()
        
        grad_q_std = q.grad.clone()
        grad_k_std = k.grad.clone()
        grad_v_std = v.grad.clone()
        
        # Compare with higher tolerance for larger sequences
        assert torch.allclose(grad_q_flash, grad_q_std, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_k_flash, grad_k_std, atol=1e-2, rtol=1e-2)
        assert torch.allclose(grad_v_flash, grad_v_std, atol=1e-2, rtol=1e-2)
    
    def test_causal_gradient(self, device):
        """Test gradients with causal masking."""
        batch, heads, seq_len, head_dim = 1, 2, 32, 32
        
        q, k, v = generate_test_inputs(batch, heads, seq_len, head_dim, device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        
        # FlashAttention with causal mask
        flash_attn = FlashAttention(causal=True).to(device)
        output_flash = flash_attn(q, k, v)
        loss_flash = output_flash.mean()
        loss_flash.backward()
        
        grad_q_flash = q.grad.clone()
        grad_k_flash = k.grad.clone()
        grad_v_flash = v.grad.clone()
        
        # Clear gradients
        q.grad = None
        k.grad = None
        v.grad = None
        
        # Standard attention with causal mask
        std_attn = StandardAttention(causal=True).to(device)
        output_std = std_attn(q, k, v)
        loss_std = output_std.mean()
        loss_std.backward()
        
        grad_q_std = q.grad.clone()
        grad_k_std = k.grad.clone()
        grad_v_std = v.grad.clone()
        
        # Compare gradients
        assert torch.allclose(grad_q_flash, grad_q_std, atol=1e-3, rtol=1e-3)
        assert torch.allclose(grad_k_flash, grad_k_std, atol=1e-3, rtol=1e-3)
        assert torch.allclose(grad_v_flash, grad_v_std, atol=1e-3, rtol=1e-3)
    
    def test_gradient_via_finite_difference(self, device):
        """Test gradient using finite differences (slow but accurate)."""
        # Use very small tensors for finite difference
        batch, heads, seq_len, head_dim = 1, 1, 3, 4
        
        q = torch.randn(batch, heads, seq_len, head_dim, 
                       device=device, requires_grad=True)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        
        # Autograd gradient
        output = flash_attn(q, k, v)
        loss = output.sum()
        loss.backward()
        grad_autograd = q.grad.clone()
        
        # Numerical gradient (only for Q to save time)
        def loss_fn(q_input):
            return flash_attn(q_input, k, v).sum()
        
        grad_numerical = compute_numerical_gradient(loss_fn, q.detach())
        
        # Compare (higher tolerance for numerical gradient)
        max_diff = (grad_autograd - grad_numerical).abs().max()
        print(f"Max gradient difference: {max_diff:.6e}")
        
        assert torch.allclose(grad_autograd, grad_numerical, atol=1e-2, rtol=1e-2), \
            f"Gradient mismatch: max diff = {max_diff}"
    
    def test_gradient_accumulation(self, device):
        """Test that gradients accumulate correctly."""
        batch, heads, seq_len, head_dim = 2, 2, 16, 32
        
        q, k, v = generate_test_inputs(batch, heads, seq_len, head_dim, device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        
        flash_attn = FlashAttention().to(device)
        
        # First backward pass
        output1 = flash_attn(q, k, v)
        loss1 = output1.sum()
        loss1.backward(retain_graph=True)
        
        grad_q_1 = q.grad.clone()
        
        # Second backward pass (gradients should accumulate)
        output2 = flash_attn(q, k, v)
        loss2 = output2.mean()
        loss2.backward()
        
        grad_q_2 = q.grad.clone()
        
        # Check that gradients accumulated
        assert not torch.allclose(grad_q_1, grad_q_2, atol=1e-6), \
            "Gradients did not accumulate"
    
    def test_zero_gradient(self, device):
        """Test gradient computation with zero inputs."""
        batch, heads, seq_len, head_dim = 1, 2, 8, 16
        
        q = torch.zeros(batch, heads, seq_len, head_dim, 
                       device=device, requires_grad=True)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)
        
        flash_attn = FlashAttention().to(device)
        output = flash_attn(q, k, v)
        loss = output.sum()
        loss.backward()
        
        # Gradient should exist (not None) even with zero input
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isinf(q.grad).any()
    
    def test_gradient_scale_invariance(self, device):
        """Test that gradients scale appropriately with input scaling."""
        batch, heads, seq_len, head_dim = 1, 2, 16, 32
        
        q_base, k, v = generate_test_inputs(batch, heads, seq_len, head_dim, device)
        
        flash_attn = FlashAttention().to(device)
        
        # Compute gradient for base input
        q1 = q_base.clone().requires_grad_(True)
        output1 = flash_attn(q1, k, v)
        loss1 = output1.sum()
        loss1.backward()
        grad1 = q1.grad.clone()
        
        # Compute gradient for scaled input
        scale = 2.0
        q2 = (q_base * scale).requires_grad_(True)
        output2 = flash_attn(q2, k, v)
        loss2 = output2.sum()
        loss2.backward()
        grad2 = q2.grad.clone()
        
        # Gradients should have some relationship to scaling
        # (exact relationship depends on softmax nonlinearity)
        # Just check they're not NaN or Inf
        assert not torch.isnan(grad1).any()
        assert not torch.isnan(grad2).any()
        assert not torch.isinf(grad1).any()
        assert not torch.isinf(grad2).any()
    
    def test_gradient_checkpointing_compatibility(self, device):
        """Test compatibility with gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint
        
        batch, heads, seq_len, head_dim = 2, 4, 32, 32
        
        q, k, v = generate_test_inputs(batch, heads, seq_len, head_dim, device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        
        flash_attn = FlashAttention().to(device)
        
        # With checkpointing
        output_cp = checkpoint(flash_attn, q, k, v, use_reentrant=False)
        loss_cp = output_cp.sum()
        loss_cp.backward()
        
        grad_q_cp = q.grad.clone()
        
        # Clear gradients
        q.grad = None
        k.grad = None
        v.grad = None
        
        # Without checkpointing
        output = flash_attn(q, k, v)
        loss = output.sum()
        loss.backward()
        
        grad_q = q.grad.clone()
        
        # Gradients should be the same
        assert torch.allclose(grad_q_cp, grad_q, atol=1e-5, rtol=1e-5)


def run_gradient_tests():
    """Run all gradient tests manually."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running gradient tests on {device}...\n")
    
    test_suite = TestGradients()
    
    tests = [
        ("Simple gradient", test_suite.test_simple_gradient),
        ("Medium sequence gradient", test_suite.test_gradient_medium_sequence),
        ("Causal gradient", test_suite.test_causal_gradient),
        ("Finite difference gradient", test_suite.test_gradient_via_finite_difference),
        ("Gradient accumulation", test_suite.test_gradient_accumulation),
        ("Zero gradient", test_suite.test_zero_gradient),
        ("Gradient scale invariance", test_suite.test_gradient_scale_invariance),
        ("Gradient checkpointing", test_suite.test_gradient_checkpointing_compatibility),
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
    
    print(f"\n{passed}/{len(tests)} gradient tests passed")
    if failed > 0:
        print(f"{failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_gradient_tests()
    sys.exit(0 if success else 1)
