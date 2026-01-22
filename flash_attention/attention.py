import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import flash_attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension not built. Falling back to PyTorch implementation.")


class FlashAttention(nn.Module):
    """
    FlashAttention: Memory-efficient attention using tiled computation.
    
    Reduces memory from O(N^2) to O(N) by computing attention in blocks
    and fusing operations to avoid materializing the full attention matrix.
    
    Args:
        causal: Whether to apply causal masking (for autoregressive models)
    """
    
    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal
        
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        causal: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Forward pass of FlashAttention.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            causal: Override instance causal setting
            
        Returns:
            Output tensor [batch, heads, seq_len, head_dim]
        """
        if causal is None:
            causal = self.causal
            
        # Check if CUDA extension is available
        if CUDA_AVAILABLE and q.is_cuda:
            return flash_attention_cuda.forward(q, k, v, causal)
        else:
            # Fallback to PyTorch implementation
            return self._pytorch_attention(q, k, v, causal)
    
    def _pytorch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        causal: bool
    ) -> torch.Tensor:
        """PyTorch fallback implementation for testing/comparison."""
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
        
        # Softmax and attention-weighted sum
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        return output


class StandardAttention(nn.Module):
    """
    Standard attention implementation for benchmarking.
    Materializes the full attention matrix (high memory usage).
    """
    
    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: Optional[bool] = None
    ) -> torch.Tensor:
        """Standard attention: O(N^2) memory."""
        if causal is None:
            causal = self.causal
            
        batch, heads, seq_len, head_dim = q.shape
        scale = head_dim ** -0.5
        
        # This materializes the full NxN attention matrix
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output


class MultiHeadFlashAttention(nn.Module):
    """
    Multi-head attention using FlashAttention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        causal: Whether to use causal masking
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attention = FlashAttention(causal=causal)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape to [batch, heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply FlashAttention
        attn_output = self.attention(q, k, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


def test_attention():
    """Quick test to verify attention works."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)
    
    # Test FlashAttention
    flash_attn = FlashAttention().to(device)
    output_flash = flash_attn(q, k, v)
    
    # Test StandardAttention
    std_attn = StandardAttention().to(device)
    output_std = std_attn(q, k, v)
    
    # Check they produce similar results
    diff = (output_flash - output_std).abs().max()
    print(f"Max difference: {diff:.6f}")
    print(f"Are close? {torch.allclose(output_flash, output_std, atol=1e-3)}")
    
    return output_flash, output_std


if __name__ == "__main__":
    test_attention()
