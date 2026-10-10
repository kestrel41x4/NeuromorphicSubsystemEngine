import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nse.model.rope import apply_rope

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Rotary Position Embeddings.
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections for Q, K, V in a single fused linear layer for memory efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask: tokens cannot attend to subsequent positions
        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Fused linear projection followed by split into Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape to [batch, seq_len, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Transpose for scaled dot-product: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.mask[:seq_len, :seq_len]
        attn_weights = F.softmax(scores, dim=-1)

        # Context aggregation
        context = torch.matmul(attn_weights, v)
        # Reshape back to [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)