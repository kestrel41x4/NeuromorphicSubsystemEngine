import torch
import torch.nn as nn
from dataclasses import dataclass
from nse.model.norm import RMSNorm
from nse.model.rope import precompute_rope_frequencies
from nse.model.attention import CausalSelfAttention
from nse.model.mlp import SwiGLU

@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    hidden_dim: int = 1365
    max_seq_len: int = 512

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.max_seq_len)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.hidden_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class NSEModel(nn.Module):
    """
    Neuromorphic Subsystem Engine - ~45 Million Parameter Architecture.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights to save ~8.4M parameters
        self.tok_embeddings.weight = self.lm_head.weight

        # Precompute RoPE table
        head_dim = config.d_model // config.n_heads
        freqs_cis = precompute_rope_frequencies(head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        x = self.tok_embeddings(input_ids)
        
        for block in self.blocks:
            x = block(x, self.freqs_cis)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten across batch and sequence dimensions for cross-entropy
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())