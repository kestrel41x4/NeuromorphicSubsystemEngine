import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit feed-forward network.
    """
    def __init__(self, d_model: int = 512, hidden_dim: int = 1365):
        super().__init__()
        # In SwiGLU, 1365 hidden_dim gives ~same parameters as 2048 in standard MLP
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path filtered by SiLU (Swish) multiplied element-wise by up-projection
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))