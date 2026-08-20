import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Unlike a standard FFN (which has 2 layers and a ReLU), SwiGLU has 3 layers.
    It takes the input, projects it in two ways, passes one through SiLU, multiplies them,
    and then projects back down.
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 32):
        super().__init__()
        
        # In LLaMA, hidden_dim is usually 8/3 * dim, rounded to a multiple of a parameter (e.g. 32)
        if hidden_dim is None:
            hidden_dim = int(2 * dim / 3)
            # Find the nearest multiple of `multiple_of`
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # "gate" projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # "up" projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # "down" projection

    def forward(self, x):
        # SwiGLU math: (SiLU(x @ w1) * (x @ w3)) @ w2
        # F.silu is the Swish activation function (x * sigmoid(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
