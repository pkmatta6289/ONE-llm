import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    
    Standard LayerNorm centers the data (subtracts the mean) and scales it (divides by variance).
    RMSNorm simplifies this by only scaling the data (divides by Root Mean Square).
    It works just as well for LLMs but is computationally cheaper.
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
        # Learnable parameter 'weight' (gamma) of shape (dim,) initialized to ones.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Compute the RMSNorm of x without applying the learnable weight.
        Math: x / sqrt(mean(x^2) + eps)
        """
        # 1. Square the input x
        # 2. Compute the mean over the last dimension (keepdim=True)
        # 3. Add self.eps for numerical stability
        # 4. Take the inverse square root (torch.rsqrt)
        # 5. Multiply x by this scaling factor
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply RMSNorm and scale by the learnable weight.
        """
        return self._norm(x) * self.weight
