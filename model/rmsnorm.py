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
        
        # TODO: Define a learnable parameter 'weight' (often called gamma)
        # It should be an nn.Parameter of shape (dim,) initialized to ones.
        raise NotImplementedError("YOUR TURN: Define self.weight")

    def _norm(self, x):
        """
        Compute the RMSNorm of x without applying the learnable weight.
        
        Math: x / sqrt(mean(x^2) + eps)
        """
        # TODO: 
        # 1. Square the input x
        # 2. Compute the mean over the last dimension (keepdim=True)
        # 3. Add self.eps for numerical stability
        # 4. Take the inverse square root (torch.rsqrt is numerically better than 1/sqrt)
        # 5. Multiply x by this scaling factor
        raise NotImplementedError("YOUR TURN: Implement _norm()")

    def forward(self, x):
        """
        Apply RMSNorm and scale by the learnable weight.
        """
        # TODO: Return self._norm(x) multiplied by self.weight
        raise NotImplementedError("YOUR TURN: Implement forward()")
