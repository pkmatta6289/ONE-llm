import torch
import torch.nn as nn
from .attention import Attention
from .feedforward import FeedForward
from .rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        
        # Self-Attention
        self.attention = Attention(args)
        
        # Feed-Forward Network
        # LLaMA uses a hidden dimension roughly 8/3 of the embedding dimension
        hidden_dim = int(4 * args.dim / 3) 
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=hidden_dim, multiple_of=32)
        
        # Pre-normalization layers
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        """
        x: (batch, seq_len, dim)
        freqs_cis: (seq_len, head_dim // 2) for RoPE
        mask: (seq_len, seq_len) causal mask
        """
        # 1. Pre-norm and Attention (with residual connection)
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        
        # 2. Pre-norm and Feed-Forward (with residual connection)
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out
