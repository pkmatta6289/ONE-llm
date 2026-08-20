import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .rope import apply_rotary_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key/value heads for Grouped Query Attention (GQA).
    If n_rep is 1, this does nothing (standard MHA).
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    # Expand and reshape to copy the KV heads
    x = x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim)
    return x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    """
    Grouped Query Attention (GQA) with KV caching.
    """
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        bsz, seqlen, _ = x.shape
        
        # 1. Linear projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 2. Reshape to (batch, seq, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # 3. Apply Rotary Position Embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # 4. Repeat KV heads for GQA (n_kv_heads -> n_heads)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        
        # 5. Transpose to (batch, n_heads, seq, head_dim) for batch matrix multiplication
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 6. Scaled Dot-Product Attention
        # scores = Q @ K.T / sqrt(head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Mask is (seqlen, seqlen), broadcast it
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # output = scores @ V
        output = torch.matmul(scores, xv)
        
        # 7. Restore shape: (batch, seq, n_heads, head_dim) -> (batch, seq, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # 8. Output projection
        return self.wo(output)
