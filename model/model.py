import torch
import torch.nn as nn
from .config import ModelConfig
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .rope import precompute_freqs_cis

class OneLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        
        # Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer Blocks
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config))
            
        # Final Norm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # Output Projection (using weight tying with tok_embeddings)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, 
            config.max_seq_len * 2 # Safety margin
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        bsz, seqlen = tokens.shape
        
        # 1. Token Embeddings
        h = self.tok_embeddings(tokens)
        
        # 2. Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)
        
        # 3. Create causal mask (triangle)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1)
        
        # 4. Pass through all transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
            
        # 5. Final norm
        h = self.norm(h)
        
        # 6. Output projection to vocab size
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            # Shift the targets so token N predicts token N+1
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), 
                targets.view(-1)
            )
            
        return logits, loss

    def get_num_params(self):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        # Subtract the tied weights
        n_params -= self.tok_embeddings.weight.numel()
        return n_params
