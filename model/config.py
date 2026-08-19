from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for the ONE-LLM model.
    Default values describe a small ~124M parameter model (similar to GPT-2 Small),
    but using modern architecture (GQA, RoPE, RMSNorm).
    """
    dim: int = 768               # Hidden dimension size
    n_layers: int = 12           # Number of transformer blocks
    n_heads: int = 12            # Number of query attention heads
    n_kv_heads: int = 4          # Number of key/value heads for GQA (must divide n_heads)
    vocab_size: int = 50260      # 50,257 GPT-2 + 3 special tokens
    norm_eps: float = 1e-5       # Epsilon for RMSNorm to prevent division by zero
    max_seq_len: int = 1024      # Maximum context length
