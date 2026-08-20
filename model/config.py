from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for the ONE-LLM model.
    Default values describe a Micro ~20M parameter model for fast local training.
    """
    dim: int = 288               # Hidden dimension size
    n_layers: int = 6            # Number of transformer blocks
    n_heads: int = 6             # Number of query attention heads
    n_kv_heads: int = 2          # Number of key/value heads for GQA (must divide n_heads)
    vocab_size: int = 50260      # 50,257 GPT-2 + 3 special tokens
    norm_eps: float = 1e-5       # Epsilon for RMSNorm to prevent division by zero
    max_seq_len: int = 256       # Maximum context length for testing
