"""
ONE-LLM: Pretrained Tokenizer Wrapper
=======================================

Wraps tiktoken (GPT-2's tokenizer) to provide the same interface
as our hand-built tokenizer. This gives us a production-quality
50,257-token vocabulary for actual model training.

Your hand-built tokenizer in bpe.py and tokenizer.py is kept for learning.
This file is what we'll actually use for pretraining.

Usage:
    from tokenizer.pretrained import PretrainedTokenizer
    tok = PretrainedTokenizer()
    ids = tok.encode("Hello, world!")   # [15496, 11, 995, 0]
    text = tok.decode(ids)              # "Hello, world!"
"""

import tiktoken


class PretrainedTokenizer:
    """
    Wrapper around tiktoken's GPT-2 tokenizer.

    Provides the same interface as our hand-built Tokenizer class:
    encode(), decode(), vocab_size, bos_id, eos_id, pad_id.
    """

    def __init__(self):
        # Load GPT-2's BPE tokenizer via tiktoken
        self.enc = tiktoken.get_encoding("gpt2")

        # GPT-2's vocab has 50,257 tokens. We add 3 special tokens at the end.
        # GPT-2 already has <|endoftext|> at ID 50256, but we define our own
        # special tokens for clarity.
        self._vocab_size = self.enc.n_vocab  # 50257

        # Define special token IDs (after the existing vocab)
        self.special_tokens = {
            "<|bos|>": self._vocab_size,      # 50257
            "<|eos|>": self._vocab_size + 1,  # 50258
            "<|pad|>": self._vocab_size + 2,  # 50259
        }
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}

    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token IDs."""
        ids = self.enc.encode(text, allowed_special=set())

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        # Filter out our custom special tokens before passing to tiktoken
        regular_ids = [id for id in token_ids if id not in self.inverse_special]
        return self.enc.decode(regular_ids)

    @property
    def vocab_size(self):
        """Total vocab size including special tokens."""
        return self._vocab_size + len(self.special_tokens)  # 50260

    @property
    def bos_id(self):
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self):
        return self.special_tokens["<|eos|>"]

    @property
    def pad_id(self):
        return self.special_tokens["<|pad|>"]
