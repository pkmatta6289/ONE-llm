"""
ONE-LLM: Tokenizer — Encode, Decode, and Manage Vocabulary
============================================================

YOUR DAY 1 TASK (Part 2): Build the tokenizer that USES the trained BPE merges.

TRAINING vs USING:
    - bpe.py TRAINS the tokenizer (finds the merge rules)
    - This file USES those merge rules to encode/decode text

ENCODING (text → token IDs):
    1. Pre-tokenize text with the regex pattern (same as training)
    2. For each chunk, convert to bytes
    3. Apply merges in PRIORITY ORDER (merge rule 0 first, then 1, then 2, ...)
       - For each merge (pair → new_id), scan the chunk and merge all occurrences
       - This is the greedy application of merges
    4. Flatten all chunks into a single list of token IDs

DECODING (token IDs → text):
    1. For each token ID, look up its byte sequence in the vocab
    2. Concatenate all byte sequences
    3. Decode bytes to text with errors='replace' (handle invalid UTF-8 gracefully)

SPECIAL TOKENS:
    - <|bos|> (beginning of sequence): prepended to every input
    - <|eos|> (end of sequence): marks the end of a document/response
    - <|pad|> (padding): used to pad sequences to equal length in batches
    These get IDs AFTER the BPE vocab: if you have 256 + num_merges regular tokens,
    special tokens start at 256 + num_merges.
"""

import json
import regex

from .bpe import GPT2_SPLIT_PATTERN, get_pair_counts, merge


class Tokenizer:
    """
    A byte-level BPE tokenizer.

    Attributes:
        merges: dict[(int, int), int] — learned merge rules from training
        vocab: dict[int, bytes] — token ID to byte sequence mapping
        special_tokens: dict[str, int] — special token strings to IDs
        inverse_special: dict[int, str] — special token IDs to strings
    """

    # The special tokens we support
    SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]

    def __init__(self, merges=None, vocab=None):
        """
        Initialize the tokenizer.

        If merges and vocab are provided (from training), set them up.
        Otherwise, create an empty tokenizer (call load() to populate).

        TODO: Implement this.
            1. Store merges and vocab
            2. Compile the regex pattern
            3. Register special tokens:
               - Start IDs at len(vocab) (or 256 if vocab is None)
               - Create self.special_tokens = {"<|bos|>": id, ...}
               - Create self.inverse_special = {id: "<|bos|>", ...}
               - Add special tokens to vocab: vocab[id] = token_string.encode('utf-8')
        """
        raise NotImplementedError("YOUR TURN: Initialize the tokenizer")

    def encode(self, text, add_bos=False, add_eos=False):
        """
        Encode text into a list of token IDs.

        Args:
            text: str — input text to tokenize
            add_bos: bool — if True, prepend <|bos|> token
            add_eos: bool — if True, append <|eos|> token

        Returns:
            list[int] — token IDs

        ALGORITHM:
        1. Handle special tokens first: check if any special token strings
           appear in the text. If so, split around them and handle them separately.
           (For simplicity, you can skip this initially and assume no special
           tokens appear in the raw text.)
        2. Pre-tokenize: split text using the regex pattern
        3. For each chunk:
           a. Convert to bytes: list(chunk.encode('utf-8'))
           b. Apply merges in order: iterate through self.merges (which is ordered
              by priority — first merge learned = highest priority)
              - For each (pair, new_id), call merge(chunk_ids, pair, new_id)
           c. Collect resulting token IDs
        4. Prepend BOS / append EOS if requested
        5. Return the full list of token IDs

        TODO: Implement encoding.

        HINT: The merges dict should be iterated in insertion order.
        In Python 3.7+, dicts maintain insertion order.
        """
        raise NotImplementedError("YOUR TURN: Implement encoding")

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into text.

        Args:
            token_ids: list[int] — token IDs to decode

        Returns:
            str — decoded text

        ALGORITHM:
        1. For each token ID:
           a. If it's a special token (in self.inverse_special), get the string
           b. Otherwise, look up vocab[token_id] to get bytes
        2. Concatenate all bytes
        3. Decode to string: bytes_sequence.decode('utf-8', errors='replace')

        TODO: Implement decoding.
        """
        raise NotImplementedError("YOUR TURN: Implement decoding")

    @property
    def vocab_size(self):
        """Return the total vocabulary size including special tokens."""
        return len(self.vocab)

    @property
    def bos_id(self):
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self):
        return self.special_tokens["<|eos|>"]

    @property
    def pad_id(self):
        return self.special_tokens["<|pad|>"]

    def save(self, path):
        """
        Save the tokenizer to a JSON file.

        Save:
            - merges: convert tuple keys to strings like "97,98"
            - vocab: convert bytes values to lists of ints
            - special_tokens: save as-is

        TODO: Implement saving.

        HINT: JSON can't have tuple keys, so convert (97, 98) → "97,98"
        HINT: JSON can't have bytes values, so convert b'hello' → [104, 101, ...]
        """
        raise NotImplementedError("YOUR TURN: Implement save")

    def load(self, path):
        """
        Load a tokenizer from a JSON file.

        Reverse the save() process:
            - Convert "97,98" keys back to (97, 98) tuples
            - Convert int lists back to bytes
            - Reconstruct special tokens

        TODO: Implement loading.
        """
        raise NotImplementedError("YOUR TURN: Implement load")
