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
        """
        self.merges = merges if merges is not None else {}
        self.vocab = vocab if vocab is not None else {}
        self.pattern = regex.compile(GPT2_SPLIT_PATTERN)

        # Register special tokens — IDs start after the BPE vocab
        base_id = max(len(self.vocab), 256)
        self.special_tokens = {}
        self.inverse_special = {}
        for i, token_str in enumerate(self.SPECIAL_TOKENS):
            token_id = base_id + i
            self.special_tokens[token_str] = token_id
            self.inverse_special[token_id] = token_str
            self.vocab[token_id] = token_str.encode("utf-8")

    def encode(self, text, add_bos=False, add_eos=False):
        """
        Encode text into a list of token IDs.

        Args:
            text: str — input text to tokenize
            add_bos: bool — if True, prepend <|bos|> token
            add_eos: bool — if True, append <|eos|> token

        Returns:
            list[int] — token IDs
        """
        # Step 1: Pre-tokenize text into chunks using the regex
        chunks = self.pattern.findall(text)

        # Step 2: For each chunk, convert to bytes and apply merges
        tokens = []
        for chunk in chunks:
            # Convert string to list of byte values (0-255)
            token = list(chunk.encode("utf-8"))

            # Apply every merge rule in priority order
            for pair, new_id in self.merges.items():
                token = merge(token, pair, new_id)

            # Extend (not append!) to keep a flat list of IDs
            tokens.extend(token)

        # Step 3: Add BOS/EOS if requested
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into text.

        Args:
            token_ids: list[int] — token IDs to decode

        Returns:
            str — decoded text
        """
        byte_pieces = []
        for token_id in token_ids:
            if token_id in self.inverse_special:
                # Special token — encode its string to bytes
                byte_pieces.append(self.inverse_special[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                # Regular token — look up its byte sequence
                byte_pieces.append(self.vocab[token_id])
            else:
                byte_pieces.append(b"?")

        # Join all bytes and decode to string
        return b"".join(byte_pieces).decode("utf-8", errors="replace")

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
        """
        # Convert merges: tuple keys → string keys (JSON can't have tuple keys)
        merges_out = {}
        for (a, b), new_id in self.merges.items():
            merges_out[f"{a},{b}"] = new_id

        # Convert vocab: bytes values → int lists (JSON can't have bytes)
        vocab_out = {}
        for token_id, byte_seq in self.vocab.items():
            if token_id in self.inverse_special:
                continue  # special tokens get reconstructed on load
            vocab_out[str(token_id)] = list(byte_seq)

        data = {
            "merges": merges_out,
            "vocab": vocab_out,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        """
        Load a tokenizer from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct merges: "97,98" → (97, 98)
        self.merges = {}
        for key_str, new_id in data["merges"].items():
            a, b = key_str.split(",")
            self.merges[(int(a), int(b))] = new_id

        # Reconstruct vocab: string keys → int keys, int lists → bytes
        self.vocab = {}
        for token_id_str, byte_list in data["vocab"].items():
            self.vocab[int(token_id_str)] = bytes(byte_list)

        # Reconstruct special tokens
        self.special_tokens = {}
        self.inverse_special = {}
        for token_str, token_id in data["special_tokens"].items():
            self.special_tokens[token_str] = token_id
            self.inverse_special[token_id] = token_str
            self.vocab[token_id] = token_str.encode("utf-8")
