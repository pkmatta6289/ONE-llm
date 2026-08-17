"""
ONE-LLM: Byte-Level BPE Tokenizer — Training Algorithm
========================================================

YOUR DAY 1 TASK: Implement the BPE training algorithm from scratch.

WHAT IS BPE?
------------
Byte Pair Encoding starts with a base vocabulary of 256 tokens (one per byte).
It then iteratively finds the most frequent pair of adjacent tokens in the
training corpus and merges them into a new token. After N merges, you have
a vocabulary of size 256 + N.

EXAMPLE:
    Corpus: "aaabdaaabac"
    Bytes:  [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]

    Step 1: Most frequent pair = (97, 97) i.e. "aa". Merge → new token 256
            Corpus becomes: [256, 97, 98, 100, 256, 97, 98, 97, 99]

    Step 2: Most frequent pair = (256, 97) i.e. "aaa". Merge → new token 257
            Corpus becomes: [257, 98, 100, 257, 98, 97, 99]

    ...and so on for however many merges you want.

THE ALGORITHM:
    1. Convert training text to a list of bytes (integers 0-255)
    2. Repeat for `num_merges` times:
       a. Count frequency of every adjacent pair
       b. Find the pair with highest frequency
       c. Replace all occurrences of that pair with a new token ID
       d. Record the merge: (token_a, token_b) → new_token_id
    3. Return the list of merges (this IS your trained tokenizer)

WHAT TO IMPLEMENT BELOW:
    - get_pair_counts(token_ids): Count frequencies of adjacent pairs
    - merge(token_ids, pair, new_id): Replace all occurrences of `pair` with `new_id`
    - BPETrainer.train(text, num_merges): The full training loop
"""

import regex  # `pip install regex` — supports Unicode categories like \p{L}


# GPT-2 style regex pattern for pre-tokenization.
# This splits text into chunks BEFORE we apply BPE merges.
#
# WHY? Without this, BPE might merge the space in "dog cat" with "c" to create
# a " c" token, which bleeds across word boundaries. The regex ensures merges
# only happen WITHIN these chunks:
#   - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
#   - Words (with optional leading space): " Hello", " world"
#   - Numbers (with optional leading space): " 42", " 3"
#   - Punctuation runs: "!!!", "..."
#   - Whitespace that ISN'T followed by non-whitespace
#
# TODO: Study this pattern. Try it on sample text to see what chunks it produces.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pair_counts(token_ids):
    """
    Count the frequency of every adjacent pair in `token_ids`.

    Args:
        token_ids: list[int] — a sequence of token IDs

    Returns:
        dict[(int, int), int] — mapping from (token_a, token_b) to count

    Example:
        >>> get_pair_counts([1, 2, 3, 1, 2])
        {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    pair_counts = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def merge(token_ids, pair, new_id):
    """
    Replace all occurrences of `pair` in `token_ids` with `new_id`.

    This is the core merge operation. Scan through the list and whenever you see
    pair[0] followed by pair[1], replace them with new_id.

    Args:
        token_ids: list[int] — current sequence of token IDs
        pair: tuple(int, int) — the pair to merge
        new_id: int — the new token ID to replace the pair with

    Returns:
        list[int] — new sequence with the pair merged

    Example:
        >>> merge([1, 2, 3, 1, 2, 4], pair=(1, 2), new_id=99)
        [99, 3, 99, 4]

    CAREFUL: After merging (1,2)→99, don't accidentally also try to merge
    the 99 with the next token in the same pass.
    """
    result = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(token_ids[i])
            i += 1
    return result


class BPETrainer:
    """
    Trains a Byte-Level BPE tokenizer.

    Usage:
        trainer = BPETrainer()
        merges, vocab = trainer.train("your training text here", num_merges=1000)
        # merges: dict[(int,int), int] — maps pairs to new token IDs
        # vocab: dict[int, bytes] — maps token IDs to their byte sequences
    """

    def __init__(self):
        self.pattern = regex.compile(GPT2_SPLIT_PATTERN)

    def train(self, text, num_merges=1000, verbose=False):
        """
        Train BPE on the given text.

        THE ALGORITHM (implement this step by step):

        1. Pre-tokenize: Split `text` using self.pattern into chunks
        2. Convert each chunk to a list of bytes (integers 0-255)
           - Each chunk becomes its own list: "hello" → [104, 101, 108, 108, 111]
           - Keep chunks separate! Merges should NOT cross chunk boundaries.
        3. Initialize merges = {} (empty dict)
        4. For i in range(num_merges):
           a. Count pair frequencies across ALL chunks (use get_pair_counts on each)
           b. Find the pair with the highest total count
           c. If no pairs found (all chunks are length 1), stop early
           d. Create new_id = 256 + i
           e. Replace all occurrences of that pair in all chunks (use merge())
           f. Record: merges[(pair[0], pair[1])] = new_id
           g. (Optional) Print progress: merge #i, the pair, the count
        5. Build vocab: map each token ID to its byte representation
           - Base: vocab[i] = bytes([i]) for i in 0..255
           - Merged: vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        6. Return (merges, vocab)

        Args:
            text: str — training text
            num_merges: int — number of BPE merges to perform
            verbose: bool — if True, print each merge

        Returns:
            tuple(dict, dict):
                - merges: {(int, int): int} — the learned merge rules
                - vocab: {int: bytes} — token ID to byte sequence mapping
        """
        # Step 1: Pre-tokenize text into chunks using the regex pattern
        chunks = self.pattern.findall(text)

        # Step 2: Convert each chunk (string) to a list of bytes (integers 0-255)
        tokens = []
        for chunk in chunks:
            token = []
            for c in chunk:
                token.append(ord(c))
            tokens.append(token)

        # Step 3: Initialize merges dict and base vocab
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        # Step 4: The merge loop
        for i in range(num_merges):
            # 4a: Count pair frequencies across ALL token lists
            pair_count = {}
            for token_list in tokens:
                count = get_pair_counts(token_list)
                for c in count:
                    if c not in pair_count:
                        pair_count[c] = 0
                    pair_count[c] += count[c]

            # 4c: If no pairs found, stop early
            if not pair_count:
                break

            # 4b: Find the most frequent pair
            best_pair = max(pair_count, key=pair_count.get)

            # 4d: Create new token ID
            new_id = 256 + i

            # 4e: Replace all occurrences in all token lists
            for j in range(len(tokens)):
                tokens[j] = merge(tokens[j], best_pair, new_id)

            # 4f: Record the merge rule (pair → new_id)
            merges[best_pair] = new_id

            # 5: Build vocab entry for the new token
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            # 4g: Optional verbose output
            if verbose:
                print(f"merge {i+1}/{num_merges}: {best_pair} -> {new_id} (count={pair_count[best_pair]})")

        # Step 6: Return merges and vocab
        return (merges, vocab)
