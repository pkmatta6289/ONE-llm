"""
Tests for the BPE tokenizer.

Run with: python -m pytest tests/test_tokenizer.py -v

These tests verify your implementation is correct. Don't modify the tests —
if a test fails, fix your code, not the test.
"""

import tempfile
import os
from pathlib import Path

from tokenizer.bpe import get_pair_counts, merge, BPETrainer
from tokenizer.tokenizer import Tokenizer


class TestGetPairCounts:
    """Tests for the pair counting function."""

    def test_simple(self):
        counts = get_pair_counts([1, 2, 3, 1, 2])
        assert counts[(1, 2)] == 2
        assert counts[(2, 3)] == 1
        assert counts[(3, 1)] == 1

    def test_repeated_pair(self):
        counts = get_pair_counts([5, 5, 5, 5])
        assert counts[(5, 5)] == 3  # positions 0-1, 1-2, 2-3

    def test_single_element(self):
        counts = get_pair_counts([42])
        assert counts == {}

    def test_empty(self):
        counts = get_pair_counts([])
        assert counts == {}

    def test_two_elements(self):
        counts = get_pair_counts([10, 20])
        assert counts == {(10, 20): 1}


class TestMerge:
    """Tests for the merge operation."""

    def test_simple_merge(self):
        result = merge([1, 2, 3, 1, 2, 4], pair=(1, 2), new_id=99)
        assert result == [99, 3, 99, 4]

    def test_no_match(self):
        result = merge([1, 2, 3], pair=(4, 5), new_id=99)
        assert result == [1, 2, 3]

    def test_adjacent_overlapping(self):
        # [1, 1, 1] with pair (1,1) should merge the FIRST occurrence
        # Result: [99, 1] (greedy left-to-right)
        result = merge([1, 1, 1], pair=(1, 1), new_id=99)
        assert result == [99, 1]

    def test_merge_at_end(self):
        result = merge([3, 1, 2], pair=(1, 2), new_id=99)
        assert result == [3, 99]

    def test_consecutive_merges(self):
        result = merge([1, 2, 1, 2, 1, 2], pair=(1, 2), new_id=99)
        assert result == [99, 99, 99]


class TestBPETrainer:
    """Tests for the BPE training algorithm."""

    def test_basic_training(self):
        trainer = BPETrainer()
        text = "aaabdaaabac"
        merges, vocab = trainer.train(text, num_merges=3, verbose=False)

        # Should have learned 3 merges
        assert len(merges) == 3

        # First merge should be the most frequent pair
        first_merge_pair = list(merges.keys())[0]
        first_merge_id = merges[first_merge_pair]
        assert first_merge_id == 256  # First new token

        # Vocab should have 256 base + 3 merged tokens
        assert len(vocab) == 259

    def test_vocab_contains_all_bytes(self):
        trainer = BPETrainer()
        _, vocab = trainer.train("hello", num_merges=2)
        # All 256 byte values should be in vocab
        for i in range(256):
            assert i in vocab
            assert vocab[i] == bytes([i])

    def test_zero_merges(self):
        trainer = BPETrainer()
        merges, vocab = trainer.train("hello", num_merges=0)
        assert len(merges) == 0
        assert len(vocab) == 256


class TestTokenizer:
    """Tests for the full tokenizer (encode/decode)."""

    def _make_tokenizer(self, text="aaabdaaabac" * 100, num_merges=10):
        """Helper: train a small tokenizer for testing."""
        trainer = BPETrainer()
        merges, vocab = trainer.train(text, num_merges=num_merges)
        return Tokenizer(merges=merges, vocab=vocab)

    def test_roundtrip_ascii(self):
        """Encode then decode should return the original text."""
        tok = self._make_tokenizer()
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_unicode(self):
        """Must handle non-ASCII (emoji, CJK, etc.) without crashing."""
        tok = self._make_tokenizer()
        text = "Hello 🌍 世界"
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_code(self):
        """Must handle code with special characters."""
        tok = self._make_tokenizer()
        text = "def f(x):\n    return x + 1\n"
        assert tok.decode(tok.encode(text)) == text

    def test_special_tokens(self):
        """BOS and EOS tokens should be added correctly."""
        tok = self._make_tokenizer()
        ids = tok.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id

    def test_encode_produces_fewer_tokens_than_bytes(self):
        """BPE should compress: fewer tokens than raw bytes."""
        tok = self._make_tokenizer(text="abcabc" * 1000, num_merges=50)
        text = "abcabc" * 10
        ids = tok.encode(text)
        raw_bytes = list(text.encode("utf-8"))
        assert len(ids) <= len(raw_bytes), "BPE should compress repeated patterns"

    def test_vocab_size_includes_special(self):
        tok = self._make_tokenizer()
        # 256 base + 10 merges + 3 special tokens
        assert tok.vocab_size == 256 + 10 + 3

    def test_save_load_roundtrip(self):
        """Save and load should produce identical tokenizer."""
        tok = self._make_tokenizer()
        text = "The quick brown fox jumps over the lazy dog."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            tok.save(path)

            tok2 = Tokenizer()
            tok2.load(path)

            ids1 = tok.encode(text)
            ids2 = tok2.encode(text)
            assert ids1 == ids2, "Loaded tokenizer should produce same encoding"
            assert tok2.decode(ids2) == text
        finally:
            os.unlink(path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
