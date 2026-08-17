"""
Train a BPE tokenizer on a text file.

Usage:
    python -m tokenizer.train_tokenizer --data path/to/text.txt --num-merges 1000 --output tokenizer.json

This script:
    1. Reads a text file
    2. Trains BPE with the specified number of merges
    3. Saves the trained tokenizer
    4. Runs a quick encode/decode test to verify correctness
"""

import argparse
import time
from pathlib import Path

from .bpe import BPETrainer
from .tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--num-merges", type=int, default=1000, help="Number of BPE merges")
    parser.add_argument("--output", type=str, default="tokenizer.json", help="Output path")
    parser.add_argument("--verbose", action="store_true", help="Print each merge")
    args = parser.parse_args()

    # Read training data
    print(f"Reading {args.data}...")
    text = Path(args.data).read_text(encoding="utf-8")
    print(f"  {len(text):,} characters")

    # Train
    print(f"\nTraining BPE with {args.num_merges} merges...")
    trainer = BPETrainer()
    t0 = time.time()
    merges, vocab = trainer.train(text, num_merges=args.num_merges, verbose=args.verbose)
    dt = time.time() - t0
    print(f"  Training took {dt:.2f}s")
    print(f"  Vocab size: {len(vocab)} (256 base + {args.num_merges} merges)")

    # Create tokenizer and save
    tokenizer = Tokenizer(merges=merges, vocab=vocab)
    tokenizer.save(args.output)
    print(f"\nSaved tokenizer to {args.output}")
    print(f"  Total vocab size (with special tokens): {tokenizer.vocab_size}")

    # Quick test
    print("\n--- Quick Test ---")
    test_strings = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "BPE tokenization is working! 🎉",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]

    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        status = "✅" if decoded == s else "❌"
        print(f"  {status} \"{s[:50]}{'...' if len(s) > 50 else ''}\"")
        print(f"     → {len(ids)} tokens: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        if decoded != s:
            print(f"     ⚠️  Decoded: \"{decoded[:50]}\"")
    print()


if __name__ == "__main__":
    main()
