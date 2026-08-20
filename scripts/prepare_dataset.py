import os
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

# Configuration
DATASET_NAME = "togethercomputer/RedPajama-Data-1T-Sample"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_BIN = os.path.join(OUTPUT_DIR, "train.bin")
VAL_BIN = os.path.join(OUTPUT_DIR, "val.bin")
VAL_RATIO = 0.05 # 5% for validation

def prepare_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Downloading {DATASET_NAME}...")
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split='train')
    
    # We will use tiktoken (GPT-2 encoding) for speed and simplicity.
    # It has a vocab size of 50257.
    enc = tiktoken.get_encoding("gpt2")
    
    print("Tokenizing the dataset...")
    # We tokenize everything and stream it to a list
    # For massive datasets, you'd write directly to the memmap chunk by chunk,
    # but for a 1B sample, we can process in batches.
    
    # To avoid memory issues even with 1B tokens, we write chunks to disk.
    
    # Split into train and val
    split_dataset = dataset.train_test_split(test_size=VAL_RATIO, seed=42, shuffle=True)
    
    for split, dset in split_dataset.items():
        bin_file = TRAIN_BIN if split == 'train' else VAL_BIN
        print(f"\nProcessing {split} split to {bin_file}...")
        
        # Calculate total tokens (approximate first, then exact)
        # We will write directly to a numpy memmap file
        
        arr_len = 0
        
        # First pass: count total tokens to initialize the memmap correctly
        print(f"Counting tokens for {split}...")
        for example in tqdm(dset):
            tokens = enc.encode_ordinary(example['text'])
            arr_len += len(tokens)
            
        print(f"Total tokens in {split}: {arr_len}")
        
        # Initialize memmap
        dtype = np.uint16 # GPT-2 vocab fits in uint16 (max 65535)
        mmap = np.memmap(bin_file, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Second pass: write tokens
        print(f"Writing tokens to {bin_file}...")
        idx = 0
        for example in tqdm(dset):
            tokens = enc.encode_ordinary(example['text'])
            tokens_np = np.array(tokens, dtype=dtype)
            mmap[idx:idx+len(tokens_np)] = tokens_np
            idx += len(tokens_np)
            
        mmap.flush()
        print(f"Finished writing {split} split.")

if __name__ == "__main__":
    prepare_dataset()
