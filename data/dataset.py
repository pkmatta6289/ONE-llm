import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MemmapDataset(Dataset):
    """
    A PyTorch Dataset that reads efficiently from a numpy memmap file on disk.
    This uses almost zero RAM, making it perfect for Colab training.
    """
    def __init__(self, bin_path, block_size):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}. Run prepare_dataset.py first.")
            
        # We load with mode='r' (read-only)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        
    def __len__(self):
        # The number of valid starting indices for a chunk of length block_size + 1
        return len(self.data) - self.block_size - 1
        
    def __getitem__(self, idx):
        # Grab a chunk of size block_size + 1
        # The +1 is because y is x shifted by 1
        chunk = self.data[idx:idx + self.block_size + 1].astype(np.int64)
        
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        return x, y
