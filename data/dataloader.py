import torch
from torch.utils.data import DataLoader
from .dataset import MemmapDataset
import os

def create_dataloaders(data_dir, block_size, batch_size, num_workers=2):
    """
    Creates PyTorch DataLoaders for the train and validation sets.
    """
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    
    # We set pin_memory=True to speed up CPU to GPU data transfer
    
    # Train Loader
    if os.path.exists(train_bin):
        train_dataset = MemmapDataset(train_bin, block_size)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,          # Shuffle grabs random blocks from the memmap
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = None
        
    # Val Loader
    if os.path.exists(val_bin):
        val_dataset = MemmapDataset(val_bin, block_size)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        val_loader = None
        
    return train_loader, val_loader
