import torch
import time
import math
from model.config import ModelConfig
from model.model import OneLLM
from data.dataset import get_dataloader

def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    # 1. Setup Device (MPS for Mac, CUDA for PC, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU, this will be slow.")

    # 2. Initialize Model
    config = ModelConfig()
    model = OneLLM(config).to(device)
    print(f"Model Parameters: {model.get_num_params():,} (Micro Config)")

    # 3. Data Pipeline
    batch_size = 16
    dataloader = get_dataloader("data/shakespeare.txt", batch_size, config.max_seq_len)
    
    # 4. Training Hyperparameters
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_steps = 10
    max_steps = 100  # We will just run 100 steps to get metrics
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
    
    print("\nStarting Training (100 steps)...")
    
    t0 = time.time()
    
    model.train()
    for step, (x, y) in enumerate(dataloader):
        if step >= max_steps:
            break
            
        x, y = x.to(device), y.to(device)
        
        # Adjust learning rate
        lr = get_lr(step, max_steps, max_lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if step % 10 == 0 or step == max_steps - 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            tokens_processed = batch_size * config.max_seq_len * (10 if step > 0 else 1)
            tokens_per_sec = tokens_processed / dt
            
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")
            
    print("\nTraining complete!")
    print(f"Final Loss: {loss.item():.4f}")
    print(f"Final Perplexity: {math.exp(loss.item()):.2f}")

if __name__ == "__main__":
    main()
