import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm

from data.mess3_generator import NonErgodicMess3Dataset, Mess3Process

def get_toy_config(vocab_size: int = 3, d_model: int = 64, n_ctx: int = 16) -> HookedTransformerConfig:
    """
    Creates a configuration for a very small GPT-2 style model.
    """
    return HookedTransformerConfig(
        n_layers=2,
        d_model=d_model,
        d_head=16,
        n_heads=4,
        d_mlp=d_model * 4,
        d_vocab=vocab_size,
        n_ctx=n_ctx,
        act_fn="gelu",
        normalization_type="LN",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def train_model(
    model: HookedTransformer, 
    train_loader: DataLoader, 
    epochs: int = 10, 
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list[float]:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y, _ in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            logits = model(batch_x)
            
            # Reshape to (batch_size * seq_len, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            targets = batch_y.contiguous().view(-1)
            
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    return loss_history

if __name__ == "__main__":
    # 1. Setup Data (Matches our generator)
    p1 = Mess3Process(alpha=0.60, x=0.15)
    p2 = Mess3Process(alpha=0.79, x=0.11)
    p3 = Mess3Process(alpha=0.60, x=0.50)
    dataset = NonErgodicMess3Dataset(num_samples=5000, seq_length=17, processes=[p1, p2, p3])
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cfg = get_toy_config(vocab_size=3, d_model=64, n_ctx=16)
    model = HookedTransformer(cfg).to(device)
    
    # 3. Train
    history = train_model(model, train_loader, epochs=5, lr=1e-3)
    print("Training setup is ready to go.")