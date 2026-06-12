import torch
import torch.nn.functional as F
from src.model import SmolVLA
from src.train import BridgeEmbeddingDataset, loss_fn
import os

def evaluate(checkpoint_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SmolVLA()
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state"]
        new_state_dict = {k[10:] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        print("Checkpoint not found!")
        return

    model.to(device).eval()
    
    dataset = BridgeEmbeddingDataset(data_path)
    
    # Check overfit sample (index 10)
    sample = dataset[10]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    
    with torch.no_grad():
        batch_loss = 0
        for _ in range(100):
            batch_loss += loss_fn(model, batch, device).item()
        batch_loss /= 100
        
        print(f"Sample 10 (Overfit Target) Loss: {batch_loss:.6f}")

if __name__ == "__main__":
    evaluate("robotmodel/models/checkpoints/latest.pt", "/home/jupyter-238w1a5447/robotmodel/data/processed/train.parquet")
