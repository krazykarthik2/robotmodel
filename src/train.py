import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from src.model import SmolVLA
from src.muon import MuonWithAuxAdam
import pandas as pd
import numpy as np
import hashlib
import shutil
import cv2
import imageio
import mujoco
import glob
import matplotlib.pyplot as plt
import json

def get_architecture_hash(model):
    """Generates a hash based on the model architecture and hyperparameters."""
    model_str = str(model)
    return hashlib.sha256(model_str.encode()).hexdigest()[:12]

class BridgeEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        if os.path.isdir(data_path):
            files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            print(f"Loading dataset from {len(files)} files in {data_path}")
            dfs = []
            for f in files:
                dfs.append(pd.read_parquet(f))
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.read_parquet(data_path)
        print(f"Dataset loaded with {len(self.df)} samples.")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            return {
                "vision": torch.tensor(np.array(row['vision_embedding'].tolist() if hasattr(row['vision_embedding'], 'tolist') else row['vision_embedding'], dtype=np.float32)),
                "state": torch.tensor(np.array(row['current_eef'].tolist() if hasattr(row['current_eef'], 'tolist') else row['current_eef'], dtype=np.float32)),
                "input_ids": torch.tensor(np.array(row['input_ids'].tolist() if hasattr(row['input_ids'], 'tolist') else row['input_ids'], dtype=np.int64)),
                "target": torch.tensor(np.array(row['future_trajectory'].tolist() if hasattr(row['future_trajectory'], 'tolist') else row['future_trajectory'], dtype=np.float32)).view(-1)
            }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            print(f"future_trajectory type: {type(row['future_trajectory'])}")
            if hasattr(row['future_trajectory'], 'shape'):
                print(f"future_trajectory shape: {row['future_trajectory'].shape}")
            raise e

def loss_fn(model, batch, device):
    vision = batch['vision'].to(device)
    state = batch['state'].to(device)
    input_ids = batch['input_ids'].to(device)
    x1 = batch['target'].to(device) # Target trajectory [B, 64]
    
    batch_size = x1.shape[0]
    
    # 1. Sample tau ~ Uniform(0, 1)
    tau = torch.rand(batch_size, 1, device=device)
    
    # 2. Sample noise x0 ~ N(0, I)
    x0 = torch.randn_like(x1)
    
    # 3. Compute noisy action x_tau = tau * x1 + (1 - tau) * x0
    xt = tau * x1 + (1.0 - tau) * x0
    
    # 4. Predict velocity field
    # SmolVLA forward returns velocity when noisy_actions and tau are provided
    pred_v = model(vision, state, input_ids, noisy_actions=xt, tau=tau)
    
    # 5. Target velocity is (x1 - x0)
    target_v = x1 - x0
    
    # Loss is MSE between predicted and target velocity
    # Flow matching is typically calculated in FP32
    loss = F.mse_loss(pred_v.float(), target_v.float())
    
    return loss

def visualize_trajectory(model, batch, device, step, output_dir="viz"):
    """Generates a visualization plot comparing predicted and target trajectories."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use uncompiled model if available
    if hasattr(model, '_orig_mod'):
        eval_model = model._orig_mod
    else:
        eval_model = model
        
    eval_model.eval()
    with torch.no_grad():
        vision = batch['vision'][0:1].to(device)
        state = batch['state'][0:1].to(device)
        input_ids = batch['input_ids'][0:1].to(device)
        
        # Use Euler integration for inference
        pred = eval_model.predict_action(vision, state, input_ids, num_steps=16)
        pred = pred.view(16, 4).cpu().numpy()
        target = batch['target'][0].view(16, 4).cpu().numpy()
        start_pos = state[0, :3].cpu().numpy()

    # De-normalize (Quantile stats)
    with open("norm_stats.json", "r") as f:
        stats = json.load(f)
    Q01 = np.array(stats["q01"], dtype=np.float32)[:3]
    Q99 = np.array(stats["q99"], dtype=np.float32)[:3]
    
    # pred_pos = (pred[:, :3] + 1.0) / 2.0 * (Q99 - Q01) + Q01
    pred_pos = (pred[:, :3] + 1.0) * (Q99 - Q01) / 2.0 + Q01
    target_pos = (target[:, :3] + 1.0) * (Q99 - Q01) / 2.0 + Q01

    # Integrate deltas
    px = start_pos[0] + np.cumsum(pred_pos[:, 0])
    py = start_pos[1] + np.cumsum(pred_pos[:, 1])
    pz = start_pos[2] + np.cumsum(pred_pos[:, 2])
    
    tx = start_pos[0] + np.cumsum(target_pos[:, 0])
    ty = start_pos[1] + np.cumsum(target_pos[:, 1])
    tz = start_pos[2] + np.cumsum(target_pos[:, 2])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tx, ty, tz, 'b--x', label='Ground Truth', alpha=0.6)
    ax.plot(px, py, pz, 'r-o', label='Predicted')
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=100)
    ax.set_title(f'Step {step} Trajectory (Flow Matching)')
    ax.legend()
    
    plt.savefig(f"{output_dir}/step_{step}.png")
    plt.close(fig)
    print(f"Visualization saved to {output_dir}/step_{step}.png")
    model.train()

import time
from tqdm import tqdm

# Fix for torch.compile issues in some environments
if hasattr(torch, "_dynamo"):
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

def train(overfit=False):
    # Performance Optimization: Enable TensorFloat32 for matmuls on Ampere/Lovelace
    torch.set_float32_matmul_precision('high')
    
    # Use BF16 precision for L4 GPUs (Ada Lovelace)
    accelerator = Accelerator(mixed_precision="bf16") 
    device = accelerator.device
    
    model = SmolVLA()
    
    if overfit:
        print("!!! RUNNING IN OVERFIT MODE (Single Sample) !!!")
        # Compile is slower for single sample tests, skip it
    elif hasattr(torch, "compile"):
        print("Compiling model for maximum throughput...")
        model = torch.compile(model)
    
    arch_version = get_architecture_hash(model)
    checkpoint_dir = "robotmodel/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    start_step = 0
    
    # Versioning & Checkpoint Management (Skip for overfit)
    if not overfit and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if checkpoint.get("arch_version") == arch_version:
            print(f"Resuming from checkpoint (version {arch_version})")
            model.load_state_dict(checkpoint["model_state"])
            start_step = checkpoint["step"]
        else:
            print(f"Architecture mismatch (Old: {checkpoint.get('arch_version')}, New: {arch_version}). Deleting old checkpoints.")
            shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = MuonWithAuxAdam(model, lr=1e-3, adam_lr=1e-4) # Lower Adam LR for stability
    
    # Dataset
    data_path = "/home/jupyter-238w1a5447/robotmodel/data/processed/train.parquet"
    if not os.path.exists(data_path):
        data_path = "/home/jupyter-238w1a5447/robotmodel/data/processed"
        
    train_dataset = BridgeEmbeddingDataset(data_path)
    
    if overfit:
        # Overfit on a single specific sample
        sample = train_dataset[10]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        print(f"Target values (first 4): {batch['target'][0][:4].tolist()}")
        
        model.train()
        model.to(device)
        # Increase steps for CFM which is slower than regression
        pbar = tqdm(range(10000), desc="Overfitting")
        for step in pbar:
            optimizer.zero_grad()
            loss = loss_fn(model, batch, device)
            accelerator.backward(loss)
            optimizer.step()
            if step % 500 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        # Save one visualization of the overfit
        visualize_trajectory(model, batch, device, "overfit")
        
        # Save Checkpoint after overfit
        checkpoint = {
            "step": 500,
            "model_state": accelerator.get_state_dict(model),
            "arch_version": arch_version
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Overfit complete. Result saved to viz/step_overfit.png and {ckpt_path}")
        return

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    print(f"Starting training (Version: {arch_version}) on {device}")
    
    pbar = tqdm(range(start_step, 20000), desc="Training", disable=not accelerator.is_main_process)
    train_iter = iter(train_loader)
    
    model.train()
    for step in pbar:
        start_time = time.time()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        loss = loss_fn(model, batch, device)
        accelerator.backward(loss)
        optimizer.step()
        
        # Real-time updates
        if accelerator.is_main_process:
            dt = time.time() - start_time
            samples_per_sec = 128 / dt
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "speed": f"{samples_per_sec:.1f} spl/s"
            })
        
        # Periodic Tasks
        if step % 1000 == 0 and step > start_step and accelerator.is_main_process:
            # Save Checkpoint
            checkpoint = {
                "step": step,
                "model_state": accelerator.get_state_dict(model),
                "arch_version": arch_version
            }
            torch.save(checkpoint, ckpt_path)
            print(f"\nCheckpoint saved at step {step}")
            
            # Visualization
            try:
                visualize_trajectory(model, batch, device, step)
            except Exception as e:
                print(f"Visualization failed: {e}")

if __name__ == "__main__":
    import argparse
    import sys
    print(f"DEBUG: Command line arguments: {sys.argv}")
    parser = argparse.ArgumentParser()
    # Default is now OVERFIT mode as requested by user
    parser.add_argument("--full", action="store_true", help="Run full training on entire dataset (instead of default overfit)")
    args = parser.parse_args()
    
    if args.full:
        print("!!! MODE: FULL TRAINING ON ENTIRE DATASET !!!")
    else:
        print("!!! MODE: OVERFIT ON SINGLE SAMPLE !!!")
        
    train(overfit=not args.full)
