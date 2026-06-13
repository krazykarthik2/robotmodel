import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoTokenizer, AutoProcessor
from src.model import SmolVLA
import pandas as pd
import imageio
import glob
import random
import warnings

# Suppress noisy environment warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Can't initialize NVML")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def generate_visualizations(checkpoint_path, data_dir, output_prefix="viz/random_sample"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("viz", exist_ok=True)
    
    # 1. Find a random sample
    if os.path.isdir(data_dir):
        files = glob.glob(os.path.join(data_dir, "*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        data_path = random.choice(files)
    else:
        data_path = data_dir
        
    print(f"Using sample from: {data_path}")
    df = pd.read_parquet(data_path)
    # Pick a random row that isn't at the very end (to ensure future trajectory exists)
    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx]
    
    # 2. Load Model
    model = SmolVLA()
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")
    else:
        # Explicitly setting weights_only=False to satisfy FutureWarnings
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    model.to(device)
    model.eval()
    
    # 3. Prepare Input
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    tokenizer = processor.tokenizer
    
    vision = torch.tensor(np.array(row['vision_embedding'], dtype=np.float32)).unsqueeze(0).to(device)
    state = torch.tensor(np.array(row['current_eef'], dtype=np.float32)).unsqueeze(0).to(device)
    input_ids = torch.tensor(np.array(row['input_ids'], dtype=np.int64)).unsqueeze(0).to(device)
    
    # 4. Predict
    with torch.no_grad():
        # Use predict_action (Euler Solver) instead of forward()
        pred = model.predict_action(vision, state, input_ids, num_steps=16)
        pred = pred.view(16, 4).cpu().numpy()
        
        # Ground truth (ensure 16x4)
        target = np.array(row['future_trajectory']).reshape(16, 4)

    # 5. Integrate Deltas for 3D Plotting
    # Canonical Scaling (1.0 = workspace half-range)
    SCALES = np.array([0.2, 0.4, 0.25], dtype=np.float32)
    OFFSET = np.array([0.4, 0.0, 0.25], dtype=np.float32)
    
    start_pos_norm = state[0, :3].cpu().numpy()
    start_pos_phys = (start_pos_norm * SCALES) + OFFSET
    
    # Action Scaling (Using real movement stats ~3cm range)
    # Using the same Q01/Q99 as in src/sim_viz.py
    Q01 = np.array([-0.0290, -0.0449, -0.0303], dtype=np.float32)
    Q99 = np.array([0.0273, 0.0456, 0.0522], dtype=np.float32)
    range_act = Q99 - Q01
    
    pred_pos_phys = (pred[:, :3] + 1.0) / 2.0 * range_act + Q01
    target_pos_phys = (target[:, :3] + 1.0) / 2.0 * range_act + Q01
    
    tx = start_pos_phys[0] + np.cumsum(target_pos_phys[:, 0])
    ty = start_pos_phys[1] + np.cumsum(target_pos_phys[:, 1])
    tz = start_pos_phys[2] + np.cumsum(target_pos_phys[:, 2])
    
    px = start_pos_phys[0] + np.cumsum(pred_pos_phys[:, 0])
    py = start_pos_phys[1] + np.cumsum(pred_pos_phys[:, 1])
    pz = start_pos_phys[2] + np.cumsum(pred_pos_phys[:, 2])

    # Figure size 10.24x8 results in 1024x800 pixels (divisible by 16)
    fig = plt.figure(figsize=(10.24, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(tx, ty, tz, 'b--x', label='Ground Truth', markersize=4, alpha=0.6)
    ax.plot(px, py, pz, 'r-o', label='Predicted', markersize=4)
    ax.scatter(start_pos_phys[0], start_pos_phys[1], start_pos_phys[2], color='green', s=100, label='Start')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()
    
    plot_path = f"{output_prefix}.png"
    plt.savefig(plot_path)
    print(f"Static plot saved to {plot_path}")

    # 6. Create Animated Video (Rotating View)
    frames = []
    print("Generating video frames...")
    for angle in range(0, 360, 5):
        ax.view_init(elev=20, azim=angle)
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(image)
    
    video_path = f"{output_prefix}.mp4"
    imageio.mimsave(video_path, frames, fps=20)
    print(f"Video saved to {video_path}")
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="robotmodel/models/checkpoints/latest.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="viz/random_sample")
    args = parser.parse_args()
    
    generate_visualizations(args.checkpoint, args.data_dir, args.output)
