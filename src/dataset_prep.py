import torch
import numpy as np
import pandas as pd
import imageio
import json
from tqdm import tqdm
import os
import glob
from accelerate import Accelerator
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer

def process_dataset(data_root, output_dir, limit_episodes=None):
    accelerator = Accelerator()
    device = accelerator.device

    tasks_path = os.path.join(data_root, "meta/tasks.jsonl")
    info_path = os.path.join(data_root, "meta/info.json")

    print(f"Loading tasks from {tasks_path}")
    tasks = {}
    if os.path.exists(tasks_path):
        with open(tasks_path, 'r') as f:
            for line in f:
                t = json.loads(line)
                tasks[t['task_index']] = t['task']
    else:
        print("Warning: tasks.jsonl not found. Using empty tasks.")

    print("Loading models...")
    # Load the native SmolVLM processor and model (we only need the vision tower for embeddings)
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    # Load the full model to access its specific vision encoder
    vlm_model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct", 
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    # SmolVLM uses SmolLM2 under the hood, but let's use the processor's tokenizer to be safe
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files in data/chunk-*/
    parquet_files = sorted(glob.glob(os.path.join(data_root, "data/chunk-*/episode_*.parquet")))
    
    if not parquet_files:
        # Fallback search for other structures
        parquet_files = sorted(glob.glob(os.path.join(data_root, "**", "episode_*.parquet"), recursive=True))

    if limit_episodes:
        parquet_files = parquet_files[:limit_episodes]
        
    print(f"Found {len(parquet_files)} episodes to process.")

    # Pre-index video files for faster lookup
    print("Indexing video files...")
    video_map = {}
    for vp in glob.glob(os.path.join(data_root, "videos", "**", "*.mp4"), recursive=True):
        v_name = os.path.basename(vp).replace(".mp4", "")
        # Store all occurrences; prioritize the one in the same chunk
        if v_name not in video_map:
            video_map[v_name] = []
        video_map[v_name].append(vp)
    
    missing_count = 0
    for parquet_path in tqdm(parquet_files, desc="Processing Episodes"):
        episode_id = os.path.basename(parquet_path).replace(".parquet", "")
        # Extract chunk_id from path
        path_parts = parquet_path.split(os.sep)
        chunk_id = "unknown"
        for part in reversed(path_parts):
            if "chunk-" in part:
                chunk_id = part
                break
        
        # Determine output path
        episode_output_path = os.path.join(output_dir, f"{chunk_id}_{episode_id}.parquet")
        if os.path.exists(episode_output_path):
            continue
            
        # Robust video lookup
        video_path = None
        if episode_id in video_map:
            # Try to find video in the same chunk
            for vp in video_map[episode_id]:
                if chunk_id in vp:
                    video_path = vp
                    break
            # Fallback to the first found video for this episode ID
            if not video_path:
                video_path = video_map[episode_id][0]
        
        if not video_path:
            missing_count += 1
            if missing_count <= 5: # Only warn for the first 5
                print(f"Warning: Video not found for {parquet_path}. Skipping.")
            elif missing_count == 6:
                print("Further missing video warnings suppressed...")
            continue
        
        try:
            df = pd.read_parquet(parquet_path)
            reader = imageio.get_reader(video_path)
            
            processed_data = []
            frames_iter = reader.iter_data()
            
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    frame = next(frames_iter)
                except StopIteration:
                    break
                    
                inputs = processor(images=[frame], size={"longest_edge": 512}, return_tensors="pt").to(device, dtype=torch.bfloat16)
                with torch.no_grad():
                    # Extract vision embeddings directly from the SmolVLM vision tower
                    # Ensure 4D shape: [B * num_images, C, H, W]
                    pixel_values = inputs.pixel_values.view(-1, inputs.pixel_values.shape[-3], inputs.pixel_values.shape[-2], inputs.pixel_values.shape[-1])
                    vision_outputs = vlm_model.model.vision_model(pixel_values=pixel_values)
                    # Pass through the native multi-modal projector (reduces 729 tokens to 81 tokens via pixel shuffle)
                    v_tokens = vlm_model.model.connector(vision_outputs.last_hidden_state)
                    vision_emb = v_tokens.cpu().to(torch.float32).numpy().flatten()
                    
                instruction = tasks.get(row['task_index'], "unknown task")
                input_ids = tokenizer(instruction, return_tensors="pt", padding='max_length', max_length=32, truncation=True).input_ids.numpy().flatten()
                
                # LeRobot State is 7D: [x, y, z, roll, pitch, yaw, gripper]
                # We need 4D for our 4-DOF robot: [x, y, z, gripper]
                full_state = np.array(row['observation.state'], dtype=np.float32)
                current_eef = np.array([full_state[0], full_state[1], full_state[2], full_state[6]], dtype=np.float32) 
                
                # Action Trajectory (next 16 steps)
                future_actions = df.iloc[i:i+16]['action'].tolist()
                traj = np.zeros((16, 4), dtype=np.float32)
                
                # Normalization stats
                with open("norm_stats.json", "r") as f:
                    stats = json.load(f)
                
                Q01 = np.array(stats["q01"], dtype=np.float32)
                Q99 = np.array(stats["q99"], dtype=np.float32)
                
                for j, act in enumerate(future_actions):
                    # act is [x, y, z, roll, pitch, yaw, gripper]
                    # We only care about [x, y, z, gripper]
                    raw_act = np.array([act[0], act[1], act[2], act[6]], dtype=np.float32)
                    
                    # Quantile Normalize to [-1, 1]
                    # formula: 2 * (x - Q01) / (Q99 - Q01) - 1
                    # Robust version to avoid division by zero
                    range_val = Q99 - Q01
                    range_val[range_val == 0] = 1.0
                    norm_act = 2.0 * (raw_act - Q01) / range_val - 1.0
                    
                    # Clip to [-1, 1] as per pi0 methodology
                    norm_act = np.clip(norm_act, -1.0, 1.0)
                    
                    traj[j] = norm_act
                if len(future_actions) < 16:
                    # Pad with zero deltas (stay in place)
                    for j in range(len(future_actions), 16):
                        traj[j] = np.zeros(4, dtype=np.float32)
                        
                processed_data.append({
                    "vision_embedding": vision_emb.astype(np.float32).tolist(),
                    "current_eef": current_eef.astype(np.float32).tolist(),
                    "input_ids": input_ids.astype(np.int64).tolist(),
                    "future_trajectory": traj.flatten().astype(np.float32).tolist()
                })
            
            if processed_data:
                df_out = pd.DataFrame(processed_data)
                df_out.to_parquet(episode_output_path)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {parquet_path}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/jupyter-238w1a5447/bridge_v2_data")
    parser.add_argument("--output_dir", type=str, default="/home/jupyter-238w1a5447/robotmodel/data/processed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    process_dataset(args.data_root, args.output_dir, limit_episodes=args.limit)
