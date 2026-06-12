import pandas as pd
import numpy as np
import os
import glob
import json

def convert_normalization(input_path, output_path):
    df = pd.read_parquet(input_path)
    
    # Old Z-score stats
    OLD_MEAN = np.array([0.00043198, 0.00029432, 0.00088205], dtype=np.float32)
    OLD_STD  = np.array([0.01033815, 0.01570009, 0.01481095], dtype=np.float32)
    
    # New Quantile stats
    with open("norm_stats.json", "r") as f:
        stats = json.load(f)
    Q01 = np.array(stats["q01"], dtype=np.float32)
    Q99 = np.array(stats["q99"], dtype=np.float32)
    
    def re_norm(traj_flat):
        # traj_flat is a list of 16 arrays, each of size 4
        traj = np.stack(traj_flat) # Should be (16, 4)
        
        # 1. De-normalize from Z-score to raw
        # Old Z-score only applied to first 3 dims (x, y, z)
        raw_pos = (traj[:, :3] * OLD_STD) + OLD_MEAN
        raw_gripper = traj[:, 3] # was raw 0-1
        
        raw_act = np.concatenate([raw_pos, raw_gripper[:, None]], axis=1)
        
        # 2. Re-normalize to Quantile [-1, 1]
        range_val = Q99 - Q01
        range_val[range_val == 0] = 1.0
        norm_act = 2.0 * (raw_act - Q01) / range_val - 1.0
        norm_act = np.clip(norm_act, -1.0, 1.0)
        
        return norm_act.flatten().astype(np.float32).tolist()

    df['future_trajectory'] = df['future_trajectory'].apply(re_norm)
    df.to_parquet(output_path)
    print(f"Converted {input_path} -> {output_path}")

if __name__ == "__main__":
    if os.path.exists("data/train_embeddings.parquet"):
        convert_normalization("data/train_embeddings.parquet", "data/train_embeddings_quantile.parquet")
    
    files = glob.glob("data/processed/*.parquet")
    if files:
        os.makedirs("data/processed_quantile", exist_ok=True)
        for f in files:
            out = os.path.join("data/processed_quantile", os.path.basename(f))
            convert_normalization(f, out)
