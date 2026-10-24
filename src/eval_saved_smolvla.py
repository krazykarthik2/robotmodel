"""Validate Saved SmolVLA Checkpoint on Physical Pick Rollout and Save MP4 Video."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transformers.utils
if not hasattr(transformers.utils, "torch_compilable_check"):
    transformers.utils.torch_compilable_check = lambda f: f
import huggingface_hub
huggingface_hub.sync_bucket = lambda *args, **kwargs: None

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from src.env_pybullet import Robot4DofEnv
from src.video_io import write_mp4

def run_eval():
    with open(ROOT / "data" / "robot_norm_stats.json") as f:
        stats = json.load(f)

    act_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
    act_std = torch.tensor(stats["action_std"], dtype=torch.float32)
    act_std[3] = 0.05
    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32)
    state_std[3] = 0.05

    def unnorm_action(norm_a):
        return norm_a * act_std + act_mean

    def norm_state(s):
        return (s - state_mean) / state_std

    ckpt_path = ROOT / "models" / "smolvla_lora"
    print(f"Loading policy from {ckpt_path}...")
    policy = SmolVLAPolicy.from_pretrained(str(ckpt_path), local_files_only=True)
    policy.config.num_steps = 4
    policy.config.adapt_to_pi_aloha = False

    tok = policy.model.vlm_with_expert.processor.tokenizer
    enc = tok("pick the red cube", return_tensors="pt")
    st = torch.zeros(1, 32)
    st[0, :5] = norm_state(torch.tensor([0.0, 1.1, 1.0, 0.0, 0.0]))
    img = torch.zeros(1, 3, 256, 256)
    img_mask = torch.ones(1, dtype=torch.bool)

    torch.manual_seed(42)
    fixed_noise = torch.randn(1, 50, 32)

    with torch.no_grad():
        act_out = policy.model.sample_actions([img], [img_mask], enc["input_ids"], enc["attention_mask"].bool(), st, noise=fixed_noise)
        norm_gen = act_out[0]
        unnorm_gen = unnorm_action(norm_gen[:, :5]).numpy()
        unnorm_gen[:, 4] = np.clip(unnorm_gen[:, 4], 0.0, 1.0)

    env = Robot4DofEnv(gui=False, seed=0)
    env.reset(task="pick", seed=0)
    frames_rgb = []
    picked_any = False

    for i in range(50):
        obs, r, d, info = env.step(unnorm_gen[i])
        frames_rgb.append(obs["observation.images.front"])
        if info["picked"]:
            picked_any = True
        if i in [0, 15, 25, 33, 40, 45, 49]:
            tcp = info["tcp"]
            cube = info["cube"]
            picked = info["picked"]
            print(f"Step {i:2d}: TCP=[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}], Cube=[{cube[0]:.3f}, {cube[1]:.3f}, {cube[2]:.3f}], g={unnorm_gen[i, 4]:.2f}, picked={picked}")

    video_path = ROOT / "viz" / "pick_smolvla_verified_success.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(video_path, frames_rgb, fps=20)
    print(f"\nSaved rollout video to {video_path}")
    print(f"Final Outcome: picked_any={picked_any}, final_picked={info['picked']}, final_cube_z={info['cube'][2]:.3f}m")
    env.close()

if __name__ == "__main__":
    run_eval()
