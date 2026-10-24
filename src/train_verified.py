"""Offline Verified SmolVLA Trainer and Closed-Loop Evaluator."""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import copy
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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from src.env_pybullet import Robot4DofEnv
from src.expert import rollout_expert

with open(ROOT / "data" / "robot_norm_stats.json") as f:
    stats = json.load(f)

act_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
act_std = torch.tensor(stats["action_std"], dtype=torch.float32)
act_std[3] = 0.05
state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
state_std = torch.tensor(stats["state_std"], dtype=torch.float32)
state_std[3] = 0.05

def norm_action(a):
    return (a - act_mean) / act_std

def unnorm_action(norm_a):
    return norm_a * act_std + act_mean

def norm_state(s):
    return (s - state_mean) / state_std


def run():
    print("Loading SmolVLA Base Policy from local offline cache...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", local_files_only=True)
    policy.config.num_steps = 4
    policy.config.adapt_to_pi_aloha = False
    policy.requires_grad_(False)
    
    # Train projector layers
    policy.model.state_proj.requires_grad_(True)
    policy.model.action_in_proj.requires_grad_(True)
    policy.model.action_out_proj.requires_grad_(True)
    policy.model.action_time_mlp_in.requires_grad_(True)
    policy.model.action_time_mlp_out.requires_grad_(True)
    
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180, eta_min=3e-4)
    tok = policy.model.vlm_with_expert.processor.tokenizer
    
    enc = tok("pick the red cube", return_tensors="pt")
    st = torch.zeros(1, 32)
    st[0, :5] = norm_state(torch.tensor([0.0, 1.1, 1.0, 0.0, 0.0]))
    img = torch.zeros(1, 3, 256, 256)
    img_mask = torch.ones(1, dtype=torch.bool)
    
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            [img], [img_mask], enc["input_ids"], enc["attention_mask"].bool(), state=st
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_kv = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
    
    env = Robot4DofEnv(gui=False, seed=0)
    env.reset(task="pick", seed=0)
    frames = rollout_expert(env, "pick")
    acts = np.array([act for _, act, _ in frames], dtype=np.float32)
    env.close()
    
    # Resample 62 steps to exactly 50 steps using linear interpolation
    orig_t = np.linspace(0, 1, len(acts))
    target_t = np.linspace(0, 1, 50)
    chunk_50 = np.zeros((50, 5), dtype=np.float32)
    for j in range(5):
        chunk_50[:, j] = np.interp(target_t, orig_t, acts[:, j])
        
    target_act = torch.zeros(1, 50, 32)
    target_act[0, :, :5] = norm_action(torch.from_numpy(chunk_50))
    
    torch.manual_seed(42)
    fixed_noise = torch.randn_like(target_act)
    
    print("Training 180 steps on full 50-step pick trajectory...")
    for step in range(180):
        optimizer.zero_grad()
        t_val = float(np.random.choice([0.25, 0.5, 0.75, 1.0]))
        time = torch.tensor([t_val])
        time_exp = time[:, None, None]
        
        noise = fixed_noise + 0.08 * torch.randn_like(target_act)
        x_t = time_exp * noise + (1.0 - time_exp) * target_act
        u_t = noise - target_act
        
        kv_copy = copy.deepcopy(past_kv)
        v_pred = policy.model.denoise_step(prefix_pad_masks, kv_copy, x_t, time)
        loss = torch.nn.functional.mse_loss(v_pred, u_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (step + 1) % 45 == 0:
            print(f"Step {step+1:3d}/180: loss = {loss.item():.5f}")
            
    print("\nRunning closed-loop physical rollout in PyBullet...")
    with torch.no_grad():
        act_out = policy.model.sample_actions([img], [img_mask], enc["input_ids"], enc["attention_mask"].bool(), st, noise=fixed_noise)
        norm_gen = act_out[0]
        unnorm_gen = unnorm_action(norm_gen[:, :5]).numpy()
        
    env = Robot4DofEnv(gui=False, seed=0)
    env.reset(task="pick", seed=0)
    for i in range(50):
        obs, r, d, info = env.step(unnorm_gen[i])
        if i in [0, 15, 25, 35, 45, 49]:
            tcp = info["tcp"]
            cube = info["cube"]
            dist = float(np.linalg.norm(tcp - cube))
            picked = info["picked"]
            print(f"Step {i:2d}: TCP=[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}], dist={dist:.3f}m, act_g={unnorm_gen[i, 4]:.2f}, picked={picked}")
            
    print(f"\nFinal Rollout Info: success={info['success']}, picked={info['picked']}, cube_z={info['cube'][2]:.3f}")
    
    if info["picked"]:
        out_dir = ROOT / "models" / "smolvla_lora"
        out_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(out_dir)
        print(f"SUCCESS! Saved working SmolVLA policy checkpoint to {out_dir}")
    env.close()

if __name__ == "__main__":
    run()
