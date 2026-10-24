"""Direct Trajectory-Conditioned Flow Matching Trainer for SmolVLA.

Key Insights from Diagnosis:
1. In `sample_actions`, LeRobot executes Euler integration starting from pure noise `x_0 ~ N(0, I)`
   and steps `x_{t+dt} = x_t + v_t * dt` over 4 discrete steps with `timestep` values in [1.0, 0.75, 0.5, 0.25].
2. For the generated trajectory `x_1` to track the genuine 50-step expert trajectory from start to finish,
   the velocity field must be trained across discrete timesteps `tau in [0.2, 0.4, 0.6, 0.8, 1.0]`
   specifically pulling noisy trajectories towards `u_t = noise - target_act`.
3. In this script, we train for 200 epochs on a focused batch with smooth trajectories,
   so that at tau=0 the predicted velocity field aligns with the true smooth trajectory chunks!
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transformers.utils
if not hasattr(transformers.utils, "torch_compilable_check"):
    transformers.utils.torch_compilable_check = lambda f: f
import huggingface_hub
huggingface_hub.sync_bucket = lambda *args, **kwargs: None

import torch
import torch.nn.functional as F
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from src.env_pybullet import Robot4DofEnv
from src.expert import rollout_expert


def collect_demos(episodes_per_task: int = 4):
    print("Collecting expert demonstrations...")
    env = Robot4DofEnv(gui=False, seed=42)
    trajectories = []
    
    for task in ["pick", "place", "push"]:
        for ep in range(episodes_per_task):
            seed = 100 * (ep + 1) + ["pick", "place", "push"].index(task)
            env.reset(task=task, seed=seed)
            frames = rollout_expert(env, task)
            states = np.array([obs["observation.state"][:5] for obs, act, info in frames], dtype=np.float32)
            actions = np.array([act[:5] for obs, act, info in frames], dtype=np.float32)
            
            # Slide window across the episode
            for i in range(0, len(actions) - 1, 2):
                chunk = actions[i : i + 50]
                if len(chunk) < 50:
                    pad = np.tile(chunk[-1:], (50 - len(chunk), 1))
                    chunk = np.vstack([chunk, pad])
                trajectories.append((task, states[i], chunk))
    env.close()
    print(f"Collected {len(trajectories)} trajectory chunks.")
    return trajectories


def train(num_epochs: int = 150):
    trajectories = collect_demos(episodes_per_task=4)
    
    print("\nLoading SmolVLA Base Policy...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.config.adapt_to_pi_aloha = False
    policy.config.num_steps = 4
    policy.requires_grad_(False)
    
    # Train action heads
    policy.model.action_in_proj.requires_grad_(True)
    policy.model.action_out_proj.requires_grad_(True)
    policy.model.action_time_mlp_in.requires_grad_(True)
    policy.model.action_time_mlp_out.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=2e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    
    tok = policy.model.vlm_with_expert.processor.tokenizer
    max_action_dim = policy.config.max_action_dim
    max_state_dim = policy.config.max_state_dim
    chunk_size = policy.config.chunk_size
    
    print("\nPre-computing KV prompt caches...")
    cached_kv = {}
    for task in ["pick", "place", "push"]:
        enc = tok(f"{task} the red cube", return_tensors="pt")
        st = torch.zeros(1, max_state_dim)
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
        cached_kv[task] = {
            "pad_masks": prefix_pad_masks,
            "past_kv": past_kv,
        }
    
    print("Pre-computation complete. Training flow field matching...")
    n_samples = len(trajectories)
    
    for epoch in range(num_epochs):
        indices = np.random.permutation(n_samples)[:16]
        losses = []
        optimizer.zero_grad()
        
        for idx in indices:
            task, st_raw, act_chunk = trajectories[idx]
            
            # Format target action
            target_act = torch.zeros(1, chunk_size, max_action_dim)
            target_act[0, :, :5] = torch.from_numpy(act_chunk)
            
            # Sample discrete Euler time step matching inference: [0.25, 0.5, 0.75, 1.0]
            timesteps = [0.25, 0.5, 0.75, 1.0]
            t_val = float(np.random.choice(timesteps))
            time = torch.tensor([t_val])
            time_exp = time[:, None, None]
            
            noise = torch.randn_like(target_act)
            x_t = time_exp * noise + (1.0 - time_exp) * target_act
            u_t = noise - target_act
            
            kv_info = cached_kv[task]
            prefix_pad_masks = kv_info["pad_masks"]
            past_kv = copy.deepcopy(kv_info["past_kv"])
            
            v_pred = policy.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_kv,
                x_t=x_t,
                timestep=time,
            )
            
            # Velocity loss only on 5 active arm DOF
            loss_i = F.mse_loss(v_pred[:, :, :5], u_t[:, :, :5])
            (loss_i / len(indices)).backward()
            losses.append(loss_i.item())
            
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} - Euler Flow Loss: {np.mean(losses):.5f} (lr={scheduler.get_last_lr()[0]:.6f})")
            
    out_dir = ROOT / "models" / "smolvla_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    print(f"\nTraining Complete! Policy saved to {out_dir}")


if __name__ == "__main__":
    train(num_epochs=120)
