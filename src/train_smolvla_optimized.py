"""Ultra-fast, mathematically exact KV-Cached Flow Matching Trainer for SmolVLA on 4-DOF URDF.

Methodology:
1. Extract and precompute KV prefix embeddings across task variations (pick, place, push)
   using the frozen SmolVLM vision & language backbone once.
2. Form target 50-step action trajectories directly matching the model's action chunk size.
3. Compute exact conditional flow matching (CFM) vector field loss through the expert transformer
   with isolated, deepcopied KV states.
4. Update the CFM velocity field parameters (action_in_proj, action_out_proj, action_time_mlps).
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


def collect_expert_demonstrations(num_episodes_per_task: int = 4):
    print(f"Collecting {num_episodes_per_task} expert trajectories per task (pick, place, push)...")
    env = Robot4DofEnv(gui=False, seed=42)
    dataset = []
    tasks = ["pick", "place", "push"]
    
    for task in tasks:
        for ep in range(num_episodes_per_task):
            seed = 100 * (ep + 1) + tasks.index(task)
            env.reset(task=task, seed=seed)
            prompt = f"{task} the red cube"
            env.prompt = prompt
            frames = rollout_expert(env, task)
            traj_states = []
            traj_actions = []
            images = []
            for obs, act, info in frames:
                traj_states.append(obs["observation.state"][:5])
                traj_actions.append(act[:5])
                images.append(obs["observation.images.front"])
            
            traj_states = np.array(traj_states, dtype=np.float32)
            traj_actions = np.array(traj_actions, dtype=np.float32)
            dataset.append({
                "task": task,
                "prompt": prompt,
                "states": traj_states,
                "actions": traj_actions,
                "images": images,
            })
            print(f"  [{task.upper()} Ep {ep+1}] Recorded {len(traj_states)} steps, success: {info.get('success')}")
    env.close()
    return dataset


def train_smolvla_kv_cached(num_epochs: int = 150):
    dataset = collect_expert_demonstrations(num_episodes_per_task=4)
    
    print("\nLoading SmolVLA Base Policy...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.config.adapt_to_pi_aloha = False
    policy.config.num_steps = 4
    policy.requires_grad_(False)
    
    # Train action Flow Matching parameters
    policy.model.action_in_proj.requires_grad_(True)
    policy.model.action_out_proj.requires_grad_(True)
    policy.model.action_time_mlp_in.requires_grad_(True)
    policy.model.action_time_mlp_out.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=3e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    
    tok = policy.model.vlm_with_expert.processor.tokenizer
    chunk_size = policy.config.chunk_size  # 50
    max_action_dim = policy.config.max_action_dim  # 32
    max_state_dim = policy.config.max_state_dim  # 32
    
    print("\nPre-computing KV caches for all tasks...")
    cached_prompts = {}
    for task_name in ["pick", "place", "push"]:
        prompt_text = f"{task_name} the red cube"
        enc = tok(prompt_text, return_tensors="pt")
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
        cached_prompts[task_name] = {
            "pad_masks": prefix_pad_masks,
            "past_kv": past_kv,
        }
    print("Pre-computation done! Initializing ultra-fast CFM training loop...\n")

    for epoch in range(num_epochs):
        epoch_losses = []
        for ep_data in dataset:
            task = ep_data["task"]
            actions = ep_data["actions"]
            n_frames = len(actions)
            if n_frames < 5:
                continue
            
            # Select random time window of length chunk_size (padded if needed)
            start_t = np.random.randint(0, max(1, n_frames - 10))
            chunk = actions[start_t : start_t + chunk_size]
            if len(chunk) < chunk_size:
                # pad last action
                pad = np.tile(chunk[-1:], (chunk_size - len(chunk), 1))
                chunk = np.vstack([chunk, pad])
            
            target_act = torch.zeros(1, chunk_size, max_action_dim)
            target_act[0, :, :5] = torch.from_numpy(chunk)
            
            # Sample CFM flow noise & time
            noise = torch.randn_like(target_act)
            time = torch.rand(1)
            time_exp = time[:, None, None]
            x_t = time_exp * noise + (1.0 - time_exp) * target_act
            u_t = noise - target_act
            
            prompt_info = cached_prompts[task]
            prefix_pad_masks = prompt_info["pad_masks"]
            past_kv = copy.deepcopy(prompt_info["past_kv"])
            
            v_pred = policy.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_kv,
                x_t=x_t,
                timestep=time,
            )
            
            # Loss only on the active 5 robot joints
            loss = F.mse_loss(v_pred[:, :, :5], u_t[:, :, :5])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        scheduler.step()
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} - CFM Vector Field Loss: {np.mean(epoch_losses):.5f} (lr={scheduler.get_last_lr()[0]:.6f})")

    out_dir = ROOT / "models" / "smolvla_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    print(f"\nTraining Complete! Policy saved to {out_dir}")


if __name__ == "__main__":
    train_smolvla_kv_cached(num_epochs=120)
