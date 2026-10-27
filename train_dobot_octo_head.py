import os
import sys
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import optax
import numpy as np

import octo_patch
from octo.model.octo_model import OctoModel

def train_dobot_octo_head(
    dataset_path="dobot_front_demonstrations.pkl",
    checkpoint_dir="octo_dobot_front_checkpoint",
    num_epochs=35,
    batch_size=8,
    learning_rate=3e-4
):
    print("=" * 70)
    print("OCTO PURE NEURAL DIFFUSION HEAD FINE-TUNING")
    print("Front-Elevated Camera View | Dobot Suction Cup Kinematics")
    print(f"Dataset: {dataset_path} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print("=" * 70)

    # 1. Load Pretrained Octo Model
    print("Loading pretrained Octo Small weights...")
    model = OctoModel.load_pretrained("octo_small_checkpoint")
    stats = model.dataset_statistics["bridge_dataset"]["action"]
    print("Octo Small loaded successfully!")

    # 2. Load Demonstrations
    with open(dataset_path, "rb") as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} demonstration episodes.")

    # 3. Create training samples
    # Window size: 2 steps of observations, Horizon: 4 steps of actions
    training_samples = []
    for ep in episodes:
        obs = ep["observations"] # (T, 256, 256, 3)
        acts = ep["actions"]     # (T, 7)
        instr = ep["instruction"]
        T = len(acts)
        for t in range(1, T - 4):
            obs_win = obs[t-1:t+1] # (2, 256, 256, 3)
            # action horizon: 4 steps starting at t
            act_horizon = acts[t:t+4] # (4, 7)
            # Normalize actions with Bridge stats:
            act_norm = (act_horizon - stats["mean"]) / stats["std"]
            training_samples.append({
                "obs": obs_win,
                "act_norm": act_norm,
                "text": instr,
                "timestep": np.array([t-1, t], dtype=np.int32)
            })

    print(f"Total training windows: {len(training_samples)}")
    np.random.shuffle(training_samples)

    # 4. Optimizer: AdamW on heads_action parameters only (freezing transformer backbone)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    head_params = model.params["heads_action"]
    opt_state = optimizer.init(head_params)

    action_head_module = model.module.heads["action"]

    # 5. Precompute or batch forward training step
    @jax.jit
    def train_step(params, opt_state, transformer_embeddings, actions_norm, timestep_pad_mask, action_pad_mask, rng):
        def loss_fn(p):
            loss, metrics = action_head_module.apply(
                {"params": p},
                transformer_embeddings,
                actions_norm,
                timestep_pad_mask=timestep_pad_mask,
                action_pad_mask=action_pad_mask,
                train=True,
                rngs={"dropout": rng},
                method="loss"
            )
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    rng = jax.random.PRNGKey(42)
    n_samples = len(training_samples)
    n_batches = n_samples // batch_size

    print("\nStarting Diffusion Action Head Adaptation Loop...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        np.random.shuffle(training_samples)

        for b_idx in range(n_batches):
            batch_data = training_samples[b_idx * batch_size : (b_idx + 1) * batch_size]
            b_obs = np.stack([d["obs"] for d in batch_data]) # (B, 2, 256, 256, 3)
            b_acts = np.stack([d["act_norm"] for d in batch_data]) # (B, 4, 7)
            # Replicate window dimension for action tensor: (B, 2, 4, 7)
            b_acts_4d = np.repeat(b_acts[:, None, :, :], 2, axis=1)
            b_texts = [d["text"] for d in batch_data]
            b_timesteps = np.stack([d["timestep"] for d in batch_data]) # (B, 2)

            obs_dict = {
                "image_primary": b_obs,
                "image_wrist": np.zeros((batch_size, 2, 128, 128, 3), dtype=np.uint8),
                "pad_mask_dict": {
                    "image_primary": np.ones((batch_size, 2), dtype=bool),
                    "image_wrist": np.zeros((batch_size, 2), dtype=bool),
                    "timestep": np.ones((batch_size, 2), dtype=bool)
                },
                "timestep": b_timesteps,
                "timestep_pad_mask": np.ones((batch_size, 2), dtype=bool)
            }
            task_dict = model.create_tasks(texts=b_texts)

            # 1. Run frozen transformer backbone:
            transformer_embeddings = model.run_transformer(
                obs_dict,
                task_dict,
                timestep_pad_mask=obs_dict["timestep_pad_mask"],
                train=False
            )

            # 2. Update diffusion head parameters:
            rng, step_rng = jax.random.split(rng)
            action_pad_mask = np.ones((batch_size, 2, 4, 7), dtype=bool)

            head_params, opt_state, loss_val = train_step(
                head_params,
                opt_state,
                transformer_embeddings,
                b_acts_4d,
                obs_dict["timestep_pad_mask"],
                action_pad_mask,
                step_rng
            )
            epoch_loss += float(loss_val)

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | Diffusion Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    # 6. Save adapted checkpoint
    print(f"\nSaving fine-tuned model checkpoint to {checkpoint_dir}...")
    new_params = copy.copy(model.params)
    new_params["heads_action"] = head_params
    finetuned_model = model.replace(params=new_params)
    finetuned_model.save_pretrained(checkpoint_dir)
    print("Fine-tuned checkpoint successfully saved!")
    print("Zero-handholding neural policy adaptation complete.")

if __name__ == "__main__":
    train_dobot_octo_head(num_epochs=20, batch_size=8)
