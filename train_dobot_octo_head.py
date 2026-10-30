import os
import sys
import pickle
import time
import copy
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np

import octo_patch
from octo.model.octo_model import OctoModel

def train_dobot_octo(
    dataset_path="dobot_front_demonstrations.pkl",
    checkpoint_dir="octo_dobot_front_checkpoint",
    num_epochs=40,
    batch_size=32,
    learning_rate=3e-4,
    val_split=0.10,
    early_stopping_patience=4
):
    print("=" * 75)
    print("OCTO VLA PRODUCTION FINE-TUNING PIPELINE")
    print(f"Dataset: {dataset_path} | Batch Size: {batch_size} | Base LR: {learning_rate}")
    print(f"Train/Val Split: {1.0 - val_split:.0%}/{val_split:.0%} | Early Stopping Patience: {early_stopping_patience}")
    print("=" * 75)

    # 1. Load Pretrained Octo Model
    print("Loading pretrained Octo Small weights...")
    # Check if local checkpoint exists, otherwise load directly from HuggingFace
    if os.path.exists("octo_small_checkpoint/config.json"):
        model_source = "octo_small_checkpoint"
        # Always specify step 270000 for octo-small so Orbax restores the verified 270000 checkpoint
        if os.path.exists("octo_small_checkpoint/270000"):
            load_step = 270000
        elif os.path.exists("octo_small_checkpoint/300000"):
            load_step = 300000
        else:
            load_step = 270000
    else:
        print("Local octo_small_checkpoint not found on disk. Auto-downloading from HuggingFace (hf://rail-berkeley/octo-small)...")
        model_source = "hf://rail-berkeley/octo-small"
        load_step = 270000

    model = OctoModel.load_pretrained(model_source, step=load_step)
    stats = model.dataset_statistics["bridge_dataset"]["action"]
    print("Octo Small loaded successfully.")

    # 2. Load and Stratify Demonstrations
    with open(dataset_path, "rb") as f:
        episodes = pickle.load(f)

    # Stratified Split by Task
    task_groups = {}
    for ep in episodes:
        instr = ep["instruction"]
        task_groups.setdefault(instr, []).append(ep)

    train_episodes = []
    val_episodes = []

    np.random.seed(42)
    for instr, eps in task_groups.items():
        n_eps = len(eps)
        perm = np.random.permutation(n_eps)
        n_val = max(1, int(round(n_eps * val_split)))
        val_eps = [eps[i] for i in perm[:n_val]]
        tr_eps = [eps[i] for i in perm[n_val:]]
        val_episodes.extend(val_eps)
        train_episodes.extend(tr_eps)

    print(f"Stratified Dataset Split: {len(train_episodes)} Training Episodes | {len(val_episodes)} Validation Episodes")

    def create_samples(ep_list):
        import io
        from PIL import Image
        samples = []
        for ep in ep_list:
            # Handle both raw arrays and JPEG compressed bytes
            if "observations_jpeg" in ep:
                obs = [np.array(Image.open(io.BytesIO(b))) for b in ep["observations_jpeg"]]
            else:
                obs = ep["observations"]
            acts = ep["actions"]
            instr = ep["instruction"]
            T = len(acts)
            for t in range(1, T - 4):
                obs_win = obs[t-1:t+1]
                act_horizon = acts[t:t+4]
                act_norm = (act_horizon - stats["mean"]) / stats["std"]
                samples.append({
                    "obs": obs_win,
                    "act_norm": act_norm,
                    "text": instr,
                    "timestep": np.array([t-1, t], dtype=np.int32)
                })
        return samples

    train_samples = create_samples(train_episodes)
    val_samples = create_samples(val_episodes)
    print(f"Created Samples -> Train Windows: {len(train_samples)} | Validation Windows: {len(val_samples)}")

    # 3. Learning Rate Schedule: Cosine Decay with Linear Warmup
    steps_per_epoch = len(train_samples) // batch_size
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(200, int(0.05 * total_steps))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6
    )

    # Optimizer with Gradient Clipping (protect against gradient explosion)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    )

    head_params = model.params["heads_action"]
    opt_state = optimizer.init(head_params)
    action_head_module = model.module.heads["action"]

    # JIT Compiled Train Step
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
        grad_norm = optax.global_norm(grads)
        return new_params, new_opt_state, loss, grad_norm

    # JIT Compiled Validation Loss Step
    @jax.jit
    def eval_step(params, transformer_embeddings, actions_norm, timestep_pad_mask, action_pad_mask, rng):
        loss, _ = action_head_module.apply(
            {"params": params},
            transformer_embeddings,
            actions_norm,
            timestep_pad_mask=timestep_pad_mask,
            action_pad_mask=action_pad_mask,
            train=False,
            rngs={"dropout": rng},
            method="loss"
        )
        return loss

    rng = jax.random.PRNGKey(42)
    best_val_loss = float("inf")
    patience_counter = 0
    best_params = None

    print("\nStarting Training & Validation Loop...")
    start_time = time.time()

    for epoch in range(num_epochs):
        # 1. Training Phase
        np.random.shuffle(train_samples)
        epoch_loss = 0.0
        n_batches = len(train_samples) // batch_size

        for b_idx in range(n_batches):
            batch_data = train_samples[b_idx * batch_size : (b_idx + 1) * batch_size]
            b_obs = np.stack([d["obs"] for d in batch_data])
            b_acts = np.stack([d["act_norm"] for d in batch_data])
            b_acts_4d = np.repeat(b_acts[:, None, :, :], 2, axis=1)
            b_texts = [d["text"] for d in batch_data]
            b_timesteps = np.stack([d["timestep"] for d in batch_data])

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

            transformer_embeddings = model.run_transformer(
                obs_dict,
                task_dict,
                timestep_pad_mask=obs_dict["timestep_pad_mask"],
                train=False
            )

            rng, step_rng = jax.random.split(rng)
            action_pad_mask = np.ones((batch_size, 2, 4, 7), dtype=bool)

            head_params, opt_state, loss_val, grad_norm = train_step(
                head_params,
                opt_state,
                transformer_embeddings,
                b_acts_4d,
                obs_dict["timestep_pad_mask"],
                action_pad_mask,
                step_rng
            )
            epoch_loss += float(loss_val)

            if (b_idx + 1) % 5 == 0 or (b_idx + 1) == n_batches:
                sys.stdout.write(
                    f"\r  Epoch {epoch+1:02d}/{num_epochs:02d} | Batch {b_idx+1:03d}/{n_batches:03d} "
                    f"| Loss: {float(loss_val):.4f} | GradNorm: {float(grad_norm):.3f} | Elapsed: {time.time()-start_time:.1f}s"
                )
                sys.stdout.flush()

        avg_train_loss = epoch_loss / n_batches

        # 2. Validation Phase (Unseen tasks/episodes)
        val_loss_total = 0.0
        n_val_batches = len(val_samples) // batch_size
        if n_val_batches == 0:
            val_batches = [val_samples]
        else:
            val_batches = [val_samples[i * batch_size : (i + 1) * batch_size] for i in range(n_val_batches)]

        for v_batch in val_batches:
            cur_bs = len(v_batch)
            vb_obs = np.stack([d["obs"] for d in v_batch])
            vb_acts = np.stack([d["act_norm"] for d in v_batch])
            vb_acts_4d = np.repeat(vb_acts[:, None, :, :], 2, axis=1)
            vb_texts = [d["text"] for d in v_batch]
            vb_timesteps = np.stack([d["timestep"] for d in v_batch])

            v_obs_dict = {
                "image_primary": vb_obs,
                "image_wrist": np.zeros((cur_bs, 2, 128, 128, 3), dtype=np.uint8),
                "pad_mask_dict": {
                    "image_primary": np.ones((cur_bs, 2), dtype=bool),
                    "image_wrist": np.zeros((cur_bs, 2), dtype=bool),
                    "timestep": np.ones((cur_bs, 2), dtype=bool)
                },
                "timestep": vb_timesteps,
                "timestep_pad_mask": np.ones((cur_bs, 2), dtype=bool)
            }
            v_task_dict = model.create_tasks(texts=vb_texts)

            v_embeddings = model.run_transformer(
                v_obs_dict,
                v_task_dict,
                timestep_pad_mask=v_obs_dict["timestep_pad_mask"],
                train=False
            )

            rng, step_rng = jax.random.split(rng)
            v_act_pad_mask = np.ones((cur_bs, 2, 4, 7), dtype=bool)
            v_loss = eval_step(head_params, v_embeddings, vb_acts_4d, v_obs_dict["timestep_pad_mask"], v_act_pad_mask, step_rng)
            val_loss_total += float(v_loss)

        avg_val_loss = val_loss_total / len(val_batches)
        print(f"\n=> Epoch [{epoch+1:02d}/{num_epochs:02d}] Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpoint Saving & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_params = copy.copy(head_params)
            print(f"  [*] New Best Model Found! (Val Loss: {best_val_loss:.4f}) Saving checkpoint...")
            save_ckpt(model, best_params, checkpoint_dir)
        else:
            patience_counter += 1
            print(f"  [!] Val loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation loss plateaued for {early_stopping_patience} consecutive epochs.")
                print(f"Restoring best checkpoint with Val Loss: {best_val_loss:.4f}")
                break

    print("\n" + "=" * 75)
    print("TRAINING RUN COMPLETE")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Optimal Checkpoint Stored in: {checkpoint_dir}")
    print("=" * 75)

def save_ckpt(model, head_params, checkpoint_dir):
    new_params = copy.copy(model.params)
    new_params["heads_action"] = head_params
    finetuned = model.replace(params=new_params)
    finetuned.save_pretrained(checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dobot_front_demonstrations.pkl")
    parser.add_argument("--checkpoint_dir", type=str, default="octo_dobot_front_checkpoint")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--early_stop", type=int, default=4)
    args = parser.parse_args()

    train_dobot_octo(
        dataset_path=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.early_stop
    )
