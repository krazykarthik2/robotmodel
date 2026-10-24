"""LoRA fine-tune path: tiny FastFlow policy always; official SmolVLA when LeRobot is installed."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fast_policy import DEFAULT_CKPT, FastFlowPolicy, load_arrays
from src.record_lerobot import DEFAULT_ROOT, record_dataset


def train_fast(dataset_dir: Path, steps: int, ckpt: Path) -> Path:
    states, skills, actions = load_arrays(dataset_dir)
    print(f"fast_policy samples={len(states)} action_dim={actions.shape[-1]}")
    policy = FastFlowPolicy(hidden=128, rank=16)
    policy.train_steps(states, skills, actions, steps=steps, batch=64)
    policy.save(ckpt)
    print(f"saved {ckpt}")
    return ckpt


def try_lerobot_lora(dataset_dir: Path, steps: int, output_dir: Path) -> bool:
    """Official HuggingFace LeRobot SmolVLA + PEFT LoRA. Skips if not installed."""
    try:
        import lerobot  # noqa: F401
    except Exception as exc:
        print(f"lerobot not installed ({exc}); skipping smolvla_base LoRA download/train")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.train",
        "--policy.path=lerobot/smolvla_base",
        f"--dataset.root={dataset_dir}",
        "--dataset.repo_id=local/robot4dof_vla",
        "--policy.output_features=null",
        "--policy.input_features=null",
        "--peft.method_type=LORA",
        "--peft.r=16",
        "--peft.lora_alpha=16",
        "--batch_size=1",
        f"--steps={steps}",
        "--policy.optimizer_lr=1e-3",
        "--policy.n_action_steps=8",
        "--policy.num_steps=4",
        f"--output_dir={output_dir}",
    ]
    print("running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=False, env=env, cwd=str(ROOT))
        return True
    except Exception as exc:
        print(f"lerobot-train failed: {exc}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_ROOT))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--smolvla-steps", type=int, default=200)
    ap.add_argument("--record", action="store_true", help="record a tiny dataset if missing")
    ap.add_argument("--skip-smolvla", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if args.record or not (dataset / "meta" / "info.json").exists():
        record_dataset(dataset, episodes_per_task=4)

    train_fast(dataset, steps=args.steps, ckpt=DEFAULT_CKPT)
    if not args.skip_smolvla:
        try_lerobot_lora(dataset, steps=args.smolvla_steps, output_dir=ROOT / "models" / "smolvla_lora")


if __name__ == "__main__":
    main()
