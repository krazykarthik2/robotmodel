# LeRobot SmolVLA on `urdf/robot.urdf`

Language-conditioned **pick / place / push / reach** on this 4-DOF arm in **PyBullet**. No ROS2. No MuJoCo. No multi-GB robot datasets.

## What runs

- Robot: [`urdf/robot.urdf`](urdf/robot.urdf) — 4 revolute joints + 1-DOF parallel gripper.
- Control: 5D `[j1, j2, j3, j4, gripper]` (gripper `0` open, `1` closed).
- Policy: pretrained **SmolVLA** (`lerobot/smolvla_base`) with **LoRA r=16** when LeRobot is installed; otherwise a tiny **4-step flow / LoRA-head** clone trained on a local synthetic LeRobot dataset.
- Demos: scripted IK expert records data and can finish a prompt so videos actually complete the skill.

## Setup

```bash
pip install -r requirements.txt
# optional, disk-heavy (~1GB weights):
# pip install "lerobot[smolvla,peft]"
```

## Record a tiny dataset (no Bridge / LIBERO)

```bash
python src/record_lerobot.py --episodes-per-task 4
```

Writes `data/robot4dof_vla/` (LeRobot v2.1 layout: parquet + mp4 + `meta/`).

## Train (LoRA, laptop)

```bash
python src/train_smolvla_lora.py --record --steps 400 --skip-smolvla
# or, if lerobot + GPU/disk allow the official checkpoint:
python src/train_smolvla_lora.py --record --steps 400 --smolvla-steps 50
```

`scripts/train_lora.sh` wraps the same commands.

Few-step inference uses `num_steps=4`. Official train flags (when LeRobot is present): `--peft.r=16 --policy.num_steps=4 --policy.n_action_steps=8`.

## Prompt eval + videos

```bash
python src/eval_prompt.py --suite --mode auto
python src/eval_prompt.py --prompt "push the cube to the side" --video viz/push.mp4
```

`--mode auto` runs the learned policy, then the expert if the skill is unfinished (so pick/place/push videos complete). `--mode expert` is IK-only. `--mode fast` is the tiny flow policy only.

Outputs: `viz/pick.mp4`, `viz/place.mp4`, `viz/push.mp4`, `viz/pick_place.mp4`.

## Tests

```bash
python src/test_urdf_pipeline.py
```

## Layout

- `src/env_pybullet.py` — URDF tabletop env
- `src/ik.py` — analytic 4-DOF IK
- `src/expert.py` — scripted skills
- `src/record_lerobot.py` — LeRobot dataset writer
- `src/fast_policy.py` — LoRA-rank flow head
- `src/train_smolvla_lora.py` / `src/eval_prompt.py` / `src/prompt_skills.py`

Legacy MuJoCo scripts remain in the tree but are **not** the execution path.
