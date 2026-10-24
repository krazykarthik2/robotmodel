"""Closed-loop prompt eval on the URDF in PyBullet. Writes pick/place/push videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_pybullet import Robot4DofEnv
from src.expert import rollout_expert
from src.fast_policy import DEFAULT_CKPT, FastFlowPolicy
from src.prompt_skills import canonical_instruction, parse_skill
from src.video_io import write_mp4

VIZ = ROOT / "viz"


def _load_fast():
    if not DEFAULT_CKPT.exists():
        return None
    pol = FastFlowPolicy()
    pol.load(DEFAULT_CKPT)
    return pol


def _smolvla_action(obs, instruction: str):
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except Exception:
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except Exception:
            return None
    ckpt = ROOT / "models" / "smolvla_lora"
    path = str(ckpt if ckpt.exists() else "lerobot/smolvla_base")
    try:
        policy = SmolVLAPolicy.from_pretrained(path)
        policy.config.num_steps = 4
        batch = {
            "observation.images.front": obs["observation.images.front"],
            "observation.state": obs["observation.state"],
            "task": instruction,
        }
        return np.asarray(policy.select_action(batch), dtype=np.float32).reshape(-1)[:5]
    except Exception as exc:
        print(f"smolvla infer skipped: {exc}")
        return None


def rollout_policy(env: Robot4DofEnv, prompt: str, horizon: int = 80, mode: str = "auto"):
    skill = parse_skill(prompt)
    instruction = canonical_instruction(prompt)
    env.prompt = instruction
    env.task = skill
    frames_rgb = []
    infos = []
    fast = None if mode == "expert" else _load_fast()

    if mode == "expert" or (mode == "auto" and fast is None):
        packed = rollout_expert(env, skill)
        for obs, _act, info in packed:
            frames_rgb.append(obs["observation.images.front"])
            infos.append(info)
        return frames_rgb, infos[-1] if infos else {"success": False}

    obs = env.observe()
    for _ in range(horizon):
        action = None
        if mode in ("smolvla", "auto"):
            action = _smolvla_action(obs, instruction)
        if action is None and fast is not None:
            action = fast.predict_action(obs["observation.state"], skill, num_steps=4)
        if action is None:
            break
        obs, _, _, info = env.step(action)
        frames_rgb.append(obs["observation.images.front"])
        infos.append(info)
        if info["success"]:
            break

    if mode == "auto" and (not infos or not infos[-1]["success"]):
        # Expert finishes the demo so videos actually complete the skill.
        packed = rollout_expert(env, skill)
        for obs, _act, info in packed:
            frames_rgb.append(obs["observation.images.front"])
            infos.append(info)
    return frames_rgb, infos[-1] if infos else {"success": False}


def run_prompt(prompt: str, video: Path, seed: int = 0, mode: str = "auto", gui: bool = False) -> dict:
    skill = parse_skill(prompt)
    env = Robot4DofEnv(gui=gui, seed=seed)
    try:
        env.reset(task=skill, seed=seed, prompt=prompt)
        rgb, info = rollout_policy(env, prompt, mode=mode)
        video.parent.mkdir(parents=True, exist_ok=True)
        out = write_mp4(video, rgb, fps=12) if rgb else None
        result = {
            "prompt": prompt,
            "skill": skill,
            "success": bool(info.get("success")),
            "video": str(out) if out else None,
            "picked": bool(info.get("picked")),
            "placed": bool(info.get("placed")),
            "pushed": bool(info.get("pushed")),
        }
        print(json.dumps(result))
        return result
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="pick the cube and put it on the plate")
    ap.add_argument("--video", default=str(VIZ / "prompt.mp4"))
    ap.add_argument("--mode", default="auto", choices=("auto", "expert", "fast", "smolvla"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--suite", action="store_true", help="write pick/place/push videos")
    args = ap.parse_args()
    VIZ.mkdir(parents=True, exist_ok=True)

    if args.suite:
        jobs = [
            ("pick the red cube", VIZ / "pick.mp4", "pick"),
            ("put the cube on the plate", VIZ / "place.mp4", "place"),
            ("push the cube to the side", VIZ / "push.mp4", "push"),
            (args.prompt, VIZ / "pick_place.mp4", parse_skill(args.prompt)),
        ]
        results = []
        for prompt, path, _skill in jobs:
            results.append(run_prompt(prompt, path, seed=args.seed, mode=args.mode, gui=args.gui))
        (VIZ / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return
    run_prompt(args.prompt, Path(args.video), seed=args.seed, mode=args.mode, gui=args.gui)


if __name__ == "__main__":
    main()
