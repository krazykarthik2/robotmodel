"""CPU tests: URDF, IK, PyBullet expert, dataset schema, fast policy action dim."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_pybullet import ACTION_DIM, Robot4DofEnv
from src.expert import expert_success_rate, rollout_expert
from src.ik import ARM_JOINT_NAMES, RobotIK, forward_kinematics, solve_ik
from src.prompt_skills import parse_skill
from src.record_lerobot import record_dataset


def test_ik_reaches():
    ik = RobotIK()
    target = np.array([0.26, 0.04, 0.09])
    q = ik.solve_ik(target)
    assert q.shape == (4,)
    reached = forward_kinematics(q)
    err = np.linalg.norm(reached - target)
    assert err < 0.05, f"IK err {err} reached={reached} target={target}"
    assert len(ARM_JOINT_NAMES) == 4


def test_urdf_loads():
    env = Robot4DofEnv(gui=False, seed=0)
    try:
        assert len(env.arm_ids) == 4
        assert len(env.finger_ids) == 2
        obs = env.reset(task="pick", seed=1)
        assert obs["observation.state"].shape == (ACTION_DIM,)
        assert obs["observation.images.front"].ndim == 3
        q = env.ik_joints([0.25, 0.0, 0.08])
        obs, _, _, info = env.step(np.concatenate([q, [0.0]]))
        assert "success" in info
    finally:
        env.close()


def test_prompt_parser():
    assert parse_skill("pick the red cube") == "pick"
    assert parse_skill("put the cube on the plate") == "place"
    assert parse_skill("push the block left") == "push"
    assert parse_skill("reach the cube") == "reach"


def test_expert_skills():
    rates = expert_success_rate(n=2, tasks=("pick", "place", "push"))
    print("expert rates", rates)
    for k in ("pick", "place", "push"):
        assert rates[k] >= 0.5, rates


def test_dataset_schema():
    with tempfile.TemporaryDirectory() as td:
        out = record_dataset(Path(td) / "ds", episodes_per_task=1, fps=8, image_size=64)
        info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
        assert info["features"]["action"]["shape"] == [5]
        assert info["total_episodes"] >= 4
        assert (out / "videos" / "observation.images.front" / "chunk-000").exists()


def test_fast_policy_action_dim():
    try:
        import torch  # noqa: F401
    except Exception:
        print("torch missing, skip fast policy")
        return
    from src.fast_policy import FastFlowPolicy

    pol = FastFlowPolicy(hidden=32, rank=4)
    states = np.random.randn(32, 5).astype(np.float32)
    skills = np.eye(4, dtype=np.float32)[np.random.randint(0, 4, 32)]
    actions = np.clip(states + 0.01, -1, 1).astype(np.float32)
    actions[:, 4] = 0.5
    pol.train_steps(states, skills, actions, steps=5, batch=8)
    a = pol.predict_action(states[0], "pick", num_steps=4)
    assert a.shape == (5,), a.shape


def test_closed_loop_pick_video(tmp_path=None):
    from src.video_io import write_mp4

    env = Robot4DofEnv(gui=False, seed=2)
    try:
        env.reset(task="pick", seed=2)
        frames = rollout_expert(env, "pick")
        rgb = [f[0]["observation.images.front"] for f in frames]
        assert frames[-1][2]["picked"] or frames[-1][2]["success"]
        dest = ROOT / "viz" / "_test_pick.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_mp4(dest, rgb, fps=8)
        assert dest.exists() or dest.with_suffix(".gif").exists()
    finally:
        env.close()


def run_all():
    tests = [
        test_ik_reaches,
        test_prompt_parser,
        test_urdf_loads,
        test_expert_skills,
        test_dataset_schema,
        test_fast_policy_action_dim,
        test_closed_loop_pick_video,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        raise SystemExit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    run_all()
