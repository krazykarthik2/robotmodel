"""Scripted IK expert used to record demos and as eval fallback."""

from __future__ import annotations

import numpy as np

from src.env_pybullet import REST_Q, Robot4DofEnv
from src.ik import solve_ik


def _interp(a, b, n: int):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if n <= 1:
        return [b]
    return [a + (b - a) * (i / (n - 1)) for i in range(n)]


def _hold(env: Robot4DofEnv, q, g: float, steps: int, frames: list):
    action = np.concatenate([np.asarray(q, dtype=np.float64).reshape(4), [g]])
    last = None
    for _ in range(steps):
        obs, _, _, info = env.step(action)
        last = (obs, action.astype(np.float32), info)
        frames.append(last)
    return last


def _move(env: Robot4DofEnv, q_end, g: float, n: int, frames: list):
    q0 = env.get_arm_q()
    last = None
    for q in _interp(q0, q_end, n):
        action = np.concatenate([np.asarray(q, dtype=np.float64).reshape(4), [g]])
        obs, _, _, info = env.step(action)
        last = (obs, action.astype(np.float32), info)
        frames.append(last)
    return last


def rollout_expert(env: Robot4DofEnv, task: str | None = None) -> list:
    task = task or env.task
    frames: list = []
    cube = env.cube_pose()
    hover = cube + np.array([0.0, 0.0, 0.09])
    grasp = cube + np.array([0.0, 0.0, 0.035])
    lift = cube + np.array([0.0, 0.0, 0.14])
    plate = np.array([env._plate_xy[0], env._plate_xy[1], 0.04])
    plate_hover = plate + np.array([0.0, 0.0, 0.12])
    push_goal = np.array([env._push_goal[0], env._push_goal[1], cube[2] + 0.03])
    side = cube + np.array([0.0, -0.07 if cube[1] >= 0 else 0.07, 0.03])

    q_hover = env.ik_joints(hover)
    q_grasp = env.ik_joints(grasp, seed=q_hover)
    q_lift = env.ik_joints(lift, seed=q_grasp)

    if task == "reach":
        _move(env, q_hover, 0.0, 18, frames)
        _move(env, q_grasp, 0.0, 12, frames)
        _hold(env, q_grasp, 0.0, 6, frames)
        return frames

    if task == "push":
        sign = 1.0 if env._push_goal[1] >= cube[1] else -1.0
        hover_b = cube + np.array([0.0, -sign * 0.08, 0.08])
        behind = cube + np.array([0.0, -sign * 0.06, 0.015])
        target = cube + np.array([0.0, sign * 0.14, 0.015])
        q_hb = env.ik_joints(hover_b)
        q_b = env.ik_joints(behind, seed=q_hb)
        q_t = env.ik_joints(target, seed=q_b)
        _move(env, q_hb, 0.85, 12, frames)
        _move(env, q_b, 0.85, 8, frames)
        _move(env, q_t, 0.85, 24, frames)
        _hold(env, q_t, 0.85, 8, frames)
        return frames

    # pick (and place continues after lift)
    _move(env, q_hover, 0.0, 16, frames)
    _move(env, q_grasp, 0.0, 12, frames)
    _hold(env, q_grasp, 0.0, 4, frames)
    _hold(env, q_grasp, 1.0, 10, frames)
    _move(env, q_lift, 1.0, 14, frames)
    _hold(env, q_lift, 1.0, 6, frames)
    if task == "pick":
        return frames

    q_ph = env.ik_joints(plate_hover, seed=q_lift)
    q_pl = env.ik_joints(plate, seed=q_ph)
    _move(env, q_ph, 1.0, 16, frames)
    _move(env, q_pl, 1.0, 12, frames)
    _hold(env, q_pl, 0.0, 10, frames)
    _move(env, q_ph, 0.0, 10, frames)
    _hold(env, REST_Q, 0.0, 8, frames)
    return frames


def expert_success_rate(n: int = 8, tasks=None, gui: bool = False) -> dict:
    tasks = list(tasks or ("pick", "place", "push", "reach"))
    out = {t: [] for t in tasks}
    env = Robot4DofEnv(gui=gui, seed=0)
    try:
        for i, task in enumerate(tasks * max(1, n // len(tasks))):
            if len(out[task]) >= n:
                continue
            env.reset(task=task, seed=1000 + i)
            frames = rollout_expert(env, task)
            ok = bool(frames[-1][2]["success"]) if frames else False
            out[task].append(ok)
    finally:
        env.close()
    rates = {t: float(np.mean(v)) if v else 0.0 for t, v in out.items()}
    rates["overall"] = float(np.mean([x for v in out.values() for x in v])) if any(out.values()) else 0.0
    return rates


# silence unused import if solvers are referenced from notebooks
_ = solve_ik
