"""Record a tiny LeRobot-format dataset from the PyBullet expert. No huge downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_pybullet import ACTION_DIM, IMAGE_SIZE, Robot4DofEnv
from src.expert import rollout_expert
from src.prompt_skills import sample_instruction
from src.video_io import write_mp4

DEFAULT_ROOT = ROOT / "data" / "robot4dof_vla"


def _write_parquet(path: Path, rows: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    except Exception:
        pass
    np.savez_compressed(path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in rows.items()})


def record_dataset(
    out_dir: str | Path = DEFAULT_ROOT,
    episodes_per_task: int = 6,
    fps: int = 12,
    image_size: int = IMAGE_SIZE,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    video_dir = out_dir / "videos" / "observation.images.front" / "chunk-000"
    data_dir = out_dir / "data" / "chunk-000"
    meta_dir = out_dir / "meta"
    for d in (video_dir, data_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    tasks = ("pick", "place", "push", "reach")
    env = Robot4DofEnv(gui=False, image_size=image_size, seed=seed)
    episodes = []
    tasks_jsonl = []
    total_frames = 0
    task_to_index = {t: i for i, t in enumerate(tasks)}
    rng = np.random.default_rng(seed)

    try:
        ep_idx = 0
        for task in tasks:
            for k in range(episodes_per_task):
                prompt = sample_instruction(task, rng)
                env.reset(task=task, seed=seed + 17 * ep_idx + k, prompt=prompt)
                frames = rollout_expert(env, task)
                if not frames:
                    continue
                rgb = [f[0]["observation.images.front"] for f in frames]
                states = np.stack([f[0]["observation.state"] for f in frames]).astype(np.float32)
                actions = np.stack([f[1] for f in frames]).astype(np.float32)
                n = len(frames)
                ep_name = f"episode_{ep_idx:06d}"
                write_mp4(video_dir / f"{ep_name}.mp4", rgb, fps=fps)
                rows = {
                    "observation.state": [s.tolist() for s in states],
                    "action": [a.tolist() for a in actions],
                    "timestamp": (np.arange(n, dtype=np.float32) / fps).tolist(),
                    "frame_index": np.arange(n, dtype=np.int64).tolist(),
                    "episode_index": np.full(n, ep_idx, dtype=np.int64).tolist(),
                    "index": np.arange(total_frames, total_frames + n, dtype=np.int64).tolist(),
                    "task_index": np.full(n, task_to_index[task], dtype=np.int64).tolist(),
                }
                _write_parquet(data_dir / f"{ep_name}.parquet", rows)
                episodes.append(
                    {
                        "episode_index": ep_idx,
                        "tasks": [prompt],
                        "length": n,
                    }
                )
                tasks_jsonl.append({"task_index": task_to_index[task], "task": prompt})
                total_frames += n
                ep_idx += 1
    finally:
        env.close()

    info = {
        "codebase_version": "v2.1",
        "robot_type": "robot_4dof",
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"},
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [image_size, image_size, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.fps": fps, "video.codec": "mp4v"},
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ["j1", "j2", "j3", "j4", "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ["j1", "j2", "j3", "j4", "gripper"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes:
            f.write(json.dumps(row) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        seen = set()
        for row in tasks_jsonl:
            key = (row["task_index"], row["task"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(row) + "\n")
    stats = {
        "observation.state": {
            "min": [-3.14, -1.57, -1.57, -3.14, 0.0],
            "max": [3.14, 1.57, 1.57, 3.14, 1.0],
            "mean": [0.0, -0.7, 1.0, 0.0, 0.4],
            "std": [0.6, 0.4, 0.4, 0.4, 0.4],
        },
        "action": {
            "min": [-3.14, -1.57, -1.57, -3.14, 0.0],
            "max": [3.14, 1.57, 1.57, 3.14, 1.0],
            "mean": [0.0, -0.7, 1.0, 0.0, 0.4],
            "std": [0.6, 0.4, 0.4, 0.4, 0.4],
        },
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {len(episodes)} episodes, {total_frames} frames -> {out_dir}")
    return out_dir


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_ROOT))
    ap.add_argument("--episodes-per-task", type=int, default=4)
    args = ap.parse_args()
    record_dataset(args.out, episodes_per_task=args.episodes_per_task)


if __name__ == "__main__":
    main()
