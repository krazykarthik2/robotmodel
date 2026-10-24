"""Tiny skill-conditioned flow policy for 5D joint actions (laptop-friendly).

This is not the 450M SmolVLA checkpoint. It clones expert joint trajectories
with a low-rank (LoRA-style) MLP and 4-step Euler integration so we can train
and run without downloading multi-GB datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "models" / "fast_bc.pt"
SKILLS = ("pick", "place", "push", "reach")


def _no_grad(fn):
    def wrapper(self, *args, **kwargs):
        with self.torch.no_grad():
            return fn(self, *args, **kwargs)

    return wrapper


def _skill_onehot(name: str) -> np.ndarray:
    y = np.zeros(len(SKILLS), dtype=np.float32)
    y[SKILLS.index(name) if name in SKILLS else 0] = 1.0
    return y


class FastFlowPolicy:
    """Conditional flow-matching on flattened 5D actions, LoRA-rank adapters."""

    def __init__(self, hidden: int = 128, rank: int = 16, device: str | None = None):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        in_dim = 5 + len(SKILLS) + 1 + 5  # state + skill + tau + x_tau
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        ).to(self.device)
        # LoRA-style low-rank action head (frozen-wide + adapter).
        self.base_out = nn.Linear(hidden, 5, bias=True).to(self.device)
        self.lora_a = nn.Linear(hidden, rank, bias=False).to(self.device)
        self.lora_b = nn.Linear(rank, 5, bias=False).to(self.device)
        nn.init.zeros_(self.lora_b.weight)
        self.opt = torch.optim.AdamW(
            list(self.net.parameters())
            + list(self.base_out.parameters())
            + list(self.lora_a.parameters())
            + list(self.lora_b.parameters()),
            lr=3e-3,
        )

    def _velocity(self, state, skill, tau, x_tau):
        import torch

        cond = torch.cat([state, skill, tau, x_tau], dim=-1)
        h = self.net(cond)
        # Residual CFM: velocity ~ action-x_tau, with LoRA correction.
        v = self.base_out(h) + self.lora_b(self.lora_a(h))
        return v

    def train_steps(self, states, skills, actions, steps: int = 400, batch: int = 64) -> float:
        import torch

        x1 = torch.tensor(actions, dtype=torch.float32, device=self.device)
        st = torch.tensor(states, dtype=torch.float32, device=self.device)
        sk = torch.tensor(skills, dtype=torch.float32, device=self.device)
        n = x1.shape[0]
        last = 0.0
        for i in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), device=self.device)
            tau = torch.rand(idx.shape[0], 1, device=self.device)
            x0 = torch.randn_like(x1[idx])
            x_tau = (1 - tau) * x0 + tau * x1[idx]
            target = x1[idx] - x0
            pred = self._velocity(st[idx], sk[idx], tau, x_tau)
            loss = torch.nn.functional.huber_loss(pred, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last = float(loss.item())
            if i % 100 == 0:
                print(f"fast_policy step {i}/{steps} loss={last:.4f}")
        return last

    @_no_grad
    def predict_action(self, state, skill: str, num_steps: int = 4) -> np.ndarray:
        import torch

        st = torch.tensor(np.asarray(state, dtype=np.float32).reshape(1, 5), device=self.device)
        sk = torch.tensor(_skill_onehot(skill).reshape(1, 4), device=self.device)
        x = torch.randn(1, 5, device=self.device)
        dt = 1.0 / num_steps
        for k in range(num_steps):
            tau = torch.full((1, 1), k * dt, device=self.device)
            v = self._velocity(st, sk, tau, x)
            x = x + dt * v
        out = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        out[:4] = np.clip(out[:4], [-3.14, -1.57, -1.57, -3.14], [3.14, 1.57, 1.57, 3.14])
        out[4] = np.clip(out[4], 0.0, 1.0)
        return out

    def save(self, path: str | Path = DEFAULT_CKPT):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "net": self.net.state_dict(),
                "base_out": self.base_out.state_dict(),
                "lora_a": self.lora_a.state_dict(),
                "lora_b": self.lora_b.state_dict(),
            },
            path,
        )
        meta = {"kind": "fast_flow_lora", "action_dim": 5, "num_steps": 4}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def load(self, path: str | Path = DEFAULT_CKPT):
        blob = self.torch.load(path, map_location=self.device)
        self.net.load_state_dict(blob["net"])
        self.base_out.load_state_dict(blob["base_out"])
        self.lora_a.load_state_dict(blob["lora_a"])
        self.lora_b.load_state_dict(blob["lora_b"])
        self.net.eval()
        return self


def load_arrays(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)
    data_dir = dataset_dir / "data" / "chunk-000"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    prompts = []
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.append(json.loads(line)["tasks"][0])
    states, actions, skills = [], [], []
    files = sorted(data_dir.glob("episode_*"))
    from src.prompt_skills import parse_skill

    for i, f in enumerate(files):
        skill_name = parse_skill(prompts[i] if i < len(prompts) else "pick")
        oh = _skill_onehot(skill_name)
        if f.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(f)
            st = np.stack(df["observation.state"].to_list()).astype(np.float32)
            ac = np.stack(df["action"].to_list()).astype(np.float32)
        else:
            z = np.load(f)
            st = np.asarray(z["observation.state"], dtype=np.float32)
            ac = np.asarray(z["action"], dtype=np.float32)
        states.append(st)
        actions.append(ac)
        skills.append(np.repeat(oh[None, :], st.shape[0], axis=0))
    return np.concatenate(states), np.concatenate(skills), np.concatenate(actions)
