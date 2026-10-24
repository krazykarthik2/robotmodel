"""Write mp4 without depending on MuJoCo."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_mp4(path: str | Path, frames: list, fps: int = 15) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames")
    arr = [np.asarray(f, dtype=np.uint8) for f in frames]
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, arr, fps=fps)
        return path
    except Exception:
        pass
    try:
        import cv2

        h, w = arr[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError("cv2 VideoWriter failed")
        for f in arr:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return path
    except Exception:
        pass
    # GIF fallback keeps a watchable artifact if codecs are missing.
    gif = path.with_suffix(".gif")
    try:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in arr]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        return gif
    except Exception as exc:
        raise RuntimeError(f"could not write video to {path}") from exc
