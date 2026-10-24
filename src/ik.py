"""Position IK for urdf/robot.urdf: 4 arm joints + unused gripper links."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Match urdf/robot.urdf (meters). Positive joint2 pitches the upper arm toward +X.
SHOULDER_Z = 0.25  # joint1 origin 0.05 + link1 0.20
L_UPPER = 0.20
L_FORE = 0.20
TCP_ALONG_FORE = 0.08  # joint4 -> gripper TCP along last-link +Z
L_TCP = L_FORE + TCP_ALONG_FORE  # 0.28

J_LIMITS = np.array(
    [
        [-np.pi, np.pi],
        [-1.57, 1.57],
        [-1.57, 1.57],
        [-np.pi, np.pi],
    ],
    dtype=np.float64,
)

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")


def _clip_joints(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.clip(q, J_LIMITS[:, 0], J_LIMITS[:, 1])


def _fk_plane(j2: float, j3: float) -> tuple[float, float]:
    """Radial r and world z of the TCP in the arm plane."""
    r = L_UPPER * np.sin(j2) + L_TCP * np.sin(j2 + j3)
    z = SHOULDER_Z + L_UPPER * np.cos(j2) + L_TCP * np.cos(j2 + j3)
    return float(r), float(z)


def forward_kinematics(joint_angles) -> np.ndarray:
    """Return TCP xyz for arm joints [j1, j2, j3, j4]. joint4 does not move TCP."""
    j1, j2, j3, _j4 = _clip_joints(joint_angles)
    r, z = _fk_plane(j2, j3)
    return np.array([r * np.cos(j1), r * np.sin(j1), z], dtype=np.float64)


def _solve_plane(r_des: float, z_des: float, seed_j2j3=None) -> np.ndarray:
    seeds = []
    if seed_j2j3 is not None:
        seeds.append(np.asarray(seed_j2j3, dtype=np.float64).reshape(2))
    seeds.extend(
        [
            np.array([1.35, 1.25]),
            np.array([1.2, 1.4]),
            np.array([1.0, 1.2]),
            np.array([1.4, 0.8]),
            np.array([0.9, -1.1]),
            np.array([0.6, 1.3]),
            np.array([-0.8, 1.3]),
        ]
    )
    for j2s in np.linspace(0.4, 1.5, 5):
        for j3s in np.linspace(0.4, 1.5, 5):
            seeds.append(np.array([j2s, j3s], dtype=np.float64))
    best_q = seeds[0]
    best_err = 1e9
    for seed in seeds:
        j2, j3 = float(seed[0]), float(seed[1])
        for _ in range(40):
            r, z = _fk_plane(j2, j3)
            err = np.array([r_des - r, z_des - z])
            e = float(np.linalg.norm(err))
            if e < best_err:
                best_err = e
                best_q = np.array([j2, j3], dtype=np.float64)
            if e < 1e-4:
                break
            eps = 1e-4
            j = np.array(
                [
                    [
                        (_fk_plane(j2 + eps, j3)[0] - r) / eps,
                        (_fk_plane(j2, j3 + eps)[0] - r) / eps,
                    ],
                    [
                        (_fk_plane(j2 + eps, j3)[1] - z) / eps,
                        (_fk_plane(j2, j3 + eps)[1] - z) / eps,
                    ],
                ]
            )
            try:
                dq = np.linalg.pinv(j, rcond=1e-4) @ err
            except np.linalg.LinAlgError:
                break
            j2 = float(np.clip(j2 + 0.7 * dq[0], -1.57, 1.57))
            j3 = float(np.clip(j3 + 0.7 * dq[1], -1.57, 1.57))
        r, z = _fk_plane(j2, j3)
        e = float(np.hypot(r_des - r, z_des - z))
        if e < best_err:
            best_err = e
            best_q = np.array([j2, j3], dtype=np.float64)
        if best_err < 0.008:
            break
    return best_q


def solve_ik(target_position, seed=None, elbow="down") -> np.ndarray:
    """Numeric 4-DOF position IK. Wrist yaw is copied from seed (or 0)."""
    del elbow
    tx, ty, tz = np.asarray(target_position, dtype=np.float64).reshape(3)
    j1 = float(np.arctan2(ty, tx))
    r = float(np.hypot(tx, ty))
    seed_j = None
    j4 = 0.0
    if seed is not None:
        seed = np.asarray(seed, dtype=np.float64).reshape(-1)
        seed_j = seed[1:3]
        if seed.size > 3:
            j4 = float(seed[3])
    j2, j3 = _solve_plane(r, tz, seed_j2j3=seed_j)
    return _clip_joints([j1, j2, j3, j4])


class RobotIK:
    """IK helper used by the PyBullet env and tests. Optional ikpy chain for debugging."""

    def __init__(self, urdf_path: str | Path | None = None):
        if urdf_path is None:
            urdf_path = Path(__file__).resolve().parents[1] / "urdf" / "robot.urdf"
        self.urdf_path = Path(urdf_path)
        self.chain = None
        try:
            import ikpy.chain

            chain = ikpy.chain.Chain.from_urdf_file(str(self.urdf_path))
            mask = []
            for link in chain.links:
                name = (link.name or "").lower()
                mask.append(any(j in name for j in ("joint1", "joint2", "joint3", "joint4")))
            if any(mask):
                chain.active_links_mask = mask
                self.chain = chain
        except Exception:
            self.chain = None

    def solve_ik(self, target_position, target_orientation=None, seed=None):
        del target_orientation
        return solve_ik(target_position, seed=seed)

    def forward_kinematics(self, joint_angles):
        q = np.asarray(joint_angles, dtype=np.float64).reshape(-1)
        return forward_kinematics(q[:4])


if __name__ == "__main__":
    ik = RobotIK()
    target = [0.26, 0.0, 0.09]
    angles = ik.solve_ik(target)
    reached = ik.forward_kinematics(angles)
    print(f"target={target} joints={angles} reached={reached} err={np.linalg.norm(reached-target)}")
