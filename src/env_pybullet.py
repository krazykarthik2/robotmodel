"""PyBullet tabletop env for urdf/robot.urdf. No ROS2, no MuJoCo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ik import J_LIMITS, RobotIK, forward_kinematics, solve_ik

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "urdf" / "robot.urdf"

ARM_JOINTS = 4
ACTION_DIM = 5  # j1,j2,j3,j4,gripper  (0=open, 1=closed)
IMAGE_SIZE = 256
TABLE_Z = 0.0
TABLE_HALF = 0.35
REST_Q = np.array([0.0, 1.1, 1.0, 0.0], dtype=np.float64)
GRIPPER_TRAVEL = 0.022


class Robot4DofEnv:
    def __init__(
        self,
        gui: bool = False,
        image_size: int = IMAGE_SIZE,
        urdf_path: str | Path | None = None,
        seed: int | None = None,
    ):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.image_size = int(image_size)
        self.urdf_path = Path(urdf_path) if urdf_path else URDF_PATH
        self.rng = np.random.default_rng(seed)
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(
            str(self.urdf_path),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        self._index_joints()
        self.ik = RobotIK(self.urdf_path)
        self.table_id = None
        self.cube_id = None
        self.plate_id = None
        self.grasp_cid = None
        self.task = "pick"
        self.prompt = "pick the cube"
        self._cube_start = np.zeros(3)
        self._plate_xy = np.array([0.32, 0.12], dtype=np.float64)
        self._push_goal = np.zeros(2)
        self._build_table()
        self.reset()

    def _index_joints(self):
        p = self.p
        self.arm_ids = []
        self.finger_ids = []
        self.ee_link_index = -1
        self.gripper_base_index = -1
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            name = info[1].decode()
            link = info[12].decode()
            if name in ("joint1", "joint2", "joint3", "joint4"):
                self.arm_ids.append(i)
            elif name in ("gripper_left_joint", "gripper_right_joint"):
                self.finger_ids.append(i)
            if link == "ee_link":
                self.ee_link_index = i
            if link == "gripper_base":
                self.gripper_base_index = i
        if len(self.arm_ids) != 4:
            raise RuntimeError(f"expected 4 arm joints, got {self.arm_ids}")
        if len(self.finger_ids) != 2:
            raise RuntimeError(f"expected 2 finger joints, got {self.finger_ids}")
        self.tcp_link = self.gripper_base_index if self.gripper_base_index >= 0 else self.ee_link_index

    def _build_table(self):
        p = self.p
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[TABLE_HALF, TABLE_HALF, 0.02])
        table_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[TABLE_HALF, TABLE_HALF, 0.02], rgbaColor=[0.72, 0.62, 0.48, 1]
        )
        self.table_id = p.createMultiBody(
            0, table_col, table_vis, basePosition=[0.28, 0.0, -0.02]
        )

    def close(self):
        if self.client is not None:
            self.p.disconnect(self.client)
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def reset(self, task: str = "pick", seed: int | None = None, prompt: str | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task = task
        self._release_grasp()
        p = self.p
        for body in (self.cube_id, self.plate_id):
            if body is not None:
                p.removeBody(body)
        self.cube_id = None
        self.plate_id = None

        cube_xy = np.array(
            [self.rng.uniform(0.26, 0.34), self.rng.uniform(-0.10, 0.06)], dtype=np.float64
        )
        plate_xy = np.array(
            [self.rng.uniform(0.28, 0.36), self.rng.uniform(0.08, 0.16)], dtype=np.float64
        )
        if np.linalg.norm(cube_xy - plate_xy) < 0.08:
            plate_xy[1] = cube_xy[1] + 0.12
        self._cube_start = np.array([cube_xy[0], cube_xy[1], 0.025])
        self._plate_xy = plate_xy
        self._push_goal = cube_xy + np.array([0.0, 0.10 if cube_xy[1] <= 0 else -0.10])

        cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        cube_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0.85, 0.15, 0.12, 1]
        )
        self.cube_id = p.createMultiBody(
            0.05, cube_col, cube_vis, basePosition=self._cube_start.tolist()
        )
        p.changeDynamics(self.cube_id, -1, lateralFriction=1.2, rollingFriction=0.01)

        plate_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.045, height=0.01)
        plate_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.045, length=0.01, rgbaColor=[0.9, 0.9, 0.85, 1]
        )
        self.plate_id = p.createMultiBody(
            0, plate_col, plate_vis, basePosition=[plate_xy[0], plate_xy[1], 0.005]
        )

        self._set_arm(REST_Q, gripper=0.0, reset=True)
        for _ in range(30):
            p.stepSimulation()

        self.prompt = prompt if prompt is not None else self._default_prompt(task)
        return self.observe()

    def _default_prompt(self, task: str) -> str:
        if task == "place":
            return "put the cube on the plate"
        if task == "push":
            return "push the cube to the side"
        if task == "reach":
            return "reach the cube"
        return "pick the red cube"

    def _set_arm(self, q, gripper: float, reset: bool = False):
        p = self.p
        q = np.clip(np.asarray(q, dtype=np.float64).reshape(4), J_LIMITS[:, 0], J_LIMITS[:, 1])
        g = float(np.clip(gripper, 0.0, 1.0))
        travel = g * GRIPPER_TRAVEL
        forces = [80, 80, 80, 40]
        for jid, angle, force in zip(self.arm_ids, q, forces):
            if reset:
                p.resetJointState(self.robot, jid, float(angle))
            p.setJointMotorControl2(
                self.robot, jid, p.POSITION_CONTROL, targetPosition=float(angle), force=force
            )
        for jid in self.finger_ids:
            if reset:
                p.resetJointState(self.robot, jid, travel)
            p.setJointMotorControl2(
                self.robot, jid, p.POSITION_CONTROL, targetPosition=travel, force=40
            )

    def get_arm_q(self) -> np.ndarray:
        return np.array(
            [self.p.getJointState(self.robot, j)[0] for j in self.arm_ids], dtype=np.float64
        )

    def get_gripper(self) -> float:
        pos = self.p.getJointState(self.robot, self.finger_ids[0])[0]
        return float(np.clip(pos / GRIPPER_TRAVEL, 0.0, 1.0))

    def tcp_pose(self) -> np.ndarray:
        st = self.p.getLinkState(self.robot, self.tcp_link, computeForwardKinematics=True)
        return np.array(st[0], dtype=np.float64)

    def cube_pose(self) -> np.ndarray:
        pos, _ = self.p.getBasePositionAndOrientation(self.cube_id)
        return np.array(pos, dtype=np.float64)

    def render_image(self) -> np.ndarray:
        p = self.p
        eye = [0.55, -0.35, 0.42]
        target = [0.26, 0.0, 0.04]
        view = p.computeViewMatrix(eye, target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(
            fov=55, aspect=1.0, nearVal=0.05, farVal=1.8
        )
        w = h = self.image_size
        _, _, rgb, _, _ = p.getCameraImage(
            w,
            h,
            view,
            proj,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.asarray(rgb)
        if rgb.ndim == 1:
            rgb = np.reshape(rgb, (h, w, 4))
        img = rgb[:, :, :3].astype(np.uint8)
        return np.ascontiguousarray(img)

    def observe(self) -> dict:
        q = self.get_arm_q()
        g = self.get_gripper()
        state = np.concatenate([q, [g]]).astype(np.float32)
        return {
            "observation.images.front": self.render_image(),
            "observation.state": state,
            "action": state.copy(),
            "prompt": self.prompt,
            "task": self.task,
        }

    def _maybe_grasp(self):
        if self.grasp_cid is not None:
            return
        if self.get_gripper() < 0.72:
            return
        tcp = self.tcp_pose()
        cube = self.cube_pose()
        if np.linalg.norm(tcp - cube) < 0.055:
            self.grasp_cid = self.p.createConstraint(
                self.robot,
                self.tcp_link,
                self.cube_id,
                -1,
                self.p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0.04],
                [0, 0, 0],
            )

    def _release_grasp(self):
        if self.grasp_cid is not None:
            self.p.removeConstraint(self.grasp_cid)
            self.grasp_cid = None

    def step(self, action, n_substeps: int = 12):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"action dim {action.shape} != {ACTION_DIM}")
        prev_g = self.get_gripper()
        self._set_arm(action[:4], action[4])
        for _ in range(n_substeps):
            self.p.stepSimulation()
        g = self.get_gripper()
        if g >= 0.72:
            self._maybe_grasp()
        elif prev_g >= 0.5 and g < 0.45:
            self._release_grasp()
        obs = self.observe()
        info = self.success_info()
        return obs, float(info["success"]), False, info

    def success_info(self) -> dict:
        cube = self.cube_pose()
        picked = bool(cube[2] > 0.07)
        placed = bool(
            np.linalg.norm(cube[:2] - self._plate_xy) < 0.055 and cube[2] < 0.06
        )
        pushed = bool(np.linalg.norm(cube[:2] - self._cube_start[:2]) > 0.07)
        reached = bool(np.linalg.norm(self.tcp_pose() - cube) < 0.06)
        if self.task == "place":
            success = placed
        elif self.task == "push":
            success = pushed
        elif self.task == "reach":
            success = reached
        else:
            success = picked
        return {
            "success": success,
            "picked": picked,
            "placed": placed,
            "pushed": pushed,
            "reached": reached,
            "cube": cube,
            "tcp": self.tcp_pose(),
        }

    def ik_joints(self, xyz, seed=None) -> np.ndarray:
        seed = self.get_arm_q() if seed is None else seed
        q = solve_ik(xyz, seed=seed)
        # Snap with PyBullet IK if the analytic miss is large.
        err = np.linalg.norm(forward_kinematics(q) - np.asarray(xyz, dtype=np.float64))
        if err > 0.03:
            pb_q = self.p.calculateInverseKinematics(
                self.robot,
                self.tcp_link,
                xyz,
                maxNumIterations=80,
                residualThreshold=0.004,
            )
            q = np.array(pb_q[:4], dtype=np.float64)
        return np.clip(q, J_LIMITS[:, 0], J_LIMITS[:, 1])
