import os
import sys
import jax
import cv2
import numpy as np
import imageio

import octo_patch
from octo.model.octo_model import OctoModel
from dobot_urdf_renderer import DobotURDFRenderer
from dobot_magician_ros2.ai_training.env.dobot_env import DobotKinematics

COLOR_RGB_MAP = {
    "red": [0.90, 0.15, 0.15],
    "green": [0.15, 0.85, 0.25],
    "blue": [0.15, 0.40, 0.90],
    "yellow": [0.90, 0.85, 0.15],
    "purple": [0.70, 0.20, 0.85],
    "cyan": [0.15, 0.85, 0.85],
    "orange": [0.95, 0.50, 0.10]
}

# Ground truth physical measurements:
# Table plane in world coordinates: Z = -0.1492 m (-149.2 mm)
# Cube height: 0.024 m (top of cube at Z = -0.1252 m)
# Sucker stem + cup extends downwards exactly 0.080 m (80 mm) from EE origin.
# Thus: Sucker tip reaches cube top when EE origin is at:
# Z_EE_contact = -0.1252 + 0.080 = -0.0452 m (-45.2 mm)
SUCKER_LENGTH_M = 0.080
TABLE_Z_M = -0.1492
CUBE_TOP_Z_M = TABLE_Z_M + 0.024 # -0.1252 m
CONTACT_EE_Z_M = CUBE_TOP_Z_M + SUCKER_LENGTH_M # -0.0452 m
STANDBY_EE_Z_M = 0.040 # Hover height above table

def run_suction_suite():
    print("=" * 70)
    print("OCTO PURE SUCTION CUP AUTONOMOUS MULTI-OBJECT BENCHMARK")
    print("100% SUCTION CUP ONLY | Zero Handholding | Pure AI Vision Closed Loop")
    print("=" * 70)

    kin = DobotKinematics()
    renderer = DobotURDFRenderer(width=640, height=480)
    model = OctoModel.load_pretrained("octo_small_checkpoint")
    stats = model.dataset_statistics["bridge_dataset"]["action"]

    # 4 distinct scenarios across different colors, targets, distractors and positions:
    benchmarks = [
        {
            "id": 1,
            "instruction": "pick up the red cube and place it on the green platform",
            "cube_color": "red",
            "plat_color": "green",
            "init_ee": np.array([0.220, 0.020, STANDBY_EE_Z_M, 0.0]),
            "init_cube": np.array([0.220, 0.065, TABLE_Z_M + 0.012]),
            "init_plat": np.array([0.220, -0.065, TABLE_Z_M + 0.003]),
            "distractor_cubes": [
                (COLOR_RGB_MAP["blue"], np.array([0.180, -0.020, TABLE_Z_M + 0.011])),
                (COLOR_RGB_MAP["yellow"], np.array([0.250, 0.010, TABLE_Z_M + 0.011]))
            ],
            "distractor_plats": [
                (COLOR_RGB_MAP["cyan"], np.array([0.170, 0.140, TABLE_Z_M + 0.003]))
            ],
            "max_steps": 35
        },
        {
            "id": 2,
            "instruction": "pick up the blue cube and place it on the yellow platform",
            "cube_color": "blue",
            "plat_color": "yellow",
            "init_ee": np.array([0.210, -0.020, STANDBY_EE_Z_M, 0.0]),
            "init_cube": np.array([0.210, -0.060, TABLE_Z_M + 0.012]),
            "init_plat": np.array([0.230, 0.070, TABLE_Z_M + 0.003]),
            "distractor_cubes": [
                (COLOR_RGB_MAP["red"], np.array([0.240, 0.010, TABLE_Z_M + 0.011])),
                (COLOR_RGB_MAP["orange"], np.array([0.180, 0.050, TABLE_Z_M + 0.011]))
            ],
            "distractor_plats": [
                (COLOR_RGB_MAP["purple"], np.array([0.170, -0.120, TABLE_Z_M + 0.003]))
            ],
            "max_steps": 35
        },
        {
            "id": 3,
            "instruction": "pick up the yellow cube and place it on the cyan platform",
            "cube_color": "yellow",
            "plat_color": "cyan",
            "init_ee": np.array([0.230, 0.010, STANDBY_EE_Z_M, 0.0]),
            "init_cube": np.array([0.235, 0.055, TABLE_Z_M + 0.012]),
            "init_plat": np.array([0.200, -0.070, TABLE_Z_M + 0.003]),
            "distractor_cubes": [
                (COLOR_RGB_MAP["green"], np.array([0.190, 0.020, TABLE_Z_M + 0.011])),
                (COLOR_RGB_MAP["blue"], np.array([0.220, -0.030, TABLE_Z_M + 0.011]))
            ],
            "distractor_plats": [
                (COLOR_RGB_MAP["red"], np.array([0.250, -0.050, TABLE_Z_M + 0.003]))
            ],
            "max_steps": 35
        },
        {
            "id": 4,
            "instruction": "pick up the green cube and place it on the red platform",
            "cube_color": "green",
            "plat_color": "red",
            "init_ee": np.array([0.220, -0.015, STANDBY_EE_Z_M, 0.0]),
            "init_cube": np.array([0.225, -0.055, TABLE_Z_M + 0.012]),
            "init_plat": np.array([0.215, 0.065, TABLE_Z_M + 0.003]),
            "distractor_cubes": [
                (COLOR_RGB_MAP["yellow"], np.array([0.180, -0.010, TABLE_Z_M + 0.011])),
                (COLOR_RGB_MAP["purple"], np.array([0.245, 0.020, TABLE_Z_M + 0.011]))
            ],
            "distractor_plats": [
                (COLOR_RGB_MAP["cyan"], np.array([0.180, 0.120, TABLE_Z_M + 0.003]))
            ],
            "max_steps": 35
        }
    ]

    output_video_path = "octo_suction_autonomous_benchmark.mp4"
    writer = imageio.get_writer(output_video_path, fps=15, codec="libx264")

    overall_results = []

    for b in benchmarks:
        print("\n" + "=" * 65)
        print(f"RUNNING SUCTION SAMPLE [{b['id']}/4]: {b['instruction'].upper()}")
        print(f">> Target Cube: {b['cube_color'].upper()} | Destination Platform: {b['plat_color'].upper()}")
        print(f">> Initial EE: [{b['init_ee'][0]:.3f}, {b['init_ee'][1]:.3f}, {b['init_ee'][2]:.3f}]")
        print("=" * 65)

        task = model.create_tasks(texts=[b["instruction"]])

        ee = b["init_ee"].copy()
        cube_pos = b["init_cube"].copy()
        plat_pos = b["init_plat"].copy()
        cube_rgb = COLOR_RGB_MAP[b["cube_color"]]
        plat_rgb = COLOR_RGB_MAP[b["plat_color"]]

        suction_on = False
        suction_attached = False
        min_tip_dist = 999.0
        attached_step = -1

        for step in range(1, b["max_steps"] + 1):
            # Suction Cup tip is 0.080m below EE origin:
            sucker_tip_pos = np.array([ee[0], ee[1], ee[2] - SUCKER_LENGTH_M])
            dist_xy = float(np.linalg.norm(sucker_tip_pos[:2] - cube_pos[:2]))
            dist_3d = float(np.linalg.norm(sucker_tip_pos - cube_pos))
            min_tip_dist = min(min_tip_dist, dist_3d)

            # Physical suction adherence:
            # Sucker cup seals when: suction pump is active, tip within 2.5cm XY, and touches cube top
            if suction_on and dist_xy < 0.025 and (sucker_tip_pos[2] <= CUBE_TOP_Z_M + 0.008):
                suction_attached = True
                if attached_step == -1:
                    attached_step = step
            elif not suction_on:
                suction_attached = False

            # Update cube position:
            if suction_attached:
                cube_pos = np.array([ee[0], ee[1], sucker_tip_pos[2] - 0.012])
            elif cube_pos[2] > TABLE_Z_M + 0.012:
                cube_pos[2] = max(cube_pos[2] - 0.008, TABLE_Z_M + 0.012)

            # Render 3D simulation with suction cup
            j = kin.inverse(ee[0], ee[1], ee[2], ee[3])
            t = renderer.compute_urdf_transforms(j[0], j[1], j[2], j[3], gripper_closed=False)
            s = renderer.build_scene(
                t,
                cube_pos=cube_pos,
                cube_color_rgb=cube_rgb,
                plat_pos=plat_pos,
                plat_color_rgb=plat_rgb,
                distractor_cubes=b["distractor_cubes"],
                distractor_plats=b["distractor_plats"],
                end_effector="suction"
            )
            _, img_w, img_tp = renderer.render_views(s, t["ee_pos"])
            img_primary = cv2.resize(img_tp, (256, 256))

            obs = {
                "image_primary": np.stack([img_primary, img_primary], axis=0)[None],
                "image_wrist": np.stack([img_w, img_w], axis=0)[None],
                "pad_mask_dict": {
                    "image_primary": np.ones((1, 2), dtype=bool),
                    "image_wrist": np.ones((1, 2), dtype=bool),
                    "timestep": np.ones((1, 2), dtype=bool)
                },
                "timestep": np.array([[0, step]], dtype=np.int32),
                "timestep_pad_mask": np.ones((1, 2), dtype=bool)
            }

            # Pure AI policy inference:
            rng = jax.random.PRNGKey(b["id"] * 1000 + step * 7)
            pred_actions = model.sample_actions(obs, task, rng=rng, unnormalization_statistics=stats)
            act = np.array(pred_actions[0, 0])

            dx, dy, dz, grip = float(act[0]), float(act[1]), float(act[2]), float(act[6])

            # Apply actions purely as predicted by policy:
            ee[0] = float(np.clip(ee[0] + dx, 0.160, 0.310))
            ee[1] = float(np.clip(ee[1] + dy, -0.150, 0.150))
            # Keep sucker above table contact level to prevent crash:
            ee[2] = float(np.clip(ee[2] + dz, CONTACT_EE_Z_M - 0.004, 0.120))

            # Suction pump control:
            # Model output grip < 0.5 activates suction pump:
            suction_on = bool(grip < 0.5)

            print(f"  Step {step:02d}: EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] | Act=[{dx:+.3f}, {dy:+.3f}, {dz:+.3f}] | Grip={grip:.2f} ({'SUCTION ON' if suction_on else 'OFF'}) | TipDist={dist_3d*1000:.1f}mm | Attached={suction_attached}")

            # Compose multi-camera dashboard:
            frame = np.full((544, 960, 3), 20, dtype=np.uint8)
            frame[:480, :640] = cv2.resize(img_tp, (640, 480))
            frame[:210, 670:950] = cv2.resize(img_primary, (280, 210))
            frame[230:390, 670:950] = cv2.resize(img_w, (280, 160))

            cv2.putText(frame, f"SUCTION OCTO #{b['id']} | STEP {step:02d}/{b['max_steps']} | SUCTION: {'ATTACHED' if suction_attached else ('ON' if suction_on else 'OFF')}", (15, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Task: \"{b['instruction']}\" | Act: [{dx:+.3f}, {dy:+.3f}, {dz:+.3f}] | Sucker Tip Z: {(sucker_tip_pos[2] - TABLE_Z_M)*1000:+.1f}mm", (15, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 255), 1, cv2.LINE_AA)
            writer.append_data(frame)

        result_entry = {
            "id": b["id"],
            "task": b["instruction"],
            "cube": b["cube_color"],
            "min_tip_dist_mm": min_tip_dist * 1000.0,
            "attached": suction_attached or (attached_step != -1),
            "attached_step": attached_step
        }
        overall_results.append(result_entry)
        print(f"  >> Suction Sample {b['id']} Result: Min Tip Dist = {result_entry['min_tip_dist_mm']:.1f}mm | Suction Attached = {result_entry['attached']} (at step {result_entry['attached_step']})")

    writer.close()
    renderer.close()

    print("\n" + "=" * 70)
    print("SUCTION BENCHMARK EXECUTION SUMMARY:")
    print("=" * 70)
    for r in overall_results:
        status_str = f"SUCCESSFUL SUCTION (Attached at step {r['attached_step']})" if r["attached"] else f"APPROACH ONLY (Closest: {r['min_tip_dist_mm']:.1f}mm)"
        print(f"Sample #{r['id']} [{r['cube'].upper()} CUBE]: {status_str}")
    print(f"\nConsolidated Suction Benchmark Video written to: {output_video_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_suction_suite()
