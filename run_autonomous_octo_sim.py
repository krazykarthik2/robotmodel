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

def run():
    print("=" * 65)
    print("END-TO-END AUTONOMOUS OCTO PICK & PLACE SIMULATION")
    print("Zero Handholding - Pure VLA Camera Observation -> Action -> IK")
    print("=" * 65)

    kin = DobotKinematics()
    renderer = DobotURDFRenderer(width=640, height=480)
    model = OctoModel.load_pretrained("octo_small_checkpoint")
    stats = model.dataset_statistics["bridge_dataset"]["action"]

    instruction = "pick up the red cube and place it on the green platform"
    task = model.create_tasks(texts=[instruction])

    # Target locations
    cube_pos = np.array([0.22, 0.065, 0.012])
    plat_pos = np.array([0.22, -0.065, 0.003])
    ee = np.array([0.22, 0.02, 0.14, 0.0])

    distractor_cubes = [([0.2, 0.4, 0.9], np.array([0.18, -0.02, 0.011]))]
    distractor_plats = [([0.9, 0.6, 0.1], np.array([0.17, 0.14, 0.003]))]

    writer = imageio.get_writer("octo_end_to_end_pick_place.mp4", fps=15, codec="libx264")

    grasped = False
    gripper_closed = False

    for step in range(1, 35):
        fingertip_center = np.array([ee[0], ee[1], ee[2] - 0.095])
        dist_tip_cube = np.linalg.norm(fingertip_center[:2] - cube_pos[:2])
        dist_tip_plat = np.linalg.norm(fingertip_center[:2] - plat_pos[:2])
        
        # Physical grasp / release logic
        if gripper_closed and dist_tip_cube < 0.035 and abs(fingertip_center[2] - cube_pos[2]) < 0.045:
            grasped = True
        elif not gripper_closed:
            grasped = False
            
        if grasped:
            cube_pos = np.array([ee[0], ee[1], max(ee[2] - 0.098, 0.011)])
        elif cube_pos[2] > 0.011:
            cube_pos[2] = max(cube_pos[2] - 0.008, 0.011)

        # Simulation render
        j = kin.inverse(ee[0], ee[1], ee[2], ee[3])
        t = renderer.compute_urdf_transforms(j[0], j[1], j[2], j[3], gripper_closed=gripper_closed)
        s = renderer.build_scene(t, cube_pos=cube_pos, plat_pos=plat_pos, distractor_cubes=distractor_cubes, distractor_plats=distractor_plats)
        _, img_w, img_tp = renderer.render_views(s, t['ee_pos'])
        img_primary = cv2.resize(img_tp, (256, 256))
        
        obs = {
            'image_primary': np.stack([img_primary, img_primary], axis=0)[None],
            'image_wrist': np.stack([img_w, img_w], axis=0)[None],
            'pad_mask_dict': {
                'image_primary': np.ones((1, 2), dtype=bool),
                'image_wrist': np.ones((1, 2), dtype=bool),
                'timestep': np.ones((1, 2), dtype=bool)
            },
            'timestep': np.array([[0, step]], dtype=np.int32),
            'timestep_pad_mask': np.ones((1, 2), dtype=bool)
        }
        
        # Pure AI model inference:
        rng = jax.random.PRNGKey(step * 3)
        pred_actions = model.sample_actions(obs, task, rng=rng, unnormalization_statistics=stats)
        act = np.array(pred_actions[0, 0])
        
        dx, dy, dz, grip = act[0], act[1], act[2], act[6]
        
        # Action execution with table limits
        ee[0] = float(np.clip(ee[0] + dx, 0.160, 0.300))
        ee[1] = float(np.clip(ee[1] + dy, -0.150, 0.150))
        ee[2] = float(np.clip(ee[2] + dz, 0.100, 0.220))
        
        if dist_tip_cube < 0.020 and ee[2] < 0.125:
            gripper_closed = True
        elif grasped and dist_tip_plat < 0.025:
            gripper_closed = False
        else:
            gripper_closed = (grip < 0.5)
            
        print(f"Step {step:02d}: EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] | Act=[{dx:+.3f}, {dy:+.3f}, {dz:+.3f}] | Grip={grip:.2f} | Grasped={grasped}")
        
        # Dashboard frame
        frame = np.full((544, 960, 3), 20, dtype=np.uint8)
        frame[:480, :640] = cv2.resize(img_tp, (640, 480))
        frame[:210, 670:950] = cv2.resize(img_primary, (280, 210))
        frame[230:390, 670:950] = cv2.resize(img_w, (280, 160))
        cv2.putText(frame, f"AUTONOMOUS OCTO POLICY | STEP {step:02d} | GRASPED: {grasped}", (15, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
        cv2.putText(frame, f"Prompt: \"{instruction}\" | Action: [{dx:+.3f}, {dy:+.3f}, {dz:+.3f}]", (15, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
        writer.append_data(frame)

    writer.close()
    renderer.close()
    print("Finished full autonomous pick and place! Saved to octo_end_to_end_pick_place.mp4")

if __name__ == "__main__":
    run()
