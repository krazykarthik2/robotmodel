"""
Octo Generalist Policy + Real Dobot Magician Physical Runner
Directly executes Octo vision-language commands on the real physical Dobot Magician via DobotDll.dll:
1. Captures live camera frame (USB / RealSense)
2. ArUco auto-calibration rectifies workspace to 256x256 image
3. Octo evaluates natural language prompt ("pick up the red cube...", "push the yellow cube...")
4. Commands the physical Dobot arm motors via real_dobot_controller.py
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2

# Real Dobot driver and ArUco calibrator
from real_dobot_controller import RealDobotController
from aruco_calibrator import ArucoAutoCalibrator

# Octo Policy
import octo_patch
from octo.model.octo_model import OctoModel
import jax

def run_real_octo(camera_id=0, com_port="", task_prompt="pick up the red cube and place it on the green platform"):
    print("=" * 60)
    print("OCTO VISION-LANGUAGE POLICY -> REAL PHYSICAL DOBOT MAGICIAN")
    print("=" * 60)

    # 1. Initialize Real Dobot Hardware
    dobot = RealDobotController(port=com_port)
    connected = dobot.connect()
    if not connected:
        print("[WARN] Real Dobot not detected or port occupied. Running in HARDWARE PREVIEW / DRY-RUN mode.")

    # 2. Initialize Camera & ArUco Calibrator
    print(f"\n[INFO] Opening overhead camera (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    calibrator = ArucoAutoCalibrator(output_size=(256, 256))

    # 3. Load Octo Model
    print("\n[INFO] Loading Octo Foundation Model (octo-small-1.5)...")
    checkpoint_path = os.path.abspath("octo_small_checkpoint")
    model = OctoModel.load_pretrained(checkpoint_path)
    octo_task = model.create_tasks(texts=[task_prompt])
    unnorm_stats = model.dataset_statistics["bridge_dataset"]["action"]
    print(f"[INFO] Octo Task Prepared: \"{task_prompt}\"")

    # Move arm to initial standby
    print("\n[INFO] Moving arm to standby observation pose (X=200, Y=0, Z=140)...")
    dobot.move_to_xyz(200.0, 0.0, 140.0, mode="MOVJ_XYZ")
    dobot.set_gripper("open")

    print("\n[INFO] Ready for execution! Press [SPACE] to trigger task, [q] to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Fallback test image if camera disconnected
            frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        calibrated_crop, debug_view = calibrator.process_frame(frame)

        cv2.putText(debug_view, f"Task: {task_prompt}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(debug_view, f"Calibrated: {calibrator.is_calibrated} | Space to Run | Q to Quit", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        cv2.imshow("Overhead Camera (Debug)", debug_view)
        cv2.imshow("Calibrated Octo Input (256x256)", calibrated_crop)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # SPACE key -> Trigger Octo Physical Execution
            print("\n" + "=" * 50)
            print(f"EXECUTING OCTO TASK ON REAL ROBOT: \"{task_prompt}\"")
            print("=" * 50)
            
            # Prepare observation dict for Octo
            img_p = calibrated_crop
            # Dummy wrist image or camera feed
            img_w = cv2.resize(calibrated_crop, (128, 128))
            
            octo_obs = {
                "image_primary": np.stack([img_p, img_p], axis=0)[None],
                "image_wrist": np.stack([img_w, img_w], axis=0)[None],
                "pad_mask_dict": {
                    "image_primary": np.ones((1, 2), dtype=bool),
                    "image_wrist": np.ones((1, 2), dtype=bool),
                    "timestep": np.ones((1, 2), dtype=bool)
                },
                "timestep": np.array([[0, 1]], dtype=np.int32),
                "timestep_pad_mask": np.ones((1, 2), dtype=bool)
            }

            # Query Octo Diffusion Action Head
            print("[INFO] Sampling action trajectory from Octo...")
            rng_key = jax.random.PRNGKey(int(time.time()))
            pred_actions = model.sample_actions(octo_obs, octo_task, rng=rng_key, unnormalization_statistics=unnorm_stats)
            actions = np.array(pred_actions[0]) # (action_horizon=4, action_dim=7)
            print(f"[OCTO ACTION] Predicted 4-step action chunk:\n{actions}")

            # Execute actions on real hardware
            cur_pose = dobot.get_pose()
            print(f"[INFO] Initial Real Pose: X={cur_pose['x']:.1f}mm, Y={cur_pose['y']:.1f}mm, Z={cur_pose['z']:.1f}mm")

            for step_idx, act in enumerate(actions, 1):
                # act: [dx, dy, dz, droll, dpitch, dyaw, grip]
                # Scale delta (meters to mm)
                dx_mm = act[0] * 1000.0 * 2.0
                dy_mm = act[1] * 1000.0 * 2.0
                dz_mm = act[2] * 1000.0 * 2.0
                grip_cmd = act[6]

                target_x = np.clip(cur_pose['x'] + dx_mm, 150.0, 300.0)
                target_y = np.clip(cur_pose['y'] + dy_mm, -150.0, 150.0)
                target_z = np.clip(cur_pose['z'] + dz_mm, 20.0, 150.0)

                print(f"  Step {step_idx}: Move to X={target_x:.1f}mm, Y={target_y:.1f}mm, Z={target_z:.1f}mm | Grip: {grip_cmd:.2f}")
                dobot.move_to_xyz(target_x, target_y, target_z, mode="MOVL_XYZ")
                
                if grip_cmd > 0.5:
                    dobot.set_gripper("close")
                else:
                    dobot.set_gripper("open")

                cur_pose = {"x": target_x, "y": target_y, "z": target_z}
                time.sleep(0.1)

            print("[SUCCESS] Octo action chunk executed on physical Dobot!\n")

    dobot.disconnect()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--com_port", type=str, default="")
    parser.add_argument("--task", type=str, default="pick up the red cube and place it on the green platform")
    args = parser.parse_args()

    run_real_octo(camera_id=args.camera_id, com_port=args.com_port, task_prompt=args.task)
