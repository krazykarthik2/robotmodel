"""
Interactive Robot Command & Calibration Center:
1. Displays the calibrated Octo POV (256x256) side-by-side with raw camera feed
2. Auto-connects and turns on the real Dobot Magician (native USB serial / alarms cleared)
3. Accepts tasks either by:
   - Voice Microphone (Press [v] or [m] and speak your instruction)
   - Typing in the console (Press [t] and type your command)
4. Evaluates the Octo Foundation Model on the live calibrated camera observation
5. Moves the physical robot to execute the vision-language task
"""
import os
import sys
import time
import argparse
import threading
import numpy as np
import cv2
import pygame

# Hardware drivers & ArUco
from dobot_unified_driver import DobotUnifiedDriver
from aruco_calibrator import ArucoAutoCalibrator
from voice_instruction import listen_for_instruction

# Octo Policy
import octo_patch
from octo.model.octo_model import OctoModel
import jax

def cv2_to_pygame_surface(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def open_camera(cam_src, force_mjpeg=True):
    if isinstance(cam_src, str) and (cam_src.startswith("http://") or cam_src.startswith("rtsp://")):
        return cv2.VideoCapture(cam_src)
    idx = int(cam_src) if str(cam_src).isdigit() else 0
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        if force_mjpeg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, _ = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def main():
    parser = argparse.ArgumentParser(description="Voice & Text Controlled Octo Robot System")
    parser.add_argument("--camera_id", default="0", help="Camera index (0, 1) or stream URL")
    parser.add_argument("--secondary_cam", default="1", help="Second camera index")
    args = parser.parse_args()

    # 1. Connect Real Dobot Hardware
    print("\n" + "=" * 60)
    print("STEP 1: CONNECTING & INITIALIZING REAL DOBOT ROBOT")
    print("=" * 60)
    dobot = DobotUnifiedDriver(prefer_driver="auto")
    is_connected = dobot.connect()
    
    if is_connected:
        print("[INFO] Moving arm to standby observation pose (X=200, Y=0, Z=140)...")
        dobot.move_to_xyz(200.0, 0.0, 140.0)
        dobot.set_gripper("open")
    else:
        print("[NOTICE] Real robot not detected or offline. Running in DRY-RUN / SIMULATION mode.")

    # 2. Open Camera
    cam_sources = [args.camera_id, args.secondary_cam]
    src_idx = 0
    active_src = cam_sources[src_idx]
    cap = open_camera(active_src)
    if not cap.isOpened():
        src_idx = 1
        active_src = cam_sources[src_idx]
        cap = open_camera(active_src)
        if not cap.isOpened():
            print("[ERROR] Could not open camera. Please check connection.")
            return

    # 3. ArUco Calibrator
    calibrator = ArucoAutoCalibrator(output_size=(256, 256), marker_size_mm=80.0)

    # 4. Load Octo Model
    print("\n" + "=" * 60)
    print("STEP 2: LOADING OCTO GENERALIST POLICY")
    print("=" * 60)
    checkpoint_path = os.path.abspath("octo_small_checkpoint")
    model = OctoModel.load_pretrained(checkpoint_path)
    unnorm_stats = model.dataset_statistics["bridge_dataset"]["action"]
    print("[SUCCESS] Octo Model loaded and ready for instructions!")

    # 5. Initialize Pygame Dashboard
    pygame.init()
    win_w, win_h = 1000, 560
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("OCTO ROBOT CONTROL: Live Calibrated POV & Voice Interface")
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    font_large = pygame.font.SysFont("Arial", 22, bold=True)
    clock = pygame.time.Clock()

    current_instruction = "pick up the red cube and place it on the green platform"
    status_text = "READY - WAITING FOR COMMAND"
    is_executing = False
    last_crop = None

    print("\n" + "=" * 60)
    print("SYSTEM READY: HOW TO GIVE INSTRUCTIONS")
    print("=" * 60)
    print("  [v] or [m] -> SPEAK via MICROPHONE (e.g. 'push yellow cube onto cyan target')")
    print("  [t]        -> TYPE instruction into this terminal console")
    print("  [SPACE]    -> EXECUTE current instruction with Octo on Dobot")
    print("  [c]        -> SWITCH camera (Camera 0 <-> Camera 1)")
    print("  [r]        -> RESET ArUco calibration")
    print("  [q]        -> QUIT")
    print("=" * 60 + "\n")

    running = True
    frame_count = 0
    t0 = time.time()

    def voice_listener_worker():
        nonlocal current_instruction, status_text
        status_text = "LISTENING TO MICROPHONE..."
        spoken = listen_for_instruction()
        if spoken:
            current_instruction = spoken
            status_text = f"VOICE RECEIVED: \"{spoken}\""
            print(f"\n[UPDATED INSTRUCTION]: \"{current_instruction}\"")
        else:
            status_text = "MIC TIMEOUT (NO SPEECH DETECTED)"

    def execute_octo_on_robot():
        nonlocal is_executing, status_text
        if last_crop is None:
            return
        is_executing = True
        status_text = f"OCTO EVALUATING: \"{current_instruction}\"..."
        print(f"\n[EXECUTION STARTED] Task: \"{current_instruction}\"")

        # Encode prompt
        octo_task = model.create_tasks(texts=[current_instruction])

        # Prepare 256x256 overhead and 128x128 wrist
        img_p = last_crop
        img_w = cv2.resize(last_crop, (128, 128))

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

        rng_key = jax.random.PRNGKey(int(time.time()))
        pred_actions = model.sample_actions(octo_obs, octo_task, rng=rng_key, unnormalization_statistics=unnorm_stats)
        actions = np.array(pred_actions[0]) # (4, 7)
        print(f"[OCTO PREDICTION] Action chunk (4 steps):\n{actions}")

        cur_pose = dobot.get_pose()
        print(f"[ROBOT POSE] X={cur_pose['x']:.1f}mm, Y={cur_pose['y']:.1f}mm, Z={cur_pose['z']:.1f}mm")

        for s_idx, act in enumerate(actions, 1):
            status_text = f"EXECUTING ROBOT STEP {s_idx}/4..."
            dx_mm = act[0] * 1000.0 * 2.0
            dy_mm = act[1] * 1000.0 * 2.0
            dz_mm = act[2] * 1000.0 * 2.0
            grip_val = act[6]

            target_x = np.clip(cur_pose['x'] + dx_mm, 150.0, 300.0)
            target_y = np.clip(cur_pose['y'] + dy_mm, -150.0, 150.0)
            target_z = np.clip(cur_pose['z'] + dz_mm, 35.0, 160.0) # Keeps safely above table

            print(f"  Step {s_idx}: Move to X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f} | Gripper: {grip_val:.2f}")
            dobot.move_to_xyz(target_x, target_y, target_z)
            
            if grip_val > 0.5:
                dobot.set_gripper("close")
            else:
                dobot.set_gripper("open")

            cur_pose = {"x": target_x, "y": target_y, "z": target_z}
            time.sleep(0.1)

        status_text = "ACTION CHUNK FINISHED!"
        is_executing = False
        print("[SUCCESS] Task execution cycle completed.\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key in [pygame.K_v, pygame.K_m]:
                    # Voice input triggered
                    threading.Thread(target=voice_listener_worker, daemon=True).start()
                elif event.key == pygame.K_t:
                    # Type input in console
                    status_text = "ENTER INSTRUCTION IN CONSOLE..."
                    print("\n" + ">" * 15 + " ENTER YOUR INSTRUCTION IN TERMINAL " + "<" * 15)
                    try:
                        typed = input("Type command and press ENTER: ").strip()
                        if typed:
                            current_instruction = typed
                            status_text = f"INSTRUCTION SET: \"{typed}\""
                            print(f"[UPDATED]: \"{current_instruction}\"")
                    except Exception as e:
                        pass
                elif event.key == pygame.K_SPACE:
                    # Execute
                    if not is_executing:
                        threading.Thread(target=execute_octo_on_robot, daemon=True).start()
                elif event.key == pygame.K_c:
                    # Switch Camera
                    src_idx = (src_idx + 1) % len(cam_sources)
                    active_src = cam_sources[src_idx]
                    print(f"\n[INFO] Switching to Camera: '{active_src}'...")
                    cap.release()
                    cap = open_camera(active_src)
                    calibrator.is_calibrated = False
                elif event.key == pygame.K_r:
                    calibrator.is_calibrated = False
                    print("[INFO] Calibration reset.")

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=True)
        last_crop = calibrated_crop

        frame_count += 1
        fps = frame_count / (time.time() - t0 + 1e-5)

        # Pygame Drawing
        screen.fill((20, 22, 28))

        # Left View: Raw Camera Feed with ArUco detections (640x480)
        cam_surf = cv2_to_pygame_surface(debug_view)
        screen.blit(cam_surf, (15, 20))

        # Right View: Calibrated Octo POV (256x256 -> displayed as 280x280)
        crop_display = cv2.resize(calibrated_crop, (280, 280))
        crop_surf = cv2_to_pygame_surface(crop_display)
        screen.blit(crop_surf, (685, 20))
        pygame.draw.rect(screen, (0, 240, 180), (684, 19, 282, 282), 2)

        # Overlays & Telemetry
        lbl_octo = font_bold.render("OCTO CALIBRATED POV (256x256)", True, (0, 240, 180))
        screen.blit(lbl_octo, (685, 310))

        cal_status = "ARUCO: CALIBRATED (16x16cm Table)" if calibrator.is_calibrated else "ARUCO: SEARCHING FOR MARKERS 0, 1, 2, 3"
        cal_col = (50, 255, 120) if calibrator.is_calibrated else (255, 90, 90)
        screen.blit(font.render(cal_status, True, cal_col), (685, 335))

        rob_status = f"ROBOT: {'CONNECTED (LIVE HARDWARE)' if dobot.connected else 'SIMULATION / DRY-RUN'}"
        rob_col = (0, 220, 255) if dobot.connected else (255, 200, 80)
        screen.blit(font.render(rob_status, True, rob_col), (685, 360))

        # Status text
        screen.blit(font.render(f"STATUS: {status_text}", True, (255, 255, 255)), (15, 515))

        # Current Instruction Box
        pygame.draw.rect(screen, (32, 36, 48), (15, 470, 950, 40), border_radius=4)
        lbl_inst = font_bold.render(f"TASK: \"{current_instruction}\"", True, (255, 215, 0))
        screen.blit(lbl_inst, (25, 480))

        # Hotkeys help bar
        help_str = "[v/m] Speak Mic  |  [t] Type Text  |  [SPACE] Run Octo  |  [c] Switch Cam  |  [q] Quit"
        screen.blit(font.render(help_str, True, (160, 175, 200)), (685, 410))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    dobot.close()
    pygame.quit()
    print("[INFO] Application closed.")

if __name__ == "__main__":
    main()
