"""
Interactive Robot Command & Calibration Center:
- Normalizes whatever the camera resolution, aspect ratio, or angle might be to exactly 640x480 inside the main viewport
- Calibrates the perspective distortion using ArUco and shows the rectified Octo POV (256x256 / 300x300)
- Auto-connects and powers on the real Dobot Magician
- Accepts tasks by Voice Microphone [v/m] or Console Typing [t]
- Executes the Octo policy on the physical robot with [SPACE]
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
    """Converts a BGR or RGB OpenCV image to a Pygame Surface."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def fit_to_640x480(frame):
    """
    Guarantees the raw camera feed fits inside a 640x480 frame:
    - Maintains original aspect ratio
    - Centers image with clean letterboxing
    - Works with any resolution (720p, 1080p, 4:3, 16:9, tilted angle)
    """
    target_w, target_h = 640, 480
    h, w = frame.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create clean dark canvas
    canvas = np.full((target_h, target_w, 3), 18, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def open_camera(cam_src, force_mjpeg=True):
    """
    Opens camera device (index int or stream URL str) with MJPEG fourcc support:
    - Supports full IP stream URLs: e.g. http://192.168.1.50:8080/video or http://192.168.x.x:4747/mjpegfeed
    - Supports hardware indexes (0, 1, ...) with MJPEG FourCC requested
    """
    print(f"[INFO] Opening Camera source '{cam_src}' (force_mjpeg={force_mjpeg})...")
    sys.stdout.flush()

    # Network / IP stream URL (e.g. IP Webcam, DroidCam, ESP32-CAM, RTSP)
    if isinstance(cam_src, str) and (cam_src.startswith("http://") or cam_src.startswith("https://") or cam_src.startswith("rtsp://") or ".mjpg" in cam_src):
        cap = cv2.VideoCapture(cam_src)
        return cap

    # Hardware camera index
    idx = int(cam_src) if isinstance(cam_src, (int, str)) and str(cam_src).isdigit() else 0
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)

    if cap.isOpened():
        if force_mjpeg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Quick test read
        ret, _ = cap.read()
        if not ret:
            # Fallback if specific hardware doesn't support MJPG fourcc request
            cap.release()
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap

def main():
    parser = argparse.ArgumentParser(description="Voice & Text Controlled Octo Robot System")
    parser.add_argument("--port", "--com", default=None, help="Dobot COM port (e.g. COM13). If omitted, auto-detects by hardware description.")
    parser.add_argument("--camera_id", default="0", help="Initial camera index (0, 1) or MJPEG stream URL (e.g. http://192.168.1.X:8080/video)")
    parser.add_argument("--secondary_cam", default="1", help="Second camera index or MJPEG IP address and path (e.g. http://192.168.1.X:8080/video)")
    parser.add_argument("--workspace_mm", type=float, default=360.0, help="Workspace table span in mm (default: 360mm)")
    parser.add_argument("--num_chunks", type=int, default=10, help="Number of continuous Octo policy rollout cycles to execute (default: 10)")
    parser.add_argument("--step_gain", type=float, default=15.0, help="Scaling gain from Octo delta to physical mm (default: 15.0)")
    parser.add_argument("--no_mjpeg", action="store_true", help="Disable MJPEG fourcc request")
    args = parser.parse_args()

    use_mjpeg = not args.no_mjpeg

    # 1. Connect Real Dobot Hardware
    print("\n" + "=" * 60)
    print("STEP 1: CONNECTING & INITIALIZING REAL DOBOT ROBOT")
    print("=" * 60)
    sys.stdout.flush()
    dobot = DobotUnifiedDriver(port=args.port, prefer_driver="auto")
    is_connected = dobot.connect()
    
    if is_connected:
        print("[INFO] Setting suction OFF and preparing standby pose...")
        sys.stdout.flush()
        dobot.set_suction(False)
        dobot.set_gripper("off")
        print("[INFO] Moving arm to standby observation pose (X=200, Y=0, Z=140)...")
        sys.stdout.flush()
        dobot.move_to_xyz(200.0, 0.0, 140.0, wait=False)
        print("[INFO] Standby position commanded successfully.")
        sys.stdout.flush()
    else:
        print("[NOTICE] Real robot not detected or offline. Running in DRY-RUN / SIMULATION mode.")
        sys.stdout.flush()

    # 2. Open Camera
    print("[INFO] Opening camera feed...")
    sys.stdout.flush()
    cam_sources = [args.camera_id, args.secondary_cam]
    src_idx = 0
    active_src = cam_sources[src_idx]
    cap = open_camera(active_src, force_mjpeg=use_mjpeg)
    if not cap.isOpened():
        print(f"[WARN] Could not open camera '{active_src}'. Trying secondary camera '{cam_sources[1]}'...")
        src_idx = 1
        active_src = cam_sources[src_idx]
        cap = open_camera(active_src, force_mjpeg=use_mjpeg)
        if not cap.isOpened():
            print("[ERROR] Could not open camera. Please check connection.")
            return
    print(f"[INFO] Camera successfully opened (Source {active_src}).")
    sys.stdout.flush()

    # 3. ArUco Calibrator (Full 360x360mm Table Workspace -> 256x256 for Octo)
    calibrator = ArucoAutoCalibrator(output_size=(256, 256), marker_size_mm=80.0)
    calibrator.ws_w_mm = args.workspace_mm
    calibrator.ws_h_mm = args.workspace_mm

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
    # Left: 640x480 (Raw Camera), Right: 300x300 (Calibrated Octo) + UI panels
    win_w, win_h = 1020, 560
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("OCTO ROBOT CONTROL: Live Calibrated POV & Voice Interface")
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    current_instruction = "pick up the red cube and place it on the green platform"
    status_text = "READY - WAITING FOR COMMAND"
    is_executing = False
    last_crop = None

    print("\n" + "=" * 60)
    print("SYSTEM READY: HOW TO GIVE INSTRUCTIONS")
    print("=" * 60)
    print("  [v] or [m] -> SPEAK via MICROPHONE (e.g. 'pick up the red cube and place on green')")
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
        total_chunks = args.num_chunks
        print(f"\n{'=' * 60}")
        print(f"[OCTO ROLLOUT STARTED] Task: \"{current_instruction}\"")
        print(f"[INFO] Executing closed-loop policy loop for {total_chunks} chunks ({total_chunks * 4} steps total)...")
        print(f"{'=' * 60}\n")

        # Encode prompt once
        octo_task = model.create_tasks(texts=[current_instruction])

        for chunk_idx in range(1, total_chunks + 1):
            if not is_executing or not running:
                break

            status_text = f"OCTO INFERENCE (CHUNK {chunk_idx}/{total_chunks})..."
            cur_pose = dobot.get_pose()

            # Fresh observation from live camera feed
            img_p = last_crop.copy()
            img_w = cv2.resize(img_p, (128, 128))

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

            rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 100000)
            pred_actions = model.sample_actions(octo_obs, octo_task, rng=rng_key, unnormalization_statistics=unnorm_stats)
            actions = np.array(pred_actions[0]) # (4, 7)
            
            print(f"\n--- Chunk [{chunk_idx}/{total_chunks}] --- Robot Pose: X={cur_pose['x']:.1f}, Y={cur_pose['y']:.1f}, Z={cur_pose['z']:.1f}")

            for s_idx, act in enumerate(actions, 1):
                if not running:
                    break
                status_text = f"EXECUTING STEP {s_idx}/4 (CHUNK {chunk_idx}/{total_chunks})..."
                
                # Octo actions are normalized deltas in meters.
                # Scaling gain translates them into physical millimetres on Dobot's workspace
                dx_mm = act[0] * 1000.0 * args.step_gain
                dy_mm = act[1] * 1000.0 * args.step_gain
                dz_mm = act[2] * 1000.0 * args.step_gain
                grip_val = act[6]

                target_x = float(np.clip(cur_pose['x'] + dx_mm, 150.0, 310.0))
                target_y = float(np.clip(cur_pose['y'] + dy_mm, -160.0, 160.0))
                # Tool tip clearance safety: prevent end effector from plunging into table
                target_z = float(np.clip(cur_pose['z'] + dz_mm, 20.0, 160.0))

                print(f"  Step {s_idx}: dX={dx_mm:+.1f} dY={dy_mm:+.1f} dZ={dz_mm:+.1f} -> X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f} | Grip: {grip_val:.2f}")
                dobot.move_to_xyz(target_x, target_y, target_z)

                if grip_val > 0.5:
                    dobot.set_gripper("close")
                else:
                    dobot.set_gripper("open")

                cur_pose = {"x": target_x, "y": target_y, "z": target_z}
                time.sleep(0.15)

        # Explicitly stop suction and release gripper after task rollout finishes
        dobot.set_suction(False)
        dobot.set_gripper("off")
        status_text = "TASK ROLLOUT FINISHED! (SUCTION STOPPED)"
        is_executing = False
        print(f"\n[SUCCESS] Completed {total_chunks} policy rollout cycles. Suction stopped.\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key in [pygame.K_v, pygame.K_m]:
                    threading.Thread(target=voice_listener_worker, daemon=True).start()
                elif event.key == pygame.K_t:
                    status_text = "ENTER INSTRUCTION IN CONSOLE..."
                    print("\n" + ">" * 15 + " ENTER YOUR INSTRUCTION IN TERMINAL " + "<" * 15)
                    try:
                        typed = input("Type command and press ENTER: ").strip()
                        if typed:
                            current_instruction = typed
                            status_text = f"INSTRUCTION SET: \"{typed}\""
                            print(f"[UPDATED]: \"{current_instruction}\"")
                    except Exception:
                        pass
                elif event.key == pygame.K_SPACE:
                    if not is_executing:
                        threading.Thread(target=execute_octo_on_robot, daemon=True).start()
                elif event.key == pygame.K_c:
                    src_idx = (src_idx + 1) % len(cam_sources)
                    active_src = cam_sources[src_idx]
                    print(f"\n[INFO] Switching to Camera: '{active_src}'...")
                    cap.release()
                    cap = open_camera(active_src, force_mjpeg=use_mjpeg)
                    calibrator.is_calibrated = False
                elif event.key == pygame.K_r:
                    calibrator.is_calibrated = False
                    print("[INFO] Calibration reset.")

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 1. Process calibration on native resolution
        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=True)
        last_crop = calibrated_crop

        # 2. Guarantee debug camera feed fits strictly inside 640x480 box
        camera_640x480 = fit_to_640x480(debug_view)

        frame_count += 1
        fps = frame_count / (time.time() - t0 + 1e-5)

        # Pygame Rendering
        screen.fill((18, 20, 26))

        # Left View: Standardized 640x480 Viewport
        cam_surf = cv2_to_pygame_surface(camera_640x480)
        screen.blit(cam_surf, (15, 20))
        pygame.draw.rect(screen, (60, 65, 80), (14, 19, 642, 482), 1)

        # Right View: Calibrated Octo POV (shown as 320x320)
        crop_display = cv2.resize(calibrated_crop, (320, 320))
        crop_surf = cv2_to_pygame_surface(crop_display)
        screen.blit(crop_surf, (675, 20))
        pygame.draw.rect(screen, (0, 240, 180), (674, 19, 322, 322), 2)

        # Overlays & Telemetry
        lbl_octo = font_bold.render("OCTO CALIBRATED POV (256x256)", True, (0, 240, 180))
        screen.blit(lbl_octo, (675, 350))

        cal_status = "ARUCO: CALIBRATED (16x16cm Rectified)" if calibrator.is_calibrated else "ARUCO: DETECTING 4 MARKERS..."
        cal_col = (50, 255, 120) if calibrator.is_calibrated else (255, 90, 90)
        screen.blit(font.render(cal_status, True, cal_col), (675, 375))

        rob_status = f"ROBOT: {'CONNECTED (LIVE HARDWARE)' if dobot.connected else 'SIMULATION / DRY-RUN'}"
        rob_col = (0, 220, 255) if dobot.connected else (255, 200, 80)
        screen.blit(font.render(rob_status, True, rob_col), (675, 400))

        # Bottom Bar: Task Instruction & Status
        pygame.draw.rect(screen, (28, 32, 42), (15, 510, 985, 40), border_radius=4)
        lbl_inst = font_bold.render(f"TASK: \"{current_instruction}\"", True, (255, 215, 0))
        screen.blit(lbl_inst, (25, 520))

        # Hotkeys help bar
        help_str = "[v/m] Speak Mic  |  [t] Type Text  |  [SPACE] Run Octo  |  [c] Switch Cam  |  [q] Quit"
        screen.blit(font.render(help_str, True, (160, 175, 200)), (675, 440))
        screen.blit(font.render(f"STATUS: {status_text}", True, (255, 255, 255)), (15, 502))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    dobot.close()
    pygame.quit()
    print("[INFO] Application closed.")

if __name__ == "__main__":
    main()
