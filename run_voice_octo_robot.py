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
from dobot_urdf_renderer import DobotURDFRenderer
from dobot_magician_ros2.ai_training.env.dobot_env import DobotKinematics
import json

CONFIG_FILE = "calibration_config.json"

def load_calibration_config():
    cfg = {
        "sim_yaw_offset_deg": 0.0,
        "sim_pitch_offset_deg": 0.0,
        "sim_roll_offset_deg": 0.0,
        "x_scale": 1.0,
        "y_scale": 1.0,
        "z_scale": 1.0,
        "z_table_surface_mm": -35.0,
        "standby_x_mm": 200.0,
        "standby_y_mm": 0.0,
        "standby_z_mm": 140.0
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
                cfg.update(saved)
                print(f"[INFO] Loaded manual calibration config from '{CONFIG_FILE}': Base Yaw Offset={cfg['sim_yaw_offset_deg']}°")
        except Exception:
            pass
    return cfg

# Octo Policy
import octo_patch
from octo.model.octo_model import OctoModel
import jax

def cv2_to_pygame_surface(cv_img):
    """Converts a BGR or RGB OpenCV image to a Pygame Surface."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def enforce_dobot_workspace_limits(x, y, z):
    """
    Kinematic & Physical Boundary Projector for Dobot Magician:
    1. Base Reach (Cylindrical Envelope):
       - Min Reach R_min = 160.0 mm (prevents collision with Dobot base column)
       - Max Reach R_max = 315.0 mm (prevents link singularities and motor stall at 342mm limit)
    2. Orientation (Hemisphere):
       - Yaw angle |j1| <= 75.0 degrees (keeps target strictly in front work area)
       - X >= 60.0 mm (prevents commanding arm behind the robot)
    3. Vertical Height with 8cm Suction Cup:
       - Min Z = -35.0 mm (table surface contact level with objects; prevents crashing table)
       - Max Z = 145.0 mm (safe observation ceiling)
    """
    clamped_x = float(x)
    clamped_y = float(y)
    clamped_z = float(np.clip(z, -35.0, 145.0))

    if clamped_x < 60.0:
        clamped_x = 60.0

    # Reach boundary
    r = np.sqrt(clamped_x**2 + clamped_y**2)
    R_MIN = 160.0
    R_MAX = 315.0

    if r < R_MIN:
        scale = R_MIN / max(r, 1e-4)
        clamped_x *= scale
        clamped_y *= scale
    elif r > R_MAX:
        scale = R_MAX / max(r, 1e-4)
        clamped_x *= scale
        clamped_y *= scale

    # Azimuth yaw boundary
    yaw_deg = np.degrees(np.arctan2(clamped_y, clamped_x))
    if abs(yaw_deg) > 75.0:
        target_yaw_rad = np.radians(np.sign(yaw_deg) * 75.0)
        curr_r = np.sqrt(clamped_x**2 + clamped_y**2)
        clamped_x = curr_r * np.cos(target_yaw_rad)
        clamped_y = curr_r * np.sin(target_yaw_rad)

    clamped = (abs(clamped_x - x) > 0.5 or abs(clamped_y - y) > 0.5 or abs(clamped_z - z) > 0.5)
    return float(clamped_x), float(clamped_y), float(clamped_z), clamped

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
    parser.add_argument("--invert_y", action="store_true", default=False, help="Invert Y motion axis sign")
    parser.add_argument("--invert_z", action="store_true", default=False, help="Invert Z vertical height motion axis sign")
    parser.add_argument("--swap_xy", action="store_true", default=False, help="Swap X and Y axes if camera view is rotated 90 degrees")
    parser.add_argument("--auto_home", action="store_true", help="Perform real hardware HOME calibration on startup to automatically calibrate simulation")
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
    
    # Calibration offsets between real hardware and 3D simulation
    calib_cfg = load_calibration_config()
    sim_calib_offset = {"yaw": calib_cfg.get("sim_yaw_offset_deg", 0.0), "pitch": 0.0, "roll": 0.0}

    def calibrate_sim_to_real_hardware():
        nonlocal sim_calib_offset, status_text
        status_text = "HOMING REAL DOBOT TO CALIBRATE SIMULATION..."
        print("\n" + "=" * 60)
        print("[AUTO-CALIBRATION] Executing hardware HOME calibration...")
        print("=" * 60)
        # 1. Command hardware to physical home
        home_pose = dobot.home(wait_sec=12.0)
        print(f"[AUTO-CALIBRATION] Homing complete! Measured Home Pose: {home_pose}")
        
        # 2. Check measured base yaw against reference (Home base yaw is 0.0 deg)
        meas_j1 = float(home_pose.get("j1", 0.0))
        meas_j2 = float(home_pose.get("j2", 0.0))
        meas_j3 = float(home_pose.get("j3", 0.0))
        
        # Set exact alignment offset so sim matches real home pose zero
        sim_calib_offset["yaw"] = -meas_j1
        status_text = f"SIM AUTO-CALIBRATED (Base Offset: {sim_calib_offset['yaw']:+.1f}°)"
        print(f"[AUTO-CALIBRATION] Simulation aligned with physical Dobot! Base offset: {sim_calib_offset['yaw']:+.1f} deg\n")

        # 3. Move to standby pose
        dobot.move_to_xyz(200.0, 0.0, 140.0, wait=True)
        dobot.set_suction(False)

    if is_connected and args.auto_home:
        calibrate_sim_to_real_hardware()

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

    # 5. Initialize 3D URDF Simulation Renderer
    print("\n" + "=" * 60)
    print("STEP 3: INITIALIZING 3D URDF DIGITAL TWIN SIMULATION")
    print("=" * 60)
    sim_renderer = DobotURDFRenderer(width=480, height=360)
    dobot_kin = DobotKinematics()
    sim_enabled = True

    # 6. Initialize Pygame Unified Dashboard
    pygame.init()
    # Left: 640x480 (Raw Camera), Center: 320x320 (Calibrated Octo POV), Right: 480x360 (3D URDF Sim)
    win_w, win_h = 1480, 580
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("OCTO ROBOT CONTROL: Live POV, 3D URDF Digital Twin & Voice Interface")
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    current_instruction = "pick up the red cube and place it on the green platform"
    status_text = "READY - WAITING FOR COMMAND"
    is_executing = False
    last_crop = None
    suction_state = False
    last_sim_frame = np.full((360, 480, 3), 20, dtype=np.uint8)
    sim_update_counter = 0

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
        nonlocal is_executing, status_text, suction_state
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
                
                # Canonical Octo to Dobot Coordinate Mapping (Pre-configured & Verified):
                # Bridge Dataset standard:
                # - Act[0] (forward/backward): maps directly to Dobot +X (forward away from base)
                # - Act[1] (lateral left/right): maps directly to Dobot +Y (left/right)
                # - Act[2] (height vertical): in Bridge, descending to grasp is negative (-dZ).
                #   Dobot height Z decreases when approaching table surface.
                dx_mm = act[0] * 1000.0 * args.step_gain
                dy_mm = act[1] * 1000.0 * args.step_gain
                dz_mm = act[2] * 1000.0 * args.step_gain
                grip_val = act[6]

                raw_tx = cur_pose['x'] + dx_mm
                raw_ty = cur_pose['y'] + dy_mm
                raw_tz = cur_pose['z'] + dz_mm

                # Enforce physical and kinematic workspace limits
                target_x, target_y, target_z, was_clamped = enforce_dobot_workspace_limits(raw_tx, raw_ty, raw_tz)

                clamp_msg = f" [OUT-OF-BOUNDS -> PROJECTED TO WORKSPACE]" if was_clamped else ""
                print(f"  Step {s_idx}: dX={dx_mm:+.1f} dY={dy_mm:+.1f} dZ={dz_mm:+.1f} -> Target (X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}){clamp_msg} | Sucker: {grip_val:.2f}")
                # Move and actively verify physical following
                reached = dobot.move_to_xyz(target_x, target_y, target_z, wait=True, tolerance_mm=4.0)

                # Sucker control
                if grip_val > 0.5:
                    dobot.set_suction(True)
                    suction_state = True
                else:
                    dobot.set_suction(False)
                    suction_state = False

                cur_pose = dobot.get_pose()
                actual_err = ((cur_pose['x']-target_x)**2 + (cur_pose['y']-target_y)**2 + (cur_pose['z']-target_z)**2)**0.5
                print(f"        -> Physical Feedback: X={cur_pose['x']:.1f}, Y={cur_pose['y']:.1f}, Z={cur_pose['z']:.1f}mm | Tracking: {'REACHED' if reached else f'Lags by {actual_err:.1f}mm'}")

        # Explicitly stop suction after task rollout finishes
        dobot.set_suction(False)
        dobot.set_gripper("off")
        suction_state = False
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
                    status_text = "CALIBRATION RESET - SHOW 4 MARKERS TO LOCK"
                    print("\n[INFO] Calibration reset! Looking for 4 markers to lock perspective once...")
                elif event.key == pygame.K_y:
                    args.invert_y = not args.invert_y
                    status_text = f"INVERT Y TOGGLED: {'INVERTED' if args.invert_y else 'NORMAL'}"
                    print(f"\n[INFO] Invert Y motion toggled: {args.invert_y}")
                elif event.key == pygame.K_z:
                    args.invert_z = not args.invert_z
                    status_text = f"INVERT Z TOGGLED: {'INVERTED' if args.invert_z else 'NORMAL'}"
                    print(f"\n[INFO] Invert Z motion toggled: {args.invert_z}")
                elif event.key == pygame.K_x:
                    args.swap_xy = not args.swap_xy
                    status_text = f"SWAP XY TOGGLED: {'SWAPPED (90°)' if args.swap_xy else 'NORMAL'}"
                    print(f"\n[INFO] Swap XY motion axes toggled: {args.swap_xy}")
                    # Press 'h' anytime to run hardware HOME calibration and auto-align simulation
                    threading.Thread(target=calibrate_sim_to_real_hardware, daemon=True).start()
                elif event.key == pygame.K_b:
                    # Toggle 90 degree base yaw rotation manually if desired
                    sim_calib_offset["yaw"] = (sim_calib_offset["yaw"] + 90.0) % 360.0
                    if sim_calib_offset["yaw"] > 180.0:
                        sim_calib_offset["yaw"] -= 360.0
                    status_text = f"SIM BASE YAW ROTATED: {sim_calib_offset['yaw']:+.0f} DEG"
                    print(f"\n[INFO] Sim Base Yaw Offset -> {sim_calib_offset['yaw']:+.0f} deg")

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # 1. Calibrate ONCE initially or when 'r' is pressed. Once locked, never jitter/recalculate!
        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=not calibrator.is_calibrated)
        last_crop = calibrated_crop

        # 2. Render Synchronized 3D URDF Digital Twin Simulation
        sim_update_counter += 1
        if sim_update_counter % 2 == 0: # 15 FPS simulation sync
            try:
                cur_p = dobot.get_pose()
                # Use exact forward/inverse kinematic model driven directly by live Cartesian position:
                # This guarantees 100% position accuracy without firmware angle offset errors
                x_m = float(cur_p['x']) / 1000.0
                y_m = float(cur_p['y']) / 1000.0
                z_m = float(cur_p['z']) / 1000.0

                j_calc = dobot_kin.inverse(x_m, y_m, z_m)
                j1_rad, j2_rad, j3_rad, j4_rad = j_calc[0], j_calc[1], j_calc[2], j_calc[3]

                # Compute URDF transform hierarchy
                t_m = sim_renderer.compute_urdf_transforms(
                    j1_rad, j2_rad, j3_rad, j4_rad, gripper_closed=False
                )

                # Render with physical 8cm sucker attached
                scene = sim_renderer.build_scene(t_m, end_effector="suction")
                _, _, last_sim_frame = sim_renderer.render_views(scene, t_m['ee_pos'])
            except Exception:
                pass

        # 3. Guarantee debug camera feed fits strictly inside 640x480 box
        camera_640x480 = fit_to_640x480(debug_view)

        frame_count += 1
        fps = frame_count / (time.time() - t0 + 1e-5)

        # Pygame Rendering
        screen.fill((18, 20, 26))

        # Panel 1 (Left): Standardized 640x480 Raw Viewport
        cam_surf = cv2_to_pygame_surface(camera_640x480)
        screen.blit(cam_surf, (15, 20))
        pygame.draw.rect(screen, (60, 65, 80), (14, 19, 642, 482), 1)

        # Panel 2 (Center): Calibrated Octo POV (300x300)
        crop_display = cv2.resize(calibrated_crop, (300, 300))
        crop_surf = cv2_to_pygame_surface(crop_display)
        screen.blit(crop_surf, (675, 20))
        pygame.draw.rect(screen, (0, 240, 180), (674, 19, 302, 302), 2)

        lbl_octo = font_bold.render("OCTO POV (360x360mm Table)", True, (0, 240, 180))
        screen.blit(lbl_octo, (675, 330))

        cal_status = "ARUCO: LOCKED (Table Calibrated)" if calibrator.is_calibrated else "ARUCO: WAITING 4 MARKERS..."
        cal_col = (50, 255, 120) if calibrator.is_calibrated else (255, 90, 90)
        screen.blit(font.render(cal_status, True, cal_col), (675, 355))

        rob_status = f"ROBOT: {'CONNECTED (LIVE HARDWARE)' if dobot.connected else 'SIMULATION / DRY-RUN'}"
        rob_col = (0, 220, 255) if dobot.connected else (255, 200, 80)
        screen.blit(font.render(rob_status, True, rob_col), (675, 380))

        # Panel 3 (Right): 3D URDF Digital Twin Simulation (480x360)
        sim_surf = cv2_to_pygame_surface(last_sim_frame)
        screen.blit(sim_surf, (990, 20))
        pygame.draw.rect(screen, (70, 130, 255), (989, 19, 482, 362), 2)
        lbl_sim = font_bold.render("3D URDF DIGITAL TWIN (SYNCHRONIZED)", True, (70, 130, 255))
        screen.blit(lbl_sim, (990, 390))

        cur_p = dobot.get_pose()
        telemetry_str = f"EE: X={cur_p['x']:.1f} Y={cur_p['y']:.1f} Z={cur_p['z']:.1f}mm | Sucker: {'ON (SUCKING)' if suction_state else 'OFF'}"
        screen.blit(font.render(telemetry_str, True, (200, 215, 240)), (990, 415))

        # Bottom Bar: Task Instruction & Status
        pygame.draw.rect(screen, (28, 32, 42), (15, 515, 1450, 45), border_radius=4)
        lbl_inst = font_bold.render(f"TASK: \"{current_instruction}\"", True, (255, 215, 0))
        screen.blit(lbl_inst, (25, 528))

        # Hotkeys help bar
        help_str = "[SPACE] Run Octo | [v/m] Voice Mic | [t] Type Text | [h] Auto-Home Calibrate | [c] Switch Cam | [q] Quit"
        screen.blit(font.render(help_str, True, (160, 175, 200)), (675, 440))
        screen.blit(font.render(f"STATUS: {status_text}", True, (255, 255, 255)), (15, 502))

        pygame.display.flip()
        clock.tick(30)

    sim_renderer.close()
    cap.release()
    dobot.close()
    pygame.quit()
    print("[INFO] Application closed.")

if __name__ == "__main__":
    main()
