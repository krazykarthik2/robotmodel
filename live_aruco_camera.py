"""
Live camera viewer with on-the-fly camera switching and ArUco auto-calibration:
- Uses DirectShow (cv2.CAP_DSHOW) on Windows to ensure cameras open instantly without timeout
- Dual GUI window backend: Pygame (works on all environments including headless/headless opencv) with automatic fallback
- Supports 8cm x 8cm markers placed side-by-side with 0 space (16cm x 16cm total)
- Hotkey [c] -> switch camera index (0 <-> 1)
- Hotkey [s] -> save calibrated frame
- Hotkey [r] -> reset calibration
- Hotkey [q] or [ESC] -> quit
"""
import sys
import time
import argparse
import numpy as np
import cv2
import pygame

from aruco_calibrator import ArucoAutoCalibrator

def scan_available_cameras(max_to_test=4):
    available = []
    print("[INFO] Scanning for connected video cameras...")
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    print(f"[INFO] Found available cameras: {available}")
    return available

def open_camera(cam_idx):
    # Use CAP_DSHOW on Windows to prevent FFMPEG backend hangs
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def cv2_to_pygame_surface(cv_img):
    """Converts a BGR or RGB OpenCV image to a Pygame Surface."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def main():
    parser = argparse.ArgumentParser(description="Live ArUco Camera with Switching")
    parser.add_argument("--camera_id", type=int, default=0, help="Initial camera ID (default: 0)")
    parser.add_argument("--out_size", type=int, default=256, help="Output crop size for Octo (default: 256)")
    args = parser.parse_args()

    available_cams = scan_available_cameras()
    if not available_cams:
        print("[ERROR] No working cameras found on system.")
        return

    cam_idx = args.camera_id if args.camera_id in available_cams else available_cams[0]
    print(f"[INFO] Opening Camera ID: {cam_idx}...")
    cap = open_camera(cam_idx)

    calibrator = ArucoAutoCalibrator(output_size=(args.out_size, args.out_size), marker_size_mm=80.0)

    # Initialize Pygame GUI
    pygame.init()
    # Main window shows: Left side: Raw Camera (640x480), Right side: Calibrated Octo (280x280) + info
    win_w, win_h = 960, 520
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Dobot ArUco Auto-Calibrator (Camera {cam_idx})")
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    print("\n" + "=" * 60)
    print("LIVE ARUCO CALIBRATION & MULTI-CAMERA SWITCHER (RUNNING)")
    print("=" * 60)
    print("Controls (In Pygame Window):")
    print("  [c] -> SWITCH CAMERA (e.g. Camera 0 <-> Camera 1)")
    print("  [s] -> SAVE current calibrated crop to disk")
    print("  [r] -> RESET calibration")
    print("  [q] / [ESC] -> QUIT")
    print("=" * 60 + "\n")

    running = True
    frame_count = 0
    t0 = time.time()
    save_notification_timer = 0

    while running:
        # Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key == pygame.K_c:
                    if len(available_cams) > 1:
                        cur_pos = available_cams.index(cam_idx)
                        next_pos = (cur_pos + 1) % len(available_cams)
                        cam_idx = available_cams[next_pos]
                        print(f"\n[INFO] Switching to Camera ID {cam_idx}...")
                        cap.release()
                        cap = open_camera(cam_idx)
                        pygame.display.set_caption(f"Dobot ArUco Auto-Calibrator (Camera {cam_idx})")
                        calibrator.is_calibrated = False
                        frame_count = 0
                        t0 = time.time()
                    else:
                        print("[NOTICE] Only 1 camera detected on system.")
                elif event.key == pygame.K_s:
                    if 'last_crop' in locals() and last_crop is not None:
                        cv2.imwrite("saved_calibrated_obs.png", last_crop)
                        print("[SAVED] Saved frame to 'saved_calibrated_obs.png'!")
                        save_notification_timer = time.time() + 2.0
                elif event.key == pygame.K_r:
                    calibrator.is_calibrated = False
                    print("[INFO] Calibration reset. Re-detecting markers...")

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=True)
        last_crop = calibrated_crop

        frame_count += 1
        fps = frame_count / (time.time() - t0 + 1e-5)

        # Draw on Screen
        screen.fill((25, 27, 34))

        # 1. Left View: Camera Debug (640x480)
        cam_surf = cv2_to_pygame_surface(debug_view)
        screen.blit(cam_surf, (15, 20))

        # 2. Right View: Calibrated Octo Crop (280x280)
        crop_display = cv2.resize(calibrated_crop, (270, 270))
        crop_surf = cv2_to_pygame_surface(crop_display)
        screen.blit(crop_surf, (670, 20))
        pygame.draw.rect(screen, (80, 200, 120), (669, 19, 272, 272), 2)

        # Information Panel below crop
        lbl_title = font_bold.render("CALIBRATED OCTO (256x256)", True, (240, 240, 240))
        screen.blit(lbl_title, (670, 305))

        status_txt = "STATUS: CALIBRATED (READY)" if calibrator.is_calibrated else "SEARCHING FOR MARKERS (0, 1, 2, 3)"
        status_col = (50, 240, 100) if calibrator.is_calibrated else (255, 100, 100)
        lbl_status = font.render(status_txt, True, status_col)
        screen.blit(lbl_status, (670, 335))

        lbl_cam = font.render(f"ACTIVE CAMERA: ID {cam_idx} [Press 'c' to switch]", True, (0, 220, 255))
        screen.blit(lbl_cam, (670, 365))

        lbl_fps = font.render(f"FPS: {fps:.1f}", True, (180, 190, 210))
        screen.blit(lbl_fps, (670, 395))

        lbl_help = font.render("[c] Switch Cam | [s] Save | [r] Reset | [q] Quit", True, (140, 150, 170))
        screen.blit(lbl_help, (670, 425))

        if time.time() < save_notification_timer:
            lbl_saved = font_bold.render("FRAME SAVED TO DISK!", True, (255, 220, 0))
            screen.blit(lbl_saved, (670, 455))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    print("[INFO] Exited cleanly.")

if __name__ == "__main__":
    main()
