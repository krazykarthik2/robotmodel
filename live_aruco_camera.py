"""
Live camera viewer with on-the-fly camera switching and ArUco auto-calibration:
- Direct, instant camera opening for Camera 0 and Camera 1 (no slow/failing index scanning)
- Uses DirectShow (cv2.CAP_DSHOW) on Windows with safety fallbacks
- Pygame GUI window showing both camera feed and rectified Octo observation
- Hotkey [c] -> SWITCH CAMERA (Camera 0 <-> Camera 1)
- Hotkey [s] -> SAVE current calibrated crop to disk
- Hotkey [r] -> RESET calibration
- Hotkey [q] or [ESC] -> QUIT
"""
import sys
import time
import argparse
import numpy as np
import cv2
import pygame

from aruco_calibrator import ArucoAutoCalibrator

def open_camera(cam_idx):
    """Safely opens camera with DirectShow backend."""
    print(f"[INFO] Opening Camera ID: {cam_idx}...")
    try:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)
    except Exception as e:
        print(f"[WARN] Error with DSHOW, using default backend: {e}")
        cap = cv2.VideoCapture(cam_idx)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def cv2_to_pygame_surface(cv_img):
    """Converts a BGR or RGB OpenCV image to a Pygame Surface."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

def main():
    parser = argparse.ArgumentParser(description="Live ArUco Camera with Instant Switching")
    parser.add_argument("--camera_id", type=int, default=0, help="Initial camera ID: 0 or 1 (default: 0)")
    parser.add_argument("--out_size", type=int, default=256, help="Output crop size for Octo (default: 256)")
    args = parser.parse_args()

    # Pre-configured available cameras list
    known_cameras = [0, 1]
    cam_idx = args.camera_id if args.camera_id in known_cameras else 0

    cap = open_camera(cam_idx)
    if not cap.isOpened():
        print(f"[WARN] Could not open camera {cam_idx}. Trying other camera...")
        cam_idx = 1 if cam_idx == 0 else 0
        cap = open_camera(cam_idx)
        if not cap.isOpened():
            print("[ERROR] Neither Camera 0 nor Camera 1 could be opened. Please verify USB connection.")
            return

    calibrator = ArucoAutoCalibrator(output_size=(args.out_size, args.out_size), marker_size_mm=80.0)

    # Initialize Pygame GUI window
    pygame.init()
    win_w, win_h = 960, 520
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Dobot ArUco Auto-Calibrator (Camera {cam_idx})")
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    print("\n" + "=" * 60)
    print("LIVE ARUCO CALIBRATION & CAMERA SWITCHER (ACTIVE)")
    print("=" * 60)
    print("Controls (In Pygame Window):")
    print("  [c] -> SWITCH CAMERA (Camera 0 <-> Camera 1)")
    print("  [s] -> SAVE current calibrated crop to disk")
    print("  [r] -> RESET calibration")
    print("  [q] / [ESC] -> QUIT")
    print("=" * 60 + "\n")

    running = True
    frame_count = 0
    t0 = time.time()
    save_notification_timer = 0
    last_crop = None

    while running:
        # Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key == pygame.K_c:
                    # Switch Camera 0 <-> Camera 1
                    next_idx = 1 if cam_idx == 0 else 0
                    print(f"\n[INFO] Switching to Camera ID {next_idx}...")
                    cap.release()
                    new_cap = open_camera(next_idx)
                    if new_cap.isOpened():
                        cap = new_cap
                        cam_idx = next_idx
                        pygame.display.set_caption(f"Dobot ArUco Auto-Calibrator (Camera {cam_idx})")
                        calibrator.is_calibrated = False
                        frame_count = 0
                        t0 = time.time()
                    else:
                        print(f"[WARN] Failed to open Camera {next_idx}, keeping Camera {cam_idx}")
                        cap = open_camera(cam_idx)
                elif event.key == pygame.K_s:
                    if last_crop is not None:
                        cv2.imwrite("saved_calibrated_obs.png", last_crop)
                        print("[SAVED] Saved frame to 'saved_calibrated_obs.png'!")
                        save_notification_timer = time.time() + 2.0
                elif event.key == pygame.K_r:
                    calibrator.is_calibrated = False
                    print("[INFO] Calibration reset. Re-detecting markers...")

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=True)
        last_crop = calibrated_crop

        frame_count += 1
        fps = frame_count / (time.time() - t0 + 1e-5)

        # Draw UI
        screen.fill((25, 27, 34))

        # 1. Left View: Camera Debug (640x480)
        cam_surf = cv2_to_pygame_surface(debug_view)
        screen.blit(cam_surf, (15, 20))

        # 2. Right View: Calibrated Octo Crop (270x270)
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
