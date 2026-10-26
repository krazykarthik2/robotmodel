"""
Live camera viewer with on-the-fly camera switching and ArUco auto-calibration:
- Supports 8cm x 8cm markers placed side-by-side with 0 space (16cm x 16cm total)
- Hotkey [c] to switch camera index (0 -> 1 -> 2...)
- Hotkey [s] to save calibrated frame
- Hotkey [r] to reset calibration
- Hotkey [q] to quit
"""
import cv2
import argparse
import time
from aruco_calibrator import ArucoAutoCalibrator

def scan_available_cameras(max_to_test=5):
    available = []
    print("[INFO] Scanning for connected video cameras...")
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    print(f"[INFO] Found available cameras: {available}")
    return available

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
    print(f"[INFO] Opening Camera ID: {cam_idx}")
    cap = cv2.VideoCapture(cam_idx)

    calibrator = ArucoAutoCalibrator(output_size=(args.out_size, args.out_size), marker_size_mm=80.0)

    print("\n" + "=" * 55)
    print("LIVE ARUCO CALIBRATION & MULTI-CAMERA SWITCHER")
    print("=" * 55)
    print("Hotkeys:")
    print("  [c] -> SWITCH to another camera (e.g. Cam 0 <-> Cam 1)")
    print("  [s] -> SAVE current calibrated crop to disk")
    print("  [r] -> RESET / RE-DETECT calibration")
    print("  [q] -> QUIT")
    print("=" * 55 + "\n")

    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Failed to read from Camera {cam_idx}. Trying to reconnect...")
            time.sleep(0.5)
            continue

        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=True)

        frame_count += 1
        fps = frame_count / (time.time() - t0 + 1e-5)

        # Overlay Camera Info
        cv2.putText(debug_view, f"Camera ID: {cam_idx}  (Press 'c' to switch)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        status_color = (0, 255, 0) if calibrator.is_calibrated else (0, 0, 255)
        status_text = "CALIBRATED (Ready for Octo)" if calibrator.is_calibrated else "SEARCHING FOR MARKERS (0, 1, 2, 3)"
        cv2.putText(debug_view, status_text, (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        cv2.putText(debug_view, f"FPS: {fps:.1f}", (debug_view.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Overhead Camera (Debug & Markers)", debug_view)
        cv2.imshow("Calibrated Octo Crop (256x256)", calibrated_crop)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Switch to next camera
            if len(available_cams) > 1:
                cur_pos = available_cams.index(cam_idx)
                next_pos = (cur_pos + 1) % len(available_cams)
                cam_idx = available_cams[next_pos]
                print(f"\n[INFO] Switching to Camera ID {cam_idx}...")
                cap.release()
                cap = cv2.VideoCapture(cam_idx)
                calibrator.is_calibrated = False
                frame_count = 0
                t0 = time.time()
            else:
                print("[NOTICE] Only 1 camera detected on system.")
        elif key == ord('s'):
            cv2.imwrite("saved_calibrated_obs.png", calibrated_crop)
            print("[SAVED] Saved frame to 'saved_calibrated_obs.png'!")
        elif key == ord('r'):
            calibrator.is_calibrated = False
            print("[INFO] Calibration reset. Re-detecting markers...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
