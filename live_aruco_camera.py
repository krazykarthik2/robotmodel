"""
Live camera runner for Dobot Magician with ArUco auto-calibration.
Run this script with your webcam connected (e.g. RealSense, USB webcam):
    python live_aruco_camera.py --camera_id 0
Press 'q' to quit, 's' to save current calibrated frame.
"""
import cv2
import argparse
import time
from aruco_calibrator import ArucoAutoCalibrator

def main():
    parser = argparse.ArgumentParser(description="Live Dobot ArUco Auto-Calibration Pipeline")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--out_size", type=int, default=256, help="Output crop size for Octo (default: 256)")
    args = parser.parse_args()

    print(f"[INFO] Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera_id}. Check connection or index.")
        return

    calibrator = ArucoAutoCalibrator(output_size=(args.out_size, args.out_size))
    print("[INFO] ArUco Auto-Calibrator initialized. Point camera at the workspace.")
    print("[INFO] Hotkeys: [q] Quit  |  [s] Save Frame  |  [r] Reset Calibration")

    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            time.sleep(0.05)
            continue

        calibrated_crop, debug_view = calibrator.process_frame(frame, auto_calibrate=True)
        
        # Calculate FPS
        frame_count += 1
        fps = frame_count / (time.time() - t0)
        cv2.putText(debug_view, f"FPS: {fps:.1f}", (debug_view.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show feeds
        cv2.imshow("Original Camera Feed (Debug)", debug_view)
        cv2.imshow("Calibrated Octo Crop (256x256)", calibrated_crop)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("saved_calibrated_obs.png", calibrated_crop)
            print("[SAVED] Saved current frame to saved_calibrated_obs.png")
        elif key == ord('r'):
            calibrator.is_calibrated = False
            print("[INFO] Calibration reset.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
