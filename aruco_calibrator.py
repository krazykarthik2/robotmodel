"""
ArucoAutoCalibrator:
Detects 4 ArUco markers (IDs 0, 1, 2, 3), computes perspective homography,
crops the workspace to an exact square (256x256 for Octo or any custom size),
orients the coordinate frame so +X points forward and +Y points left (standard robot frame),
and maps pixel coordinates to physical Dobot coordinates (mm or meters).
"""
import cv2
import numpy as np

class ArucoAutoCalibrator:
    def __init__(self, output_size=(256, 256), dict_type=cv2.aruco.DICT_4X4_50):
        """
        Marker ID Placement (Dobot base at bottom center):
            ID 0: Top-Left     [X = +300 mm, Y = -150 mm]
            ID 1: Top-Right    [X = +300 mm, Y = +150 mm]
            ID 2: Bottom-Right [X = +150 mm, Y = +150 mm]
            ID 3: Bottom-Left  [X = +150 mm, Y = -150 mm]
        """
        self.output_size = output_size # (width, height)
        self.dict_type = dict_type
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Physical workspace boundaries corresponding to the 4 corners (in mm)
        self.x_min_mm = 150.0
        self.x_max_mm = 300.0
        self.y_min_mm = -150.0
        self.y_max_mm = 150.0

        # Destination pixel coordinates for [ID 0, ID 1, ID 2, ID 3]
        # In image: row 0 is top (X_max = +300), col 0 is left (Y_min = -150)
        # ID 0 (Top-Left): (0, 0)
        # ID 1 (Top-Right): (output_w - 1, 0)
        # ID 2 (Bottom-Right): (output_w - 1, output_h - 1)
        # ID 3 (Bottom-Left): (0, output_h - 1)
        w, h = output_size
        self.dst_pts = np.array([
            [0, 0],         # ID 0
            [w - 1, 0],     # ID 1
            [w - 1, h - 1], # ID 2
            [0, h - 1]      # ID 3
        ], dtype=np.float32)

        self.homography_matrix = None
        self.is_calibrated = False

    def detect_markers(self, frame):
        """Detects ArUco markers and returns corner dict {id: center_point}."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        marker_centers = {}
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0] # 4 corners: [top-left, top-right, bottom-right, bottom-left]
                center_x = float(np.mean(c[:, 0]))
                center_y = float(np.mean(c[:, 1]))
                marker_centers[int(marker_id)] = (center_x, center_y)
        return marker_centers, corners, ids

    def calibrate(self, frame):
        """Computes homography warp matrix from the 4 detected ArUco markers."""
        centers, _, _ = self.detect_markers(frame)
        required_ids = [0, 1, 2, 3]
        
        missing = [mid for mid in required_ids if mid not in centers]
        if missing:
            return False, f"Missing ArUco markers: IDs {missing}. Found: {list(centers.keys())}"

        src_pts = np.array([centers[mid] for mid in required_ids], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, self.dst_pts)
        self.homography_matrix = H
        self.is_calibrated = True
        return True, "Calibration successful! Homography computed."

    def process_frame(self, frame, auto_calibrate=True):
        """
        Crops, rectifies, and rotates the camera frame into calibrated workspace space.
        Returns:
            calibrated_rgb: Exactly (H, W, 3) oriented to robot workspace (256x256 default).
            debug_view: Raw camera frame annotated with detected markers and calibration box.
        """
        debug_view = frame.copy()
        centers, corners, ids = self.detect_markers(frame)
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(debug_view, corners, ids)
            for mid, (cx, cy) in centers.items():
                cv2.circle(debug_view, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.putText(debug_view, f"ID {mid}", (int(cx) + 10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if auto_calibrate or not self.is_calibrated:
            if all(mid in centers for mid in [0, 1, 2, 3]):
                self.calibrate(frame)

        if self.is_calibrated:
            w, h = self.output_size
            calibrated = cv2.warpPerspective(frame, self.homography_matrix, (w, h))
            
            # Draw boundary polygon on debug view
            inv_H = np.linalg.inv(self.homography_matrix)
            poly_pts = cv2.perspectiveTransform(self.dst_pts[None, :, :], inv_H)[0].astype(np.int32)
            cv2.polylines(debug_view, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(debug_view, "CALIBRATED WORKSPACE", (poly_pts[0][0], poly_pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            return calibrated, debug_view
        else:
            # Fallback if not all markers visible yet
            fallback = cv2.resize(frame, self.output_size)
            cv2.putText(debug_view, "WAITING FOR ARUCO 0, 1, 2, 3...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return fallback, debug_view

    def pixel_to_dobot_xyz(self, u, v, z_surface=0.0):
        """
        Converts pixel coordinate (u, v) in calibrated output frame to Dobot physical coordinates (x, y, z) in meters.
        u: 0 -> w-1 (left to right)
        v: 0 -> h-1 (top to bottom)
        """
        w, h = self.output_size
        u_norm = np.clip(u / (w - 1), 0.0, 1.0)
        v_norm = np.clip(v / (h - 1), 0.0, 1.0)
        
        # v: 0 is X_max (+300mm), 1 is X_min (+150mm)
        x_mm = self.x_max_mm - v_norm * (self.x_max_mm - self.x_min_mm)
        # u: 0 is Y_min (-150mm), 1 is Y_max (+150mm)
        y_mm = self.y_min_mm + u_norm * (self.y_max_mm - self.y_min_mm)
        
        return np.array([x_mm / 1000.0, y_mm / 1000.0, z_surface], dtype=np.float32)

    def dobot_xyz_to_pixel(self, x_m, y_m):
        """Converts Dobot physical coordinate in meters to (u, v) in calibrated image."""
        w, h = self.output_size
        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0
        
        v_norm = (self.x_max_mm - x_mm) / (self.x_max_mm - self.x_min_mm)
        u_norm = (y_mm - self.y_min_mm) / (self.y_max_mm - self.y_min_mm)
        
        u = int(np.clip(u_norm * (w - 1), 0, w - 1))
        v = int(np.clip(v_norm * (h - 1), 0, h - 1))
        return u, v
