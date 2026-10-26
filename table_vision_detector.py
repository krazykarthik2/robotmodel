import cv2
import numpy as np

# Color ranges in HSV format
COLOR_HSV_RANGES = {
    "red": [
        (np.array([0, 90, 80]), np.array([10, 255, 255])),
        (np.array([165, 90, 80]), np.array([180, 255, 255]))
    ],
    "green": [
        (np.array([35, 80, 70]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([95, 100, 70]), np.array([130, 255, 255]))
    ],
    "yellow": [
        (np.array([18, 90, 90]), np.array([35, 255, 255]))
    ],
    "orange": [
        (np.array([10, 100, 90]), np.array([22, 255, 255]))
    ],
    "purple": [
        (np.array([130, 70, 60]), np.array([160, 255, 255]))
    ]
}

def detect_colored_objects(crop_img, min_area=80):
    """
    Detects colored cubes/objects in the calibrated Octo POV image.
    Returns dict: {color_name: {"pixel": (u, v), "area": int, "phys_mm": (x_mm, y_mm)}}
    """
    if crop_img is None or crop_img.size == 0:
        return {}

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    h, w = crop_img.shape[:2]
    detected = {}

    for color_name, ranges in COLOR_HSV_RANGES.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, m)

        # Morphological opening to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Pick largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area >= min_area:
            M = cv2.moments(largest)
            if M["m00"] > 0:
                u = float(M["m10"] / M["m00"])
                v = float(M["m01"] / M["m00"])
                # Inverse homography pixel to physical mm:
                # Table spans: X: [40mm, 400mm], Y: [-180mm, +180mm]
                y_mm = 180.0 - (u / (w - 1.0)) * 360.0
                x_mm = 400.0 - (v / (h - 1.0)) * 360.0
                detected[color_name] = {
                    "pixel": (int(u), int(v)),
                    "area": int(area),
                    "phys_mm": (float(x_mm), float(y_mm))
                }

    return detected
