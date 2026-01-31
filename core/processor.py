from __future__ import annotations

import cv2
import numpy as np


class ImageProcessor:
    """
    Pure image operations (OpenCV).
    This class is intentionally stateless; GUI/model provide state.
    """

    @staticmethod
    def grayscale(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def blur(bgr: np.ndarray, intensity: int) -> np.ndarray:
        # intensity: 0..25 (mapped to odd kernel sizes)
        k = max(1, int(intensity))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(bgr, (k, k), 0)

    @staticmethod
    def edge_detect(bgr: np.ndarray, threshold1: int) -> np.ndarray:
        # Simple: threshold2 is derived
        t1 = int(threshold1)
        t2 = min(255, t1 * 3)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t1, t2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def adjust_brightness(bgr: np.ndarray, beta: int) -> np.ndarray:
        # beta: -100..100
        return cv2.convertScaleAbs(bgr, alpha=1.0, beta=int(beta))

    @staticmethod
    def adjust_contrast(bgr: np.ndarray, alpha: float) -> np.ndarray:
        # alpha: 0.5..3.0
        return cv2.convertScaleAbs(bgr, alpha=float(alpha), beta=0)

    @staticmethod
    def rotate(bgr: np.ndarray, degrees: int) -> np.ndarray:
        deg = int(degrees) % 360
        if deg == 90:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        if deg == 270:
            return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return bgr.copy()

    @staticmethod
    def flip(bgr: np.ndarray, mode: str) -> np.ndarray:
        # mode: "horizontal" or "vertical"
        if mode == "horizontal":
            return cv2.flip(bgr, 1)
        if mode == "vertical":
            return cv2.flip(bgr, 0)
        return bgr.copy()

    @staticmethod
    def resize_scale(bgr: np.ndarray, scale_percent: int) -> np.ndarray:
        # scale_percent: 10..200
        sp = max(10, min(200, int(scale_percent)))
        h, w = bgr.shape[:2]
        new_w = max(1, int(w * sp / 100))
        new_h = max(1, int(h * sp / 100))
        return cv2.resize(
            bgr,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if sp < 100 else cv2.INTER_LINEAR
        )
