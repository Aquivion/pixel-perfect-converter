from typing import Tuple
import cv2
import numpy as np


def ensure_3channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def auto_canny_thresholds(gray: np.ndarray, sigma: float) -> Tuple[int, int]:
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if upper - lower < 10:
        lower = max(0, lower - 5)
        upper = min(255, upper + 5)
    return lower, upper


def gaussian_blur(gray: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return gray
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(gray, (int(k), int(k)), 0)


def canny_edges(
    gray: np.ndarray, use_sigma: bool, sigma: float, low: int, high: int
) -> Tuple[np.ndarray, int, int]:
    if use_sigma:
        low_t, high_t = auto_canny_thresholds(gray, float(sigma))
    else:
        low_t, high_t = int(low), int(high)
    edges = cv2.Canny(gray, low_t, high_t)
    return edges, low_t, high_t


def morphology(
    edges: np.ndarray,
    close_k: int,
    close_iter: int,
    dilate_iter: int,
    kernel_shape: str,
) -> np.ndarray:
    out = edges
    if close_k and close_k > 0:
        shape = {
            "rect": cv2.MORPH_RECT,
            "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS,
        }.get(kernel_shape, cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(shape, (int(close_k), int(close_k)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
    if dilate_iter and dilate_iter > 0:
        kernel_d = np.ones((3, 3), np.uint8)
        out = cv2.dilate(out, kernel_d, iterations=int(dilate_iter))
    return out
