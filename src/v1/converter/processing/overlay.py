from io import BytesIO
from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def draw_edges_overlay(
    original_bgr: np.ndarray,
    edges: np.ndarray,
    color_rgb: Tuple[int, int, int],
    thickness=1,
    alpha=0.8,
) -> np.ndarray:
    overlay = original_bgr.copy()
    ys, xs = np.where(edges != 0)
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB→BGR
    if thickness <= 1:
        overlay[ys, xs] = color_bgr
    else:
        mask = np.zeros_like(overlay)
        for x, y in zip(xs, ys):
            cv2.circle(mask, (int(x), int(y)), max(1, thickness // 2), color_bgr, -1)
        nonzero = np.any(mask != 0, axis=2)
        overlay[nonzero] = mask[nonzero]
    blended = cv2.addWeighted(overlay, alpha, original_bgr, 1 - alpha, 0)
    return blended


def draw_grid_overlay(
    bgr: np.ndarray, clustered_x: list, clustered_y: list
) -> np.ndarray:
    grid_overlay = bgr.copy()
    for x in clustered_x:
        xi = int(round(x))
        cv2.line(
            grid_overlay,
            (xi, 0),
            (xi, grid_overlay.shape[0] - 1),
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    for y in clustered_y:
        yi = int(round(y))
        cv2.line(
            grid_overlay,
            (0, yi),
            (grid_overlay.shape[1] - 1, yi),
            (255, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    return grid_overlay


def to_png_bytes(img_bgr: np.ndarray) -> bytes:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    bio = BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()
