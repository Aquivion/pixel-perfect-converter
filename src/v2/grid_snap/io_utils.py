from pathlib import Path
import cv2
import numpy as np


def imread_rgba(path: str) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        a = img[..., 3:4] / 255.0
        rgb = img[..., :3]
        bg = np.ones_like(rgb) * 255
        img = (rgb * a + bg * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_png_rgb(path: str, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
