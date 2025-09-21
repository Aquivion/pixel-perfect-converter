import numpy as np


def draw_grid(rgb: np.ndarray, grid, color=(255, 255, 255)) -> np.ndarray:
    out = rgb.copy()
    for kind, v in grid:
        if kind == "v":
            x = int(round(v))
            if 0 <= x < out.shape[1]:
                out[:, x : x + 1] = color
        else:
            y = int(round(v))
            if 0 <= y < out.shape[0]:
                out[y : y + 1, :] = color
    return out
