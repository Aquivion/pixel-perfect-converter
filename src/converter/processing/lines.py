from typing import List, Tuple
import cv2
import numpy as np


def segment_angle_degrees(x1, y1, x2, y2) -> float:
    dx, dy = (x2 - x1), (y2 - y1)
    return float(np.degrees(np.arctan2(abs(dy), abs(dx))))  # 0..90


def cluster_1d(values: List[float], eps: float) -> List[float]:
    if not values:
        return []
    vals = sorted(values)
    clusters, current = [], [vals[0]]
    for v in vals[1:]:
        if abs(v - current[-1]) <= eps:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    clusters.append(current)
    return [float(np.mean(c)) for c in clusters]


def hough_lines_filtered(
    edges: np.ndarray,
    rho: float,
    theta_deg: float,
    threshold: int,
    min_line_length: int,
    max_line_gap: int,
    angle_tol_deg: float,
) -> Tuple[List[dict], List[float], List[float]]:
    kept_lines: List[dict] = []
    grid_x, grid_y = [], []
    lines = cv2.HoughLinesP(
        edges,
        rho=float(rho),
        theta=np.deg2rad(float(theta_deg)),
        threshold=int(threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )
    if lines is None:
        return kept_lines, grid_x, grid_y

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        ang = segment_angle_degrees(x1, y1, x2, y2)
        is_horizontal = ang <= float(angle_tol_deg)
        is_vertical = abs(90.0 - ang) <= float(angle_tol_deg)
        if not (is_horizontal or is_vertical):
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        kept_lines.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "angle_deg": float(ang),
                "length": length,
                "orientation": "horizontal" if is_horizontal else "vertical",
            }
        )
        if is_vertical:
            grid_x.append((x1 + x2) / 2.0)
        else:
            grid_y.append((y1 + y2) / 2.0)

    return kept_lines, grid_x, grid_y
