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


def estimate_spacing(positions: List[float]) -> Tuple[float, List[float]]:
    """
    Estimate grid spacing using median of inlier diffs (robust to outliers).
    Returns (spacing, inlier_diffs). spacing = 0 if cannot estimate.
    """
    if len(positions) < 2:
        return 0.0, []
    ps = np.sort(np.array(positions, dtype=float))
    diffs = np.diff(ps)
    # Remove zero / near-zero accidental duplicates
    diffs = diffs[diffs > 0.5]
    if len(diffs) == 0:
        return 0.0, []
    med = float(np.median(diffs))
    abs_dev = np.abs(diffs - med)
    mad = float(np.median(abs_dev))
    if mad == 0:
        # Fallback: keep diffs within 25% of median
        inliers = diffs[np.abs(diffs - med) <= 0.25 * med]
    else:
        # Standard robust threshold ~2.5 * MAD
        inliers = diffs[abs_dev <= 2.5 * mad]
    if len(inliers) == 0:
        return 0.0, []
    spacing = float(np.median(inliers))
    return spacing, list(map(float, inliers))


def complete_grid(
    positions: List[float],
    spacing: float,
    min_bound: float,
    max_bound: float,
    jitter_frac: float = 0.3,
    max_multi: int = 6,
) -> List[float]:
    """
    Build a completed grid:
      1. Optionally fill internal gaps that are multiples of spacing (within tolerance).
      2. Extend backward / forward to bounds.
      3. Merge near-duplicates.
    jitter_frac: fractional tolerance vs spacing for merging & accepting subdivided gaps.
    max_multi: maximum gap multiple to try to subdivide (prevents runaway insertion).
    """
    if spacing <= 0 or not positions:
        return positions

    existing = sorted(float(p) for p in positions)
    tol = max(1.0, spacing * jitter_frac)

    # ---- 1. Fill internal gaps ----
    filled = [existing[0]]
    for curr in existing[1:]:
        prev = filled[-1]
        gap = curr - prev
        if gap > spacing + tol:
            n = int(round(gap / spacing))
            if 2 <= n <= max_multi:
                candidate_interval = gap / n
                if abs(candidate_interval - spacing) <= spacing * jitter_frac:
                    # Insert n-1 intermediate lines
                    for k in range(1, n):
                        filled.append(prev + k * candidate_interval)
        filled.append(curr)

    # ---- 2. Extend backward & forward ----
    first = filled[0]
    back = []
    v = first
    while v - spacing >= min_bound - tol:
        v -= spacing
        back.append(v)

    last = filled[-1]
    forward = []
    v = last
    while v + spacing <= max_bound + tol:
        v += spacing
        forward.append(v)

    all_lines = sorted(back + filled + forward)

    # ---- 3. Merge close lines & clamp ----
    merged: List[float] = []
    for val in all_lines:
        if not merged or abs(val - merged[-1]) > tol:
            merged.append(val)
        else:
            merged[-1] = (merged[-1] + val) / 2.0  # average cluster

    clamped = [float(min(max(val, min_bound), max_bound)) for val in merged]

    # Remove any residual near duplicates after clamping (tight threshold)
    final: List[float] = []
    for v in clamped:
        if not final or abs(v - final[-1]) > 0.5:
            final.append(v)

    return final
