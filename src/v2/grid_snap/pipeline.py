from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from .params import Params


@dataclass
class Result:
    x_centers: np.ndarray
    y_centers: np.ndarray
    spacing_x: Optional[float]
    spacing_y: Optional[float]
    origin_x: float
    origin_y: float
    grid: List[Tuple[str, float]]  # ('v' or 'h', value)
    rgb_up: np.ndarray
    edges: np.ndarray


def upscale_nn(img, s):
    if s == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * s, h * s), interpolation=cv2.INTER_NEAREST)


def canny_edges(img_rgb, low, high):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low, high, L2gradient=True)


def hough_rect_lines(edge, P: Params):
    lines = cv2.HoughLinesP(
        edge,
        P.hough_rho,
        np.deg2rad(P.hough_theta_deg),
        P.hough_thresh,
        minLineLength=P.min_line_len,
        maxLineGap=P.max_line_gap,
    )
    segs = []
    if lines is None:
        return segs
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx))
        ang = (ang + 180) % 180
        ang = 180 - ang if ang > 90 else ang
        if ang <= P.angle_tol_deg or abs(90 - ang) <= P.angle_tol_deg:
            segs.append((int(x1), int(y1), int(x2), int(y2)))
    return segs


def cluster_positions(vals, eps, weights=None):
    if not vals:
        return np.array([]), []
    order = np.argsort(vals)
    v = np.array(vals)[order]
    w = (
        np.ones_like(v, float)
        if weights is None
        else np.array(weights)[order].astype(float)
    )
    clusters = []
    cur, wcur = [v[0]], [w[0]]
    for vi, wi in zip(v[1:], w[1:]):
        if abs(vi - cur[-1]) <= eps:
            cur.append(vi)
            wcur.append(wi)
        else:
            clusters.append((np.average(cur, weights=wcur), cur, wcur))
            cur, wcur = [vi], [wi]
    clusters.append((np.average(cur, weights=wcur), cur, wcur))
    centers = np.array([c[0] for c in clusters])
    return centers, clusters


def filter_by_support(centers, clusters, min_support):
    out = []
    for center, pts, ws in clusters:
        if sum(ws) >= min_support:
            out.append(center)
    return np.array(sorted(out))


def estimate_spacing(centers):
    if len(centers) < 3:
        return None
    c = np.sort(centers)
    diffs = np.diff(c)
    diffs = diffs[diffs > 1]
    if len(diffs) == 0:
        return None
    base = np.median(diffs)
    candidates = []
    for m in np.linspace(max(2, 0.6 * base), 1.6 * base, 25):
        k = np.round(diffs / m)
        err = np.abs(diffs - k * m)
        score = np.mean(err / (m + 1e-6))
        candidates.append((score, m))
    return float(min(candidates)[1])


def circular_median_mod(values, spacing):
    if spacing is None or len(values) == 0:
        return 0.0
    r = np.mod(values, spacing)
    r.sort()
    # simple circular median
    mid = r[len(r) // 2]
    return float(mid)


def build_grid(xc, yc, w, h, sx, sy):
    if sx is None or sy is None:
        return []
    ox = float(np.min(xc)) % sx if len(xc) else 0.0
    oy = float(np.min(yc)) % sy if len(yc) else 0.0
    xs = np.arange(-ox, w, sx)
    ys = np.arange(-oy, h, sy)
    grid = [("v", float(x)) for x in xs if 0 <= x < w] + [
        ("h", float(y)) for y in ys if 0 <= y < h
    ]
    return grid, ox, oy


def run(rgb, P: Params) -> Result:
    rgb_up = upscale_nn(rgb, P.upscale)
    edges = canny_edges(rgb_up, P.canny_low, P.canny_high)
    segs = hough_rect_lines(edges, P)

    xv, xw, yv, yw = [], [], [], []
    for x1, y1, x2, y2 in segs:
        if abs(y2 - y1) <= abs(x2 - x1):
            yv.append((y1 + y2) / 2)
            yw.append(abs(x2 - x1) + 1)
        else:
            xv.append((x1 + x2) / 2)
            xw.append(abs(y2 - y1) + 1)

    x_centers, x_cl = cluster_positions(xv, P.cluster_eps, xw)
    y_centers, y_cl = cluster_positions(yv, P.cluster_eps, yw)
    x_centers = filter_by_support(x_centers, x_cl, P.min_support)
    y_centers = filter_by_support(y_centers, y_cl, P.min_support)

    sx = estimate_spacing(x_centers)
    sy = estimate_spacing(y_centers)
    if P.force_square_cells and sx and sy:
        s = float(np.median([sx, sy]))
        sx = sy = s

    grid, ox, oy = build_grid(
        x_centers, y_centers, rgb_up.shape[1], rgb_up.shape[0], sx, sy
    )

    return Result(
        x_centers=x_centers,
        y_centers=y_centers,
        spacing_x=sx,
        spacing_y=sy,
        origin_x=ox,
        origin_y=oy,
        grid=grid,
        rgb_up=rgb_up,
        edges=edges,
    )
