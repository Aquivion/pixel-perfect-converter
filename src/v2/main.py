# Algorithm + demo: detect rectilinear edges in pixel art and recover a snapped grid.
#
# Steps in code:
# 1) Load image, optionally NN-upscale for thicker edges.
# 2) Grayscale + Canny.
# 3) HoughLinesP → keep near-horizontal/vertical segments.
# 4) Cluster X/Y line positions (simple 1D clustering with gap threshold).
# 5) Estimate base grid spacing from consecutive gaps (robust median).
# 6) Build rectilinear grid snapped to origin/spacing; save overlay for you.
#
# The output will show:
#  - Edges
#  - Detected line clusters
#  - Final grid overlay
#
# You can tweak EPS_PIX, MIN_SUPPORT, and HOUGH params at the top.

import cv2
import numpy as np
import os

IMG_PATH = "./images/magical_weapons/input.png"

# --- Tunables (good defaults for chunky pixel art) ---
UPSCALE = 2  # Nearest-neighbor scale factor
CANNY_LOW, CANNY_HIGH = 40, 120
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESH = 60
MIN_LINE_LEN = 30
MAX_LINE_GAP = 3

ANGLE_TOL_DEG = 4.0  # accept lines within ±tol of 0° or 90°
EPS_PIX = 4  # cluster tolerance in pixels (after upscaling)
MIN_SUPPORT = 6  # minimum total segment length supporting a grid line


# --- Helpers ---
def imread_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[-1] == 4:
        # composite on opaque background to avoid edge artifacts from alpha
        alpha = img[..., 3:4] / 255.0
        rgb = img[..., :3]
        bg = np.ones_like(rgb) * 255
        img = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def upscale_nn(img, s):
    if s == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * s, h * s), interpolation=cv2.INTER_NEAREST)


def canny_edges(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, L2gradient=True)
    return edges


def hough_rect_lines(edge):
    lines = cv2.HoughLinesP(
        edge,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESH,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP,
    )
    segs = []
    if lines is None:
        return segs
    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(int, l)
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        # normalize to [-90,90]
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        if abs(angle) <= ANGLE_TOL_DEG or abs(abs(angle) - 90) <= ANGLE_TOL_DEG:
            segs.append((x1, y1, x2, y2))
    return segs


def cluster_positions(vals, eps=EPS_PIX, weights=None):
    """1D clustering by merging sorted vals with gap<eps. Returns (centers, clusters)."""
    if len(vals) == 0:
        return np.array([]), []
    order = np.argsort(vals)
    vals_sorted = np.array(vals)[order]
    if weights is None:
        weights_sorted = np.ones_like(vals_sorted, dtype=float)
    else:
        weights_sorted = np.array(weights)[order].astype(float)

    clusters = []
    cur = [vals_sorted[0]]
    wcur = [weights_sorted[0]]
    for v, w in zip(vals_sorted[1:], weights_sorted[1:]):
        if abs(v - cur[-1]) <= eps:
            cur.append(v)
            wcur.append(w)
        else:
            clusters.append((np.average(cur, weights=wcur), cur, wcur))
            cur, wcur = [v], [w]
    clusters.append((np.average(cur, weights=wcur), cur, wcur))
    centers = np.array([c[0] for c in clusters])
    return centers, clusters


def estimate_spacing(centers):
    """Estimate base spacing from sorted centers using robust median of diffs & GCD-like refinement."""
    if len(centers) < 3:
        return None
    centers = np.sort(centers)
    diffs = np.diff(centers)
    diffs = diffs[diffs > 1]  # ignore 1px near-duplicate
    if len(diffs) == 0:
        return None
    base = np.median(diffs)
    # refine: try to find a spacing that explains most diffs as near multiples
    candidates = []
    for m in np.linspace(max(2, base * 0.6), base * 1.6, 25):
        k = np.round(diffs / m)
        err = np.abs(diffs - k * m)
        score = np.mean(err / (m + 1e-6))
        candidates.append((score, m))
    spacing = min(candidates)[1]
    return float(spacing)


def build_grid(x_centers, y_centers, img_w, img_h, spacing_x, spacing_y):
    if spacing_x is None or spacing_y is None:
        return []
    # choose origin as the grid line near the image border
    x0 = float(np.min(x_centers)) % spacing_x
    y0 = float(np.min(y_centers)) % spacing_y
    xs = np.arange(-x0, img_w, spacing_x)
    ys = np.arange(-y0, img_h, spacing_y)
    grid = []
    for x in xs:
        if 0 <= x < img_w:
            grid.append(("v", float(x)))
    for y in ys:
        if 0 <= y < img_h:
            grid.append(("h", float(y)))
    return grid


# --- Pipeline ---
rgb = imread_rgba(IMG_PATH)
rgb_up = upscale_nn(rgb, UPSCALE)
edges = canny_edges(rgb_up)
segs = hough_rect_lines(edges)

# Separate horizontal vs vertical, compute weighted positions (by segment length)
x_vals, x_w = [], []
y_vals, y_w = [], []
for x1, y1, x2, y2 in segs:
    if abs(y2 - y1) <= abs(x2 - x1):  # horizontal-ish
        y_vals.append((y1 + y2) / 2)
        y_w.append(abs(x2 - x1) + 1)
    else:  # vertical-ish
        x_vals.append((x1 + x2) / 2)
        x_w.append(abs(y2 - y1) + 1)

x_centers, x_clusters = cluster_positions(x_vals, EPS_PIX, x_w)
y_centers, y_clusters = cluster_positions(y_vals, EPS_PIX, y_w)


# Filter weak lines
def filter_by_support(centers, clusters, min_support=MIN_SUPPORT):
    out = []
    for c, (center, pts, ws) in enumerate(clusters):
        support = sum(ws)
        if support >= min_support:
            out.append(center)
    return np.array(sorted(out))


x_centers_f = filter_by_support(x_centers, x_clusters)
y_centers_f = filter_by_support(y_centers, y_clusters)

sx = estimate_spacing(x_centers_f)
sy = estimate_spacing(y_centers_f)

grid = build_grid(x_centers_f, y_centers_f, rgb_up.shape[1], rgb_up.shape[0], sx, sy)

# --- Save overlay next to input image ---
overlay = rgb_up.copy()
grid_img = overlay.copy()
for kind, v in grid:
    if kind == "v":
        x = int(round(v))
        grid_img[:, x : x + 1] = (255, 255, 255)
    else:
        y = int(round(v))
        grid_img[y : y + 1, :] = (255, 255, 255)

out_dir = os.path.dirname(IMG_PATH)
out_path = os.path.join(out_dir, "tavern_grid_overlay.png")
cv2.imwrite(out_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
print(f"Wrote grid overlay to: {out_path}")
