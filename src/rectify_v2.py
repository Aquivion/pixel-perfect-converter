import sys
import math
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def auto_canny(img_gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with automatic thresholds based on image median."""
    v = np.median(img_gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_gray, lower, upper, L2gradient=True)


def hough_lines(
    bin_img: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 150,
):
    """Standard Hough transform returning (rho, theta)."""
    lines = cv2.HoughLines(bin_img, rho, theta, threshold)
    if lines is None:
        return []
    return [tuple(l[0]) for l in lines]  # flatten to [(rho, theta), ...]


def separate_lines(lines: List[Tuple[float, float]], angle_tol_deg: float = 10.0):
    """Split into vertical-like and horizontal-like using angle tolerance."""
    vertical = []
    horizontal = []
    angle_tol = math.radians(angle_tol_deg)

    for rho, theta in lines:
        # Vertical: theta ~ 0 or ~ pi
        if abs(theta - 0) < angle_tol or abs(theta - math.pi) < angle_tol:
            vertical.append((rho, theta))
        # Horizontal: theta ~ pi/2
        elif abs(theta - math.pi / 2) < angle_tol:
            horizontal.append((rho, theta))

    return vertical, horizontal


def lines_to_positions_vertical(lines: List[Tuple[float, float]]) -> List[float]:
    """Convert (rho, theta) to x = rho / cos(theta) for vertical lines."""
    xs = []
    for rho, theta in lines:
        c = math.cos(theta)
        if abs(c) < 1e-6:
            continue
        xs.append(rho / c)
    return xs


def lines_to_positions_horizontal(lines: List[Tuple[float, float]]) -> List[float]:
    """Convert (rho, theta) to y = rho / sin(theta) for horizontal lines."""
    ys = []
    for rho, theta in lines:
        s = math.sin(theta)
        if abs(s) < 1e-6:
            continue
        ys.append(rho / s)
    return ys


def cluster_positions(vals: List[float], tol: float) -> List[float]:
    """1D clustering: merge values that are within `tol`, return cluster means."""
    if not vals:
        return []
    vals = sorted(vals)
    clusters = []
    cur = [vals[0]]
    for v in vals[1:]:
        if abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            clusters.append(np.mean(cur))
            cur = [v]
    clusters.append(np.mean(cur))
    return clusters


def refine_with_morph(
    edges: np.ndarray, w: int, h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use morphology to enhance long vertical/horizontal segments.
    Returns (vertical_map, horizontal_map) binary maps.
    """
    # scale kernels to image size (tweak if necessary)
    vert_kernel_len = max(10, h // 50)
    hori_kernel_len = max(10, w // 50)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hori_kernel_len, 1))

    vertical_map = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vert_kernel, iterations=1)
    horizontal_map = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, hori_kernel, iterations=1)
    return vertical_map, horizontal_map


def detect_grid_lines(
    img_bgr: np.ndarray,
    angle_tol_deg: float = 10.0,
    hough_threshold: int = 180,
    cluster_tol_px_factor: float = 0.0125,
):
    """
    Detect vertical and horizontal grid line positions in pixels.

    Returns:
        x_positions (list of x in px), y_positions (list of y in px)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(
        gray, d=9, sigmaColor=50, sigmaSpace=50
    )  # preserves edges better than Gaussian
    edges = auto_canny(gray, sigma=0.33)

    # Strengthen long line segments in each direction
    vert_map, hori_map = refine_with_morph(edges, w, h)
    merged = cv2.bitwise_or(vert_map, hori_map)

    # Hough on the merged map
    lines = hough_lines(merged, rho=1, theta=np.pi / 180, threshold=hough_threshold)

    # If detection is weak, fall back to Hough on plain edges
    if not lines:
        lines = hough_lines(
            edges, rho=1, theta=np.pi / 180, threshold=max(100, hough_threshold // 2)
        )

    vertical, horizontal = separate_lines(lines, angle_tol_deg=angle_tol_deg)

    xs = lines_to_positions_vertical(vertical)
    ys = lines_to_positions_horizontal(horizontal)

    # Cluster lines to merge near-duplicates
    x_tol = max(2.0, w * cluster_tol_px_factor)  # e.g. ~1.25% of width
    y_tol = max(2.0, h * cluster_tol_px_factor)

    xs_merged = cluster_positions(xs, x_tol)
    ys_merged = cluster_positions(ys, y_tol)

    # Keep only those within image bounds after clustering
    xs_final = [float(x) for x in xs_merged if -5 <= x <= w + 5]
    ys_final = [float(y) for y in ys_merged if -5 <= y <= h + 5]

    return xs_final, ys_final


def draw_overlay(
    img_bgr: np.ndarray,
    xs: List[float],
    ys: List[float],
    line_thickness: int = 2,
    alpha: float = 0.75,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Draw vertical (xs) and horizontal (ys) lines as an overlay on the image.

    Uses cv2.addWeighted to avoid broadcasting issues.
    """
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # Draw lines on overlay
    for x in xs:
        xi = int(round(x))
        cv2.line(
            overlay, (xi, 0), (xi, h - 1), color, line_thickness, lineType=cv2.LINE_AA
        )

    for y in ys:
        yi = int(round(y))
        cv2.line(
            overlay, (0, yi), (w - 1, yi), color, line_thickness, lineType=cv2.LINE_AA
        )

    # Blend overlay with original
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Detect and overlay grid lines on an image."
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save output image with overlay")
    parser.add_argument(
        "--angle_tol_deg",
        type=float,
        default=180,
        help="Angle tolerance for vertical/horizontal classification",
    )
    parser.add_argument(
        "--hough_threshold", type=int, default=2, help="Hough accumulator threshold"
    )
    parser.add_argument(
        "--cluster_tol_pct",
        type=float,
        default=1.25,
        help="Clustering tolerance as percent of image size (1D). Default 1.25",
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Overlay grid line thickness in pixels"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.75, help="Overlay opacity (0..1)"
    )
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: Could not read input image '{args.input}'")
        sys.exit(1)

    h, w = img.shape[:2]
    xs, ys = detect_grid_lines(
        img,
        angle_tol_deg=args.angle_tol_deg,
        hough_threshold=args.hough_threshold,
        cluster_tol_px_factor=(args.cluster_tol_pct / 100.0),
    )

    out = draw_overlay(
        img,
        xs,
        ys,
        line_thickness=args.thickness,
        alpha=args.alpha,
        color=(0, 0, 255),  # red in BGR
    )

    # Optional: visualize where lines were found in console
    print(f"Detected vertical lines at x ≈ {', '.join(f'{x:.1f}' for x in xs)}")
    print(f"Detected horizontal lines at y ≈ {', '.join(f'{y:.1f}' for y in ys)}")
    cv2.imwrite(args.output, out)
    print(f"Saved overlay to '{args.output}'")


if __name__ == "__main__":
    main()
