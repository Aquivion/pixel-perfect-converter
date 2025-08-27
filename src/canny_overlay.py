import json
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# ---------- Utilities from your script ----------
def ensure_3channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def auto_canny_thresholds(gray: np.ndarray, sigma: float) -> Tuple[int, int]:
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if upper - lower < 10:
        lower = max(0, lower - 5)
        upper = min(255, upper + 5)
    return lower, upper


def segment_angle_degrees(x1, y1, x2, y2) -> float:
    dx, dy = (x2 - x1), (y2 - y1)
    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))  # 0..90
    return angle


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


def draw_edges_overlay(original_bgr, edges, color_rgb, thickness=1, alpha=0.8):
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


def to_png_bytes(img_bgr: np.ndarray) -> bytes:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    bio = BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Canny + Hough Grid Tuner", layout="wide")
st.title("Canny + Hough Grid Tuner")

uploaded = st.file_uploader(
    "Upload an image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"]
)

with st.sidebar:
    st.header("Preprocess")
    scale = st.number_input(
        "Scale (nearest-neighbor integer)", min_value=1, max_value=8, value=2, step=1
    )
    blur = st.number_input(
        "Gaussian blur kernel (odd, 0=off)", min_value=0, max_value=31, value=3, step=1
    )

    st.header("Canny")
    use_sigma = st.checkbox("Use auto thresholds (sigma)", value=True)
    sigma = st.slider("sigma", 0.00, 1.00, 0.12, 0.01)
    low = st.slider("low", 0, 255, 50, 1)
    high = st.slider("high", 0, 255, 150, 1)

    st.header("Morphology")
    close_k = st.number_input(
        "Closing kernel size (0=off)", min_value=0, max_value=51, value=5, step=1
    )
    close_iter = st.number_input(
        "Closing iterations", min_value=1, max_value=10, value=1, step=1
    )
    dilate_iter = st.number_input(
        "Dilate iterations", min_value=0, max_value=10, value=0, step=1
    )

    st.header("Hough LinesP")
    enable_hough = st.checkbox("Enable Hough", value=True)
    rho = st.number_input("rho (px)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    theta_deg = st.number_input(
        "theta (deg)", min_value=0.5, max_value=5.0, value=1.0, step=0.5
    )
    hough_thresh = st.number_input(
        "accumulator threshold", min_value=1, max_value=500, value=60, step=1
    )
    min_len = st.number_input(
        "min line length", min_value=1, max_value=4000, value=220, step=10
    )
    max_gap = st.number_input(
        "max line gap", min_value=0, max_value=200, value=10, step=1
    )
    angle_tol = st.number_input(
        "angle tolerance (deg)", min_value=0.0, max_value=15.0, value=3.5, step=0.5
    )

    st.header("Clustering")
    cluster_eps = st.number_input(
        "cluster eps (px)", min_value=1.0, max_value=50.0, value=8.0, step=1.0
    )

    st.header("Overlay")
    edge_thickness = st.number_input(
        "overlay edge thickness", min_value=1, max_value=7, value=1, step=1
    )
    alpha = st.slider("overlay alpha", 0.0, 1.0, 0.8, 0.05)
    color = st.color_picker("overlay color", "#FF0000")
    # hex -> RGB
    color_rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# Decode uploaded image
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
if img is None:
    st.error("Could not read the image.")
    st.stop()

# Processing
bgr = ensure_3channel(img)

# 1) scale (nearest neighbor)
if scale > 1:
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

# 2) grayscale (+ optional blur)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
if blur > 0:
    if blur % 2 == 0:
        blur += 1
    gray_blur = cv2.GaussianBlur(gray, (int(blur), int(blur)), 0)
else:
    gray_blur = gray

# 3) Canny thresholds
if use_sigma:
    low_t, high_t = auto_canny_thresholds(gray_blur, float(sigma))
else:
    low_t, high_t = int(low), int(high)

edges = cv2.Canny(gray_blur, low_t, high_t)

# 4) Closing
if close_k and close_k > 0:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(close_k), int(close_k)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))

# 5) Optional dilation
if dilate_iter and dilate_iter > 0:
    kernel_d = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel_d, iterations=int(dilate_iter))

# Previews: edge map and overlay
edges_vis = np.zeros_like(bgr)
edges_vis[edges != 0] = (255, 255, 255)
overlay_bgr = draw_edges_overlay(
    bgr, edges, color_rgb, thickness=int(edge_thickness), alpha=float(alpha)
)

# 6) Hough + filter + cluster
kept_lines = []
grid_x, grid_y = [], []

if enable_hough:
    lines = cv2.HoughLinesP(
        edges,
        rho=float(rho),
        theta=np.deg2rad(float(theta_deg)),
        threshold=int(hough_thresh),
        minLineLength=int(min_len),
        maxLineGap=int(max_gap),
    )
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            ang = segment_angle_degrees(x1, y1, x2, y2)
            is_horizontal = ang <= float(angle_tol)
            is_vertical = abs(90.0 - ang) <= float(angle_tol)
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

clustered_x = cluster_1d(grid_x, eps=float(cluster_eps))
clustered_y = cluster_1d(grid_y, eps=float(cluster_eps))

# Grid overlay
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

# Layout: previews
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Original (scaled)")
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader("Edges (binary)")
    st.image(cv2.cvtColor(edges_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
with col2:
    st.subheader("Edge Overlay")
    st.image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader("Clustered Grid Overlay")
    st.image(cv2.cvtColor(grid_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

# Downloads
st.markdown("---")
st.subheader("Download results")

edges_bytes = to_png_bytes(edges_vis)
overlay_bytes = to_png_bytes(overlay_bgr)
grid_bytes = to_png_bytes(grid_overlay)
raw_json = {
    "kept_lines": kept_lines,
    "notes": "Segments filtered to near-horizontal/vertical by angle tolerance.",
}
clustered_json = {
    "grid_x": clustered_x,
    "grid_y": clustered_y,
    "params": {
        "scale": scale,
        "use_sigma": use_sigma,
        "sigma": sigma if use_sigma else None,
        "low": low_t,
        "high": high_t,
        "blur": blur,
        "close_k": close_k,
        "close_iter": close_iter,
        "kernel": kernel_shape,
        "dilate_iter": dilate_iter,
        "hough": enable_hough,
        "hough_params": {
            "rho": rho,
            "theta_deg": theta_deg,
            "threshold": hough_thresh,
            "min_line_length": min_len,
            "max_line_gap": max_gap,
        },
        "angle_tol_deg": angle_tol,
        "cluster_eps": cluster_eps,
    },
}

st.download_button(
    "Download edges.png", data=edges_bytes, file_name="edges.png", mime="image/png"
)
st.download_button(
    "Download edge_overlay.png",
    data=overlay_bytes,
    file_name="edge_overlay.png",
    mime="image/png",
)
st.download_button(
    "Download grid_overlay.png",
    data=grid_bytes,
    file_name="grid_overlay.png",
    mime="image/png",
)
st.download_button(
    "Download lines_raw.json",
    data=json.dumps(raw_json, indent=2),
    file_name="lines_raw.json",
    mime="application/json",
)
st.download_button(
    "Download lines_clustered.json",
    data=json.dumps(clustered_json, indent=2),
    file_name="lines_clustered.json",
    mime="application/json",
)
