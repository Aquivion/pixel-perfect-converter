import json
from typing import Tuple
import numpy as np
import cv2
import streamlit as st

from converter.params import (
    PipelineParams,
    PreprocessParams,
    CannyParams,
    MorphParams,
    HoughParams,
    ClusterParams,
    OverlayParams,
)
from converter.pipeline import run_pipeline
from converter.processing.edges import ensure_3channel
from converter.processing.overlay import to_png_bytes

st.set_page_config(page_title="Canny + Hough Grid Tuner", layout="wide")
st.title("Canny + Hough Grid Tuner")

uploaded = st.file_uploader(
    "Upload an image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"]
)

with st.sidebar:
    st.header("Preprocess")
    scale = st.number_input(
        "Scale (nearest-neighbor integer)",
        min_value=1,
        max_value=8,
        value=PreprocessParams.scale,
        step=1,
    )
    blur = st.number_input(
        "Gaussian blur kernel (odd, 0=off)",
        min_value=0,
        max_value=31,
        value=PreprocessParams.blur,
        step=1,
    )

    st.header("Canny")
    use_sigma = st.checkbox("Use auto thresholds (sigma)", value=CannyParams.use_sigma)
    sigma = st.slider("sigma", 0.00, 1.00, CannyParams.sigma, 0.01)
    low = st.slider("low", 0, 255, CannyParams.low, 1)
    high = st.slider("high", 0, 255, CannyParams.high, 1)

    st.header("Morphology")
    close_k = st.number_input(
        "Closing kernel size (0=off)",
        min_value=0,
        max_value=51,
        value=MorphParams.close_k,
        step=1,
    )
    close_iter = st.number_input(
        "Closing iterations",
        min_value=1,
        max_value=10,
        value=MorphParams.close_iter,
        step=1,
    )
    dilate_iter = st.number_input(
        "Dilate iterations",
        min_value=0,
        max_value=10,
        value=MorphParams.dilate_iter,
        step=1,
    )
    options = ["rect", "ellipse", "cross"]
    kernel_shape = st.selectbox(
        "Kernel shape", options=options, index=options.index(MorphParams.kernel_shape)
    )

    st.header("Hough LinesP")
    enable_hough = st.checkbox("Enable Hough", value=HoughParams.enable)
    rho = st.number_input(
        "rho (px)", min_value=0.5, max_value=5.0, value=HoughParams.rho, step=0.5
    )
    theta_deg = st.number_input(
        "theta (deg)",
        min_value=0.5,
        max_value=5.0,
        value=HoughParams.theta_deg,
        step=0.5,
    )
    hough_thresh = st.number_input(
        "accumulator threshold",
        min_value=1,
        max_value=500,
        value=HoughParams.threshold,
        step=1,
    )
    min_len = st.number_input(
        "min line length",
        min_value=1,
        max_value=4000,
        value=HoughParams.min_line_length,
        step=10,
    )
    max_gap = st.number_input(
        "max line gap",
        min_value=0,
        max_value=200,
        value=HoughParams.max_line_gap,
        step=1,
    )
    angle_tol = st.number_input(
        "angle tolerance (deg)",
        min_value=0.0,
        max_value=15.0,
        value=HoughParams.angle_tol_deg,
        step=0.5,
    )

    st.header("Clustering")
    cluster_eps = st.number_input(
        "cluster eps (px)",
        min_value=1.0,
        max_value=50.0,
        value=ClusterParams.eps,
        step=1.0,
    )

    st.header("Overlay")
    edge_thickness = st.number_input(
        "overlay edge thickness",
        min_value=1,
        max_value=7,
        value=OverlayParams.edge_thickness,
        step=1,
    )
    alpha = st.slider("overlay alpha", 0.0, 1.0, OverlayParams.alpha, 0.05)
    default_color = f"#{OverlayParams.color_rgb[0]:02X}{OverlayParams.color_rgb[1]:02X}{OverlayParams.color_rgb[2]:02X}"
    color = st.color_picker("overlay color", default_color)
    color_rgb: Tuple[int, int, int] = tuple(
        int(color[i : i + 2], 16) for i in (1, 3, 5)
    )  # hex→RGB

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# Decode uploaded image (keep bytes for caching)
file_bytes = uploaded.read()
file_np = np.frombuffer(file_bytes, np.uint8)
img = cv2.imdecode(file_np, cv2.IMREAD_UNCHANGED)
if img is None:
    st.error("Could not read the image.")
    st.stop()
bgr_in = ensure_3channel(img)

params = PipelineParams(
    preprocess=PreprocessParams(scale=int(scale), blur=int(blur)),
    canny=CannyParams(
        use_sigma=bool(use_sigma), sigma=float(sigma), low=int(low), high=int(high)
    ),
    morph=MorphParams(
        close_k=int(close_k),
        close_iter=int(close_iter),
        dilate_iter=int(dilate_iter),
        kernel_shape=str(kernel_shape),
    ),
    hough=HoughParams(
        enable=bool(enable_hough),
        rho=float(rho),
        theta_deg=float(theta_deg),
        threshold=int(hough_thresh),
        min_line_length=int(min_len),
        max_line_gap=int(max_gap),
        angle_tol_deg=float(angle_tol),
    ),
    cluster=ClusterParams(eps=float(cluster_eps)),
    overlay=OverlayParams(
        edge_thickness=int(edge_thickness), alpha=float(alpha), color_rgb=color_rgb
    ),
)


@st.cache_data(show_spinner=False)
def _run(bgr_array: np.ndarray, params_dict: dict):
    p = PipelineParams(
        preprocess=PreprocessParams(**params_dict["preprocess"]),
        canny=CannyParams(**params_dict["canny"]),
        morph=MorphParams(**params_dict["morph"]),
        hough=HoughParams(**params_dict["hough"]),
        cluster=ClusterParams(**params_dict["cluster"]),
        overlay=OverlayParams(**params_dict["overlay"]),
    )

    return run_pipeline(bgr_array, p)


res = _run(
    bgr_in,
    {
        "preprocess": vars(params.preprocess),
        "canny": vars(params.canny),
        "morph": vars(params.morph),
        "hough": vars(params.hough),
        "cluster": vars(params.cluster),
        "overlay": vars(params.overlay),
    },
)

# Layout: previews
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Original (scaled)")
    st.image(cv2.cvtColor(res.bgr, cv2.COLOR_BGR2RGB), width="content")
    st.subheader("Edges (binary)")
    st.image(cv2.cvtColor(res.edges_vis, cv2.COLOR_BGR2RGB), width="content")
with col2:
    st.subheader("Edge Overlay")
    st.image(cv2.cvtColor(res.overlay_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader("Clustered Grid Overlay")
    st.image(
        cv2.cvtColor(res.grid_overlay, cv2.COLOR_BGR2RGB), use_container_width=True
    )

# --- New Layout for Final Results ---
st.markdown("---")

col3, col4 = st.columns(2, gap="large")
with col3:
    st.markdown("### Completed Grid Overlay")
    st.image(
        cv2.cvtColor(res.completed_grid_overlay, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )
    st.markdown(
        f"Estimated spacing X: {res.spacing_x:.2f} px | Y: {res.spacing_y:.2f} px "
        "(0 means insufficient data)"
    )

with col4:
    st.markdown("### Upscaled Pixel Art")
    st.image(
        cv2.cvtColor(res.upscaled_pixel_art, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )

st.markdown("### Final Pixel Art (Logical)")
st.image(cv2.cvtColor(res.final_pixel_art, cv2.COLOR_BGR2RGB), width="content")


# Downloads
st.markdown("---")
st.subheader("Download results")

edges_bytes = to_png_bytes(res.edges_vis)
overlay_bytes = to_png_bytes(res.overlay_bgr)
grid_bytes = to_png_bytes(res.grid_overlay)
completed_grid_bytes = to_png_bytes(res.completed_grid_overlay)
final_pixel_art_bytes = to_png_bytes(res.final_pixel_art)
upscaled_pixel_art_bytes = to_png_bytes(res.upscaled_pixel_art)
raw_json = {
    "kept_lines": res.kept_lines,
    "notes": "Segments filtered to near-horizontal/vertical by angle tolerance.",
}
clustered_json = {
    "grid_x": res.clustered_x,
    "grid_y": res.clustered_y,
    "spacing": {"x": res.spacing_x, "y": res.spacing_y},
    "completed_x": res.completed_x,
    "completed_y": res.completed_y,
    "params": {
        "scale": params.preprocess.scale,
        "use_sigma": params.canny.use_sigma,
        "sigma": params.canny.sigma if params.canny.use_sigma else None,
        "low": res.low_t,
        "high": res.high_t,
        "blur": params.preprocess.blur,
        "close_k": params.morph.close_k,
        "close_iter": params.morph.close_iter,
        "kernel": params.morph.kernel_shape,
        "dilate_iter": params.morph.dilate_iter,
        "hough": params.hough.enable,
        "hough_params": {
            "rho": params.hough.rho,
            "theta_deg": params.hough.theta_deg,
            "threshold": params.hough.threshold,
            "min_line_length": params.hough.min_line_length,
            "max_line_gap": params.hough.max_line_gap,
        },
        "angle_tol_deg": params.hough.angle_tol_deg,
        "cluster_eps": params.cluster.eps,
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
    "Download completed_grid_overlay.png",
    data=completed_grid_bytes,
    file_name="completed_grid_overlay.png",
    mime="image/png",
)
st.download_button(
    "Download final_pixel_art.png",
    data=final_pixel_art_bytes,
    file_name="final_pixel_art.png",
    mime="image/png",
)
st.download_button(
    "Download upscaled_pixel_art.png",
    data=upscaled_pixel_art_bytes,
    file_name="upscaled_pixel_art.png",
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
