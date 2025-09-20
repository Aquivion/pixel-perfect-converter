import json
import numpy as np
import cv2
import streamlit as st
from dataclasses import asdict

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
from converter.preset_management import (
    get_available_presets,
    save_preset,
    load_preset,
)

st.set_page_config(page_title="Canny + Hough Grid Tuner", layout="wide")

# --- State Management and Helper ---
if "params" not in st.session_state:
    st.session_state.params = asdict(PipelineParams())


# Helper to bridge widget keys with the nested session state dictionary
def update_param(category: str, param_name: str):
    key = f"{category}_{param_name}"
    if key in st.session_state:
        st.session_state.params[category][param_name] = st.session_state[key]


# --- UI ---
st.title("Canny + Hough Grid Tuner")

uploaded = st.file_uploader(
    "Upload an image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"]
)

with st.sidebar:
    st.header("Presets")
    available_presets = ["None"] + get_available_presets()

    # Use a standard key for the selectbox
    preset_to_load = st.selectbox(
        "Select Preset",
        options=available_presets,
        index=0,  # Default to "None"
    )

    # Add a dedicated button to trigger the load
    if st.button("Load Preset"):
        if preset_to_load and preset_to_load != "None":
            loaded_params = load_preset(preset_to_load)
            if loaded_params:
                st.session_state.params = loaded_params
                st.success(f"Loaded preset '{preset_to_load}'")

    preset_name_to_save = st.text_input("Save as preset name")
    if st.button("Save Preset"):
        if preset_name_to_save:
            save_preset(preset_name_to_save, st.session_state.params)
            st.success(f"Saved preset '{preset_name_to_save}'")
            # Refresh available presets list
            st.rerun()
        else:
            st.warning("Please enter a name for the preset.")

    st.header("Preprocess")
    st.number_input(
        "Scale (nearest-neighbor integer)",
        min_value=1,
        max_value=8,
        value=st.session_state.params["preprocess"]["scale"],
        key="preprocess_scale",
        on_change=update_param,
        args=("preprocess", "scale"),
    )
    st.number_input(
        "Gaussian blur kernel (odd, 0=off)",
        min_value=0,
        max_value=31,
        value=st.session_state.params["preprocess"]["blur"],
        key="preprocess_blur",
        on_change=update_param,
        args=("preprocess", "blur"),
    )

    st.header("Canny")
    st.checkbox(
        "Use auto thresholds (sigma)",
        value=st.session_state.params["canny"]["use_sigma"],
        key="canny_use_sigma",
        on_change=update_param,
        args=("canny", "use_sigma"),
    )
    st.slider(
        "sigma",
        0.00,
        1.00,
        value=st.session_state.params["canny"]["sigma"],
        key="canny_sigma",
        on_change=update_param,
        args=("canny", "sigma"),
    )
    st.slider(
        "low",
        0,
        255,
        value=st.session_state.params["canny"]["low"],
        key="canny_low",
        on_change=update_param,
        args=("canny", "low"),
    )
    st.slider(
        "high",
        0,
        255,
        value=st.session_state.params["canny"]["high"],
        key="canny_high",
        on_change=update_param,
        args=("canny", "high"),
    )

    st.header("Morphology")
    st.number_input(
        "Closing kernel size (0=off)",
        min_value=0,
        max_value=51,
        value=st.session_state.params["morph"]["close_k"],
        key="morph_close_k",
        on_change=update_param,
        args=("morph", "close_k"),
    )
    st.number_input(
        "Closing iterations",
        min_value=1,
        max_value=10,
        value=st.session_state.params["morph"]["close_iter"],
        key="morph_close_iter",
        on_change=update_param,
        args=("morph", "close_iter"),
    )
    st.number_input(
        "Dilate iterations",
        min_value=0,
        max_value=10,
        value=st.session_state.params["morph"]["dilate_iter"],
        key="morph_dilate_iter",
        on_change=update_param,
        args=("morph", "dilate_iter"),
    )
    options = ["rect", "ellipse", "cross"]
    st.selectbox(
        "Kernel shape",
        options=options,
        index=options.index(st.session_state.params["morph"]["kernel_shape"]),
        key="morph_kernel_shape",
        on_change=update_param,
        args=("morph", "kernel_shape"),
    )

    st.header("Hough LinesP")
    st.checkbox(
        "Enable Hough",
        value=st.session_state.params["hough"]["enable"],
        key="hough_enable",
        on_change=update_param,
        args=("hough", "enable"),
    )
    st.number_input(
        "rho (px)",
        min_value=0.5,
        max_value=5.0,
        step=0.5,
        value=st.session_state.params["hough"]["rho"],
        key="hough_rho",
        on_change=update_param,
        args=("hough", "rho"),
    )
    st.number_input(
        "theta (deg)",
        min_value=0.5,
        max_value=5.0,
        step=0.5,
        value=st.session_state.params["hough"]["theta_deg"],
        key="hough_theta_deg",
        on_change=update_param,
        args=("hough", "theta_deg"),
    )
    st.number_input(
        "accumulator threshold",
        min_value=1,
        max_value=500,
        value=st.session_state.params["hough"]["threshold"],
        key="hough_threshold",
        on_change=update_param,
        args=("hough", "threshold"),
    )
    st.number_input(
        "min line length",
        min_value=1,
        max_value=4000,
        step=10,
        value=st.session_state.params["hough"]["min_line_length"],
        key="hough_min_line_length",
        on_change=update_param,
        args=("hough", "min_line_length"),
    )
    st.number_input(
        "max line gap",
        min_value=0,
        max_value=200,
        value=st.session_state.params["hough"]["max_line_gap"],
        key="hough_max_line_gap",
        on_change=update_param,
        args=("hough", "max_line_gap"),
    )
    st.number_input(
        "angle tolerance (deg)",
        min_value=0.0,
        max_value=15.0,
        step=0.5,
        value=st.session_state.params["hough"]["angle_tol_deg"],
        key="hough_angle_tol_deg",
        on_change=update_param,
        args=("hough", "angle_tol_deg"),
    )

    st.header("Clustering")
    st.number_input(
        "cluster eps (px)",
        min_value=1.0,
        max_value=50.0,
        step=1.0,
        value=st.session_state.params["cluster"]["eps"],
        key="cluster_eps",
        on_change=update_param,
        args=("cluster", "eps"),
    )

    st.header("Overlay")
    st.number_input(
        "overlay edge thickness",
        min_value=1,
        max_value=7,
        value=st.session_state.params["overlay"]["edge_thickness"],
        key="overlay_edge_thickness",
        on_change=update_param,
        args=("overlay", "edge_thickness"),
    )
    st.slider(
        "overlay alpha",
        0.0,
        1.0,
        step=0.05,
        value=st.session_state.params["overlay"]["alpha"],
        key="overlay_alpha",
        on_change=update_param,
        args=("overlay", "alpha"),
    )

    # For color picker, we handle it slightly differently as it has no on_change
    color_hex = st.color_picker(
        "overlay color",
        f"#{st.session_state.params['overlay']['color_rgb'][0]:02X}{st.session_state.params['overlay']['color_rgb'][1]:02X}{st.session_state.params['overlay']['color_rgb'][2]:02X}",
    )
    st.session_state.params["overlay"]["color_rgb"] = tuple(
        int(color_hex[i : i + 2], 16) for i in (1, 3, 5)
    )


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


res = _run(bgr_in, st.session_state.params)

# Layout: previews
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Original (scaled)")
    st.image(cv2.cvtColor(res.bgr, cv2.COLOR_BGR2RGB), width="stretch")
    st.subheader("Edges (binary)")
    st.image(cv2.cvtColor(res.edges_vis, cv2.COLOR_BGR2RGB), width="stretch")
with col2:
    st.subheader("Edge Overlay")
    st.image(cv2.cvtColor(res.overlay_bgr, cv2.COLOR_BGR2RGB), width="stretch")
    st.subheader("Clustered Grid Overlay")
    st.image(cv2.cvtColor(res.grid_overlay, cv2.COLOR_BGR2RGB), width="stretch")

# --- New Layout for Final Results ---
st.markdown("---")

col3, col4 = st.columns(2, gap="large")
with col3:
    st.markdown("### Completed Grid Overlay")
    st.image(
        cv2.cvtColor(res.completed_grid_overlay, cv2.COLOR_BGR2RGB),
        width="stretch",
    )
    st.markdown(
        f"Estimated spacing X: {res.spacing_x:.2f} px | Y: {res.spacing_y:.2f} px "
        "(0 means insufficient data)"
    )

with col4:
    st.markdown("### Upscaled Pixel Art")
    st.image(
        cv2.cvtColor(res.upscaled_pixel_art, cv2.COLOR_BGR2RGB),
        width="stretch",
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
        "scale": st.session_state.params["preprocess"]["scale"],
        "use_sigma": st.session_state.params["canny"]["use_sigma"],
        "sigma": (
            st.session_state.params["canny"]["sigma"]
            if st.session_state.params["canny"]["use_sigma"]
            else None
        ),
        "low": res.low_t,
        "high": res.high_t,
        "blur": st.session_state.params["preprocess"]["blur"],
        "close_k": st.session_state.params["morph"]["close_k"],
        "close_iter": st.session_state.params["morph"]["close_iter"],
        "kernel": st.session_state.params["morph"]["kernel_shape"],
        "dilate_iter": st.session_state.params["morph"]["dilate_iter"],
        "hough": st.session_state.params["hough"]["enable"],
        "hough_params": {
            "rho": st.session_state.params["hough"]["rho"],
            "theta_deg": st.session_state.params["hough"]["theta_deg"],
            "threshold": st.session_state.params["hough"]["threshold"],
            "min_line_length": st.session_state.params["hough"]["min_line_length"],
            "max_line_gap": st.session_state.params["hough"]["max_line_gap"],
        },
        "angle_tol_deg": st.session_state.params["hough"]["angle_tol_deg"],
        "cluster_eps": st.session_state.params["cluster"]["eps"],
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
