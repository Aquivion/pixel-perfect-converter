import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import io
import json
import sys

# Ensure 'src/v2' is on sys.path so 'grid_snap' can be imported when run via Streamlit
ROOT = Path(__file__).resolve().parents[1]  # points to src/v2
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_snap.params import Params  # noqa: E402
from grid_snap.pipeline import run  # noqa: E402
from grid_snap.io_utils import imread_rgba, save_png_rgb  # noqa: E402, F401
from grid_snap.visualize import draw_grid  # noqa: E402

st.set_page_config(page_title="Pixel Grid Snap", layout="wide")

st.title("🧰 Pixel Grid Snap (rectilinear grid from pixel art)")
st.caption(
    "Detect axis-aligned edges, cluster them, and snap a grid that matches your sprite."
)

# Sidebar controls
with st.sidebar:
    st.header("Parameters")
    upscale = st.slider("Upscale (NN)", 1, 5, 2)
    canny_low = st.slider("Canny low", 0, 200, 40)
    canny_high = st.slider("Canny high", 1, 400, 120)
    angle_tol = st.slider("Angle tolerance (°)", 0.0, 10.0, 4.0, 0.5)
    cluster_eps = st.slider("Cluster ε (px after upscale)", 1, 10, 4)
    min_support = st.slider("Minimum line support", 1, 50, 6)
    force_square = st.checkbox("Force square cells", value=False)

    st.markdown("---")
    st.caption("Hough")
    h_rho = st.slider("ρ", 1, 3, 1)
    h_theta = st.slider("θ (deg)", 1, 3, 1)
    h_thr = st.slider("Threshold", 1, 200, 60)
    min_len = st.slider("Min line length", 1, 200, 30)
    max_gap = st.slider("Max line gap", 0, 20, 3)

# File upload or demo image
uploaded = st.file_uploader("Upload a PNG/JPG", type=["png", "jpg", "jpeg"])
if uploaded:
    rgb = np.array(Image.open(uploaded).convert("RGB"))
else:
    demo_path = Path(__file__).resolve().parents[1] / "images" / "medieval_tavern.png"
    rgb = imread_rgba(demo_path)

P = Params(
    upscale=upscale,
    canny_low=canny_low,
    canny_high=canny_high,
    hough_rho=h_rho,
    hough_theta_deg=h_theta,
    hough_thresh=h_thr,
    min_line_len=min_len,
    max_line_gap=max_gap,
    angle_tol_deg=angle_tol,
    cluster_eps=cluster_eps,
    min_support=min_support,
    force_square_cells=force_square,
)

if st.button("Run"):
    res = run(rgb, P)
    grid_img = draw_grid(res.rgb_up, res.grid)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Input (upscaled)")
        st.image(res.rgb_up, use_column_width=True)
        st.subheader("Edges")
        st.image(res.edges, use_column_width=True, clamp=True)

    with col2:
        st.subheader("Snapped grid overlay")
        st.image(grid_img, use_column_width=True)
        st.markdown(
            f"""
            **Spacing:** {res.spacing_x:.2f} × {res.spacing_y:.2f} px  
            **Origin:** ({res.origin_x:.2f}, {res.origin_y:.2f})  
            **Lines:** {len(res.grid)}
            """
        )

    # Downloads
    out_png = io.BytesIO()
    Image.fromarray(grid_img).save(out_png, format="PNG")
    st.download_button(
        "⬇️ Download overlay PNG",
        out_png.getvalue(),
        file_name="grid_overlay.png",
        mime="image/png",
    )

    grid_json = {
        "origin_x": res.origin_x,
        "origin_y": res.origin_y,
        "spacing_x": res.spacing_x,
        "spacing_y": res.spacing_y,
        "verticals": [v for k, v in res.grid if k == "v"],
        "horizontals": [v for k, v in res.grid if k == "h"],
    }
    st.download_button(
        "⬇️ Download grid JSON",
        json.dumps(grid_json, indent=2),
        file_name="grid.json",
        mime="application/json",
    )
