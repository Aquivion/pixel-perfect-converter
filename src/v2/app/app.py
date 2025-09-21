import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import io
import json
import sys
import hashlib

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


# --- helpers -------------------------------------------------
def bytes_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@st.cache_data(show_spinner=False)
def compute_result(image_bytes: bytes, P: Params):
    import io
    from PIL import Image

    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return run(rgb, P)  # your pipeline returns Result dataclass


# --- inputs --------------------------------------------------
uploaded = st.file_uploader(
    "Upload PNG/JPG", type=["png", "jpg", "jpeg"], key="uploader"
)
if uploaded is None:
    # fall back to demo asset
    with open("images/medieval_tavern.png", "rb") as f:
        image_bytes = f.read()
else:
    image_bytes = uploaded.read()

# Sidebar params (any change causes a rerun)
with st.sidebar:
    st.header("Parameters")
    upscale = int(st.slider("Upscale", 1, 5, 2))
    canny_low = int(st.slider("Canny low", 0, 200, 40))
    canny_high = int(st.slider("Canny high", 1, 400, 120))
    angle_tol_deg = float(st.slider("Angle tol (°)", 0.0, 10.0, 4.0, 0.5))
    cluster_eps = int(st.slider("Cluster ε", 1, 10, 4))
    min_support = int(st.slider("Min support", 1, 50, 6))
    force_square_cells = st.checkbox("Force square cells", False)
    hough_rho = int(st.slider("ρ", 1, 3, 1))
    hough_theta_deg = int(st.slider("θ (deg)", 1, 3, 1))
    hough_thresh = int(st.slider("Hough threshold", 1, 200, 60))
    min_line_len = int(st.slider("Min line length", 1, 200, 30))
    max_line_gap = int(st.slider("Max line gap", 0, 20, 3))

P = Params(
    upscale=upscale,
    canny_low=canny_low,
    canny_high=canny_high,
    hough_rho=hough_rho,
    hough_theta_deg=hough_theta_deg,
    hough_thresh=hough_thresh,
    min_line_len=min_line_len,
    max_line_gap=max_line_gap,
    angle_tol_deg=angle_tol_deg,
    cluster_eps=cluster_eps,
    min_support=min_support,
    force_square_cells=force_square_cells,
)

# --- compute (cached by image+params) ------------------------
res = compute_result(image_bytes, P)

# --- render stays visible across widget changes --------------
grid_img = draw_grid(res.rgb_up, res.grid)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input (upscaled)")
    st.image(res.rgb_up, use_column_width=True)
    st.subheader("Edges")
    st.image(res.edges, use_column_width=True, clamp=True)
with col2:
    st.subheader("Snapped grid overlay")
    st.image(grid_img, use_column_width=True)
    st.markdown(
        f"**Spacing:** {res.spacing_x:.2f} × {res.spacing_y:.2f} · **Origin:** ({res.origin_x:.2f}, {res.origin_y:.2f}) · **Lines:** {len(res.grid)}"
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
