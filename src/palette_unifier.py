# -------------------------------- palette_unifier.py ----------------------------
"""
Merge visually-similar colours in an already-rectified / scaled-down pixel-art.

Usage:
    python palette_unifier.py scaled_output.png final_output.png [threshold]

`threshold` (optional, default = 20) is the maximum Euclidean distance in RGB
for two colours to be considered “the same”.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def _cluster_palette(unique_cols: np.ndarray, thr: int) -> tuple[dict, np.ndarray]:
    """
    Greedy single-pass clustering: for every unique colour, either join an
    existing cluster (if any representative is within `thr`) or start a new one.

    Returns
    -------
    mapping : dict[(int,int,int) → int]    # maps a colour → cluster index
    reps    : np.ndarray[K,3]              # representative colour for each cluster
    """
    representatives: list[np.ndarray] = []
    clusters: list[list[np.ndarray]] = []
    mapping: dict[tuple[int, int, int], int] = {}

    for col in unique_cols:
        # Try to place `col` into the first cluster whose rep is close enough
        placed = False
        for idx, rep in enumerate(representatives):
            if np.linalg.norm(col - rep) <= thr:
                clusters[idx].append(col)
                mapping[tuple(col)] = idx
                placed = True
                break
        if not placed:
            representatives.append(col)
            clusters.append([col])
            mapping[tuple(col)] = len(representatives) - 1

    # Use the **median** of each cluster as final representative
    reps = np.array(
        [np.median(np.vstack(c), axis=0).astype(np.uint8) for c in clusters],
        dtype=np.uint8,
    )
    return mapping, reps


def merge_similar_colours(img_path: Path, out_path: Path, threshold: int = 20) -> None:
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img)
    h, w, _ = arr.shape

    # 1. Cluster unique colours
    unique_cols = np.unique(arr.reshape(-1, 3), axis=0)
    mapping, reps = _cluster_palette(unique_cols, threshold)

    # 2. Re-map every pixel through the palette mapping
    flat = arr.reshape(-1, 3)
    out_flat = np.empty_like(flat)
    for i, col in enumerate(flat):
        rep_idx = mapping[tuple(col)]
        out_flat[i] = reps[rep_idx]

    out_arr = out_flat.reshape(h, w, 3)

    # 3. Save
    Image.fromarray(out_arr, mode="RGB").save(out_path)
    print(
        f"Merged palette → {len(reps)} colours (from {len(unique_cols)}) "
        f"and wrote {out_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(
            "Usage: python palette_unifier.py <threshold>"
        )
    inp, outp = "scaled_output.png", "final_output.png"
    thr = int(sys.argv[1]) 
    merge_similar_colours(inp, outp, thr)
