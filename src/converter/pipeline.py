from dataclasses import dataclass
from typing import Dict, List
import cv2
import numpy as np

from .params import PipelineParams
from .processing.edges import ensure_3channel, gaussian_blur, canny_edges, morphology
from .processing.lines import hough_lines_filtered, cluster_1d
from .processing.overlay import draw_edges_overlay, draw_grid_overlay


@dataclass
class PipelineResult:
    bgr: np.ndarray
    gray_blur: np.ndarray
    edges: np.ndarray
    edges_vis: np.ndarray
    overlay_bgr: np.ndarray
    grid_overlay: np.ndarray
    kept_lines: List[Dict]
    clustered_x: List[float]
    clustered_y: List[float]
    low_t: int
    high_t: int


def run_pipeline(bgr_in: np.ndarray, p: PipelineParams) -> PipelineResult:
    bgr = ensure_3channel(bgr_in)
    if p.preprocess.scale > 1:
        bgr = cv2.resize(
            bgr,
            None,
            fx=p.preprocess.scale,
            fy=p.preprocess.scale,
            interpolation=cv2.INTER_NEAREST,
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = gaussian_blur(gray, p.preprocess.blur)

    edges, low_t, high_t = canny_edges(
        gray_blur, p.canny.use_sigma, p.canny.sigma, p.canny.low, p.canny.high
    )
    edges = morphology(
        edges,
        p.morph.close_k,
        p.morph.close_iter,
        p.morph.dilate_iter,
        p.morph.kernel_shape,
    )

    edges_vis = np.zeros_like(bgr)
    edges_vis[edges != 0] = (255, 255, 255)

    overlay_bgr = draw_edges_overlay(
        bgr, edges, p.overlay.color_rgb, p.overlay.edge_thickness, p.overlay.alpha
    )

    kept_lines, grid_x, grid_y = [], [], []
    if p.hough.enable:
        kept_lines, grid_x, grid_y = hough_lines_filtered(
            edges,
            p.hough.rho,
            p.hough.theta_deg,
            p.hough.threshold,
            p.hough.min_line_length,
            p.hough.max_line_gap,
            p.hough.angle_tol_deg,
        )

    clustered_x = cluster_1d(grid_x, p.cluster.eps)
    clustered_y = cluster_1d(grid_y, p.cluster.eps)

    grid_overlay = draw_grid_overlay(bgr, clustered_x, clustered_y)

    return PipelineResult(
        bgr=bgr,
        gray_blur=gray_blur,
        edges=edges,
        edges_vis=edges_vis,
        overlay_bgr=overlay_bgr,
        grid_overlay=grid_overlay,
        kept_lines=kept_lines,
        clustered_x=clustered_x,
        clustered_y=clustered_y,
        low_t=low_t,
        high_t=high_t,
    )
