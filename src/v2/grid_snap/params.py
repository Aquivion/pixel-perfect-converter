from dataclasses import dataclass


@dataclass
class Params:
    upscale: int = 2
    canny_low: int = 40
    canny_high: int = 120
    hough_rho: int = 1
    hough_theta_deg: float = 1.0  # degrees
    hough_thresh: int = 60
    min_line_len: int = 30
    max_line_gap: int = 3
    angle_tol_deg: float = 4.0
    cluster_eps: int = 4
    min_support: int = 6
    force_square_cells: bool = False
