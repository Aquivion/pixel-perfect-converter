from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PreprocessParams:
    scale: int = 1
    blur: int = 0  # odd; 0 = off


@dataclass(frozen=True)
class CannyParams:
    use_sigma: bool = False
    sigma: float = 0.12
    low: int = 50
    high: int = 150


@dataclass(frozen=True)
class MorphParams:
    close_k: int = 0
    close_iter: int = 1
    dilate_iter: int = 0
    kernel_shape: str = "rect"  # rect|ellipse|cross


@dataclass(frozen=True)
class HoughParams:
    enable: bool = True
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 60
    min_line_length: int = 50
    max_line_gap: int = 18
    angle_tol_deg: float = 3.5


@dataclass(frozen=True)
class ClusterParams:
    eps: float = 8.0


@dataclass(frozen=True)
class OverlayParams:
    edge_thickness: int = 1
    alpha: float = 0.8
    color_rgb: Tuple[int, int, int] = (255, 0, 0)


@dataclass(frozen=True)
class PipelineParams:
    preprocess: PreprocessParams
    canny: CannyParams
    morph: MorphParams
    hough: HoughParams
    cluster: ClusterParams
    overlay: OverlayParams
