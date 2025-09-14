from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PreprocessParams:
    scale: int = 1
    blur: int = 0  # odd; 0 = off


@dataclass
class CannyParams:
    use_sigma: bool = False
    sigma: float = 0.12
    low: int = 50
    high: int = 150


@dataclass
class MorphParams:
    close_k: int = 0
    close_iter: int = 1
    dilate_iter: int = 0
    kernel_shape: str = "rect"  # rect|ellipse|cross


@dataclass
class HoughParams:
    enable: bool = True
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 60
    min_line_length: int = 50
    max_line_gap: int = 18
    angle_tol_deg: float = 3.5


@dataclass
class ClusterParams:
    eps: float = 8.0


@dataclass
class OverlayParams:
    edge_thickness: int = 1
    alpha: float = 0.8
    color_rgb: Tuple[int, int, int] = (255, 0, 255)


@dataclass
class PipelineParams:
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    canny: CannyParams = field(default_factory=CannyParams)
    morph: MorphParams = field(default_factory=MorphParams)
    hough: HoughParams = field(default_factory=HoughParams)
    cluster: ClusterParams = field(default_factory=ClusterParams)
    overlay: OverlayParams = field(default_factory=OverlayParams)
