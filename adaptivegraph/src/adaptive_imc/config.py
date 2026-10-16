from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class RetrievalConfig:
    model_name: str = "vit_base_patch14_dinov2.lvd142m"
    batch_size: int = 8
    min_neighbors: int = 20
    max_neighbors: int = 40
    similarity_threshold: float = 0.20


@dataclass
class MatchingConfig:
    base_resize: int = 1280
    max_keypoints: int = 4096
    detection_threshold: float = 0.01
    lightglue_filter_threshold: float = 0.15
    width_confidence: float = 0.99
    depth_confidence: float = -1.0
    ransac_reproj_threshold: float = 4.0
    ransac_confidence: float = 0.9999
    ransac_max_iters: int = 20000
    coverage_grid: int = 4
    strong_min_inliers: int = 55
    strong_min_ratio: float = 0.22
    strong_min_coverage: float = 0.08
    easy_inliers: int = 100
    easy_ratio: float = 0.50
    easy_coverage: float = 0.25
    triangle_borderline_inliers: int = 35


@dataclass
class RescueConfig:
    enable_tiles: bool = True
    tile_resize: int = 1024
    full_tile_for_all_valid: bool = False
    tile_trigger_min_inliers: int = 18
    tile_trigger_max_inliers: int = 105
    tile_trigger_coverage: float = 0.30
    high_similarity_threshold: float = 0.48
    max_rescues_per_image: int = 8
    enable_rotation: bool = True
    rotation_similarity_threshold: float = 0.55
    rotation_max_base_inliers: int = 35
    rotation_stop_inliers: int = 80
    low_degree_rescue: bool = True
    low_degree_target: int = 2
    low_degree_extra_pairs: int = 2
    # Scene-preservation constraint (A3). When True, a RESCUED edge may densify
    # within a trusted component or attach a singleton, but may NOT merge two
    # already-established components (size >= 2) on local-match evidence alone.
    # Legitimate cross-scene links from the trusted base pass remain unaffected.
    scene_preserving: bool = False
    merge_min_cross_edges: int = 3


@dataclass
class GraphConfig:
    min_component_size: int = 2
    triangle_support_required_for_borderline: int = 1
    singleton_as_outlier: bool = True
    # Ablation switches (default True = full proposed method; set False for baselines).
    enable_triangle_support: bool = True
    enable_label_propagation: bool = True


@dataclass
class ColmapConfig:
    camera_model: str = "SIMPLE_RADIAL"
    camera_mode: str = "PER_IMAGE"
    verify_max_error: float = 4.0
    verify_min_num_inliers: int = 15
    min_num_matches: int = 15
    min_model_size: int = 3
    max_num_models: int = 25
    max_model_overlap: int = 20
    init_min_num_inliers: int = 70
    abs_pose_min_num_inliers: int = 20
    abs_pose_min_inlier_ratio: float = 0.20
    num_threads: int = -1


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 2026
    use_amp: bool = True
    keep_intermediate: bool = True


@dataclass
class PipelineConfig:
    name: str = "balanced"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    rescue: RescueConfig = field(default_factory=RescueConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = PipelineConfig()
    return _merge_dataclass(cfg, raw)


def dump_config(cfg: PipelineConfig) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(cfg)
