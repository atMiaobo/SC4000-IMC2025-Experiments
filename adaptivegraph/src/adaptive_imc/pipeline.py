from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import random
import time

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import PipelineConfig, dump_config
from .geometry import PairGeometry, pair_quality_score, verify_correspondences
from .graph_ops import EdgeStats, build_graph, triangle_support, low_degree_nodes
from .matching import AdaptiveLightGlueMatcher
from .retrieval import CandidatePair, DINOv2Retriever, make_candidate_pairs
from .reconstruction import PosePrediction, reconstruct_with_colmap


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class PairResult:
    i: int
    j: int
    similarity: float
    geom: PairGeometry
    quality: float
    route: str
    rescued: bool = False
    accepted_reason: str = ""

    def as_edge(self) -> EdgeStats:
        return EdgeStats(
            i=self.i,
            j=self.j,
            inliers=self.geom.inlier_count,
            ratio=self.geom.inlier_ratio,
            coverage=self.geom.coverage,
            similarity=self.similarity,
            quality=self.quality,
            route=self.route,
        )


def find_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(directory)
    images = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=lambda p: str(p.relative_to(directory)))
    return images


def _shape(path: Path) -> tuple[int, int]:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Could not read image: {path}")
    return int(im.shape[0]), int(im.shape[1])


def _verify(
    pts0: np.ndarray,
    pts1: np.ndarray,
    shape0: tuple[int, int],
    shape1: tuple[int, int],
    cfg: PipelineConfig,
) -> PairGeometry:
    m = cfg.matching
    return verify_correspondences(
        pts0,
        pts1,
        shape0,
        shape1,
        reproj_threshold=m.ransac_reproj_threshold,
        confidence=m.ransac_confidence,
        max_iters=m.ransac_max_iters,
        coverage_grid=m.coverage_grid,
    )


def _is_strong(geom: PairGeometry, cfg: PipelineConfig) -> bool:
    m = cfg.matching
    return (
        geom.inlier_count >= m.strong_min_inliers
        and geom.inlier_ratio >= m.strong_min_ratio
        and geom.coverage >= m.strong_min_coverage
    )


def _is_easy(geom: PairGeometry, cfg: PipelineConfig) -> bool:
    m = cfg.matching
    return (
        geom.inlier_count >= m.easy_inliers
        and geom.inlier_ratio >= m.easy_ratio
        and geom.coverage >= m.easy_coverage
    )


def _potential_for_rescue(result: PairResult, cfg: PipelineConfig) -> float:
    """Higher means 'this pair is worth spending more compute on'."""
    g = result.geom
    # High retrieval confidence + some geometric signal + poor spatial coverage is
    # precisely the regime where crops/rotations can recover useful correspondences.
    coverage_deficit = max(0.0, cfg.rescue.tile_trigger_coverage - g.coverage)
    return (
        2.0 * result.similarity
        + np.log1p(g.inlier_count)
        + 1.5 * g.inlier_ratio
        + 1.2 * coverage_deficit
    )


def _wants_tile(result: PairResult, cfg: PipelineConfig) -> bool:
    r, g = cfg.rescue, result.geom
    if not r.enable_tiles:
        return False
    if r.full_tile_for_all_valid:
        return g.inlier_count >= r.tile_trigger_min_inliers or result.similarity >= r.high_similarity_threshold
    in_band = r.tile_trigger_min_inliers <= g.inlier_count <= r.tile_trigger_max_inliers
    coverage_problem = in_band and g.coverage < r.tile_trigger_coverage
    semantic_disagreement = result.similarity >= r.high_similarity_threshold and not _is_easy(g, cfg)
    return bool(coverage_problem or semantic_disagreement)


def _wants_rotation(result: PairResult, cfg: PipelineConfig) -> bool:
    r = cfg.rescue
    return bool(
        r.enable_rotation
        and result.similarity >= r.rotation_similarity_threshold
        and result.geom.inlier_count <= r.rotation_max_base_inliers
    )


def _combine_and_verify(
    base: PairResult,
    extras: list[tuple[np.ndarray, np.ndarray, str]],
    shape0: tuple[int, int],
    shape1: tuple[int, int],
    cfg: PipelineConfig,
) -> PairResult:
    pts0 = [base.geom.pts0]
    pts1 = [base.geom.pts1]
    tags = ["base"]
    for p0, p1, tag in extras:
        if len(p0):
            pts0.append(np.asarray(p0, dtype=np.float32))
            pts1.append(np.asarray(p1, dtype=np.float32))
            tags.append(tag)
    all0 = np.concatenate(pts0, axis=0) if pts0 else np.empty((0, 2), np.float32)
    all1 = np.concatenate(pts1, axis=0) if pts1 else np.empty((0, 2), np.float32)
    geom = _verify(all0, all1, shape0, shape1, cfg)
    route = "+".join(dict.fromkeys(tags))
    return PairResult(
        i=base.i,
        j=base.j,
        similarity=base.similarity,
        geom=geom,
        quality=pair_quality_score(geom, base.similarity),
        route=route,
        rescued=True,
    )


def _rescue_pair(
    base: PairResult,
    image_paths: list[Path],
    matcher: AdaptiveLightGlueMatcher,
    shapes: list[tuple[int, int]],
    cfg: PipelineConfig,
    *,
    force_tiles: bool = False,
    force_rotation: bool = False,
) -> PairResult:
    i, j = base.i, base.j
    extras: list[tuple[np.ndarray, np.ndarray, str]] = []

    do_rotation = force_rotation or _wants_rotation(base, cfg)
    if do_rotation:
        best_rotation = None
        best_rotation_geom = None
        for k, p0, p1 in matcher.match_rotations(image_paths[i], image_paths[j]):
            rg = _verify(p0, p1, shapes[i], shapes[j], cfg)
            if best_rotation_geom is None or rg.inlier_count > best_rotation_geom.inlier_count:
                best_rotation = (p0, p1, f"rot{k * 90}")
                best_rotation_geom = rg
            if rg.inlier_count >= cfg.rescue.rotation_stop_inliers:
                break
        if best_rotation is not None and best_rotation_geom is not None:
            # Only add a rotation if it genuinely contributes non-trivial support.
            if best_rotation_geom.inlier_count >= max(12, int(0.45 * max(base.geom.inlier_count, 1))):
                extras.append(best_rotation)

    do_tiles = force_tiles or _wants_tile(base, cfg)
    if do_tiles:
        p0, p1 = matcher.match_tiled(image_paths[i], image_paths[j], include_cross_base=True)
        if len(p0):
            extras.append((p0, p1, "tiles"))

    if not extras:
        return base
    return _combine_and_verify(base, extras, shapes[i], shapes[j], cfg)


def _diag_row(result: PairResult) -> dict[str, object]:
    g = result.geom
    return {
        "i": result.i,
        "j": result.j,
        "similarity": result.similarity,
        "raw_matches": g.raw_count,
        "inliers": g.inlier_count,
        "inlier_ratio": g.inlier_ratio,
        "coverage": g.coverage,
        "median_sampson": g.median_sampson,
        "quality": result.quality,
        "route": result.route,
        "rescued": result.rescued,
        "accepted_reason": result.accepted_reason,
    }


def process_dataset(
    *,
    dataset_name: str,
    image_paths: list[Path],
    cfg: PipelineConfig,
    work_dir: str | Path,
) -> tuple[dict[int, PosePrediction], pd.DataFrame, dict[str, object]]:
    """Run the adaptive scene-graph pipeline on one Kaggle dataset/collection."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.runtime.seed)
    np.random.seed(cfg.runtime.seed)
    torch.manual_seed(cfg.runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.runtime.seed)

    t0 = time.time()
    n = len(image_paths)
    print(f"\n=== {dataset_name}: {n} images ===")
    if n == 0:
        return {}, pd.DataFrame(), {"dataset": dataset_name, "num_images": 0}
    if n == 1:
        return {0: PosePrediction("outliers", None, None)}, pd.DataFrame(), {
            "dataset": dataset_name,
            "num_images": 1,
            "num_pairs": 0,
        }

    shapes = [_shape(p) for p in image_paths]

    retriever = DINOv2Retriever(
        cfg.retrieval.model_name,
        device=cfg.runtime.device,
        use_amp=cfg.runtime.use_amp,
    )
    embeddings = retriever.encode(image_paths, batch_size=cfg.retrieval.batch_size)
    pairs, _ = make_candidate_pairs(
        embeddings,
        min_neighbors=cfg.retrieval.min_neighbors,
        max_neighbors=cfg.retrieval.max_neighbors,
        similarity_threshold=cfg.retrieval.similarity_threshold,
    )
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Candidate edges: {len(pairs)}")

    matcher = AdaptiveLightGlueMatcher(
        device=cfg.runtime.device,
        max_keypoints=cfg.matching.max_keypoints,
        detection_threshold=cfg.matching.detection_threshold,
        filter_threshold=cfg.matching.lightglue_filter_threshold,
        width_confidence=cfg.matching.width_confidence,
        depth_confidence=cfg.matching.depth_confidence,
        base_resize=cfg.matching.base_resize,
        tile_resize=cfg.rescue.tile_resize,
        use_amp=cfg.runtime.use_amp,
    )

    results: dict[tuple[int, int], PairResult] = {}
    for p in tqdm(pairs, desc=f"{dataset_name}: base matching"):
        p0, p1 = matcher.match_base(image_paths[p.i], image_paths[p.j])
        geom = _verify(p0, p1, shapes[p.i], shapes[p.j], cfg)
        res = PairResult(
            i=p.i,
            j=p.j,
            similarity=p.similarity,
            geom=geom,
            quality=pair_quality_score(geom, p.similarity),
            route="base",
        )
        results[(p.i, p.j)] = res

    # ---------- Adaptive compute allocation ----------
    # Scene-preservation (A3): snapshot the "trusted" components formed by the base
    # (pre-rescue) strong edges. A rescued edge is later allowed to densify within one
    # of these components or attach a singleton, but NOT to merge two established
    # components (size >= 2) on local-match evidence alone. Base (non-rescued) edges
    # are never blocked, so legitimate cross-scene links are preserved.
    trusted_comp: dict[int, int] = {}
    trusted_size: dict[int, int] = {}
    if cfg.rescue.scene_preserving:
        base_strong_edges = [r.as_edge() for r in results.values() if _is_strong(r.geom, cfg)]
        g_trusted = build_graph(n, base_strong_edges)
        for ci, comp in enumerate(nx.connected_components(g_trusted)):
            for node in comp:
                trusted_comp[node] = ci
            trusted_size[ci] = len(comp)

    def _blocks_scene_merge(i: int, j: int) -> bool:
        """True if admitting a rescued i-j edge would merge two established scenes."""
        ci, cj = trusted_comp.get(i), trusted_comp.get(j)
        return (
            ci is not None and cj is not None and ci != cj
            and trusted_size.get(ci, 0) >= 2 and trusted_size.get(cj, 0) >= 2
        )

    rescue_candidates = [r for r in results.values() if _wants_tile(r, cfg) or _wants_rotation(r, cfg)]
    rescue_candidates.sort(key=lambda r: _potential_for_rescue(r, cfg), reverse=True)
    rescue_count = np.zeros(n, dtype=np.int32)
    max_per_image = int(cfg.rescue.max_rescues_per_image)
    for base in tqdm(rescue_candidates, desc=f"{dataset_name}: adaptive rescue"):
        if max_per_image >= 0 and (
            rescue_count[base.i] >= max_per_image or rescue_count[base.j] >= max_per_image
        ):
            continue
        upgraded = _rescue_pair(base, image_paths, matcher, shapes, cfg)
        if upgraded is not base:
            results[(base.i, base.j)] = upgraded
            rescue_count[base.i] += 1
            rescue_count[base.j] += 1

    # Strong geometrically verified graph first.
    strong: dict[tuple[int, int], PairResult] = {}
    for key, r in results.items():
        if not _is_strong(r.geom, cfg):
            continue
        if cfg.rescue.scene_preserving and r.rescued and _blocks_scene_merge(r.i, r.j):
            r.accepted_reason = "rejected_cross_component_rescue"
            continue
        r.accepted_reason = "strong_geometry"
        strong[key] = r
    g = build_graph(n, [r.as_edge() for r in strong.values()])

    # Borderline edge rehabilitation via triangle support. This is intentionally
    # conservative: a weak pair is kept only if two already-trusted image edges
    # independently place it inside a local scene triangle.
    borderline = sorted(results.values(), key=lambda r: r.quality, reverse=True)
    for r in borderline:
        if not cfg.graph.enable_triangle_support:
            break
        key = (r.i, r.j)
        if key in strong:
            continue
        if r.geom.inlier_count < cfg.matching.triangle_borderline_inliers:
            continue
        support = triangle_support(g, r.i, r.j)
        if support >= cfg.graph.triangle_support_required_for_borderline:
            r.accepted_reason = f"triangle_support_{support}"
            strong[key] = r
            e = r.as_edge()
            g.add_edge(
                e.i, e.j,
                inliers=e.inliers, ratio=e.ratio, coverage=e.coverage,
                similarity=e.similarity, quality=e.quality, route=e.route,
            )

    # ---------- Topology-aware low-degree/bridge rescue ----------
    # Rather than spend expensive tiling on every edge, spend it on isolated nodes
    # whose missing overlap may split a real scene component.
    if cfg.rescue.low_degree_rescue:
        pair_lookup = {(p.i, p.j): p for p in pairs}
        forced_done: set[tuple[int, int]] = set()
        candidates_by_node: dict[int, list[PairResult]] = {i: [] for i in range(n)}
        for r in results.values():
            if (r.i, r.j) in strong:
                continue
            candidates_by_node[r.i].append(r)
            candidates_by_node[r.j].append(r)
        for node in low_degree_nodes(g, cfg.rescue.low_degree_target):
            alts = sorted(candidates_by_node[node], key=lambda r: (r.similarity, r.quality), reverse=True)
            tried = 0
            for base in alts:
                key = (base.i, base.j)
                if key in forced_done or key in strong:
                    continue
                forced_done.add(key)
                upgraded = _rescue_pair(
                    results[key], image_paths, matcher, shapes, cfg,
                    force_tiles=cfg.rescue.enable_tiles,
                    force_rotation=cfg.rescue.enable_rotation and results[key].similarity >= cfg.rescue.rotation_similarity_threshold,
                )
                results[key] = upgraded
                tried += 1
                admit = _is_strong(upgraded.geom, cfg)
                if admit and cfg.rescue.scene_preserving and _blocks_scene_merge(upgraded.i, upgraded.j):
                    admit = False
                if admit:
                    upgraded.accepted_reason = "low_degree_topology_rescue"
                    strong[key] = upgraded
                    e = upgraded.as_edge()
                    g.add_edge(
                        e.i, e.j,
                        inliers=e.inliers, ratio=e.ratio, coverage=e.coverage,
                        similarity=e.similarity, quality=e.quality, route=e.route,
                    )
                if g.degree[node] >= cfg.rescue.low_degree_target or tried >= cfg.rescue.low_degree_extra_pairs:
                    break

    matcher.clear_dataset_cache()
    del matcher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accepted = strong
    print(f"Accepted graph edges: {len(accepted)} / {len(results)}")
    print(f"Graph components >=2: {sum(1 for c in nx.connected_components(g) if len(c) >= 2)}")

    predictions, colmap_diag = reconstruct_with_colmap(
        image_paths=image_paths,
        accepted_results=accepted,
        match_graph=g,
        work_dir=work_dir / "colmap",
        cfg=cfg.colmap,
        enable_label_propagation=cfg.graph.enable_label_propagation,
    )

    diag_df = pd.DataFrame([_diag_row(r) for r in results.values()])
    diag_df["accepted"] = [((int(i), int(j)) in accepted) for i, j in zip(diag_df.i, diag_df.j)] if len(diag_df) else []
    diag_df.to_csv(work_dir / "pair_diagnostics.csv", index=False)

    component_sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    summary = {
        "dataset": dataset_name,
        "num_images": n,
        "num_candidate_pairs": len(pairs),
        "num_accepted_pairs": len(accepted),
        "num_rescued_pairs": int(sum(r.rescued for r in results.values())),
        "component_sizes": component_sizes,
        "elapsed_seconds": time.time() - t0,
        "colmap": colmap_diag,
        "config": dump_config(cfg),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return predictions, diag_df, summary
