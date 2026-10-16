"""MASt3R reconstruction pipeline.

Reuses AdaptiveGraph infrastructure (DINOv2 retrieval, MAGSAC verification, COLMAP SfM,
graph label propagation, submission writer) and swaps the local matcher for MASt3R.

Two modes, selected by `adaptive` flag:
  - BASELINE (adaptive=False): fixed top-K DINOv2 retrieval, MASt3R on every candidate pair,
    MAGSAC gate, triangle-support rehab + graph label propagation, COLMAP.
  - PROPOSED (adaptive=True): reconstruction-aware budget — a cheap base top-K pass builds a
    trusted graph, then EXTRA MASt3R matching is routed only to low-degree / under-connected
    nodes, while cross-component merges require multi-view (triangle) evidence
    (scene-preservation; lesson learned from the ALIKED AdaptiveGraph over-merge).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import networkx as nx
import numpy as np

# Reuse AdaptiveGraph components (added to sys.path by the runner).
from adaptive_imc.geometry import verify_correspondences, pair_quality_score, PairGeometry
from adaptive_imc.graph_ops import EdgeStats, build_graph, triangle_support, low_degree_nodes
from adaptive_imc.retrieval import DINOv2Retriever, make_candidate_pairs
from adaptive_imc.reconstruction import reconstruct_with_colmap

from .matching_mast3r import MASt3RMatcher


@dataclass
class MPairResult:
    i: int
    j: int
    similarity: float
    geom: PairGeometry
    quality: float
    def as_edge(self) -> EdgeStats:
        g = self.geom
        return EdgeStats(i=self.i, j=self.j, inliers=g.inlier_count, ratio=g.inlier_ratio,
                         coverage=g.coverage, similarity=self.similarity, quality=self.quality, route="mast3r")


def _verify(pts0, pts1, shape0, shape1, cfg):
    return verify_correspondences(
        pts0, pts1, shape0, shape1,
        reproj_threshold=cfg["ransac_reproj_threshold"],
        confidence=cfg["ransac_confidence"],
        max_iters=cfg["ransac_max_iters"],
        coverage_grid=cfg["coverage_grid"],
    )


def _is_strong(g: PairGeometry, cfg) -> bool:
    return (g.inlier_count >= cfg["strong_min_inliers"]
            and g.inlier_ratio >= cfg["strong_min_ratio"]
            and g.coverage >= cfg["strong_min_coverage"])


DEFAULT_CFG = {
    # retrieval
    "min_neighbors": 10, "max_neighbors": 20, "similarity_threshold": 0.22, "batch_size": 8,
    "dino_model": "vit_base_patch14_dinov2.lvd142m",
    # verification / gate (MASt3R is dense -> higher inlier floor than ALIKED)
    "ransac_reproj_threshold": 4.0, "ransac_confidence": 0.9999, "ransac_max_iters": 20000,
    "coverage_grid": 4, "strong_min_inliers": 60, "strong_min_ratio": 0.20, "strong_min_coverage": 0.08,
    "triangle_borderline_inliers": 40, "triangle_support_required": 1,
    # adaptive (proposed)
    "base_neighbors": 6, "low_degree_target": 3, "low_degree_extra_pairs": 6,
    # colmap
    "colmap": None,  # filled from adaptive_imc ColmapConfig defaults by runner
}


def _shape(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return int(h), int(w)


def process_dataset_mast3r(*, dataset_name, image_paths, matcher: MASt3RMatcher, cfg,
                           colmap_cfg, work_dir, adaptive: bool = False, verbose=True):
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    t0 = time.time()
    if verbose:
        print(f"\n=== [{'ADAPT' if adaptive else 'BASE'}] {dataset_name}: {n} images ===")
    if n < 2:
        from adaptive_imc.reconstruction import PosePrediction
        return ({0: PosePrediction("outliers", None, None)} if n == 1 else {}), {"dataset": dataset_name, "num_images": n}

    shapes = [_shape(p) for p in image_paths]

    # ---- DINOv2 retrieval ----
    retr = DINOv2Retriever(cfg["dino_model"], device="cuda", use_amp=True)
    emb = retr.encode(image_paths, batch_size=cfg["batch_size"])
    del retr
    kmax = cfg["base_neighbors"] if adaptive else cfg["max_neighbors"]
    pairs, sim = make_candidate_pairs(emb, min_neighbors=min(cfg["min_neighbors"], kmax),
                                      max_neighbors=kmax, similarity_threshold=cfg["similarity_threshold"])
    if verbose:
        print(f"candidate pairs: {len(pairs)} (top-{kmax})")

    # ---- MASt3R matching on candidate pairs ----
    results: dict[tuple[int, int], MPairResult] = {}
    for idx, p in enumerate(pairs):
        pts0, pts1 = matcher.match_pair(image_paths[p.i], image_paths[p.j])
        geom = _verify(pts0, pts1, shapes[p.i], shapes[p.j], cfg)
        results[(p.i, p.j)] = MPairResult(p.i, p.j, p.similarity, geom, pair_quality_score(geom, p.similarity))
        if verbose and (idx + 1) % 50 == 0:
            print(f"  matched {idx+1}/{len(pairs)}")

    def strong_set(res):
        return {k: r for k, r in res.items() if _is_strong(r.geom, cfg)}

    strong = strong_set(results)
    g = build_graph(n, [r.as_edge() for r in strong.values()])

    # trusted components snapshot (for scene-preserving adaptive merges)
    tcomp, tsize = {}, {}
    for ci, comp in enumerate(nx.connected_components(g)):
        for node in comp:
            tcomp[node] = ci
        tsize[ci] = len(comp)

    rescued = 0
    if adaptive:
        # ---- reconstruction-aware budget: extra MASt3R matching for under-connected nodes ----
        # candidate extra pairs = each node's next-best DINOv2 neighbours not yet matched.
        order = np.argsort(-sim, axis=1)
        for node in low_degree_nodes(g, cfg["low_degree_target"]):
            tried = 0
            for j in order[node]:
                j = int(j)
                if j == node:
                    continue
                key = tuple(sorted((node, j)))
                if key in results:
                    continue
                pts0, pts1 = matcher.match_pair(image_paths[key[0]], image_paths[key[1]])
                geom = _verify(pts0, pts1, shapes[key[0]], shapes[key[1]], cfg)
                r = MPairResult(key[0], key[1], float(sim[key[0], key[1]]), geom, pair_quality_score(geom, float(sim[key[0], key[1]])))
                results[key] = r
                rescued += 1
                if _is_strong(geom, cfg):
                    ci, cj = tcomp.get(key[0]), tcomp.get(key[1])
                    cross_big = (ci is not None and cj is not None and ci != cj
                                 and tsize.get(ci, 0) >= 2 and tsize.get(cj, 0) >= 2)
                    # scene-preservation: cross-component merge needs triangle (multi-view) support
                    if cross_big and triangle_support(g, key[0], key[1]) < cfg["triangle_support_required"]:
                        pass  # untrusted cross merge -> skip edge
                    else:
                        strong[key] = r
                        e = r.as_edge()
                        g.add_edge(e.i, e.j, inliers=e.inliers, ratio=e.ratio, coverage=e.coverage,
                                   similarity=e.similarity, quality=e.quality, route=e.route)
                tried += 1
                if g.degree[node] >= cfg["low_degree_target"] or tried >= cfg["low_degree_extra_pairs"]:
                    break

    # ---- triangle-support borderline rehab (both modes) ----
    for r in sorted(results.values(), key=lambda r: r.quality, reverse=True):
        key = (r.i, r.j)
        if key in strong or r.geom.inlier_count < cfg["triangle_borderline_inliers"]:
            continue
        if triangle_support(g, r.i, r.j) >= cfg["triangle_support_required"]:
            strong[key] = r
            e = r.as_edge()
            g.add_edge(e.i, e.j, inliers=e.inliers, ratio=e.ratio, coverage=e.coverage,
                       similarity=e.similarity, quality=e.quality, route=e.route)

    matcher.clear_cache()
    if verbose:
        print(f"accepted edges: {len(strong)}/{len(results)}  components>=2: "
              f"{sum(1 for c in nx.connected_components(g) if len(c) >= 2)}  rescued: {rescued}")

    preds, colmap_diag = reconstruct_with_colmap(
        image_paths=image_paths, accepted_results=strong, match_graph=g,
        work_dir=work_dir / "colmap", cfg=colmap_cfg, enable_label_propagation=True)

    summary = {
        "dataset": dataset_name, "mode": "adaptive" if adaptive else "baseline",
        "num_images": n, "num_candidate_pairs": len(pairs), "num_matched_pairs": len(results),
        "num_accepted_pairs": len(strong), "num_rescued_pairs": rescued,
        "component_sizes": sorted((len(c) for c in nx.connected_components(g)), reverse=True),
        "elapsed_seconds": time.time() - t0, "colmap": colmap_diag,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return preds, summary
