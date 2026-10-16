"""Independent scene discovery via geometrically-verified image-pair graphs.

Algorithm:
1. Extract SIFT keypoints/descriptors once per image (cached across all pairs).
2. For every image pair, run FLANN matching + MAGSAC geometric verification.
3. Compute (total_lowe_matches, magsac_inliers) for each pair.
4. Build an edge iff total_lowe_matches >= GOOD_MATCH_THRESHOLD (default 40).
   - total_lowe_matches is more discriminative than raw MAGSAC inliers because
     RANSAC overfits to few points: a cross-scene pair with 12 Lowe matches can
     have 10 MAGSAC "inliers" (83%), while a same-scene wide-baseline pair has
     30 Lowe matches with only 12 inliers (40%).
   - In practice: cross-scene total_lowe max ≈ 38; same-scene total_lowe min ≈ 40+
     for the nearest neighbors that keep each scene connected.
5. Compute connected components of the filtered graph.
6. Each component is a scene; singletons (no valid edges) are labelled "outliers".

Note: The GOOD_MATCH_THRESHOLD=40 threshold was derived from ETs training data analysis.
Cross-scene pairs (et vs another_et, two similar-looking ET statues) had at most 38
total Lowe-ratio matches. Same-scene pairs had a minimum of ~40 through the nearest
sequential neighbors that maintain connectivity.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .descriptors import global_descriptor

# Primary edge criterion: total Lowe-ratio matches before RANSAC.
# Cross-scene pairs for the ETs dataset max out at 38; same-scene connected neighbors ≥40.
GOOD_MATCH_THRESHOLD = 40

_SiftFeatures = tuple  # (list[cv2.KeyPoint], np.ndarray)


def _extract_sift(path: Path, nfeatures: int = 4000) -> Optional[_SiftFeatures]:
    """Return (keypoints, descriptors) for one image, or None on read/detect failure."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.04)
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(kp) < 8:
        return None
    return kp, des


def _match_pair(
    feat_a: _SiftFeatures,
    feat_b: _SiftFeatures,
    ratio: float = 0.75,
) -> tuple[int, int]:
    """Return (total_lowe_matches, magsac_inliers) for pre-computed SIFT features.

    total_lowe_matches: matches passing Lowe's ratio test (primary edge criterion).
    magsac_inliers: geometrically verified inliers (secondary check).
    """
    kpa, desa = feat_a
    kpb, desb = feat_b

    index_params = {"algorithm": 1, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(desa.astype(np.float32), desb.astype(np.float32), k=2)
    good = [m for m, n in raw if n is not None and m.distance < ratio * n.distance]
    if len(good) < 8:
        return 0, 0

    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    try:
        _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, ransacReprojThreshold=1.0, confidence=0.999)
    except cv2.error:
        # Degenerate geometry (e.g. all points collinear) — treat as 0 inliers
        return len(good), 0
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    return len(good), inliers


def _sift_match_counts(path_a: Path, path_b: Path, ratio: float = 0.75) -> tuple[int, int]:
    """Return (total_lowe_matches, magsac_inliers) for a SIFT match pair (paths).

    Extracts SIFT features from both images on each call.
    Prefer using the cached-feature path inside discover_scenes for bulk matching.
    """
    feat_a = _extract_sift(path_a)
    feat_b = _extract_sift(path_b)
    if feat_a is None or feat_b is None:
        return 0, 0
    return _match_pair(feat_a, feat_b, ratio=ratio)


class _UnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {n: n for n in nodes}
        self.rank: dict[str, int] = {n: 0 for n in nodes}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def discover_scenes(
    image_paths: dict[str, Path],
    good_match_threshold: int = GOOD_MATCH_THRESHOLD,
    verbose: bool = False,
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[int, int]]]:
    """Discover scene labels for all images using geometric verification.

    Returns:
        scene_map: image filename -> scene label string.
            Connected components with >= 2 nodes get "scene_N".
            Singletons (no valid edges) are labelled "outliers".
        edge_data: (name_a, name_b) -> (total_lowe_matches, magsac_inliers) for all pairs.
    """
    names = sorted(image_paths)
    if len(names) < 2:
        return {n: "scene_0" for n in names}, {}

    # Extract SIFT features once per image, then reuse across all pairs.
    if verbose:
        print(f"  extracting SIFT features for {len(names)} images …")
    features: dict[str, Optional[_SiftFeatures]] = {
        name: _extract_sift(image_paths[name]) for name in names
    }

    uf = _UnionFind(names)
    edge_data: dict[tuple[str, str], tuple[int, int]] = {}

    pairs = list(itertools.combinations(names, 2))
    for i, (na, nb) in enumerate(pairs):
        fa, fb = features[na], features[nb]
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            total_good, inliers = _match_pair(fa, fb)
        edge_data[(na, nb)] = (total_good, inliers)
        if verbose and (i + 1) % 20 == 0:
            print(f"  pairs processed: {i + 1}/{len(pairs)}")
        if total_good >= good_match_threshold:
            uf.union(na, nb)

    # Identify which nodes have at least one valid edge (they are in a scene)
    has_edge: set[str] = set()
    for (na, nb), (total_good, _) in edge_data.items():
        if total_good >= good_match_threshold:
            has_edge.add(na)
            has_edge.add(nb)

    # Assign scene labels
    component_label: dict[str, str] = {}
    scene_index = 0
    scene_map: dict[str, str] = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


# ---------------------------------------------------------------------------
# v5: LGEU-guided pair selection + MAGSAC inlier filter
# ---------------------------------------------------------------------------

MAX_EXHAUSTIVE_N = 40          # use exhaustive SIFT for N <= this
MIN_MAGSAC_INLIERS = 3        # MAGSAC inlier floor when filtering
LGEU_TOPK = 30                # top-K LGEU neighbours per image (increased from 20 for better recall)


def _lgeu_select_pairs(
    image_paths: dict[str, Path],
    names: list[str],
    lgeu_model_payload: dict,
    topk: int = LGEU_TOPK,
    verbose: bool = False,
) -> list[tuple[str, str]]:
    """Score all pairs with LGEU (cheap — global descriptors only) and return top-K per image."""
    import numpy as np

    if verbose:
        print(f"  [v5] computing LGEU global descriptors for {len(names)} images ...")

    descs: dict[str, np.ndarray] = {}
    metas: dict[str, dict] = {}
    for name in names:
        try:
            desc, meta = global_descriptor(image_paths[name])
        except Exception:
            desc = np.zeros(133, dtype=np.float32)
            meta = {"width": 1, "height": 1, "aspect": 1.0}
        descs[name] = desc
        metas[name] = meta

    N = len(names)
    desc_mat = np.stack([descs[n] for n in names])  # (N, feat)
    sim = desc_mat @ desc_mat.T                      # (N, N) cosine similarity
    # Mutual rank: rank of b in a's list, average
    ranks_fwd = np.argsort(np.argsort(-sim, axis=1), axis=1) + 1  # (N, N)
    ranks_bwd = ranks_fwd.T

    model = lgeu_model_payload["model"]
    feat_cols = lgeu_model_payload.get("feature_columns", [])

    import re

    def _num_gap(a: str, b: str) -> float:
        ta = [int(x) for x in re.findall(r"\d+", a)]
        tb = [int(x) for x in re.findall(r"\d+", b)]
        if not ta or not tb:
            return 1.0
        return min(abs(x - y) for x in ta for y in tb) / 1000.0

    def _name_jaccard(a: str, b: str) -> float:
        sa = set(a.lower().replace(".", "_").split("_"))
        sb = set(b.lower().replace(".", "_").split("_"))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    # Build features for all pairs and score with LGEU
    all_pairs: list[tuple[str, str]] = []
    all_feats: list[list[float]] = []
    for i, na in enumerate(names):
        meta_a = metas[na]
        area_a = float(meta_a["width"] * meta_a["height"])
        for j, nb in enumerate(names):
            if j <= i:
                continue
            meta_b = metas[nb]
            area_b = float(meta_b["width"] * meta_b["height"])
            cosine = float(sim[i, j])
            mrm = float((ranks_fwd[i, j] + ranks_bwd[i, j]) / 2.0)
            mrx = float(max(ranks_fwd[i, j], ranks_bwd[i, j]))
            aspect_diff = float(abs(meta_a["aspect"] - meta_b["aspect"]))
            area_rat = float(min(area_a, area_b) / max(area_a, area_b, 1.0))
            ng = _num_gap(na, nb)
            nj = _name_jaccard(na, nb)
            feat = [cosine, mrm, mrx, aspect_diff, area_rat, ng, nj]
            all_pairs.append((na, nb))
            all_feats.append(feat)

    feat_array = np.array(all_feats, dtype=np.float32)
    scores = model.predict_proba(feat_array)[:, 1]  # prob same-scene

    # Top-K per image
    img_scores: dict[str, list[tuple[float, str, str]]] = {n: [] for n in names}
    for (na, nb), score in zip(all_pairs, scores):
        img_scores[na].append((score, na, nb))
        img_scores[nb].append((score, na, nb))

    selected: set[tuple[str, str]] = set()
    for n in names:
        img_scores[n].sort(reverse=True)
        for score, na, nb in img_scores[n][:topk]:
            key = (na, nb) if na < nb else (nb, na)
            selected.add(key)

    if verbose:
        print(f"  [v5] LGEU selected {len(selected)} / {len(all_pairs)} pairs")
    return list(selected)


def discover_scenes_v5(
    image_paths: dict[str, Path],
    good_match_threshold: int = GOOD_MATCH_THRESHOLD,
    min_magsac_inliers: int = MIN_MAGSAC_INLIERS,
    lgeu_model_payload: dict | None = None,
    lgeu_topk: int = LGEU_TOPK,
    max_exhaustive_n: int = MAX_EXHAUSTIVE_N,
    verbose: bool = False,
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[int, int]]]:
    """v5 scene discovery: LGEU pair pre-filtering for large N + MAGSAC inlier filter.

    Differences from discover_scenes() (v4):
    - When N > max_exhaustive_n and lgeu_model_payload is provided, SIFT is run only
      on top-LGEU-scoring pairs (cheaper, scales to N=200+).
    - Edge criterion adds a MAGSAC inlier requirement: both Lowe count AND inlier
      count must meet their thresholds.  Falls back to Lowe-only if MAGSAC returns
      zero (degenerate geometry handled gracefully).
    """
    names = sorted(image_paths)
    if len(names) < 2:
        return {n: "scene_0" for n in names}, {}

    if verbose:
        print(f"  [v5] extracting SIFT features for {len(names)} images ...")
    features: dict[str, Optional[_SiftFeatures]] = {
        name: _extract_sift(image_paths[name]) for name in names
    }

    # Choose which pairs to SIFT-match
    use_lgeu = lgeu_model_payload is not None and len(names) > max_exhaustive_n
    if use_lgeu:
        pairs_to_match = _lgeu_select_pairs(image_paths, names, lgeu_model_payload, lgeu_topk, verbose)
    else:
        pairs_to_match = list(itertools.combinations(names, 2))

    uf = _UnionFind(names)
    edge_data: dict[tuple[str, str], tuple[int, int]] = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            total_good, inliers = _match_pair(fa, fb)
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 20 == 0:
            print(f"  [v5] pairs processed: {i + 1}/{len(pairs_to_match)}")

        # Edge criterion: sufficient Lowe matches AND MAGSAC verification.
        # Fallback: if MAGSAC returned 0 inliers on a high-Lowe pair (degenerate
        # geometry edge case) we still admit it when Lowe >= 2*threshold.
        lowe_ok = total_good >= good_match_threshold
        magsac_ok = inliers >= min_magsac_inliers or total_good >= 2 * good_match_threshold
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    # Assign scene labels (same logic as discover_scenes)
    has_edge: set[str] = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= good_match_threshold
        magsac_ok = inliers >= min_magsac_inliers or total_good >= 2 * good_match_threshold
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict[str, str] = {}
    scene_index = 0
    scene_map: dict[str, str] = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


# ---------------------------------------------------------------------------
# v6: same as v5 but graceful fallbacks for all failure modes
# ---------------------------------------------------------------------------


def discover_scenes_v6(
    image_paths: dict[str, Path],
    good_match_threshold: int = GOOD_MATCH_THRESHOLD,
    min_magsac_inliers: int = MIN_MAGSAC_INLIERS,
    lgeu_model_payload: dict | None = None,
    lgeu_topk: int = LGEU_TOPK,
    max_exhaustive_n: int = MAX_EXHAUSTIVE_N,
    verbose: bool = False,
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[int, int]]]:
    """v6 scene discovery: same as v5 with fallbacks for all failure modes.

    Differences from discover_scenes_v5:
    - Per-image SIFT extraction wrapped in try/except; failed images get None features.
    - LGEU pair selection wrapped in try/except; failure falls back to exhaustive pairs.
    - _match_pair wrapped in try/except per pair; failure counts as (0, 0).
    - Guard against empty pairs_to_match list.
    """
    names = sorted(image_paths)
    if len(names) < 2:
        return {n: "scene_0" for n in names}, {}

    if verbose:
        print(f"  [v6] extracting SIFT features for {len(names)} images ...")
    features: dict[str, Optional[_SiftFeatures]] = {}
    for name in names:
        try:
            features[name] = _extract_sift(image_paths[name])
        except Exception:
            features[name] = None

    # Select which pairs to SIFT-match
    use_lgeu = lgeu_model_payload is not None and len(names) > max_exhaustive_n
    if use_lgeu:
        try:
            pairs_to_match = _lgeu_select_pairs(
                image_paths, names, lgeu_model_payload, lgeu_topk, verbose
            )
            if verbose:
                print(f"  [v6] LGEU selected {len(pairs_to_match)} / {len(names) * (len(names) - 1) // 2} pairs")
        except Exception as exc:
            if verbose:
                print(f"  [v6] LGEU pair selection failed ({type(exc).__name__}: {exc}); falling back to exhaustive")
            pairs_to_match = list(itertools.combinations(names, 2))
    else:
        pairs_to_match = list(itertools.combinations(names, 2))

    if not pairs_to_match:
        return {n: "scene_0" for n in names}, {}

    uf = _UnionFind(names)
    edge_data: dict[tuple[str, str], tuple[int, int]] = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            try:
                total_good, inliers = _match_pair(fa, fb)
            except Exception:
                total_good, inliers = 0, 0
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [v6] pairs processed: {i + 1}/{len(pairs_to_match)}")

        lowe_ok = total_good >= good_match_threshold
        magsac_ok = inliers >= min_magsac_inliers or total_good >= 2 * good_match_threshold
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    has_edge: set[str] = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= good_match_threshold
        magsac_ok = inliers >= min_magsac_inliers or total_good >= 2 * good_match_threshold
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict[str, str] = {}
    scene_index = 0
    scene_map: dict[str, str] = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


# v7: scene clustering unchanged from v6 — improvements are in SfM only
discover_scenes_v7 = discover_scenes_v6


def discover_scenes_v9(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v9: GBM LGEU model + TOPK=20 (V7 conservative pairing, V8 model quality).

    Same as discover_scenes_v6 but forces lgeu_topk=20 regardless of the
    module-level LGEU_TOPK constant (currently 30 from V8).
    """
    return discover_scenes_v6(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        lgeu_topk=20,
        verbose=verbose,
    )


def discover_scenes_v10(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v10: GBM LGEU + TOPK=25 (midpoint grid search between V7's 20 and V8's 30).

    V7 (TOPK=20): 8/51 stairs, 27/73 total.
    V8 (TOPK=30): 3/51 stairs (over-merged), 22/73 total.
    V10 tests TOPK=25 as a potential sweet spot for hidden non-video datasets.
    """
    return discover_scenes_v6(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        lgeu_topk=25,
        verbose=verbose,
    )


def discover_scenes_v11(
    image_paths,
    good_match_threshold: int = GOOD_MATCH_THRESHOLD,
    lgeu_model_payload=None,
    verbose=False,
):
    """v11: Lower Lowe-match threshold (30 vs 40) for exhaustive-path datasets.

    ETs uses exhaustive SIFT (N=22 < 40). 2 images are in 'outliers' because they
    have only 30-39 Lowe matches with scene_0 images (below the standard 40 threshold).
    Lowering to 30 may connect these outliers to scene_0, allowing COLMAP to register them.

    For large datasets (LGEU path, N > 40): standard threshold preserved.
    For small datasets (exhaustive path, N <= 40): threshold = 30.
    """
    names = sorted(image_paths)
    N = len(names)
    threshold = 30 if N <= MAX_EXHAUSTIVE_N else good_match_threshold
    return discover_scenes_v6(
        image_paths,
        good_match_threshold=threshold,
        lgeu_model_payload=lgeu_model_payload,
        lgeu_topk=20,
        verbose=verbose,
    )


def discover_scenes_v12(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v12: Video-aware pair augmentation — guarantees temporal neighbors for video sequences.

    For video datasets (N > MAX_EXHAUSTIVE_N, timestamp filenames), LGEU selects only
    TOPK=20 pairs per image by global descriptor similarity. This can leave temporally
    isolated frames as 'outliers' when LGEU fails to select their key overlap pairs —
    the root cause of 14 stairs images getting 0 cameras registered in V9.

    V12 augments LGEU pairs with mandatory sequential neighbors (±2 frames by timestamp)
    for video sequences, ensuring each video frame connects to its nearest temporal
    neighbors before SIFT matching. Non-video and small datasets use standard V9 logic.
    """
    import re as _re

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Video sequence path: LGEU topk=20 + forced sequential temporal pairs
    def _ts_key(n):
        m = _ts_re.search(n)
        return int(m.group()) if m else 0

    sorted_by_time = sorted(names, key=_ts_key)

    try:
        lgeu_list = _lgeu_select_pairs(
            image_paths, names, lgeu_model_payload, topk=20, verbose=verbose
        )
        if verbose:
            print(f"  [v12] LGEU selected {len(lgeu_list)} pairs for video sequence")
        augmented: set[tuple[str, str]] = set()
        for na, nb in lgeu_list:
            augmented.add((na, nb) if na < nb else (nb, na))
    except Exception as exc:
        if verbose:
            print(f"  [v12] LGEU failed ({type(exc).__name__}); using sequential pairs only")
        augmented = set()

    temporal_window = 2
    temporal_added = 0
    for idx, name in enumerate(sorted_by_time):
        for delta in range(1, temporal_window + 1):
            if idx + delta < len(sorted_by_time):
                na, nb = name, sorted_by_time[idx + delta]
                key = (na, nb) if na < nb else (nb, na)
                if key not in augmented:
                    augmented.add(key)
                    temporal_added += 1

    if verbose:
        print(f"  [v12] added {temporal_added} temporal pairs; total={len(augmented)}")

    pairs_to_match = list(augmented)

    features: dict = {}
    for name in names:
        try:
            features[name] = _extract_sift(image_paths[name])
        except Exception:
            features[name] = None

    if not pairs_to_match:
        return {n: "scene_0" for n in names}, {}

    uf = _UnionFind(names)
    edge_data: dict[tuple[str, str], tuple[int, int]] = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            try:
                total_good, inliers = _match_pair(fa, fb)
            except Exception:
                total_good, inliers = 0, 0
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [v12] pairs processed: {i + 1}/{len(pairs_to_match)}")

        lowe_ok = total_good >= GOOD_MATCH_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * GOOD_MATCH_THRESHOLD
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    has_edge: set[str] = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= GOOD_MATCH_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * GOOD_MATCH_THRESHOLD
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict[str, str] = {}
    scene_index = 0
    scene_map: dict[str, str] = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


def discover_scenes_v13(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v13: Cosine-only pair ranking for video sequences — bypass LGEU num_gap penalty.

    For video datasets (N > MAX_EXHAUSTIVE_N, timestamp filenames), the LGEU model's
    num_gap feature is computed as min absolute numeric diff between filename tokens
    divided by 1000. For millisecond timestamps (~600ms gaps at normal video fps),
    this yields num_gap ≈ 0.6, which the GBM model treats as evidence of different
    scenes. This penalizes temporally distant but spatially overlapping frames,
    reducing pair recall and stranding frames as outliers.

    V13 bypasses the LGEU model for video sequences and ranks pairs by raw cosine
    similarity of global descriptors (TOPK=20 per image). Cosine similarity is the
    primary visual-appearance signal and avoids the timestamp artifact.
    Non-video and small datasets use standard V9 logic (GBM LGEU, TOPK=20).
    """
    import re as _re

    import numpy as np

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Video path: rank pairs by cosine similarity of global descriptors only
    if verbose:
        print(f"  [v13] video sequence ({N} images) — cosine-only pair ranking (bypassing LGEU num_gap)")

    descs: dict[str, np.ndarray] = {}
    for name in names:
        try:
            desc, _ = global_descriptor(image_paths[name])
        except Exception:
            desc = np.zeros(133, dtype=np.float32)
        descs[name] = desc

    desc_mat = np.stack([descs[n] for n in names])  # (N, D)
    norms = np.linalg.norm(desc_mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    desc_mat = desc_mat / norms
    sim = desc_mat @ desc_mat.T  # (N, N) cosine similarity
    np.fill_diagonal(sim, -1.0)  # exclude self-pairs

    TOPK = 20
    selected: set[tuple[str, str]] = set()
    for i, na in enumerate(names):
        top_j = np.argsort(-sim[i])[:TOPK]
        for j in top_j:
            nb = names[j]
            key = (na, nb) if na < nb else (nb, na)
            selected.add(key)

    pairs_to_match = list(selected)

    if verbose:
        print(f"  [v13] cosine selected {len(pairs_to_match)} / {N * (N - 1) // 2} pairs")

    features: dict = {}
    for name in names:
        try:
            features[name] = _extract_sift(image_paths[name])
        except Exception:
            features[name] = None

    if not pairs_to_match:
        return {n: "scene_0" for n in names}, {}

    uf = _UnionFind(names)
    edge_data: dict[tuple[str, str], tuple[int, int]] = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            try:
                total_good, inliers = _match_pair(fa, fb)
            except Exception:
                total_good, inliers = 0, 0
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [v13] pairs processed: {i + 1}/{len(pairs_to_match)}")

        lowe_ok = total_good >= GOOD_MATCH_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * GOOD_MATCH_THRESHOLD
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    has_edge: set[str] = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= GOOD_MATCH_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * GOOD_MATCH_THRESHOLD
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict[str, str] = {}
    scene_index = 0
    scene_map: dict[str, str] = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


def discover_scenes_v14(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v14: LGEU GBM scoring with num_gap neutralized for video sequences.

    The GBM model's num_gap feature = min(|token_a - token_b|) / 1000 across all
    numeric filename tokens. For millisecond Unix timestamps (~600ms between adjacent
    video frames at ~1.6 fps), num_gap ≈ 0.6. The GBM model learned that large
    num_gap correlates with cross-scene pairs (training data used sequential file
    indices where same-scene pairs have small gaps). Applied to video, this penalizes
    temporally adjacent same-scene frames and strands them as 'outliers'.

    V14 zeros out num_gap before GBM scoring for video sequences (N > MAX_EXHAUSTIVE_N,
    >=70% timestamp filenames). The model's 6 remaining features — cosine similarity,
    mutual rank mean/max, aspect_diff, area_ratio, name_jaccard — are all valid for
    video and continue to discriminate same-scene vs cross-scene pairs. Non-video and
    small datasets use standard V9 logic (full GBM, TOPK=20), unaffected.
    """
    import re as _re

    import numpy as np

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Video path: LGEU GBM scoring with num_gap zeroed out
    if verbose:
        print(f"  [v14] video sequence ({N} images) — GBM LGEU with num_gap neutralized")

    descs: dict = {}
    metas: dict = {}
    for name in names:
        try:
            desc, meta = global_descriptor(image_paths[name])
        except Exception:
            desc = np.zeros(133, dtype=np.float32)
            meta = {"width": 1, "height": 1, "aspect": 1.0}
        descs[name] = desc
        metas[name] = meta

    desc_mat = np.stack([descs[n] for n in names])  # (N, D)
    sim = desc_mat @ desc_mat.T  # (N, N) cosine similarity
    ranks_fwd = np.argsort(np.argsort(-sim, axis=1), axis=1) + 1  # (N, N)
    ranks_bwd = ranks_fwd.T

    model = lgeu_model_payload["model"]

    def _name_jaccard(a: str, b: str) -> float:
        sa = set(a.lower().replace(".", "_").split("_"))
        sb = set(b.lower().replace(".", "_").split("_"))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    all_pairs: list = []
    all_feats: list = []
    for i, na in enumerate(names):
        meta_a = metas[na]
        area_a = float(meta_a["width"] * meta_a["height"])
        for j, nb in enumerate(names):
            if j <= i:
                continue
            meta_b = metas[nb]
            area_b = float(meta_b["width"] * meta_b["height"])
            cosine = float(sim[i, j])
            mrm = float((ranks_fwd[i, j] + ranks_bwd[i, j]) / 2.0)
            mrx = float(max(ranks_fwd[i, j], ranks_bwd[i, j]))
            aspect_diff = float(abs(meta_a["aspect"] - meta_b["aspect"]))
            area_rat = float(min(area_a, area_b) / max(area_a, area_b, 1.0))
            ng = 0.0  # num_gap zeroed: timestamp gaps are artifacts, not scene evidence
            nj = _name_jaccard(na, nb)
            all_pairs.append((na, nb))
            all_feats.append([cosine, mrm, mrx, aspect_diff, area_rat, ng, nj])

    feat_array = np.array(all_feats, dtype=np.float32)
    scores = model.predict_proba(feat_array)[:, 1]  # prob same-scene

    TOPK = 20
    img_scores: dict = {n: [] for n in names}
    for (na, nb), score in zip(all_pairs, scores):
        img_scores[na].append((score, na, nb))
        img_scores[nb].append((score, na, nb))

    selected: set = set()
    for n in names:
        img_scores[n].sort(reverse=True)
        for score, na, nb in img_scores[n][:TOPK]:
            key = (na, nb) if na < nb else (nb, na)
            selected.add(key)

    pairs_to_match = list(selected)

    if verbose:
        print(f"  [v14] selected {len(pairs_to_match)} / {N * (N - 1) // 2} pairs")

    features: dict = {}
    for name in names:
        try:
            features[name] = _extract_sift(image_paths[name])
        except Exception:
            features[name] = None

    if not pairs_to_match:
        return {n: "scene_0" for n in names}, {}

    uf = _UnionFind(names)
    edge_data: dict = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            try:
                total_good, inliers = _match_pair(fa, fb)
            except Exception:
                total_good, inliers = 0, 0
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [v14] pairs processed: {i + 1}/{len(pairs_to_match)}")

        lowe_ok = total_good >= GOOD_MATCH_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * GOOD_MATCH_THRESHOLD
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    has_edge: set = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= GOOD_MATCH_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * GOOD_MATCH_THRESHOLD
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict = {}
    scene_index = 0
    scene_map: dict = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


def discover_scenes_v15(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v15: Lower Lowe threshold (30) for video sequences on the LGEU path.

    In V9 (GBM+TOPK=20), 14 stairs images end up as outliers with 0 cameras registered.
    Stairs is a video sequence (N=51 > MAX_EXHAUSTIVE_N) with forward-facing staircase
    motion; frames have limited visual overlap, so some valid same-scene pairs produce
    only 30-39 Lowe matches — just below the standard threshold of 40. These pairs are
    geometrically consistent (MAGSAC-verifiable) but are blocked from forming edges by
    the Lowe count cutoff, stranding their images as outliers.

    V15 lowers good_match_threshold to 30 for video sequences (N > MAX_EXHAUSTIVE_N,
    >=70% timestamp filenames). The LGEU pair pre-selection (TOPK=20) limits candidate
    pairs to the most visually similar ones, reducing cross-scene false-positive risk.
    MAGSAC geometric verification (MIN_MAGSAC_INLIERS=3) guards against any remaining
    cross-scene pairs that happen to produce >= 30 Lowe matches. Non-video and small
    datasets use the standard V9 path (threshold=40, TOPK=20), unaffected.

    Unlike V11, which lowered the threshold for small exhaustive-path datasets (ETs)
    and caused cross-scene merging, V15 only lowers the threshold for large video
    datasets where LGEU pre-selection already constrains the candidate set.
    """
    import re as _re

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Video path: lower Lowe threshold from 40 to 30; LGEU+MAGSAC guard false positives
    if verbose:
        print(f"  [v15] video sequence ({N} images) — Lowe threshold=30, LGEU TOPK=20")

    return discover_scenes_v6(
        image_paths,
        good_match_threshold=30,
        lgeu_model_payload=lgeu_model_payload,
        lgeu_topk=20,
        verbose=verbose,
    )


def discover_scenes_v16(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v16: GBM LGEU with num_gap=0 (V14) combined with Lowe threshold=30 (V15) for video.

    V14 zeroed num_gap before GBM scoring for video sequences, neutralising the timestamp
    artifact that causes the model to rank temporally-adjacent same-scene frames low.
    V14 result: 6/51 stairs (same as V9 — the improved pair selection was offset by the
    40-match Lowe cutoff still blocking weak-but-valid edges).

    V15 lowered good_match_threshold to 30 for video sequences, admitting pairs with
    30-39 Lowe matches that are geometrically consistent (MAGSAC-verifiable). V15 result:
    7/51 stairs — better pair admission helped but pair selection still used the biased
    num_gap feature.

    V16 combines both changes for video sequences:
    1. GBM scoring with num_gap=0 → selects pairs by visual appearance, not timestamp gaps
    2. good_match_threshold=30 → admits weak-overlap but geometrically valid same-scene edges
    MAGSAC (MIN_MAGSAC_INLIERS=3) and LGEU TOPK=20 still guard against cross-scene merging.

    Non-video and small datasets (N <= MAX_EXHAUSTIVE_N) use standard V9 logic (full GBM,
    num_gap active, threshold=40, TOPK=20), unaffected.
    """
    import re as _re

    import numpy as np

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Video path: GBM LGEU with num_gap=0 + lowered Lowe threshold
    if verbose:
        print(f"  [v16] video sequence ({N} images) — GBM num_gap=0 + Lowe threshold=30")

    descs: dict = {}
    metas: dict = {}
    for name in names:
        try:
            desc, meta = global_descriptor(image_paths[name])
        except Exception:
            desc = np.zeros(133, dtype=np.float32)
            meta = {"width": 1, "height": 1, "aspect": 1.0}
        descs[name] = desc
        metas[name] = meta

    desc_mat = np.stack([descs[n] for n in names])  # (N, D)
    sim = desc_mat @ desc_mat.T  # (N, N) cosine similarity
    ranks_fwd = np.argsort(np.argsort(-sim, axis=1), axis=1) + 1  # (N, N)
    ranks_bwd = ranks_fwd.T

    model = lgeu_model_payload["model"]

    def _name_jaccard(a: str, b: str) -> float:
        sa = set(a.lower().replace(".", "_").split("_"))
        sb = set(b.lower().replace(".", "_").split("_"))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    all_pairs: list = []
    all_feats: list = []
    for i, na in enumerate(names):
        meta_a = metas[na]
        area_a = float(meta_a["width"] * meta_a["height"])
        for j, nb in enumerate(names):
            if j <= i:
                continue
            meta_b = metas[nb]
            area_b = float(meta_b["width"] * meta_b["height"])
            cosine = float(sim[i, j])
            mrm = float((ranks_fwd[i, j] + ranks_bwd[i, j]) / 2.0)
            mrx = float(max(ranks_fwd[i, j], ranks_bwd[i, j]))
            aspect_diff = float(abs(meta_a["aspect"] - meta_b["aspect"]))
            area_rat = float(min(area_a, area_b) / max(area_a, area_b, 1.0))
            ng = 0.0  # neutralise timestamp artifact (same as V14)
            nj = _name_jaccard(na, nb)
            all_pairs.append((na, nb))
            all_feats.append([cosine, mrm, mrx, aspect_diff, area_rat, ng, nj])

    feat_array = np.array(all_feats, dtype=np.float32)
    scores = model.predict_proba(feat_array)[:, 1]  # prob same-scene

    TOPK = 20
    img_scores: dict = {n: [] for n in names}
    for (na, nb), score in zip(all_pairs, scores):
        img_scores[na].append((score, na, nb))
        img_scores[nb].append((score, na, nb))

    selected: set = set()
    for n in names:
        img_scores[n].sort(reverse=True)
        for score, na, nb in img_scores[n][:TOPK]:
            key = (na, nb) if na < nb else (nb, na)
            selected.add(key)

    pairs_to_match = list(selected)

    if verbose:
        print(f"  [v16] selected {len(pairs_to_match)} / {N * (N - 1) // 2} pairs")

    features: dict = {}
    for name in names:
        try:
            features[name] = _extract_sift(image_paths[name])
        except Exception:
            features[name] = None

    if not pairs_to_match:
        return {n: "scene_0" for n in names}, {}

    VIDEO_THRESHOLD = 30  # lowered from 40 (same as V15)

    uf = _UnionFind(names)
    edge_data: dict = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            try:
                total_good, inliers = _match_pair(fa, fb)
            except Exception:
                total_good, inliers = 0, 0
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [v16] pairs processed: {i + 1}/{len(pairs_to_match)}")

        lowe_ok = total_good >= VIDEO_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * VIDEO_THRESHOLD
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    has_edge: set = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= VIDEO_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * VIDEO_THRESHOLD
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict = {}
    scene_index = 0
    scene_map: dict = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


def discover_scenes_v17(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v17: Increased SIFT nfeatures (4000→8000) for video sequences, on top of V16.

    V16 combines GBM num_gap=0 (V14) + Lowe threshold=30 (V15) for video sequences,
    reaching 7/51 stairs cameras. Despite these improvements, 14 stairs images remain
    stranded as 'outliers' with zero valid SIFT edges — meaning ALL top-20 GBM-selected
    neighbors produce <30 Lowe matches for each outlier image.

    The untried bottleneck: SIFT feature density. The default _extract_sift uses
    nfeatures=4000. For forward-facing staircase video with small inter-frame overlap,
    the overlapping region between adjacent frames covers only a fraction of each frame.
    With 4000 features spread over the full image, the overlapping boundary area may be
    too sparsely sampled to accumulate 30+ Lowe matches with any neighbor.

    V17 doubles nfeatures to 8000 for video-path SIFT extraction. More keypoints spread
    denser coverage across each frame, increasing the probability that the overlapping
    region between frames is sampled. More candidate keypoints → more candidate FLANN
    matches → higher Lowe-ratio match counts for pairs with small but real geometric
    overlap. Combined with V16's num_gap=0 GBM scoring and threshold=30, this should
    help at least some outlier images form their first valid edge and enter reconstruction.

    Non-video and small datasets (N <= MAX_EXHAUSTIVE_N) use standard V9 logic
    (nfeatures=4000, full GBM, num_gap active, threshold=40, TOPK=20), unaffected.
    """
    import re as _re

    import numpy as np

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Video path: GBM num_gap=0 (V16) + Lowe threshold=30 (V16) + nfeatures=8000 (NEW)
    if verbose:
        print(f"  [v17] video sequence ({N} images) — nfeatures=8000, num_gap=0, Lowe=30")

    descs: dict = {}
    metas: dict = {}
    for name in names:
        try:
            desc, meta = global_descriptor(image_paths[name])
        except Exception:
            desc = np.zeros(133, dtype=np.float32)
            meta = {"width": 1, "height": 1, "aspect": 1.0}
        descs[name] = desc
        metas[name] = meta

    desc_mat = np.stack([descs[n] for n in names])  # (N, D)
    sim = desc_mat @ desc_mat.T  # (N, N) cosine similarity
    ranks_fwd = np.argsort(np.argsort(-sim, axis=1), axis=1) + 1  # (N, N)
    ranks_bwd = ranks_fwd.T

    model = lgeu_model_payload["model"]

    def _name_jaccard(a: str, b: str) -> float:
        sa = set(a.lower().replace(".", "_").split("_"))
        sb = set(b.lower().replace(".", "_").split("_"))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    all_pairs: list = []
    all_feats: list = []
    for i, na in enumerate(names):
        meta_a = metas[na]
        area_a = float(meta_a["width"] * meta_a["height"])
        for j, nb in enumerate(names):
            if j <= i:
                continue
            meta_b = metas[nb]
            area_b = float(meta_b["width"] * meta_b["height"])
            cosine = float(sim[i, j])
            mrm = float((ranks_fwd[i, j] + ranks_bwd[i, j]) / 2.0)
            mrx = float(max(ranks_fwd[i, j], ranks_bwd[i, j]))
            aspect_diff = float(abs(meta_a["aspect"] - meta_b["aspect"]))
            area_rat = float(min(area_a, area_b) / max(area_a, area_b, 1.0))
            ng = 0.0  # neutralise timestamp artifact (same as V14/V16)
            nj = _name_jaccard(na, nb)
            all_pairs.append((na, nb))
            all_feats.append([cosine, mrm, mrx, aspect_diff, area_rat, ng, nj])

    feat_array = np.array(all_feats, dtype=np.float32)
    scores = model.predict_proba(feat_array)[:, 1]  # prob same-scene

    TOPK = 20
    img_scores: dict = {n: [] for n in names}
    for (na, nb), score in zip(all_pairs, scores):
        img_scores[na].append((score, na, nb))
        img_scores[nb].append((score, na, nb))

    selected: set = set()
    for n in names:
        img_scores[n].sort(reverse=True)
        for score, na, nb in img_scores[n][:TOPK]:
            key = (na, nb) if na < nb else (nb, na)
            selected.add(key)

    pairs_to_match = list(selected)

    if verbose:
        print(f"  [v17] selected {len(pairs_to_match)} / {N * (N - 1) // 2} pairs")

    # Extract SIFT features with nfeatures=8000 for video sequences
    features: dict = {}
    for name in names:
        try:
            features[name] = _extract_sift(image_paths[name], nfeatures=8000)
        except Exception:
            features[name] = None

    if not pairs_to_match:
        return {n: "scene_0" for n in names}, {}

    VIDEO_THRESHOLD = 30  # same as V15/V16

    uf = _UnionFind(names)
    edge_data: dict = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            try:
                total_good, inliers = _match_pair(fa, fb)
            except Exception:
                total_good, inliers = 0, 0
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [v17] pairs processed: {i + 1}/{len(pairs_to_match)}")

        lowe_ok = total_good >= VIDEO_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * VIDEO_THRESHOLD
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    has_edge: set = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= VIDEO_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * VIDEO_THRESHOLD
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict = {}
    scene_index = 0
    scene_map: dict = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


def discover_scenes_v18(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v18: Scene clustering identical to V16 — improvement is in SfM mapper settings.

    V16 combined GBM num_gap=0 (V14) + Lowe threshold=30 (V15) for video sequences,
    reaching 7/51 stairs cameras. Despite these clustering improvements, within each
    correctly-identified scene COLMAP still only registers 3/15 and 3/20 cameras.
    The COLMAP SfM mapper settings (abs_pose_min_num_inliers=4, init_min_tri_angle=1.0°)
    are too strict for forward-facing staircase video where the initial 3D model is sparse
    and new frames have limited 3D point correspondences with the existing reconstruction.

    V18 keeps V16's clustering (best available) and targets the SfM step instead —
    see run_sfm_v18 in sfm.py.

    Non-video and small datasets (N <= MAX_EXHAUSTIVE_N) use standard V9 clustering,
    same as V16.
    """
    return discover_scenes_v16(image_paths, lgeu_model_payload=lgeu_model_payload, verbose=verbose)


def discover_scenes_v19(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v19: Loosened FLANN Lowe ratio (0.75→0.80) for video sequences, on top of V16.

    In V16 (GBM num_gap=0 + Lowe threshold=30 for video), 14 stairs images remain
    stranded as 'outliers' with zero valid SIFT edges — every top-20 GBM-selected
    neighbor produces fewer than 30 Lowe matches even with the lowered threshold and
    improved pair selection. The untried bottleneck is the FLANN Lowe ratio test,
    fixed at 0.75 across all versions V1-V18.

    For video sequences with repetitive staircase textures and small inter-frame overlap,
    many valid correspondences fall in the 0.75–0.80 ratio range: the nearest descriptor
    is a correct match, but the second-nearest is nearly as close due to repeated texture
    patterns. These valid matches are rejected by the strict 0.75 cutoff, artificially
    depressing Lowe match counts and preventing outlier images from forming any valid edge.

    V19 loosens the Lowe ratio from 0.75 to 0.80 specifically for video sequence pair
    matching. MAGSAC geometric verification (MIN_MAGSAC_INLIERS=3) provides the final
    guard against false positives admitted by the looser ratio. Combined with V16's pair
    selection (num_gap=0, TOPK=20) and threshold=30, this should help at least some
    outlier images accumulate enough matches to form a valid SIFT edge.

    Non-video and small datasets (N <= MAX_EXHAUSTIVE_N) use standard V9 logic (ratio=0.75,
    full GBM num_gap active, threshold=40, TOPK=20), unaffected.
    """
    import re as _re

    import numpy as np

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Video path: GBM num_gap=0 (V16) + Lowe threshold=30 (V16) + ratio=0.80 (NEW)
    if verbose:
        print(f"  [v19] video sequence ({N} images) — GBM num_gap=0, Lowe=30, ratio=0.80")

    descs: dict = {}
    metas: dict = {}
    for name in names:
        try:
            desc, meta = global_descriptor(image_paths[name])
        except Exception:
            desc = np.zeros(133, dtype=np.float32)
            meta = {"width": 1, "height": 1, "aspect": 1.0}
        descs[name] = desc
        metas[name] = meta

    desc_mat = np.stack([descs[n] for n in names])  # (N, D)
    sim = desc_mat @ desc_mat.T  # (N, N) cosine similarity
    ranks_fwd = np.argsort(np.argsort(-sim, axis=1), axis=1) + 1  # (N, N)
    ranks_bwd = ranks_fwd.T

    model = lgeu_model_payload["model"]

    def _name_jaccard(a: str, b: str) -> float:
        sa = set(a.lower().replace(".", "_").split("_"))
        sb = set(b.lower().replace(".", "_").split("_"))
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    all_pairs: list = []
    all_feats: list = []
    for i, na in enumerate(names):
        meta_a = metas[na]
        area_a = float(meta_a["width"] * meta_a["height"])
        for j, nb in enumerate(names):
            if j <= i:
                continue
            meta_b = metas[nb]
            area_b = float(meta_b["width"] * meta_b["height"])
            cosine = float(sim[i, j])
            mrm = float((ranks_fwd[i, j] + ranks_bwd[i, j]) / 2.0)
            mrx = float(max(ranks_fwd[i, j], ranks_bwd[i, j]))
            aspect_diff = float(abs(meta_a["aspect"] - meta_b["aspect"]))
            area_rat = float(min(area_a, area_b) / max(area_a, area_b, 1.0))
            ng = 0.0  # neutralise timestamp artifact (same as V14/V16)
            nj = _name_jaccard(na, nb)
            all_pairs.append((na, nb))
            all_feats.append([cosine, mrm, mrx, aspect_diff, area_rat, ng, nj])

    feat_array = np.array(all_feats, dtype=np.float32)
    scores = model.predict_proba(feat_array)[:, 1]  # prob same-scene

    TOPK = 20
    img_scores: dict = {n: [] for n in names}
    for (na, nb), score in zip(all_pairs, scores):
        img_scores[na].append((score, na, nb))
        img_scores[nb].append((score, na, nb))

    selected: set = set()
    for n in names:
        img_scores[n].sort(reverse=True)
        for score, na, nb in img_scores[n][:TOPK]:
            key = (na, nb) if na < nb else (nb, na)
            selected.add(key)

    pairs_to_match = list(selected)

    if verbose:
        print(f"  [v19] selected {len(pairs_to_match)} / {N * (N - 1) // 2} pairs")

    features: dict = {}
    for name in names:
        try:
            features[name] = _extract_sift(image_paths[name])
        except Exception:
            features[name] = None

    if not pairs_to_match:
        return {n: "scene_0" for n in names}, {}

    VIDEO_THRESHOLD = 30  # same as V15/V16
    VIDEO_RATIO = 0.80    # loosened from 0.75 standard to admit borderline video matches

    uf = _UnionFind(names)
    edge_data: dict = {}

    for i, pair in enumerate(pairs_to_match):
        na, nb = pair
        fa, fb = features.get(na), features.get(nb)
        if fa is None or fb is None:
            total_good, inliers = 0, 0
        else:
            try:
                total_good, inliers = _match_pair(fa, fb, ratio=VIDEO_RATIO)
            except Exception:
                total_good, inliers = 0, 0
        key = (na, nb) if na < nb else (nb, na)
        edge_data[key] = (total_good, inliers)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [v19] pairs processed: {i + 1}/{len(pairs_to_match)}")

        lowe_ok = total_good >= VIDEO_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * VIDEO_THRESHOLD
        if lowe_ok and magsac_ok:
            uf.union(na, nb)

    has_edge: set = set()
    for (na, nb), (total_good, inliers) in edge_data.items():
        lowe_ok = total_good >= VIDEO_THRESHOLD
        magsac_ok = inliers >= MIN_MAGSAC_INLIERS or total_good >= 2 * VIDEO_THRESHOLD
        if lowe_ok and magsac_ok:
            has_edge.add(na)
            has_edge.add(nb)

    component_label: dict = {}
    scene_index = 0
    scene_map: dict = {}
    for name in names:
        root = uf.find(name)
        if name not in has_edge:
            scene_map[name] = "outliers"
        else:
            if root not in component_label:
                component_label[root] = f"scene_{scene_index}"
                scene_index += 1
            scene_map[name] = component_label[root]

    return scene_map, edge_data


def discover_scenes_v20(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v20: V16 clustering + outlier reassignment via global descriptor similarity.

    All versions V14-V19 tried to rescue outlier stairs images by forcing SIFT edges
    (through better pair selection, lower thresholds, looser ratios, more features).
    None succeeded: these 14 images cannot accumulate even 30 Lowe matches with any
    of their top-20 GBM-selected neighbors. The SIFT edge gate is impassable for them.

    V20 accepts this and takes a different approach: after V16 clustering assigns 14
    images to "outliers", each outlier is reassigned to the scene of its nearest
    non-outlier image by cosine similarity of global descriptors. COLMAP then runs
    exhaustive matching internally on the enlarged scene — its own SIFT extraction and
    matching pipeline — giving these images a second chance at registration without any
    of our Python-side threshold constraints. If even a few outliers share real geometric
    overlap with their assigned scene, COLMAP will register them.

    For video sequences (N > MAX_EXHAUSTIVE_N, >=70% timestamp filenames), V16 pair
    selection (num_gap=0, TOPK=20, Lowe threshold=30) is applied first, then outlier
    reassignment. Non-video and small datasets use standard V9 logic, unaffected.
    If no non-outlier images exist (entire dataset is outliers), scene_map is returned
    as-is from V16.
    """
    import re as _re

    import numpy as np

    names = sorted(image_paths)
    N = len(names)

    _ts_re = _re.compile(r"\d{10,13}")
    n_match = sum(1 for n in names if _ts_re.search(n))
    is_video = N >= 3 and n_match / N >= 0.7

    if N <= MAX_EXHAUSTIVE_N or lgeu_model_payload is None or not is_video:
        return discover_scenes_v6(
            image_paths,
            lgeu_model_payload=lgeu_model_payload,
            lgeu_topk=20,
            verbose=verbose,
        )

    # Step 1: Run V16 clustering (num_gap=0 + Lowe threshold=30 for video)
    if verbose:
        print(f"  [v20] video sequence ({N} images) — V16 clustering + outlier reassignment")

    scene_map, edge_data = discover_scenes_v16(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        verbose=verbose,
    )

    # Step 2: Collect outlier images
    outlier_names = [n for n in names if scene_map.get(n) == "outliers"]
    if not outlier_names:
        if verbose:
            print("  [v20] no outliers — returning V16 result unchanged")
        return scene_map, edge_data

    non_outlier_names = [n for n in names if scene_map.get(n) != "outliers"]
    if not non_outlier_names:
        if verbose:
            print("  [v20] all images are outliers — cannot reassign, returning V16 result")
        return scene_map, edge_data

    if verbose:
        print(f"  [v20] reassigning {len(outlier_names)} outliers via global descriptor similarity")

    # Step 3: Compute global descriptors for outliers + non-outliers
    descs: dict = {}
    for name in names:
        try:
            desc, _ = global_descriptor(image_paths[name])
        except Exception:
            desc = np.zeros(133, dtype=np.float32)
        norm = float(np.linalg.norm(desc))
        descs[name] = desc / norm if norm > 1e-9 else desc

    # Step 4: For each outlier, find closest non-outlier by cosine similarity
    non_outlier_mat = np.stack([descs[n] for n in non_outlier_names])  # (M, D)
    scene_map = dict(scene_map)  # copy before mutating
    for outlier in outlier_names:
        sims = non_outlier_mat @ descs[outlier]  # (M,) cosine similarities
        best_idx = int(np.argmax(sims))
        best_neighbor = non_outlier_names[best_idx]
        assigned_scene = scene_map[best_neighbor]
        scene_map[outlier] = assigned_scene
        if verbose:
            print(
                f"  [v20] '{outlier}' → '{assigned_scene}' "
                f"(nearest: '{best_neighbor}', sim={sims[best_idx]:.3f})"
            )

    return scene_map, edge_data


def discover_scenes_v21(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v21: V16 clustering unchanged. V21's improvement is in the SfM step (abs_pose threshold).

    V16 clustering (GBM num_gap=0 + Lowe threshold=30 for video sequences with
    N > MAX_EXHAUSTIVE_N and >=70% timestamp filenames) is the best-scoring
    clustering configuration (7/51 stairs, 19/22 ETs). V21 retains it unchanged
    and targets the SfM bottleneck instead: COLMAP registers only 3/15 and 3/20
    cameras in the correctly-clustered video scenes because abs_pose_min_num_inliers=4
    blocks frames with exactly 3 inlier correspondences to the existing 3D model —
    exactly at the P3P minimum for near-degenerate forward-facing video.

    The SfM fix (abs_pose_min_num_inliers 4→3 for video) is in run_sfm_v21.
    Non-video and small datasets use standard V9 logic, unaffected.
    """
    return discover_scenes_v16(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        verbose=verbose,
    )


def discover_scenes_v22(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v22: V16 clustering unchanged. V22's improvement is in the SfM step (init_min_tri_angle).

    V21 lowered abs_pose_min_num_inliers from 4 to 3 (P3P minimum) for video scenes and
    tied V7 at 8/51 stairs. The remaining untried SfM parameter from V18 (which
    upload_failed as a 3-parameter bundle) is init_min_tri_angle. V21 keeps this at 1.0°
    (the V7 video default). For near-forward-facing staircase cameras, valid seed pairs
    often have a triangulation angle below 1.0°, causing COLMAP to discard them in favour
    of geometrically worse seed pairs — or fail initialisation entirely. V22 combines
    V21's abs_pose_min_num_inliers=3 with lowering init_min_tri_angle from 1.0° to 0.5°
    for video scenes only. A better seed pair produces a denser initial 3D point cloud,
    which allows more cameras to be registered in the extension phase.

    The SfM fix is in run_sfm_v22; this function passes through to discover_scenes_v21
    (which uses V16 clustering) unchanged.
    Non-video and small datasets use standard V9 logic, unaffected.
    """
    return discover_scenes_v21(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        verbose=verbose,
    )


def discover_scenes_v23(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v23: V16 clustering unchanged. V23's improvement is in the SfM step (CameraMode.SINGLE).

    V21 lowered abs_pose_min_num_inliers to 3 (P3P minimum) for video scenes and tied V7
    at 8/51 stairs. V22 added init_min_tri_angle=0.5 on top and regressed to 7/51.
    Both V21 and V22 (and all prior versions) use pycolmap.CameraMode.AUTO for feature
    extraction, which assigns independent camera intrinsics to each image. For staircase
    video, all frames come from the same physical camera with fixed focal length and
    principal point. AUTO mode must estimate intrinsics per image from a sparse seed pair,
    producing poorly constrained per-frame estimates and an inconsistent point cloud.

    V23 switches to CameraMode.SINGLE for video scenes: COLMAP shares one camera model
    across all frames and jointly refines intrinsics during bundle adjustment. This
    produces a more accurately calibrated reconstruction with tighter 3D point positions,
    increasing the likelihood that adjacent frames find the 3 inlier correspondences
    required by V21's abs_pose_min_num_inliers=3 setting.

    The SfM fix is in run_sfm_v23; this function passes through to discover_scenes_v21
    (which uses V16 clustering) unchanged.
    Non-video and small datasets use standard V9 logic, unaffected.
    """
    return discover_scenes_v21(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        verbose=verbose,
    )


def discover_scenes_v24(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v24: V16 clustering unchanged. V24's improvement is in the SfM step (init_min_num_inliers=3).

    V21 lowered abs_pose_min_num_inliers to 3 (P3P minimum) for video scenes and tied V7
    at 8/51 stairs. V22 added init_min_tri_angle=0.5 and regressed to 7/51.
    V18 changed three mapper parameters simultaneously (init_min_num_inliers 5→3,
    init_min_tri_angle 1.0→0.5, abs_pose_min_num_inliers 4→3) and upload_failed.

    The third parameter from V18 that has never been tested in isolation is
    init_min_num_inliers (5→3). This controls the minimum number of inliers required
    for a seed pair to be accepted during COLMAP initialization. For staircase video with
    nearly-parallel optical axes, even the best available seed pair may produce only 3–5
    verified inliers in the fundamental matrix estimation — causing COLMAP to reject valid
    seed pairs and fall back to geometrically worse alternatives (or fail initialization
    entirely). V24 reduces init_min_num_inliers from 5 to 3 for video scenes, combined
    with V21's abs_pose_min_num_inliers=3, without changing init_min_tri_angle (which
    caused regression in V22). Scene clustering uses V16 (num_gap=0, Lowe=30) unchanged.

    The SfM fix is in run_sfm_v24; this function passes through to discover_scenes_v21
    (which uses V16 clustering) unchanged.
    Non-video and small datasets use standard V9 logic, unaffected.
    """
    return discover_scenes_v21(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        verbose=verbose,
    )


def discover_scenes_v25(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v25: V16 clustering unchanged. V25's improvement is merging ALL COLMAP reconstructions.

    All prior versions V7–V24 call pycolmap.incremental_mapping() and then take only the
    LARGEST reconstruction:
        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    This silently discards every smaller sub-reconstruction. For staircase video recorded
    simultaneously by two physically separate camera rigs, COLMAP's incremental mapper
    naturally produces multiple disjoint sub-reconstructions — one per rig cluster — because
    the two rigs share no geometric overlap and cannot be merged into a single model. With
    N=35 images across scene_0 (15) and scene_2 (20), if COLMAP produces reconstructions of
    sizes [5, 4] for scene_0, the current code registers only 5 cameras and silently drops 4.

    V25 fixes this by merging poses from ALL reconstructions returned by incremental_mapping,
    sorted by descending registration count. For any image registered in multiple
    sub-reconstructions (a rare corner case), the pose from the largest reconstruction wins.
    This is the first version to exploit sub-reconstructions and directly targets the
    observation that V21 (the best version at 8/51) can only use one sub-reconstruction per
    scene despite COLMAP often finding more.

    Scene clustering uses V16 (GBM num_gap=0 + Lowe threshold=30 for video), unchanged from
    V21. SfM mapper settings are the same as V21 (abs_pose_min_num_inliers=3 for video).
    Non-video and small datasets use standard V9 logic, unaffected.
    """
    return discover_scenes_v21(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        verbose=verbose,
    )


def discover_scenes_v26(
    image_paths,
    lgeu_model_payload=None,
    verbose=False,
):
    """v26: V16 clustering unchanged. V26 combines V25 (merge all sub-recs) + V24 (init_min_num_inliers=3).

    V25 merged poses from ALL COLMAP sub-reconstructions (rather than only the largest) and
    tied V7 at 8/51 stairs cameras. Since V25 showed no improvement, COLMAP likely produces
    only one sub-reconstruction per scene under V21's mapper settings, so the merge-all
    strategy adds no benefit in practice.

    V24 reduced init_min_num_inliers from 5 to 3 for video scenes but smoke_errored before
    the SfM step could run, leaving the combined effect with V25's merge-all untested.
    The root cause of V24's smoke_error is not definitively known; the pycolmap parameter
    itself (init_min_num_inliers=3) is a valid IncrementalMapperOptions field.

    V26 combines both changes: init_min_num_inliers=3 for video (V24) + merge all
    sub-reconstructions (V25). With a lower seed-pair inlier floor, COLMAP has more valid
    seed pair candidates available and is more likely to: (a) successfully initialize in
    scenes where the best available seed pair produces only 3-4 geometric inliers, and
    (b) produce additional sub-reconstructions from image clusters that are currently
    rejected at initialization. These additional sub-reconstructions are then captured by
    V26's merge-all strategy. Together, this targets the initialization bottleneck that
    V21 alone could not address, and gives V25's sub-reconstruction merger something to
    collect beyond the single current model.

    Scene clustering uses V16 (GBM num_gap=0 + Lowe threshold=30 for video), unchanged.
    SfM improvements are in run_sfm_v26. Non-video and small datasets use standard V9
    logic, unaffected.
    """
    return discover_scenes_v21(
        image_paths,
        lgeu_model_payload=lgeu_model_payload,
        verbose=verbose,
    )


def evaluate_clustering(
    image_paths: dict[str, Path],
    ground_truth: dict[str, str],
    good_match_threshold: int = GOOD_MATCH_THRESHOLD,
    verbose: bool = True,
) -> dict:
    """Evaluate scene discovery accuracy against ground-truth labels."""
    scene_map, edge_data = discover_scenes(image_paths, good_match_threshold=good_match_threshold, verbose=verbose)

    names = sorted(image_paths)
    correct_same = total_same = correct_diff = total_diff = 0
    for na, nb in itertools.combinations(names, 2):
        gt_same = ground_truth.get(na) == ground_truth.get(nb)
        pred_same = scene_map[na] == scene_map[nb]
        if gt_same:
            total_same += 1
            if pred_same:
                correct_same += 1
        else:
            total_diff += 1
            if not pred_same:
                correct_diff += 1

    return {
        "scene_map": scene_map,
        "edge_data": edge_data,
        "same_scene_recall": round(correct_same / total_same, 4) if total_same else 0.0,
        "diff_scene_precision": round(correct_diff / total_diff, 4) if total_diff else 0.0,
        "total_same_pairs": total_same,
        "total_diff_pairs": total_diff,
        "correct_same": correct_same,
        "correct_diff": correct_diff,
    }
