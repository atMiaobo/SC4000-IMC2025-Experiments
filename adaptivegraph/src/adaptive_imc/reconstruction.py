from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
from collections import defaultdict

import numpy as np
import networkx as nx


@dataclass
class PosePrediction:
    scene: str
    rotation: np.ndarray | None
    translation: np.ndarray | None


class _KeypointBank:
    def __init__(self, quant: float = 0.5):
        self.quant = float(quant)
        self.points: list[list[float]] = []
        self.lookup: dict[tuple[int, int], int] = {}

    def add(self, p: np.ndarray) -> int:
        x, y = float(p[0]), float(p[1])
        key = (int(round(x / self.quant)), int(round(y / self.quant)))
        idx = self.lookup.get(key)
        if idx is None:
            idx = len(self.points)
            self.lookup[key] = idx
            self.points.append([x, y])
        return idx

    def array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float32).reshape(-1, 2)


def _safe_stage_images(image_paths: list[Path], stage_dir: Path) -> tuple[list[str], dict[str, int]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    name_to_idx: dict[str, int] = {}
    for idx, src in enumerate(image_paths):
        staged = f"{idx:05d}__{src.name}"
        dst = stage_dir / staged
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
        names.append(staged)
        name_to_idx[staged] = idx
    return names, name_to_idx


def reconstruct_with_colmap(
    *,
    image_paths: list[Path],
    accepted_results: dict[tuple[int, int], object],
    match_graph: nx.Graph,
    work_dir: Path,
    cfg,
    enable_label_propagation: bool = True,
) -> tuple[dict[int, PosePrediction], dict[str, object]]:
    """Import our correspondences into COLMAP, verify, and reconstruct.

    `accepted_results[(i,j)]` must expose `.geom.pts0`, `.geom.pts1` and `.quality`.
    """
    import pycolmap

    work_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = work_dir / "images"
    db_path = work_dir / "database.db"
    sfm_dir = work_dir / "sfm"
    pairs_path = work_dir / "pairs.txt"

    if db_path.exists():
        db_path.unlink()
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    staged_names, staged_to_idx = _safe_stage_images(image_paths, stage_dir)

    # PyCOLMAP import_images expects an initialized database.
    with pycolmap.Database.open(db_path):
        pass

    camera_mode = getattr(pycolmap.CameraMode, str(cfg.camera_mode))
    reader_opts = pycolmap.ImageReaderOptions(camera_model=str(cfg.camera_model))
    pycolmap.import_images(
        db_path,
        stage_dir,
        camera_mode=camera_mode,
        image_names=staged_names,
        options=reader_opts,
    )

    with pycolmap.Database.open(db_path) as db:
        db_images = db.read_all_images()
        stage_name_to_id = {img.name: int(img.image_id) for img in db_images}

        banks = [_KeypointBank(quant=0.5) for _ in image_paths]
        pair_matches: dict[tuple[int, int], np.ndarray] = {}

        for (i, j), result in accepted_results.items():
            pts0 = np.asarray(result.geom.pts0, dtype=np.float32)
            pts1 = np.asarray(result.geom.pts1, dtype=np.float32)
            rows = []
            for p0, p1 in zip(pts0, pts1):
                rows.append([banks[i].add(p0), banks[j].add(p1)])
            pair_matches[(i, j)] = np.asarray(rows, dtype=np.uint32).reshape(-1, 2)

        for idx, stage_name in enumerate(staged_names):
            image_id = stage_name_to_id[stage_name]
            kpts = banks[idx].array()
            db.write_keypoints(image_id, kpts)
            # Descriptors are not used because correspondences are already imported,
            # but a descriptor row keeps the database complete for COLMAP tooling.
            desc = np.zeros((len(kpts), 128), dtype=np.uint8)
            db.write_descriptors(image_id, desc)

        pair_lines = []
        for (i, j), matches in pair_matches.items():
            if len(matches) == 0:
                continue
            n0, n1 = staged_names[i], staged_names[j]
            id0, id1 = stage_name_to_id[n0], stage_name_to_id[n1]
            if id0 <= id1:
                db.write_matches(id0, id1, matches)
            else:
                db.write_matches(id1, id0, matches[:, ::-1].copy())
            pair_lines.append(f"{n0} {n1}")

    pairs_path.write_text("\n".join(pair_lines) + ("\n" if pair_lines else ""))

    if pair_lines:
        verify_opts = pycolmap.TwoViewGeometryOptions(
            {
                "min_num_inliers": int(cfg.verify_min_num_inliers),
                "ransac": {
                    "max_error": float(cfg.verify_max_error),
                    "min_inlier_ratio": 0.05,
                    "confidence": 0.999,
                    "max_num_trials": 10000,
                },
            }
        )
        pycolmap.verify_matches(db_path, pairs_path, options=verify_opts)

    pipeline_opts = pycolmap.IncrementalPipelineOptions(
        {
            "min_num_matches": int(cfg.min_num_matches),
            "multiple_models": True,
            "max_num_models": int(cfg.max_num_models),
            "max_model_overlap": int(cfg.max_model_overlap),
            "min_model_size": int(cfg.min_model_size),
            "num_threads": int(cfg.num_threads),
            "mapper": {
                "init_min_num_inliers": int(cfg.init_min_num_inliers),
                "abs_pose_min_num_inliers": int(cfg.abs_pose_min_num_inliers),
                "abs_pose_min_inlier_ratio": float(cfg.abs_pose_min_inlier_ratio),
            },
        }
    )

    maps = {}
    if pair_lines:
        try:
            maps = pycolmap.incremental_mapping(
                db_path,
                stage_dir,
                sfm_dir,
                options=pipeline_opts,
            )
        except Exception as exc:
            print(f"[COLMAP] incremental_mapping failed: {exc}")
            maps = {}

    predictions: dict[int, PosePrediction] = {
        i: PosePrediction(scene="outliers", rotation=None, translation=None)
        for i in range(len(image_paths))
    }

    # Register smaller models first, then let larger/more robust models overwrite.
    model_items = []
    for map_id, rec in maps.items():
        try:
            nreg = int(rec.num_reg_images())
        except Exception:
            nreg = len(rec.images)
        model_items.append((nreg, map_id, rec))
    model_items.sort(key=lambda x: x[0])

    registered_cluster: dict[int, str] = {}
    model_summaries = []
    cluster_counter = 0
    for nreg, map_id, rec in model_items:
        cluster_name = f"cluster{cluster_counter}"
        cluster_counter += 1
        model_summaries.append({"map_id": int(map_id), "registered": nreg, "cluster": cluster_name})
        for _, img in rec.images.items():
            stage_name = img.name
            if stage_name not in staged_to_idx:
                continue
            idx = staged_to_idx[stage_name]
            if not getattr(img, "has_pose", True):
                continue
            pose = img.cam_from_world()
            R = np.asarray(pose.rotation.matrix(), dtype=np.float64)
            t = np.asarray(pose.translation, dtype=np.float64).reshape(3)
            predictions[idx] = PosePrediction(cluster_name, R, t)
            registered_cluster[idx] = cluster_name

    # Graph-propagated labels: useful because the IMC metric also rewards correct
    # clustering even when a pose could not be registered. Disabled for baseline
    # ablations, in which unregistered images remain "outliers".
    propagation_components = list(nx.connected_components(match_graph)) if enable_label_propagation else []
    for comp in propagation_components:
        comp = set(comp)
        reg_labels = {registered_cluster[i] for i in comp if i in registered_cluster}
        if len(reg_labels) == 0:
            if len(comp) >= 2:
                label = f"cluster{cluster_counter}"
                cluster_counter += 1
                for i in comp:
                    predictions[i] = PosePrediction(label, None, None)
            continue
        if len(reg_labels) == 1:
            label = next(iter(reg_labels))
            for i in comp:
                if i not in registered_cluster:
                    predictions[i] = PosePrediction(label, None, None)
            continue

        # Component contains multiple reconstructed models. Assign each unregistered
        # image to the model receiving the strongest graph-edge support.
        for i in comp:
            if i in registered_cluster:
                continue
            votes = defaultdict(float)
            for j in match_graph.neighbors(i):
                label = registered_cluster.get(j)
                if label is not None:
                    votes[label] += float(match_graph[i][j].get("quality", 1.0))
            if votes:
                label = max(votes.items(), key=lambda kv: kv[1])[0]
                predictions[i] = PosePrediction(label, None, None)

    diagnostics = {
        "num_models": len(model_items),
        "models": model_summaries,
        "num_registered": sum(p.rotation is not None for p in predictions.values()),
        "num_images": len(image_paths),
        "database": str(db_path),
        "sfm_dir": str(sfm_dir),
    }
    return predictions, diagnostics
