"""Multiview reconstruction using PyCOLMAP incremental SfM.

Workflow per scene:
  1. Write images to a temp directory structure.
  2. Extract SIFT features into a COLMAP database.
  3. Run exhaustive matching with geometric verification.
  4. Run incremental mapping (triangulation + bundle adjustment).
  5. Extract world-to-camera poses (R, t) from the reconstruction.

Fallback: if COLMAP fails to register >= 2 cameras, return identity poses.
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

try:
    import pycolmap
    _HAS_PYCOLMAP = True
except ImportError:
    _HAS_PYCOLMAP = False


def _identity_pose() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)


def run_sfm(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Run incremental SfM on a set of images forming one scene.

    Args:
        image_paths: mapping from image filename (relative) to absolute Path.
        verbose: print progress messages.

    Returns:
        poses: dict mapping image filename -> (R [3x3], t [3]) world-to-camera.
        info: dict with registration statistics.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap not available"}

    if len(image_paths) < 2:
        return {n: _identity_pose() for n in image_paths}, {"registered": 0, "total": len(image_paths)}

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_"))
    try:
        # COLMAP expects all images to be physically present in one directory.
        # We symlink or copy them into tmpdir/images/.
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        # COLMAP uses the bare filename as the image name in the database.
        # We must avoid filename collisions (same basename from different paths).
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name  # use the base filename
            dst = img_dir / stem
            if not dst.exists():
                shutil.copy2(src_path, dst)
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        if verbose:
            print(f"  [SfM] extracting features for {len(image_paths)} images ...")

        pycolmap.extract_features(
            database_path=db_path,
            image_path=img_dir,
            camera_mode=pycolmap.CameraMode.AUTO,
        )

        if verbose:
            print("  [SfM] exhaustive matching ...")

        pycolmap.match_exhaustive(database_path=db_path)

        if verbose:
            print("  [SfM] incremental mapping ...")

        # Relax mapper thresholds to handle sequences with limited overlap or small baseline.
        # Default init_min_num_inliers=100 and init_min_tri_angle=16° are too strict for
        # close-range video sequences (e.g. stairs, where only 15-50 inliers are typical).
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.mapper.init_min_num_inliers = 10
        pipeline_options.mapper.init_min_tri_angle = 2.0  # allow small-baseline initialization
        pipeline_options.mapper.init_max_forward_motion = 0.99  # allow near-forward motion
        pipeline_options.mapper.abs_pose_min_num_inliers = 8

        reconstructions = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=img_dir,
            output_path=out_path,
            options=pipeline_options,
        )

        if not reconstructions:
            if verbose:
                print("  [SfM] mapping produced no reconstruction — using identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": len(image_paths), "error": "no_reconstruction"},
            )

        # Take the largest reconstruction (most registered images)
        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())

        # Build reverse map: colmap stem -> original image name
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for img_id in best_rec.reg_image_ids():
            colmap_img = best_rec.image(img_id)
            stem = Path(colmap_img.name).name
            orig_name = stem_to_name.get(stem)
            if orig_name is None:
                continue
            # cam_from_world() returns a Rigid3d (world-to-camera transform)
            cam_from_world = colmap_img.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            poses[orig_name] = (
                np.array(R, dtype=np.float64),
                np.array(t, dtype=np.float64),
            )

        # Images not registered get identity pose
        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": len(image_paths),
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
        }
        if verbose:
            print(f"  [SfM] registered {info['registered']}/{info['total']} cameras, "
                  f"{info['num_points3D']} 3D points")

        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Run SfM independently per scene, merging results.

    Args:
        image_paths: all images in collection.
        scene_map: mapping image_name -> scene_label (from scene_clustering.discover_scenes).
        verbose: print progress.

    Returns:
        poses: all images -> (R, t)
        info: per-scene registration stats
    """
    from collections import defaultdict
    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM] scene '{label}': {len(group_paths)} images")
        scene_poses, scene_info = run_sfm(group_paths, verbose=verbose)
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    # Any image not covered gets identity
    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


# ---------------------------------------------------------------------------
# v5: video-sequence detection + sequential matching
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"\d{10,13}")  # 10-13 digit timestamp in filename


def _is_video_sequence(names: list[str], min_frac: float = 0.7) -> bool:
    """Return True if most filenames look like timestamped video frames."""
    n_match = sum(1 for n in names if _TIMESTAMP_RE.search(n))
    return len(names) >= 3 and n_match / len(names) >= min_frac


def _sorted_by_timestamp(names: list[str]) -> list[str]:
    """Sort image names by the first embedded timestamp (ascending)."""
    def _key(n: str) -> int:
        m = _TIMESTAMP_RE.search(n)
        return int(m.group()) if m else 0
    return sorted(names, key=_key)


def run_sfm_v5(
    image_paths: dict[str, Path],
    verbose: bool = False,
    window_radius: int = 5,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v5 SfM: uses sequential matching for video sequences, exhaustive otherwise.

    Sequential matching matches each frame against its ``window_radius`` nearest
    temporal neighbours instead of all pairs.  This is far faster for video
    sequences (e.g. stairs) and allows COLMAP to initialise on consecutive frames
    with small but reliable baselines.

    Args:
        image_paths: mapping from image filename -> absolute Path.
        verbose: print progress messages.
        window_radius: sequential matching window (frames before/after each image).

    Returns:
        Same as run_sfm(): (poses dict, info dict).
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap not available"}

    if len(image_paths) < 2:
        return {n: _identity_pose() for n in image_paths}, {"registered": 0, "total": len(image_paths)}

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v5_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                shutil.copy2(src_path, dst)
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        if verbose:
            seq_str = " [sequential]" if is_video else " [exhaustive]"
            print(f"  [SfM-v5] extracting features for {len(image_paths)} images{seq_str} ...")

        pycolmap.extract_features(
            database_path=db_path,
            image_path=img_dir,
            camera_mode=pycolmap.CameraMode.AUTO,
        )

        if is_video:
            if verbose:
                print(f"  [SfM-v5] sequential matching (overlap={window_radius}) ...")
            pairing_opts = pycolmap.SequentialPairingOptions()
            pairing_opts.overlap = window_radius
            pairing_opts.loop_detection = False
            pycolmap.match_sequential(
                database_path=db_path,
                pairing_options=pairing_opts,
            )
        else:
            if verbose:
                print("  [SfM-v5] exhaustive matching ...")
            pycolmap.match_exhaustive(database_path=db_path)

        if verbose:
            print("  [SfM-v5] incremental mapping ...")

        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            # Video sequences: very relaxed init thresholds
            pipeline_options.mapper.init_min_num_inliers = 5
            pipeline_options.mapper.init_min_tri_angle = 1.0
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 4
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        reconstructions = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=img_dir,
            output_path=out_path,
            options=pipeline_options,
        )

        if not reconstructions:
            if verbose:
                print("  [SfM-v5] mapping produced no reconstruction — using identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": len(image_paths), "error": "no_reconstruction",
                 "is_video": is_video},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for img_id in best_rec.reg_image_ids():
            colmap_img = best_rec.image(img_id)
            stem = Path(colmap_img.name).name
            orig_name = stem_to_name.get(stem)
            if orig_name is None:
                continue
            cam_from_world = colmap_img.cam_from_world()
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            poses[orig_name] = (np.array(R, dtype=np.float64), np.array(t, dtype=np.float64))

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": len(image_paths),
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
        }
        if verbose:
            print(f"  [SfM-v5] registered {info['registered']}/{info['total']} cameras, "
                  f"{info['num_points3D']} 3D pts")
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v5(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
    window_radius: int = 5,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v5 per-scene SfM using run_sfm_v5 (sequential matching for video scenes)."""
    from collections import defaultdict
    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v5] scene '{label}': {len(group_paths)} images")
        scene_poses, scene_info = run_sfm_v5(group_paths, verbose=verbose, window_radius=window_radius)
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


# ---------------------------------------------------------------------------
# v6: all pycolmap operations wrapped in try/except; no exception can escape
# ---------------------------------------------------------------------------


def run_sfm_v6(
    image_paths: dict[str, Path],
    verbose: bool = False,
    window_radius: int = 5,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v6 SfM: same as v5 with every pycolmap call guarded.

    No exception propagates out.  On any COLMAP failure the function returns
    identity poses for all images in the scene plus an ``error`` key in info.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v6_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v6] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": len(image_paths), "error": reason, "is_video": is_video},
            )

        # Feature extraction
        if verbose:
            tag = " [sequential]" if is_video else " [exhaustive]"
            print(f"  [SfM-v6] extracting features for {len(image_paths)} images{tag} ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        # Matching
        try:
            if is_video:
                if verbose:
                    print(f"  [SfM-v6] sequential matching (overlap={window_radius}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = window_radius
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
            else:
                if verbose:
                    print("  [SfM-v6] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        # Incremental mapping
        if verbose:
            print("  [SfM-v6] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            pipeline_options.mapper.init_min_num_inliers = 5
            pipeline_options.mapper.init_min_tri_angle = 1.0
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 4
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            if verbose:
                print("  [SfM-v6] mapping produced no reconstruction — using identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": len(image_paths), "error": "no_reconstruction",
                 "is_video": is_video},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for img_id in best_rec.reg_image_ids():
                colmap_img = best_rec.image(img_id)
                stem = Path(colmap_img.name).name
                orig_name = stem_to_name.get(stem)
                if orig_name is None:
                    continue
                cam_from_world = colmap_img.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                poses[orig_name] = (
                    np.array(R, dtype=np.float64),
                    np.array(t, dtype=np.float64),
                )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": len(image_paths),
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
        }
        if verbose:
            print(
                f"  [SfM-v6] registered {info['registered']}/{info['total']} cameras, "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v6(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
    window_radius: int = 5,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v6 per-scene SfM: uses run_sfm_v6; catches any exception per scene."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v6] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v6(
                group_paths, verbose=verbose, window_radius=window_radius
            )
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v6] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


# ---------------------------------------------------------------------------
# v7: exhaustive matching for small video sequences; larger overlap otherwise
# ---------------------------------------------------------------------------


def run_sfm_v7(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v7 SfM: exhaustive COLMAP matching for video sequences with N <= 100.

    V6 used sequential matching with overlap=5 for all video sequences.  This
    caused stairs (N=51) to regress from V4's 2/51 to 0/51 because sequential
    matching with small overlap doesn't supply enough candidate initialization
    pairs.  V7 restores exhaustive matching for video N <= 100 (max 4950 pairs
    — well within COLMAP's capacity).  For larger video sequences (N > 100)
    sequential matching is used with overlap=max(15, N//4) instead of the
    V6 fixed overlap of 5.

    All robustness features from V6 (per-call try/except, identity fallbacks,
    informative error keys) are preserved.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v7_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v7] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            print(f"  [SfM-v7] extracting features for {N} images [{match_mode}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v7] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v7] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v7] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            pipeline_options.mapper.init_min_num_inliers = 5
            pipeline_options.mapper.init_min_tri_angle = 1.0
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 4
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for img_id in best_rec.reg_image_ids():
                colmap_img = best_rec.image(img_id)
                stem = Path(colmap_img.name).name
                orig_name = stem_to_name.get(stem)
                if orig_name is None:
                    continue
                cam_from_world = colmap_img.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                poses[orig_name] = (
                    np.array(R, dtype=np.float64),
                    np.array(t, dtype=np.float64),
                )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v7] registered {info['registered']}/{info['total']} cameras, "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v7(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v7 per-scene SfM: uses run_sfm_v7; catches any exception per scene."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v7] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v7(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v7] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


# v9 SfM is identical to v7 — improvements are in scene clustering (TOPK=20)
run_sfm_per_scene_v9 = run_sfm_per_scene_v7

# v10 SfM identical to v7 — uses TOPK=25 scene clustering; SfM unchanged
run_sfm_per_scene_v10 = run_sfm_per_scene_v7

# v11 SfM identical to v7 — uses lower Lowe threshold in scene clustering; SfM unchanged
run_sfm_per_scene_v11 = run_sfm_per_scene_v7

# v12 SfM identical to v7 — uses video-aware temporal pair augmentation in scene clustering
run_sfm_per_scene_v12 = run_sfm_per_scene_v7

# v13 SfM identical to v7 — uses cosine-only pair ranking for video sequences in scene clustering
run_sfm_per_scene_v13 = run_sfm_per_scene_v7

# v14 SfM identical to v7 — uses GBM LGEU with num_gap neutralized for video sequences
run_sfm_per_scene_v14 = run_sfm_per_scene_v7

# v15 SfM identical to v7 — uses lower Lowe threshold (30) for video sequences
run_sfm_per_scene_v15 = run_sfm_per_scene_v7

# v16 SfM identical to v7 — combines num_gap=0 + threshold=30 for video pair selection
run_sfm_per_scene_v16 = run_sfm_per_scene_v7

# v17 SfM identical to v7 — uses nfeatures=8000 + num_gap=0 + threshold=30 for video SIFT
run_sfm_per_scene_v17 = run_sfm_per_scene_v7


def run_sfm_v18(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v18 SfM: relaxed COLMAP mapper settings for video scenes to improve registration.

    V7–V17 all use run_sfm_v7, which sets abs_pose_min_num_inliers=4 and
    init_min_tri_angle=1.0° for video scenes. For forward-facing staircase video,
    the initial 3D reconstruction is sparse (built from 2 seed frames with limited
    baseline), so subsequent frames often have fewer than 4 3D-point correspondences
    with the existing model — causing the incremental mapper to skip them even when
    they have real geometric overlap with their temporal neighbors. The triangulation
    angle threshold of 1.0° also rejects valid seed pairs with nearly-pure forward
    motion.

    V18 relaxes all three bottleneck thresholds for video scenes:
    - init_min_num_inliers: 5 → 3 (easier seed-pair selection)
    - init_min_tri_angle:   1.0 → 0.5 degrees (allows nearly-forward seed pairs)
    - abs_pose_min_num_inliers: 4 → 3 (P3P minimum; admits sparser frame overlap)

    Non-video scenes (is_video=False) retain V7's settings unchanged to avoid
    regression on ETs and hidden structured-capture datasets.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v18_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v18] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            print(f"  [SfM-v18] extracting features for {N} images [{match_mode}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v18] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v18] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v18] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            # Relaxed thresholds for forward-facing video with sparse initial reconstruction
            pipeline_options.mapper.init_min_num_inliers = 3   # was 5 in v7
            pipeline_options.mapper.init_min_tri_angle = 0.5   # was 1.0 in v7 (degrees)
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 3  # was 4 in v7 (P3P minimum)
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for img_id in best_rec.reg_image_ids():
                colmap_img = best_rec.image(img_id)
                stem = Path(colmap_img.name).name
                orig_name = stem_to_name.get(stem)
                if orig_name is None:
                    continue
                cam_from_world = colmap_img.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                poses[orig_name] = (
                    np.array(R, dtype=np.float64),
                    np.array(t, dtype=np.float64),
                )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v18] registered {info['registered']}/{info['total']} cameras, "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# v19 SfM: uses run_sfm_v18 relaxed mapper — untested on leaderboard (V18 upload_failed)
# Combines with V19's looser Lowe ratio (0.80) for video pair matching in clustering.


def run_sfm_per_scene_v18(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v18 per-scene SfM: uses run_sfm_v18; relaxed COLMAP mapper for video scenes."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v18] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v18(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v18] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


# v19 SfM: same relaxed mapper as V18 (untested on leaderboard — V18 upload_failed)
run_sfm_per_scene_v19 = run_sfm_per_scene_v18

# v20 SfM: identical to v7 — improvement is in outlier reassignment in scene clustering
run_sfm_per_scene_v20 = run_sfm_per_scene_v7


def run_sfm_v21(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v21 SfM: abs_pose_min_num_inliers relaxed from 4 to 3 for video scenes only.

    V7-V20 (except upload_failed V18/V19) use abs_pose_min_num_inliers=4 for video
    scenes. For forward-facing staircase video, the incremental mapper initializes
    from a sparse seed pair and then tries to register additional frames. Consecutive
    frames often have only 3 inlier correspondences with the existing sparse point cloud
    — the theoretical P3P minimum — so COLMAP rejects them with the threshold of 4,
    leaving scene_0 (15 imgs) and scene_2 (20 imgs) at only 3 cameras each.

    V21 lowers this threshold to 3 (P3P minimum) for video scenes, allowing cameras
    to register with sparse-but-real geometric overlap. V18 changed three mapper
    parameters simultaneously and upload_failed; V21 isolates only this one parameter
    to cleanly attribute any effect and reduce risk.

    Non-video scenes retain V7's settings unchanged (abs_pose_min_num_inliers=8),
    protecting ETs and hidden structured-capture datasets from regression.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v21_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v21] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            print(f"  [SfM-v21] extracting features for {N} images [{match_mode}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v21] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v21] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v21] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            pipeline_options.mapper.init_min_num_inliers = 5
            pipeline_options.mapper.init_min_tri_angle = 1.0
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 3  # 4→3: P3P minimum for sparse video
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for img_id in best_rec.reg_image_ids():
                colmap_img = best_rec.image(img_id)
                stem = Path(colmap_img.name).name
                orig_name = stem_to_name.get(stem)
                if orig_name is None:
                    continue
                cam_from_world = colmap_img.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                poses[orig_name] = (
                    np.array(R, dtype=np.float64),
                    np.array(t, dtype=np.float64),
                )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v21] registered {info['registered']}/{info['total']} cameras, "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v21(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v21 per-scene SfM: uses run_sfm_v21 (abs_pose_min_num_inliers=3 for video)."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v21] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v21(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v21] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


def run_sfm_v22(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v22 SfM: V21 + init_min_tri_angle relaxed from 1.0 to 0.5 degrees for video scenes.

    V21 lowered abs_pose_min_num_inliers to 3 (P3P minimum) for video scenes and tied
    V7 at 8/51 stairs. The remaining untried parameter from V18 (3-param bundle that
    upload_failed) is init_min_tri_angle. V21 keeps this at 1.0 degrees (V7 video default).
    For near-forward-facing staircase cameras, valid seed pairs often have a triangulation
    angle below 1.0 degrees, causing COLMAP to reject them and fall back to a geometrically
    worse seed pair — or fail initialisation entirely. Lowering to 0.5 degrees allows
    COLMAP to select a better seed pair, yielding a denser initial point cloud and enabling
    more cameras to be registered in the extension phase.

    V22 keeps abs_pose_min_num_inliers=3 from V21 and adds init_min_tri_angle=0.5 degrees
    for video scenes only. Non-video scenes retain V7's settings (abs_pose_min_num_inliers=8,
    init_min_tri_angle=2.0), protecting ETs and hidden structured-capture datasets.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v22_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v22] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            print(f"  [SfM-v22] extracting features for {N} images [{match_mode}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v22] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v22] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v22] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            pipeline_options.mapper.init_min_num_inliers = 5
            pipeline_options.mapper.init_min_tri_angle = 0.5   # 1.0→0.5: allow small-baseline seed pairs
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 3  # kept from V21: P3P minimum
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for img_id in best_rec.reg_image_ids():
                colmap_img = best_rec.image(img_id)
                stem = Path(colmap_img.name).name
                orig_name = stem_to_name.get(stem)
                if orig_name is None:
                    continue
                cam_from_world = colmap_img.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                poses[orig_name] = (
                    np.array(R, dtype=np.float64),
                    np.array(t, dtype=np.float64),
                )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v22] registered {info['registered']}/{info['total']} cameras, "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v22(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v22 per-scene SfM: uses run_sfm_v22 (abs_pose=3 + init_min_tri_angle=0.5 for video)."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v22] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v22(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v22] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


def run_sfm_v23(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v23 SfM: CameraMode.SINGLE for video scenes; V21 mapper settings (abs_pose=3).

    All prior versions (V7–V22) use CameraMode.AUTO, which assigns independent camera
    intrinsics to each image. For staircase video, every frame comes from the same
    physical camera with fixed focal length and principal point. AUTO mode estimates
    intrinsics per image from a sparse seed pair, producing poorly constrained per-frame
    estimates and an inconsistent 3D point cloud. This limits how many subsequent frames
    find the 3 inlier correspondences required by abs_pose_min_num_inliers=3 (V21).

    V23 switches to CameraMode.SINGLE for video scenes: COLMAP shares one camera model
    across all frames and jointly refines intrinsics during bundle adjustment. A more
    accurately calibrated reconstruction with tighter 3D point positions increases the
    probability that adjacent forward-facing frames accumulate 3+ inlier correspondences
    with the existing sparse model, enabling more cameras to be registered.

    Mapper settings are kept identical to V21 (abs_pose_min_num_inliers=3,
    init_min_tri_angle=1.0 — V22's 0.5 value caused regression and is not carried over).
    Non-video scenes retain AUTO mode and V7 mapper settings, protecting ETs and hidden
    structured-capture datasets from regression.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v23_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"
        camera_mode = pycolmap.CameraMode.SINGLE if is_video else pycolmap.CameraMode.AUTO

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v23] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            cam_label = "SINGLE" if is_video else "AUTO"
            print(f"  [SfM-v23] extracting features for {N} images [{match_mode}, camera={cam_label}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=camera_mode,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v23] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v23] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v23] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            pipeline_options.mapper.init_min_num_inliers = 5
            pipeline_options.mapper.init_min_tri_angle = 1.0   # V21 value; V22's 0.5 caused regression
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 3  # from V21: P3P minimum
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for img_id in best_rec.reg_image_ids():
                colmap_img = best_rec.image(img_id)
                stem = Path(colmap_img.name).name
                orig_name = stem_to_name.get(stem)
                if orig_name is None:
                    continue
                cam_from_world = colmap_img.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                poses[orig_name] = (
                    np.array(R, dtype=np.float64),
                    np.array(t, dtype=np.float64),
                )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v23] registered {info['registered']}/{info['total']} cameras, "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_v24(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v24 SfM: V21 settings + init_min_num_inliers=3 for video scenes.

    V21 lowered abs_pose_min_num_inliers to 3 (P3P minimum) for video scenes and tied V7
    at 8/51 stairs. V18 changed three parameters simultaneously (init_min_num_inliers 5→3,
    init_min_tri_angle 1.0→0.5, abs_pose_min_num_inliers 4→3) and upload_failed. V21
    isolated abs_pose_min_num_inliers and confirmed it helps. V22 isolated init_min_tri_angle
    on top and regressed. The third parameter from V18 that has never been isolated is
    init_min_num_inliers (seed-pair selection floor).

    For staircase video with nearly-parallel optical axes and limited scene extent, even the
    geometrically best available seed pair may produce only 3–5 verified fundamental matrix
    inliers — causing COLMAP (with init_min_num_inliers=5) to skip it in favour of a worse
    alternative or fail initialization entirely. V24 reduces init_min_num_inliers from 5 to 3
    (the P3P minimum, same floor as abs_pose). Combined with abs_pose_min_num_inliers=3
    (from V21), this lowers both the seed-pair acceptance bar and the pose registration bar.
    init_min_tri_angle is kept at 1.0 degrees (V7 video default), NOT 0.5 as in V22
    (which caused regression and is not carried over).

    Non-video scenes retain V7 settings (init_min_num_inliers=10, abs_pose=8), protecting
    ETs and hidden structured-capture datasets from regression.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v24_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v24] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            print(f"  [SfM-v24] extracting features for {N} images [{match_mode}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v24] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v24] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v24] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            pipeline_options.mapper.init_min_num_inliers = 3   # 5→3: P3P minimum for seed-pair selection
            pipeline_options.mapper.init_min_tri_angle = 1.0   # V7 default; NOT 0.5 (V22 caused regression)
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 3  # from V21: P3P minimum for registration
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        stem_to_name = {Path(n).name: n for n in image_paths}

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for img_id in best_rec.reg_image_ids():
                colmap_img = best_rec.image(img_id)
                stem = Path(colmap_img.name).name
                orig_name = stem_to_name.get(stem)
                if orig_name is None:
                    continue
                cam_from_world = colmap_img.cam_from_world()
                R = cam_from_world.rotation.matrix()
                t = cam_from_world.translation
                poses[orig_name] = (
                    np.array(R, dtype=np.float64),
                    np.array(t, dtype=np.float64),
                )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        info = {
            "registered": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v24] registered {info['registered']}/{info['total']} cameras, "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v24(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v24 per-scene SfM: uses run_sfm_v24 (init_min_num_inliers=3 + abs_pose=3 for video)."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v24] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v24(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v24] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


def run_sfm_v25(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v25 SfM: merge poses from ALL COLMAP sub-reconstructions, not just the largest.

    All prior versions (V7–V24) call incremental_mapping() and then select only the largest
    reconstruction:
        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    This silently discards every smaller sub-reconstruction. For staircase video recorded by
    two physically separate camera rigs, COLMAP produces multiple disjoint models — one per
    rig cluster — because the rigs share no geometric overlap. The current code registers only
    the cameras in the single largest model and loses the rest.

    V25 iterates over ALL reconstructions sorted by descending registration count and merges
    their poses into one dict. For any image registered in multiple sub-reconstructions, the
    pose from the largest (most constrained) reconstruction wins. All other V21 mapper
    settings are preserved (abs_pose_min_num_inliers=3 for video, init_min_tri_angle=1.0,
    init_min_num_inliers=5). Non-video scenes retain V7 mapper settings.
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v25_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v25] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            print(f"  [SfM-v25] extracting features for {N} images [{match_mode}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v25] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v25] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v25] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            # V21 mapper settings (abs_pose=3 for P3P minimum; V22's init_min_tri_angle=0.5 caused regression)
            pipeline_options.mapper.init_min_num_inliers = 5
            pipeline_options.mapper.init_min_tri_angle = 1.0
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 3
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        # V25 KEY CHANGE: merge poses from ALL reconstructions (largest first).
        # Prior versions only used the single largest reconstruction.
        stem_to_name = {Path(n).name: n for n in image_paths}
        sorted_recs = sorted(reconstructions.values(), key=lambda r: r.num_reg_images(), reverse=True)

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for rec in sorted_recs:
                for img_id in rec.reg_image_ids():
                    colmap_img = rec.image(img_id)
                    stem = Path(colmap_img.name).name
                    orig_name = stem_to_name.get(stem)
                    if orig_name is None or orig_name in poses:
                        # Skip unknowns and already-registered (largest rec wins on conflict)
                        continue
                    cam_from_world = colmap_img.cam_from_world()
                    R = cam_from_world.rotation.matrix()
                    t = cam_from_world.translation
                    poses[orig_name] = (
                        np.array(R, dtype=np.float64),
                        np.array(t, dtype=np.float64),
                    )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        best_rec = sorted_recs[0]
        registered_count = sum(
            1 for n in image_paths
            if n in poses and not (poses[n][0] == np.eye(3)).all()
        )
        info = {
            "registered": registered_count,
            "registered_largest": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v25] registered {registered_count}/{N} cameras "
                f"(largest: {info['registered_largest']}, recs: {len(reconstructions)}), "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v25(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v25 per-scene SfM: uses run_sfm_v25 (merge all COLMAP sub-reconstructions)."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v25] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v25(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v25] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


def run_sfm_v26(
    image_paths: dict[str, Path],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v26 SfM: merge ALL COLMAP sub-reconstructions + init_min_num_inliers=3 for video.

    V25 merged poses from all sub-reconstructions returned by incremental_mapping() and tied
    V7 at 8/51 stairs cameras. V24 tried init_min_num_inliers=3 for video scenes but
    smoke_errored before the SfM step could execute, leaving this parameter untested in
    combination with V25's merge-all strategy.

    V26 adds init_min_num_inliers=3 (seed-pair selection floor) on top of V25's approach.
    With fewer geometric inliers required for a seed pair to be accepted, COLMAP can
    initialize from pairs that were previously rejected — giving the incremental mapper more
    starting points and potentially producing additional sub-reconstructions that V26's
    merge-all strategy can then collect. This directly targets the scenario where V25's
    merge-all collected nothing because COLMAP produced only one sub-reconstruction per scene.

    Mapper settings for video (V26):
      init_min_num_inliers=3   (5→3: allows seed pairs with sparse-but-real F-matrix inliers)
      init_min_tri_angle=1.0   (V7 default; 0.5 caused regression in V22)
      abs_pose_min_num_inliers=3  (V21: P3P minimum for camera registration)
      init_max_forward_motion=0.999

    Non-video mapper settings are unchanged from V7 (abs_pose=8, init_min=10, tri_angle=2.0).
    """
    if not _HAS_PYCOLMAP:
        return {n: _identity_pose() for n in image_paths}, {"error": "pycolmap_not_available"}

    if len(image_paths) < 2:
        return (
            {n: _identity_pose() for n in image_paths},
            {"registered": 0, "total": len(image_paths)},
        )

    names = sorted(image_paths)
    is_video = _is_video_sequence(names)
    N = len(names)
    use_exhaustive = not is_video or N <= 100

    tmpdir = Path(tempfile.mkdtemp(prefix="colmap_sfm_v26_"))
    try:
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        name_to_stem: dict[str, str] = {}
        for img_name, src_path in image_paths.items():
            stem = Path(img_name).name
            dst = img_dir / stem
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                except Exception:
                    pass
            name_to_stem[img_name] = stem

        db_path = tmpdir / "database.db"
        out_path = tmpdir / "sparse"
        out_path.mkdir()

        match_mode = "exhaustive" if use_exhaustive else f"sequential(overlap={max(15, N // 4)})"

        def _identity_result(reason: str) -> tuple[dict, dict]:
            if verbose:
                print(f"  [SfM-v26] {reason} — falling back to identity poses")
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": reason, "is_video": is_video,
                 "match_mode": match_mode},
            )

        if verbose:
            print(f"  [SfM-v26] extracting features for {N} images [{match_mode}] ...")
        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=img_dir,
                camera_mode=pycolmap.CameraMode.AUTO,
            )
        except Exception as exc:
            return _identity_result(f"feature_extraction_failed:{type(exc).__name__}")

        try:
            if use_exhaustive:
                if verbose:
                    print("  [SfM-v26] exhaustive matching ...")
                pycolmap.match_exhaustive(database_path=db_path)
            else:
                overlap = max(15, N // 4)
                if verbose:
                    print(f"  [SfM-v26] sequential matching (overlap={overlap}) ...")
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = overlap
                if hasattr(pairing_opts, "loop_detection"):
                    pairing_opts.loop_detection = False
                pycolmap.match_sequential(
                    database_path=db_path,
                    pairing_options=pairing_opts,
                )
        except Exception as exc:
            return _identity_result(f"matching_failed:{type(exc).__name__}")

        if verbose:
            print("  [SfM-v26] incremental mapping ...")
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        if is_video:
            # V26: init_min_num_inliers=3 (lower seed-pair floor) + V21 abs_pose=3 + V25 merge-all
            pipeline_options.mapper.init_min_num_inliers = 3   # 5→3: allow sparse seed pairs
            pipeline_options.mapper.init_min_tri_angle = 1.0   # V7 default; 0.5 caused regression in V22
            pipeline_options.mapper.init_max_forward_motion = 0.999
            pipeline_options.mapper.abs_pose_min_num_inliers = 3  # V21: P3P minimum
        else:
            pipeline_options.mapper.init_min_num_inliers = 10
            pipeline_options.mapper.init_min_tri_angle = 2.0
            pipeline_options.mapper.init_max_forward_motion = 0.99
            pipeline_options.mapper.abs_pose_min_num_inliers = 8

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=img_dir,
                output_path=out_path,
                options=pipeline_options,
            )
        except Exception as exc:
            return _identity_result(f"mapping_failed:{type(exc).__name__}")

        if not reconstructions:
            return (
                {n: _identity_pose() for n in image_paths},
                {"registered": 0, "total": N, "error": "no_reconstruction",
                 "is_video": is_video, "match_mode": match_mode},
            )

        # V26 KEY CHANGE: merge poses from ALL reconstructions (largest first).
        # init_min_num_inliers=3 may produce additional sub-reconstructions that
        # were previously rejected at seed-pair selection; collect all of them.
        stem_to_name = {Path(n).name: n for n in image_paths}
        sorted_recs = sorted(reconstructions.values(), key=lambda r: r.num_reg_images(), reverse=True)

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for rec in sorted_recs:
                for img_id in rec.reg_image_ids():
                    colmap_img = rec.image(img_id)
                    stem = Path(colmap_img.name).name
                    orig_name = stem_to_name.get(stem)
                    if orig_name is None or orig_name in poses:
                        continue
                    cam_from_world = colmap_img.cam_from_world()
                    R = cam_from_world.rotation.matrix()
                    t = cam_from_world.translation
                    poses[orig_name] = (
                        np.array(R, dtype=np.float64),
                        np.array(t, dtype=np.float64),
                    )
        except Exception as exc:
            return _identity_result(f"pose_extraction_failed:{type(exc).__name__}")

        for name in image_paths:
            if name not in poses:
                poses[name] = _identity_pose()

        best_rec = sorted_recs[0]
        registered_count = sum(
            1 for n in image_paths
            if n in poses and not (poses[n][0] == np.eye(3)).all()
        )
        info = {
            "registered": registered_count,
            "registered_largest": best_rec.num_reg_images(),
            "total": N,
            "num_points3D": best_rec.num_points3D(),
            "num_reconstructions": len(reconstructions),
            "is_video": is_video,
            "match_mode": match_mode,
        }
        if verbose:
            print(
                f"  [SfM-v26] registered {registered_count}/{N} cameras "
                f"(largest: {info['registered_largest']}, recs: {len(reconstructions)}), "
                f"{info['num_points3D']} 3D pts"
            )
        return poses, info

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_sfm_per_scene_v26(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v26 per-scene SfM: uses run_sfm_v26 (init_min_num_inliers=3 + merge all sub-recs)."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v26] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v26(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v26] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info


def run_sfm_per_scene_v23(
    image_paths: dict[str, Path],
    scene_map: dict[str, str],
    verbose: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """v23 per-scene SfM: uses run_sfm_v23 (CameraMode.SINGLE + abs_pose=3 for video)."""
    from collections import defaultdict

    scene_groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for name, label in scene_map.items():
        if name in image_paths:
            scene_groups[label][name] = image_paths[name]

    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_info: dict[str, dict] = {}

    for label, group_paths in scene_groups.items():
        if verbose:
            print(f"  [SfM-v23] scene '{label}': {len(group_paths)} images")
        try:
            scene_poses, scene_info = run_sfm_v23(group_paths, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  [SfM-v23] scene '{label}' threw {type(exc).__name__}; using identity")
            scene_poses = {n: _identity_pose() for n in group_paths}
            scene_info = {
                "error": f"{type(exc).__name__}",
                "registered": 0,
                "total": len(group_paths),
            }
        all_poses.update(scene_poses)
        all_info[label] = scene_info

    for name in image_paths:
        if name not in all_poses:
            all_poses[name] = _identity_pose()

    return all_poses, all_info
