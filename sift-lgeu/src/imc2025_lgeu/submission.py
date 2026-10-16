from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

from .contract import format_rotation, format_translation, read_sample, validate_submission, write_submission
from .geometry import identity_pose
from .scene_clustering import (
    discover_scenes, discover_scenes_v5, discover_scenes_v6, discover_scenes_v7,
    discover_scenes_v9, discover_scenes_v10, discover_scenes_v11, discover_scenes_v12,
    discover_scenes_v13, discover_scenes_v14, discover_scenes_v15, discover_scenes_v16,
    discover_scenes_v17, discover_scenes_v18, discover_scenes_v19, discover_scenes_v20,
    discover_scenes_v21, discover_scenes_v22, discover_scenes_v23, discover_scenes_v24,
    discover_scenes_v25, discover_scenes_v26,
)
from .sfm import (
    run_sfm_per_scene, run_sfm_per_scene_v5, run_sfm_per_scene_v6, run_sfm_per_scene_v7,
    run_sfm_per_scene_v9, run_sfm_per_scene_v10, run_sfm_per_scene_v11, run_sfm_per_scene_v12,
    run_sfm_per_scene_v13, run_sfm_per_scene_v14, run_sfm_per_scene_v15, run_sfm_per_scene_v16,
    run_sfm_per_scene_v17, run_sfm_per_scene_v18, run_sfm_per_scene_v19, run_sfm_per_scene_v20,
    run_sfm_per_scene_v21, run_sfm_per_scene_v22, run_sfm_per_scene_v23, run_sfm_per_scene_v24,
    run_sfm_per_scene_v25, run_sfm_per_scene_v26,
)

# v8 aliases: same code as v7 but with upgraded GBM LGEU (AUC≈0.69) + LGEU_TOPK=30
discover_scenes_v8 = discover_scenes_v7


def build_submission(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    model_path: Path | None = None,
    max_edges_per_dataset: int = 120,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    sample = read_sample(sample_path)
    out_rows = []
    diagnostics: dict = {"datasets": {}}
    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        for row in group.itertuples(index=False):
            path = image_root / "test" / dataset / row.image
            if path.is_file():
                image_paths[row.image] = path

        if len(image_paths) >= 2:
            # Step 1: discover scenes from geometric verification (connected components)
            scene_map, _edge_data = discover_scenes(image_paths, verbose=verbose)
            # Step 2: run SfM per discovered scene
            poses, sfm_info = run_sfm_per_scene(image_paths, scene_map, verbose=verbose)
        else:
            scene_map = {img: "scene_0" for img in image_paths}
            poses = {img: identity_pose() for img in image_paths}
            sfm_info = {}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            # Use discovered scene label; fall back to "scene_0" for missing images
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "scene": predicted_scene,
                    "image": row.image,
                    "rotation_matrix": format_rotation(rotation),
                    "translation_vector": format_translation(translation),
                }
            )
    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v5(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v5 pipeline: LGEU-guided pair filtering + MAGSAC inlier edges + sequential SfM for video scenes.

    Args:
        sample_path: path to sample_submission.csv (read at runtime — fully dynamic).
        image_root: competition root directory (contains test/).
        output_path: where to write submission.csv.
        lgeu_model_payload: loaded from lgeu.joblib via lgeu.load_model().  When provided,
            used for pair pre-filtering on datasets with > MAX_EXHAUSTIVE_N images.
        diagnostics_path: optional JSON diagnostics dump.
        verbose: print progress.
    """
    sample = read_sample(sample_path)
    out_rows = []
    diagnostics: dict = {"datasets": {}, "version": 5}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        for row in group.itertuples(index=False):
            path = image_root / "test" / dataset / row.image
            if path.is_file():
                image_paths[row.image] = path

        if len(image_paths) >= 2:
            scene_map, _edge_data = discover_scenes_v5(
                image_paths,
                lgeu_model_payload=lgeu_model_payload,
                verbose=verbose,
            )
            poses, sfm_info = run_sfm_per_scene_v5(image_paths, scene_map, verbose=verbose)
        else:
            scene_map = {img: "scene_0" for img in image_paths}
            poses = {img: identity_pose() for img in image_paths}
            sfm_info = {}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "scene": predicted_scene,
                    "image": row.image,
                    "rotation_matrix": format_rotation(rotation),
                    "translation_vector": format_translation(translation),
                }
            )

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v6(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v6 pipeline: same as v5 with per-dataset isolation and progressive output flushing.

    Key robustness improvements over v5:
    - Each dataset is wrapped in try/except.  A crash in one dataset (e.g. a COLMAP
      C++ exception on unusual geometry) produces identity poses for that dataset and
      does NOT prevent the remaining datasets from being processed.
    - The partial submission CSV is written to disk after every dataset so that if
      the kernel is killed mid-run a recoverable partial output still exists.
    - discover_scenes_v6 / run_sfm_per_scene_v6 add the same guards internally so
      exceptions deep inside SIFT or COLMAP routines are caught at multiple layers.
    - No exception propagates out of this function.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 6}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v6(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v6(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v6] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "scene": predicted_scene,
                    "image": row.image,
                    "rotation_matrix": format_rotation(rotation),
                    "translation_vector": format_translation(translation),
                }
            )

        # Progressive flush: write partial CSV after each dataset so mid-run
        # kills still leave a recoverable output on disk.
        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    # Final write + full validation
    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v7(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v7 pipeline: same as v6 with improved SfM matching for video sequences.

    Key change over v6:
    - run_sfm_v7 uses exhaustive COLMAP matching for video sequences with
      N <= 100 (instead of sequential with overlap=5).  This restores the
      stairs reconstruction that regressed in V5/V6 (V4 got 2/51 cameras
      with exhaustive matching; V5/V6 got 0/51 with sequential overlap=5).
    - For larger video sequences (N > 100), sequential matching is used with
      overlap=max(15, N//4) instead of the V6 fixed overlap of 5.
    - All V6 robustness features are preserved.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 7}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v7(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v7(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v7] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "scene": predicted_scene,
                    "image": row.image,
                    "rotation_matrix": format_rotation(rotation),
                    "translation_vector": format_translation(translation),
                }
            )

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v8(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v8 pipeline: GBM LGEU model + LGEU_TOPK=30 + V7 SfM improvements.

    Changes from V7:
    - LGEU trained with HistGradientBoostingClassifier (ROC AUC 0.690 vs 0.614).
    - LGEU_TOPK increased from 20 to 30: 50% more pairs selected per image,
      improving recall of same-scene pairs for large hidden datasets.
    - SfM improvements (exhaustive for video N<=100) inherited from V7.
    - All V6 robustness features preserved.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 8}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v8(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v7(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v8] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "scene": predicted_scene,
                    "image": row.image,
                    "rotation_matrix": format_rotation(rotation),
                    "translation_vector": format_translation(translation),
                }
            )

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v9(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v9 pipeline: GBM LGEU model (AUC 0.690) + LGEU_TOPK=20 (V7 conservative pairing).

    V8 used TOPK=30 which over-merged stairs sub-scenes, regressing from 27/73 (V7)
    to 22/73. V9 reverts to TOPK=20 while keeping GBM model quality for better pair
    ranking on hidden datasets.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 9}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v9(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v9(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v9] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "scene": predicted_scene,
                    "image": row.image,
                    "rotation_matrix": format_rotation(rotation),
                    "translation_vector": format_translation(translation),
                }
            )

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v10(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v10 pipeline: GBM LGEU + TOPK=25 (grid search midpoint between V7's 20 and V8's 30).

    V7 (TOPK=20): 27/73 cameras. V8 (TOPK=30): 22/73. V10 tests whether TOPK=25
    provides a better balance for hidden non-video datasets without regressing stairs.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 10}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v10(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v10(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v10] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v11(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v11 pipeline: GBM LGEU + TOPK=20 + lower Lowe threshold (30) for small datasets.

    ETs (N=22 <= 40) uses exhaustive SIFT. 2 images end up as 'outliers' because they
    only have 30-39 Lowe matches with scene_0 images. V11 lowers the edge threshold to
    30 for small datasets, potentially rescuing these outlier cameras.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 11}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v11(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v11(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v11] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v12(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v12 pipeline: GBM LGEU + TOPK=20 + temporal pair augmentation for video sequences.

    V9 (GBM+TOPK=20): 14 stairs images end up in 'outliers' with 0 cameras registered.
    Stairs is a video sequence (N=51 > 40). LGEU selects pairs by global descriptor similarity;
    for simultaneous dual-camera video, some frames may have poor LGEU-ranked neighbors and
    become stranded as outliers. V12 adds mandatory +-2 temporal neighbor pairs (by filename
    timestamp) for video sequences before SIFT matching, guaranteeing each frame connects to
    its nearest temporal neighbors regardless of LGEU ranking.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 12}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v12(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v12(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v12] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v14(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v14 pipeline: GBM LGEU with num_gap feature zeroed for video sequences.

    For video datasets (N > 40, timestamp filenames), the GBM model's num_gap feature
    (min numeric token gap / 1000) yields ~0.6 for adjacent video frames with ms timestamps.
    The model learned to treat high num_gap as a cross-scene signal (from non-video training
    data where same-scene pairs have small sequential index gaps). For video this is a spurious
    artifact that tanks pair recall and strands frames as outliers (14/51 outliers in V9).
    V14 zeroes num_gap before GBM scoring for video sequences, letting the model discriminate
    by visual similarity (cosine, mutual rank, aspect/area ratio, name_jaccard) alone.
    Non-video and small datasets use standard V9 logic unmodified.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 14}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v14(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v14(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v14] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v15(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v15 pipeline: GBM LGEU + TOPK=20 + lower Lowe threshold (30) for video sequences.

    In V9 (GBM+TOPK=20), 14 stairs images are stranded as outliers with 0 cameras.
    These images have valid same-scene pairs that produce 30-39 Lowe matches —
    just below the standard threshold of 40. V15 lowers good_match_threshold to 30
    for video sequences (N > 40, timestamp filenames) where LGEU pair pre-selection
    (TOPK=20) and MAGSAC geometric verification guard against cross-scene false positives.
    Non-video and small datasets use standard V9 logic (threshold=40, TOPK=20).
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 15}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v15(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v15(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v15] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v13(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v13 pipeline: cosine-only pair ranking for video sequences (bypass LGEU num_gap penalty).

    The LGEU model's num_gap feature penalizes pairs with large numeric (timestamp) gaps.
    For video sequences with millisecond timestamps, adjacent frames separated by ~600ms
    yield num_gap ~0.6, which the GBM model interprets as evidence of different scenes.
    V13 bypasses the LGEU model for video sequences (N > 40, timestamp filenames) and
    selects top-20 pairs per image by raw cosine similarity of global descriptors, which
    is free of timestamp artifacts. Non-video and small datasets use standard V9 logic.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 13}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v13(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v13(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v13] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v16(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v16 pipeline: GBM LGEU with num_gap=0 + Lowe threshold=30 for video sequences.

    Combines the two independent improvements from V14 and V15:
    - V14 zeroed num_gap before GBM scoring for video sequences (avoids penalising
      temporally-adjacent frames), but kept threshold=40 -> 6/51 stairs (same as V9).
    - V15 lowered good_match_threshold to 30 for video sequences (admits weak-overlap
      but geometrically valid edges), but kept num_gap active -> 7/51 stairs.
    V16 applies both simultaneously: better pair selection (num_gap=0) AND lower edge
    threshold (30), with MAGSAC (MIN_MAGSAC_INLIERS=3) and LGEU TOPK=20 as guards.
    Non-video and small datasets use standard V9 logic (full GBM, threshold=40, TOPK=20).
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 16}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v16(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v16(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v16] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v18(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v18 pipeline: V16 clustering + relaxed COLMAP mapper settings for video scenes.

    All V12-V17 improvements targeted scene clustering (pair selection, thresholds,
    feature density) but left the SfM step identical to V7. In V9 diagnostics, even
    correctly-clustered stairs scenes (15 and 20 images) only register 3 cameras each —
    COLMAP's incremental mapper rejects frames with < 4 3D-point correspondences
    (abs_pose_min_num_inliers=4) and fails to initialize from seed pairs with < 1.0°
    triangulation angle. V18 relaxes these thresholds for video scenes:
      - init_min_num_inliers: 5 → 3
      - init_min_tri_angle:   1.0 → 0.5 degrees
      - abs_pose_min_num_inliers: 4 → 3 (P3P minimum)
    Scene clustering inherits V16 (num_gap=0 + Lowe threshold=30 for video).
    Non-video and small datasets use unchanged V7 SfM settings (no regression risk).
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 18}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v18(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v18(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v18] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v17(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v17 pipeline: nfeatures=8000 + num_gap=0 + Lowe threshold=30 for video sequences.

    V16 combined GBM num_gap=0 (V14) with Lowe threshold=30 (V15) for video sequences,
    achieving 7/51 stairs cameras. Despite these improvements, 14 stairs images remain
    stranded as 'outliers' with zero valid SIFT edges. V17 targets the SIFT feature
    density bottleneck: doubling nfeatures from 4000 to 8000 for video-path extraction
    spreads keypoints more densely across each frame, increasing overlap-region sampling
    and raising Lowe match counts for pairs with small but real geometric overlap.
    Non-video and small datasets use standard V9 logic (nfeatures=4000, full GBM,
    threshold=40, TOPK=20), unaffected.

    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 17}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v17(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v17(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v17] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v19(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v19 pipeline: V16 clustering + loosened Lowe ratio (0.80) for video + V18 relaxed SfM.

    For video sequences with repetitive staircase textures and small inter-frame overlap,
    many valid SIFT correspondences fall in the 0.75-0.80 Lowe ratio range — rejected
    by all prior versions (V1-V18 all use ratio=0.75). V19 loosens the FLANN ratio to
    0.80 for video pair matching, admitting these borderline-but-geometrically-consistent
    matches and increasing Lowe match counts for outlier images that need >=30 to form an
    edge. MAGSAC (MIN_MAGSAC_INLIERS=3) guards against false positives from the looser
    ratio. V19 also applies V18's relaxed COLMAP mapper thresholds for video scenes
    (previously upload_failed and thus unscored). Non-video and small datasets (N<=40)
    use unchanged V9 logic (ratio=0.75, threshold=40, full GBM num_gap active, TOPK=20).
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 19}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v19(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v19(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v19] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v20(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v20 pipeline: V16 clustering + outlier reassignment via global descriptor similarity.

    All versions V14-V19 tried to rescue outlier stairs images by tuning SIFT edge
    formation (pair selection, Lowe threshold, Lowe ratio, feature density). The 14
    outlier stairs images cannot form sufficient SIFT edges with ANY top-20 neighbor,
    making all SIFT-gate approaches ineffective.

    V20 accepts this and uses a post-clustering outlier reassignment step: each outlier
    image is assigned to the scene containing its most visually similar non-outlier image
    (by cosine similarity of global descriptors). COLMAP then runs exhaustive matching on
    the enlarged scene internally, giving outlier images a second chance at registration
    via COLMAP's own SIFT pipeline, not our threshold-gated Python SIFT.

    Scene clustering for video sequences uses V16 logic (GBM num_gap=0, Lowe threshold=30,
    TOPK=20) before reassignment. SfM uses V7 (proven, exhaustive for N<=100 video).
    Non-video and small datasets (N<=40) use standard V9 logic, unaffected.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 20}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v20(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v20(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v20] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v21(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v21 pipeline: V16 clustering + relaxed abs_pose_min_num_inliers for video SfM.

    V7-V20 (excluding upload_failed V18/V19) use abs_pose_min_num_inliers=4 for video
    scenes. For forward-facing staircase video, consecutive frames often have only 3
    inlier 3D-point correspondences with the sparse initial reconstruction -- the P3P
    minimum -- so COLMAP skips them, yielding only 3/15 and 3/20 cameras in the two
    correctly-clustered stairs scenes.

    V21 lowers abs_pose_min_num_inliers to 3 for video scenes only, allowing cameras
    with P3P-minimum inlier counts to be registered. V18 changed three mapper parameters
    simultaneously and upload_failed; V21 isolates this single parameter to reduce risk.

    Scene clustering uses V16 (GBM num_gap=0 + Lowe threshold=30 for video, best at
    7/51 stairs). Non-video and small datasets (N<=40) use standard V9 logic, unaffected.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 21}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v21(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v21(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v21] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v22(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v22 pipeline: V16 clustering + relaxed init_min_tri_angle and abs_pose for video SfM.

    V21 lowered abs_pose_min_num_inliers to 3 (P3P minimum) for video scenes and tied V7
    at 8/51 stairs. The remaining untried SfM parameter from V18 (3-param bundle that
    upload_failed) is init_min_tri_angle. V21 keeps this at 1.0 degrees (V7 video default).
    For near-forward-facing staircase cameras, valid seed pairs often have a triangulation
    angle below 1.0 degrees, causing COLMAP to reject them in favour of a worse seed pair.
    V22 adds init_min_tri_angle=0.5 degrees for video scenes on top of V21 abs_pose=3,
    targeting improved seed-pair selection and a denser initial 3D point cloud.

    Scene clustering uses V16 (GBM num_gap=0 + Lowe threshold=30 for video, best at
    7/51 stairs). Non-video and small datasets (N<=40) use standard V9 logic, unaffected.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 22}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v22(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v22(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v22] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v23(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v23 pipeline: V16 clustering + CameraMode.SINGLE and abs_pose=3 for video SfM.

    All prior versions (V7–V22) use CameraMode.AUTO for COLMAP feature extraction,
    which assigns independent camera intrinsics to each image. For staircase video
    sequences, all frames come from the same physical camera with fixed focal length
    and principal point. AUTO mode estimates intrinsics per image from a sparse seed
    pair, producing poorly constrained per-frame estimates that limit reconstruction
    quality. V23 uses CameraMode.SINGLE for video scenes, forcing COLMAP to share one
    camera model across all frames and jointly refine intrinsics during bundle adjustment.
    A more accurately calibrated reconstruction with tighter 3D point positions increases
    the probability that adjacent frames accumulate the 3 inlier correspondences required
    by V21's abs_pose_min_num_inliers=3 setting.

    V22's init_min_tri_angle=0.5 change is NOT carried over (it caused regression 8→7
    stairs cameras); V23 reverts to V21's init_min_tri_angle=1.0. Scene clustering uses
    V16 (GBM num_gap=0 + Lowe threshold=30 for video). Non-video and small datasets
    (N<=40) use standard V9 logic, unaffected.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 23}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v23(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v23(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v23] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v25(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v25 pipeline: V16 clustering + merge ALL COLMAP sub-reconstructions.

    All prior versions (V7-V24) take only the single largest COLMAP reconstruction from
    incremental_mapping(), discarding every smaller sub-reconstruction. For staircase video
    recorded simultaneously by two physically separate camera rigs, COLMAP produces multiple
    disjoint sub-reconstructions because the rigs share no geometric overlap. By taking only
    the largest reconstruction, we miss any cameras correctly registered in secondary models.

    V25 merges poses from ALL reconstructions (sorted by descending registration count).
    For any image registered in multiple sub-reconstructions, the pose from the largest wins.
    Scene clustering uses V16 (GBM num_gap=0 + Lowe threshold=30 for video, best config).
    SfM mapper settings identical to V21 (abs_pose_min_num_inliers=3, init_min_num_inliers=5,
    init_min_tri_angle=1.0). Non-video and small datasets use standard V9 logic, unaffected.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 25}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v25(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v25(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v25] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v26(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v26 pipeline: V16 clustering + init_min_num_inliers=3 + merge ALL COLMAP sub-recs.

    V25 merged poses from ALL sub-reconstructions (rather than only the largest) and tied V7
    at 8/51 stairs cameras. Since V25 showed no improvement, COLMAP likely produces only one
    sub-reconstruction per scene under V21 mapper settings, so the merge-all strategy has
    nothing extra to collect. V24 tried init_min_num_inliers=3 for video scenes but
    smoke_errored before the SfM step could run, leaving this parameter untested with V25.
    With init_min_num_inliers=3, COLMAP accepts seed pairs with only 3 F-matrix inliers
    (the P3P minimum), giving the mapper more starting candidates and potentially producing
    additional sub-reconstructions that V26 can then merge.

    Scene clustering uses V16 (GBM num_gap=0 + Lowe threshold=30 for video, best config).
    SfM mapper settings for video: init_min_num_inliers=3 (NEW), init_min_tri_angle=1.0
    (not 0.5 -- V22 regression), abs_pose_min_num_inliers=3 (V21). Non-video and small
    datasets (N<=40) use standard V9 logic, unaffected.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 26}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v26(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v26(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v26] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission


def build_submission_v24(
    sample_path: Path,
    image_root: Path,
    output_path: Path,
    lgeu_model_payload: dict | None = None,
    diagnostics_path: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """v24 pipeline: V16 clustering + init_min_num_inliers=3 and abs_pose=3 for video SfM.

    V21 lowered abs_pose_min_num_inliers to 3 (P3P minimum) for video scenes and tied V7
    at 8/51 stairs. V18 changed three mapper parameters (init_min_num_inliers 5->3,
    init_min_tri_angle 1.0->0.5, abs_pose_min_num_inliers 4->3) simultaneously and
    upload_failed. V21 isolated abs_pose_min_num_inliers and succeeded. V22 added
    init_min_tri_angle=0.5 and regressed to 7/51.

    V24 isolates the third V18 parameter: init_min_num_inliers (seed-pair selection floor).
    V7 sets this to 5 for video scenes. For staircase video with nearly-parallel optical
    axes, even the geometrically best available seed pair may produce only 3-5 verified
    fundamental matrix inliers, causing COLMAP to reject valid seed pairs and fall back to
    geometrically worse alternatives (or fail initialization). V24 reduces
    init_min_num_inliers from 5 to 3 combined with abs_pose_min_num_inliers=3 (V21),
    while keeping init_min_tri_angle at 1.0 (V22 0.5 caused regression and is not
    carried over). Scene clustering uses V16 (GBM num_gap=0 + Lowe threshold=30 for video).
    Non-video and small datasets (N<=40) use standard V9 logic, unaffected.
    """
    sample = read_sample(sample_path)
    out_rows: list[dict] = []
    diagnostics: dict = {"datasets": {}, "version": 24}

    for dataset, group in sample.groupby("dataset", sort=False):
        image_paths: dict[str, Path] = {}
        scene_map: dict[str, str] = {}
        poses: dict[str, tuple] = {}
        sfm_info: dict = {}
        dataset_error: str | None = None

        try:
            for row in group.itertuples(index=False):
                path = image_root / "test" / dataset / row.image
                if path.is_file():
                    image_paths[row.image] = path

            if len(image_paths) >= 2:
                scene_map, _edge_data = discover_scenes_v24(
                    image_paths,
                    lgeu_model_payload=lgeu_model_payload,
                    verbose=verbose,
                )
                poses, sfm_info = run_sfm_per_scene_v24(
                    image_paths, scene_map, verbose=verbose
                )
            else:
                scene_map = {img: "scene_0" for img in image_paths}
                poses = {img: identity_pose() for img in image_paths}
                sfm_info = {}

        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [v24] ERROR in dataset '{dataset}': {dataset_error}")
                traceback.print_exc()
            scene_map = {}
            poses = {}
            sfm_info = {"error": dataset_error}

        diagnostics["datasets"][dataset] = {
            "images_in_sample": int(len(group)),
            "images_available": int(len(image_paths)),
            "scenes_discovered": sorted(set(scene_map.values())),
            "sfm_info": sfm_info,
            "error": dataset_error,
        }

        for row in group.itertuples(index=False):
            rotation, translation = poses.get(row.image, identity_pose())
            predicted_scene = scene_map.get(row.image, "scene_0")
            out_rows.append({
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": predicted_scene,
                "image": row.image,
                "rotation_matrix": format_rotation(rotation),
                "translation_vector": format_translation(translation),
            })

        try:
            write_submission(out_rows, output_path)
        except Exception:
            pass
        if diagnostics_path:
            try:
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            except Exception:
                pass

    submission = write_submission(out_rows, output_path)
    validate_submission(sample, submission)
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return submission
