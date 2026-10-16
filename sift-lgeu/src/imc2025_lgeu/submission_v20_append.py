

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
    via COLMAP's own SIFT pipeline — not our threshold-gated Python SIFT.

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
