"""CPU reconstruction stage. Deliberately does not import PyTorch or LightGlue."""
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import pycolmap


def main():
    run = Path(sys.argv[1])
    cfg = json.loads((run / "config.json").read_text())
    start = time.perf_counter()
    db_path = run / "database.db"
    if db_path.exists():
        raise SystemExit("Reconstruction database already exists. Use a fresh run directory.")
    pycolmap.Database.open(str(db_path)).close()
    pycolmap.import_images(str(db_path), cfg["image_dir"], camera_mode=pycolmap.CameraMode.PER_IMAGE,
                          image_names=cfg["images"])
    db = pycolmap.Database.open(str(db_path))
    ids = {image.name: image.image_id for image in db.read_all_images()}
    with h5py.File(run / "features.h5") as features:
        for i, name in enumerate(cfg["images"]):
            db.write_keypoints(ids[name], np.asarray(features[str(i)][:] + 0.5, dtype=np.float32))
    pairs = json.loads((run / "pairs.json").read_text())
    with h5py.File(run / "matches.h5") as matches:
        for i, j in pairs:
            db.write_matches(ids[cfg["images"][i]], ids[cfg["images"][j]], matches[f"{i}_{j}"][:])
    db.close()
    (run / "pairs.txt").write_text("\n".join(f"{cfg['images'][i]} {cfg['images'][j]}" for i, j in pairs))
    maps = {}
    if pairs:
        pycolmap.verify_matches(str(db_path), str(run / "pairs.txt"))
        options = pycolmap.IncrementalPipelineOptions()
        options.min_model_size = 3
        options.max_num_models = 25
        options.num_threads = 4
        options.random_seed = 0
        options.ba_use_gpu = False
        options.mapper.init_min_tri_angle = 8.0
        options.mapper.abs_pose_min_num_inliers = 12
        models = run / "sparse"
        models.mkdir()
        maps = pycolmap.incremental_mapping(str(db_path), cfg["image_dir"], str(models), options=options)
    predictions = {}
    sizes = []
    # Prefer the largest reconstruction when an image occurs in several models.
    for cluster, model in enumerate(sorted(maps.values(), key=lambda m: m.num_reg_images(), reverse=True)):
        sizes.append(model.num_reg_images())
        for image in model.images.values():
            if not image.has_pose or image.name in predictions:
                continue
            pose = image.cam_from_world()
            predictions[image.name] = (f"cluster{cluster}", pose.rotation.matrix(), pose.translation)
    fields = ["dataset", "scene", "image", "rotation_matrix", "translation_vector"]
    with (run / "predictions_train.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in cfg["images"]:
            cluster, rotation, translation = predictions.get(name, ("outliers", np.full((3, 3), np.nan), np.full(3, np.nan)))
            writer.writerow(dict(zip(fields, [cfg["dataset"], cluster, name,
                ";".join(f"{x:.12g}" for x in rotation.ravel()), ";".join(f"{x:.12g}" for x in translation)])))
    with sqlite3.connect(db_path) as connection:
        verified = connection.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows >= 15").fetchone()[0]
    summary = {"pycolmap": pycolmap.__version__, "images": len(cfg["images"]),
               "registered_images": len(predictions), "reconstruction_sizes": sizes,
               "verified_pairs_with_at_least_15_inliers": verified,
               "seconds": round(time.perf_counter() - start, 3), "scope": cfg["scope"]}
    (run / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
