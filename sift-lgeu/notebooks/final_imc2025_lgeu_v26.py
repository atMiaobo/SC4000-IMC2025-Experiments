"""Kaggle script notebook for the IMC2025 LGEU pipeline -- Version 26.

Changes from Version 25 (tied best at 27/73):
- V26 KEY CHANGE: combine V25 (merge all COLMAP sub-reconstructions) with V24's
  init_min_num_inliers=3 for video scenes.
  V25 merged ALL sub-reconstructions from incremental_mapping() instead of only the
  largest, but tied V7 at 8/51 stairs cameras -- COLMAP apparently produces only one
  sub-reconstruction per scene under V21's seed-pair floor of 5. V24 tried lowering this
  floor to 3 but smoke_errored before the SfM step ran, leaving the combination untested.
  V26 sets init_min_num_inliers=3 for video: COLMAP accepts seed pairs with as few as 3
  verified F-matrix inliers (the P3P minimum). More valid seed pair candidates means COLMAP
  can (a) initialize in scenes that currently fail entirely, and (b) produce additional
  sub-reconstructions from image clusters that the floor-5 threshold silently rejects.
  Those additional sub-reconstructions are then captured by V26's merge-all strategy.
- Scene clustering: unchanged from V16/V21 (GBM num_gap=0 + Lowe threshold=30 for video,
  TOPK=20). This remains the best-scoring clustering configuration.
- SfM mapper settings for video:
    init_min_num_inliers=3 (NEW: was 5 in V21/V25)
    init_min_tri_angle=1.0 (unchanged; 0.5 caused regression in V22)
    abs_pose_min_num_inliers=3 (from V21: P3P minimum for camera registration)
    init_max_forward_motion=0.999
- Non-video and small datasets (N<=40): standard V9 logic, unaffected.
- ETs (19/22): unaffected -- exhaustive path, non-video, V7 mapper settings unchanged.
- Expected: additional stairs cameras from either improved initialization in scene_0/scene_2
  or from COLMAP sub-reconstructions unlocked by the lower seed-pair floor.

Offline execution requires the same asset dataset as v7 (dataset v7):
  src.tar.gz           -- Python package (imc2025_lgeu/) including v26 modules
  lgeu.joblib          -- GBM LGEU model (retrained, AUC 0.690)
  kernel-metadata.json -- metadata with dataset_sources filled
  pycolmap-4.2.1-cpXXX-*.whl  -- pycolmap offline wheel

SMOKE_TEST=True: processes ONE dataset (<=SMOKE_MAX_IMAGES) end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASSET_DATASET_SLUG = "miaobo/imc2025-lgeu-assets"

SMOKE_TEST = False          # Set False for full competition run
SMOKE_MAX_IMAGES = 30


# ---------------------------------------------------------------------------
# Asset bootstrap (unchanged from v4/v5/v6)
# ---------------------------------------------------------------------------

def _find_dataset_root() -> Path:
    for base in ["/kaggle/input/datasets", "/kaggle/input"]:
        slug_path = Path(base) / ASSET_DATASET_SLUG
        if slug_path.is_dir():
            return slug_path
    for candidate in Path("/kaggle/input").rglob("lgeu.joblib"):
        return candidate.parent
    for candidate in Path("/kaggle/input").rglob("src.tar.gz"):
        return candidate.parent
    if "__file__" in globals():
        local = Path(__file__).resolve().parents[1]
        if (local / "src" / "imc2025_lgeu" / "__init__.py").is_file():
            return local
    raise FileNotFoundError(
        f"Cannot find asset dataset. Tried /kaggle/input/datasets/{ASSET_DATASET_SLUG} "
        f"and /kaggle/input/{ASSET_DATASET_SLUG}."
    )


def _extract_src(dataset_root: Path) -> Path:
    extract_dir = Path("/kaggle/working/src_pkg")
    init_path = extract_dir / "imc2025_lgeu" / "__init__.py"
    if init_path.exists():
        return extract_dir

    src_tar = dataset_root / "src.tar.gz"
    if src_tar.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {src_tar} -> {extract_dir} ...")
        with tarfile.open(src_tar, "r:gz") as tf:
            tf.extractall(extract_dir)
        if init_path.exists():
            return extract_dir
        raise FileNotFoundError(
            f"src.tar.gz extracted but imc2025_lgeu/__init__.py missing at {extract_dir}"
        )

    for candidate in [
        dataset_root / "src",
        dataset_root,
        Path(__file__).resolve().parents[1] / "src" if "__file__" in globals() else None,
    ]:
        if candidate and (candidate / "imc2025_lgeu" / "__init__.py").is_file():
            return candidate

    raise FileNotFoundError(
        f"Cannot find imc2025_lgeu package. Expected src.tar.gz at {src_tar}."
    )


def _install_pycolmap(dataset_root: Path) -> bool:
    try:
        import pycolmap  # noqa: F401
        print(f"pycolmap already available: {pycolmap.__version__}")
        return True
    except ImportError:
        pass

    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    preferred = dataset_root / f"pycolmap-4.2.1-{pyver}-{pyver}-manylinux_2_28_x86_64.whl"
    if preferred.exists():
        wheel = preferred
    else:
        wheels = sorted(dataset_root.glob("pycolmap-*.whl"))
        if not wheels:
            print(f"WARNING: no pycolmap wheel found under {dataset_root}")
            return False
        wheel = wheels[0]
        print(f"Preferred wheel not found; using fallback: {wheel.name}")

    print(f"Installing pycolmap from {wheel.name} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: pip install failed:\n{result.stderr[:500]}")
        return False

    try:
        import importlib
        import pycolmap  # noqa: F401
        importlib.reload(pycolmap)
        print(f"pycolmap installed: {pycolmap.__version__}")
        return True
    except ImportError as exc:
        print(f"WARNING: pycolmap import failed after install: {exc}")
        return False


def _find_competition_root() -> Path:
    for candidate in [
        Path("/kaggle/input/competitions/image-matching-challenge-2025"),
        Path("/kaggle/input/image-matching-challenge-2025"),
    ]:
        if candidate.is_dir() and (candidate / "sample_submission.csv").exists():
            return candidate

    candidates = [
        p
        for p in Path("/kaggle/input").glob("**/sample_submission.csv")
        if "image-matching-challenge-2025" in str(p) or (p.parent / "test").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not find official IMC2025 sample_submission.csv under /kaggle/input."
        )
    for c in candidates:
        if "image-matching-challenge-2025" in str(c) and "ongoing" not in str(c):
            return c.parent
    return candidates[0].parent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Bootstrap assets
    dataset_root = _find_dataset_root()
    src_pkg = _extract_src(dataset_root)
    sys.path.insert(0, str(src_pkg))

    print(f"Python version: {sys.version}")
    print(f"Python ABI tag: cp{sys.version_info.major}{sys.version_info.minor}")

    # 2. Install pycolmap offline
    pycolmap_ok = _install_pycolmap(dataset_root)
    if not pycolmap_ok:
        print("WARNING: pycolmap unavailable -- SfM will fall back to identity poses.")

    # 3. Import pipeline modules (v26 functions)
    from imc2025_lgeu.contract import (
        COMPETITION_SLUG,
        read_sample,
        validate_kernel_metadata,
        validate_submission,
    )
    from imc2025_lgeu.lgeu import load_model
    from imc2025_lgeu.submission import build_submission_v26

    # 4. Validate kernel metadata
    meta_path = dataset_root / "kernel-metadata.json"
    if meta_path.exists():
        validate_kernel_metadata(meta_path)
    else:
        print(f"WARNING: kernel-metadata.json not found at {meta_path}")

    if COMPETITION_SLUG != "image-matching-challenge-2025":
        raise RuntimeError(f"wrong competition slug: {COMPETITION_SLUG!r}")

    # 5. Locate competition data
    comp_root = _find_competition_root()
    sample_path = comp_root / "sample_submission.csv"

    # 6. Load LGEU model
    model_path = dataset_root / "lgeu.joblib"
    lgeu_payload = None
    if model_path.is_file():
        try:
            lgeu_payload = load_model(model_path)
            print(f"LGEU model loaded: {lgeu_payload['metrics'].get('roc_auc', '?'):.3f} ROC AUC")
        except Exception as exc:
            print(f"WARNING: could not load LGEU model: {exc}")

    print(f"\n--- Environment (v26) ---")
    print(f"Dataset root  : {dataset_root}")
    print(f"Src pkg root  : {src_pkg}")
    print(f"Comp root     : {comp_root}")
    print(f"LGEU model    : {'loaded' if lgeu_payload else 'absent/failed'}")
    print(f"GPU visible   : {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    # 7. Load full sample (always read at runtime from competition data)
    full_sample = read_sample(sample_path)
    datasets_in_sample = sorted(full_sample["dataset"].unique())
    print(f"Sample rows   : {len(full_sample)}")
    print(f"Datasets      : {datasets_in_sample}")

    # 8. Smoke-test: restrict to first dataset, <=SMOKE_MAX_IMAGES images
    if SMOKE_TEST:
        canary_ds = datasets_in_sample[0]
        canary_sample = (
            full_sample[full_sample["dataset"] == canary_ds]
            .head(SMOKE_MAX_IMAGES)
        )
        print(
            f"\n--- SMOKE_TEST MODE ---\n"
            f"Canary dataset : '{canary_ds}'\n"
            f"Images         : {len(canary_sample)} (max {SMOKE_MAX_IMAGES})"
        )
        canary_sample_path = Path("/kaggle/working/canary_sample.csv")
        canary_sample.to_csv(canary_sample_path, index=False)
        active_sample_path = canary_sample_path
        active_sample = canary_sample
    else:
        active_sample_path = sample_path
        active_sample = full_sample

    output = Path("/kaggle/working/submission.csv")
    diagnostics = Path("/kaggle/working/submission_diagnostics.json")

    # 9. Run v26 pipeline
    # build_submission_v26 never throws: all dataset errors are caught internally.
    submission = build_submission_v26(
        sample_path=active_sample_path,
        image_root=comp_root,
        output_path=output,
        lgeu_model_payload=lgeu_payload,
        diagnostics_path=diagnostics,
        verbose=True,
    )

    # 10. Validate and report
    validate_submission(active_sample, submission)

    total_registered = 0
    total_images = 0
    dataset_errors = []
    if diagnostics.exists():
        diag = json.loads(diagnostics.read_text())
        print("\n--- SfM diagnostics (v26) ---")
        for ds, info in diag.get("datasets", {}).items():
            if info.get("error"):
                dataset_errors.append(f"{ds}: {info['error']}")
                print(f"  {ds}: DATASET ERROR -- {info['error']}")
                continue
            sfm_info = info.get("sfm_info", {})
            for scene, sinfo in sfm_info.items():
                reg = sinfo.get("registered", 0)
                tot = sinfo.get("total", 0)
                pts = sinfo.get("num_points3D", 0)
                err = sinfo.get("error", "")
                total_registered += reg
                total_images += tot
                tags = []
                if sinfo.get("is_video"):
                    tags.append("video")
                if sinfo.get("match_mode"):
                    tags.append(sinfo["match_mode"])
                if sinfo.get("num_reconstructions"):
                    tags.append(f"recs:{sinfo['num_reconstructions']}")
                if sinfo.get("registered_largest") is not None:
                    tags.append(f"largest:{sinfo['registered_largest']}")
                if err:
                    tags.append(f"err:{err}")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                print(f"  {ds}/{scene}: {reg}/{tot} cameras, {pts} 3D pts{tag_str}")
        print(f"  TOTAL: {total_registered}/{total_images} cameras registered")
        if dataset_errors:
            print(f"  DATASET ERRORS ({len(dataset_errors)}): {dataset_errors}")

    result = {
        "version": 26,
        "smoke_test": SMOKE_TEST,
        "submission": str(output),
        "rows": len(submission),
        "diagnostics": str(diagnostics),
        "slug": COMPETITION_SLUG,
        "dataset_errors": dataset_errors,
    }

    if SMOKE_TEST:
        if total_registered > 0:
            print("\n* SMOKE_TEST CANARY PASSED (v26): nonzero camera registration confirmed.")
        else:
            print("\n* SMOKE_TEST WARNING (v26): zero cameras registered.")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
