"""Kaggle script notebook for the clean-room IMC2025 LGEU pipeline.

Offline execution (Internet OFF) requires an attached Kaggle Dataset containing:
  src.tar.gz           — Python package (imc2025_lgeu/) as a tarball
  lgeu.joblib          — trained LGEU checkpoint (kept for provenance)
  kernel-metadata.json — metadata with dataset_sources filled
  pycolmap-4.2.1-cpXXX-cpXXX-manylinux_2_28_x86_64.whl  — pycolmap wheel

The dataset slug must be listed in kernel-metadata.json::dataset_sources.
Assets are mounted at /kaggle/input/{dataset-owner}/{dataset-slug}/.

This script writes /kaggle/working/submission.csv.

SMOKE_TEST=True: processes ONE dataset (≤SMOKE_MAX_IMAGES images) and verifies
the end-to-end pipeline produces nonzero camera registrations.  Switch to False
for the full 9-hour competition inference.
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

# Smoke-test mode: process ONE competition dataset, ≤SMOKE_MAX_IMAGES images.
# Set to False for the full competition run.
SMOKE_TEST = False
SMOKE_MAX_IMAGES = 30


# ---------------------------------------------------------------------------
# Asset bootstrap helpers
# ---------------------------------------------------------------------------

def _find_dataset_root() -> Path:
    """Return the mounted Kaggle Dataset directory containing our assets.

    Kaggle mounts datasets at either:
      /kaggle/input/{owner}/{name}/          (older Kaggle format)
      /kaggle/input/datasets/{owner}/{name}/ (newer Kaggle format, 2026+)
    """
    for base in ["/kaggle/input/datasets", "/kaggle/input"]:
        slug_path = Path(base) / ASSET_DATASET_SLUG
        if slug_path.is_dir():
            return slug_path
    # Fallback: search /kaggle/input for lgeu.joblib (handles any path structure)
    for candidate in Path("/kaggle/input").rglob("lgeu.joblib"):
        return candidate.parent
    # Also search for src.tar.gz (may exist if not auto-extracted)
    for candidate in Path("/kaggle/input").rglob("src.tar.gz"):
        return candidate.parent
    # Local dev fallback (script is in notebooks/, project root is parent)
    if "__file__" in globals():
        local = Path(__file__).resolve().parents[1]
        if (local / "src" / "imc2025_lgeu" / "__init__.py").is_file():
            return local
    raise FileNotFoundError(
        f"Cannot find asset dataset. Tried /kaggle/input/datasets/{ASSET_DATASET_SLUG} "
        f"and /kaggle/input/{ASSET_DATASET_SLUG}. "
        "Ensure dataset_sources is correct in kernel-metadata.json."
    )


def _extract_src(dataset_root: Path) -> Path:
    """Extract src.tar.gz to /kaggle/working/src_pkg/; return extraction dir."""
    extract_dir = Path("/kaggle/working/src_pkg")
    init_path = extract_dir / "imc2025_lgeu" / "__init__.py"
    if init_path.exists():
        return extract_dir

    src_tar = dataset_root / "src.tar.gz"
    if src_tar.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {src_tar} → {extract_dir} …")
        with tarfile.open(src_tar, "r:gz") as tf:
            tf.extractall(extract_dir)
        if init_path.exists():
            return extract_dir
        raise FileNotFoundError(
            f"src.tar.gz extracted but imc2025_lgeu/__init__.py missing at {extract_dir}"
        )

    # Fallback: src/ directory preserved (--dir-mode tar/zip upload, or local dev)
    for candidate in [
        dataset_root / "src",
        dataset_root,
        Path(__file__).resolve().parents[1] / "src" if "__file__" in globals() else None,
    ]:
        if candidate and (candidate / "imc2025_lgeu" / "__init__.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Cannot find imc2025_lgeu package. "
        f"Expected src.tar.gz at {src_tar} or src/ under {dataset_root}."
    )


def _install_pycolmap(dataset_root: Path) -> bool:
    """Install pycolmap from local wheel if not already importable. Returns True on success."""
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

    print(f"Installing pycolmap from {wheel.name} …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)],
        capture_output=True,
        text=True,
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
    """Find the official IMC2025 competition dataset root.

    Kaggle mounts competition data at:
      /kaggle/input/competitions/{slug}/   (newer format)
      /kaggle/input/{slug}/               (older format)
    """
    # Check known paths first (fast)
    for candidate in [
        Path("/kaggle/input/competitions/image-matching-challenge-2025"),
        Path("/kaggle/input/image-matching-challenge-2025"),
    ]:
        if candidate.is_dir() and (candidate / "sample_submission.csv").exists():
            return candidate

    # Glob fallback
    candidates = [
        p
        for p in Path("/kaggle/input").glob("**/sample_submission.csv")
        if "image-matching-challenge-2025" in str(p) or (p.parent / "test").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not find official IMC2025 sample_submission.csv under /kaggle/input. "
            "Ensure competition_sources=['image-matching-challenge-2025'] in kernel-metadata.json."
        )
    for c in candidates:
        if "image-matching-challenge-2025" in str(c) and "ongoing" not in str(c):
            return c.parent
    return candidates[0].parent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Locate and bootstrap assets
    dataset_root = _find_dataset_root()
    src_pkg = _extract_src(dataset_root)
    sys.path.insert(0, str(src_pkg))

    print(f"Python version: {sys.version}")
    print(f"Python ABI tag: cp{sys.version_info.major}{sys.version_info.minor}")

    # 2. Install pycolmap offline
    pycolmap_ok = _install_pycolmap(dataset_root)
    if not pycolmap_ok:
        print("WARNING: pycolmap unavailable — SfM will fall back to identity poses.")

    # 3. Import pipeline modules
    from imc2025_lgeu.contract import (
        COMPETITION_SLUG,
        read_sample,
        validate_kernel_metadata,
        validate_submission,
    )
    from imc2025_lgeu.submission import build_submission

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

    model_path = dataset_root / "lgeu.joblib"
    print(f"\n--- Environment ---")
    print(f"Dataset root  : {dataset_root}")
    print(f"Src pkg root  : {src_pkg}")
    print(f"Comp root     : {comp_root}")
    print(f"LGEU model    : {'present' if model_path.is_file() else 'absent'}")
    print(f"GPU visible   : {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    # 6. Load full sample
    full_sample = read_sample(sample_path)
    datasets_in_sample = sorted(full_sample["dataset"].unique())
    print(f"Sample rows   : {len(full_sample)}")
    print(f"Datasets      : {datasets_in_sample}")

    # 7. Smoke-test mode: restrict to first dataset, at most SMOKE_MAX_IMAGES images
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
        # Write a restricted sample CSV so build_submission uses it
        canary_sample_path = Path("/kaggle/working/canary_sample.csv")
        canary_sample.to_csv(canary_sample_path, index=False)
        active_sample_path = canary_sample_path
        active_sample = canary_sample
    else:
        active_sample_path = sample_path
        active_sample = full_sample

    output = Path("/kaggle/working/submission.csv")
    diagnostics = Path("/kaggle/working/submission_diagnostics.json")

    # 8. Run pipeline
    submission = build_submission(
        sample_path=active_sample_path,
        image_root=comp_root,
        output_path=output,
        model_path=model_path if model_path.is_file() else None,
        diagnostics_path=diagnostics,
        verbose=True,
    )

    # 9. Validate and report
    validate_submission(active_sample, submission)

    # Print diagnostics
    total_registered = 0
    total_images = 0
    if diagnostics.exists():
        diag = json.loads(diagnostics.read_text())
        print("\n--- SfM diagnostics ---")
        for ds, info in diag.get("datasets", {}).items():
            sfm_info = info.get("sfm_info", {})
            for scene, sinfo in sfm_info.items():
                reg = sinfo.get("registered", 0)
                tot = sinfo.get("total", 0)
                pts = sinfo.get("num_points3D", 0)
                total_registered += reg
                total_images += tot
                print(f"  {ds}/{scene}: {reg}/{tot} cameras, {pts} 3D pts")
        print(f"  TOTAL: {total_registered}/{total_images} cameras registered")

    result = {
        "smoke_test": SMOKE_TEST,
        "submission": str(output),
        "rows": len(submission),
        "diagnostics": str(diagnostics),
        "slug": COMPETITION_SLUG,
    }

    if SMOKE_TEST:
        if total_registered > 0:
            print("\n✓ SMOKE_TEST CANARY PASSED: nonzero camera registration confirmed.")
        else:
            print("\n✗ SMOKE_TEST WARNING: zero cameras registered (SfM failed or identity fallback).")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
