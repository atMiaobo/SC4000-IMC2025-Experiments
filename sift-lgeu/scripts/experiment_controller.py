#!/usr/bin/env python3
"""Autonomous experiment controller for IMC2025 LGEU pipeline.

Usage:
  python scripts/experiment_controller.py run --version 7 --smoke-only
  python scripts/experiment_controller.py run --version 7
  python scripts/experiment_controller.py status
  python scripts/experiment_controller.py collect --version 7
  python scripts/experiment_controller.py history

Each run is recorded in experiments/experiment_log.json.
Supports safe resume: re-running the same version skips already-completed steps.

Phase 1 note: Kaggle API submissions return HTTP 400 (competition is closed).
Submissions must be made manually via Kaggle notebook UI (instructions printed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "experiments" / "experiment_log.json"
KERNEL_PREFIX = "imc2025-lgeu-v"
KAGGLE_USER = "miaobo"
COMPETITION = "image-matching-challenge-2025"
DATASET_SLUG = f"{KAGGLE_USER}/imc2025-lgeu-assets"


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _load_log() -> dict:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text())
    return {"experiments": {}}


def _save_log(log: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2))


def _get_entry(log: dict, version: int) -> dict:
    key = f"v{version}"
    if key not in log["experiments"]:
        log["experiments"][key] = {"version": version, "steps": {}}
    return log["experiments"][key]


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], check: bool = True, capture: bool = False, cwd=None) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, cwd=cwd)


def _run_kaggle(args: list[str]) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command via uv."""
    return _run(["uv", "run", "kaggle"] + args, capture=True, cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# Step: run local tests
# ---------------------------------------------------------------------------

def step_local_tests(version: int, entry: dict) -> bool:
    step = "local_tests"
    if entry["steps"].get(step) == "pass":
        print(f"  [skip] local tests already passed for v{version}")
        return True
    print(f"\n[v{version}] Running local tests ...")
    result = subprocess.run(
        ["uv", "run", "pytest", "-q", "--tb=short"],
        capture_output=False,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        entry["steps"][step] = "fail"
        print(f"  ✗ Local tests FAILED for v{version}")
        return False
    entry["steps"][step] = "pass"
    print(f"  ✓ Local tests PASSED for v{version}")
    return True


# ---------------------------------------------------------------------------
# Step: package src.tar.gz
# ---------------------------------------------------------------------------

def step_package_src(version: int, entry: dict) -> bool:
    step = "package_src"
    tarball = REPO_ROOT / "src.tar.gz"
    print(f"\n[v{version}] Packaging src.tar.gz ...")
    result = subprocess.run(
        ["tar", "-czf", str(tarball), "-C", str(REPO_ROOT / "src"), "imc2025_lgeu"],
        check=False,
    )
    if result.returncode != 0:
        entry["steps"][step] = "fail"
        print("  ✗ tarball creation FAILED")
        return False
    size_mb = tarball.stat().st_size / 1024 / 1024
    entry["steps"][step] = "pass"
    entry["src_tarball_mb"] = round(size_mb, 2)
    print(f"  ✓ src.tar.gz ({size_mb:.2f} MB)")
    return True


# ---------------------------------------------------------------------------
# Step: upload dataset (always include all files)
# ---------------------------------------------------------------------------

def step_upload_dataset(version: int, entry: dict) -> bool:
    step = "upload_dataset"
    if entry["steps"].get(step) == "pass":
        print(f"  [skip] dataset already uploaded for v{version}")
        return True

    # Assemble upload directory with all required files
    upload_dir = Path("/tmp/lgeu_dataset_v7_upload")
    upload_dir.mkdir(exist_ok=True)

    required = {
        "src.tar.gz": REPO_ROOT / "src.tar.gz",
        "lgeu.joblib": REPO_ROOT / "models" / "lgeu.joblib",
    }
    # Copy wheels from models directory or find them
    wheel_src_dirs = [REPO_ROOT / "models", REPO_ROOT]
    wheels_found = []
    for d in wheel_src_dirs:
        wheels_found += list(d.glob("pycolmap-*.whl"))
    wheels_found = list({w.name: w for w in wheels_found}.values())  # deduplicate by name

    print(f"\n[v{version}] Uploading dataset (all files) ...")
    missing = []
    for name, path in required.items():
        if not path.exists():
            # Try alternative locations
            alt = REPO_ROOT / name
            if alt.exists():
                required[name] = alt
            else:
                missing.append(str(path))

    if missing:
        print(f"  ✗ Missing files: {missing}")
        entry["steps"][step] = "fail_missing_files"
        return False

    if not wheels_found:
        print("  ✗ No pycolmap wheels found — cannot upload dataset")
        entry["steps"][step] = "fail_no_wheels"
        return False

    for name, path in required.items():
        shutil.copy2(str(path), str(upload_dir / name))
        print(f"  + {name} ({path.stat().st_size // 1024} KB)")

    for wheel in wheels_found:
        shutil.copy2(str(wheel), str(upload_dir / wheel.name))
        print(f"  + {wheel.name} ({wheel.stat().st_size // 1024} KB)")

    result = _run(
        ["uv", "run", "kaggle", "datasets", "version",
         "-p", str(upload_dir),
         "-m", f"v{version} source — exhaustive matching for small video scenes",
         "--dir-mode", "zip"],
        check=False,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"  ✗ Dataset upload FAILED (rc={result.returncode})")
        entry["steps"][step] = f"fail_rc{result.returncode}"
        return False

    entry["steps"][step] = "pass"
    print(f"  ✓ Dataset uploaded")
    return True


# ---------------------------------------------------------------------------
# Step: push kernel to Kaggle
# ---------------------------------------------------------------------------

def step_push_kernel(version: int, entry: dict, smoke_only: bool) -> bool:
    step = "push_smoke" if smoke_only else "push_full"
    if entry["steps"].get(step) == "pass":
        print(f"  [skip] kernel already pushed for v{version} ({'smoke' if smoke_only else 'full'})")
        return True

    notebook_path = REPO_ROOT / "notebooks" / f"final_imc2025_lgeu_v{version}.py"
    if not notebook_path.exists():
        print(f"  ✗ Notebook not found: {notebook_path}")
        entry["steps"][step] = "fail_notebook_missing"
        return False

    # Read notebook, set SMOKE_TEST flag
    src = notebook_path.read_text()
    if smoke_only:
        src = src.replace("SMOKE_TEST = False", "SMOKE_TEST = True")
    else:
        src = src.replace("SMOKE_TEST = True", "SMOKE_TEST = False")

    kernel_slug = f"{KERNEL_PREFIX}{version}"
    kernel_full_slug = f"{KAGGLE_USER}/{kernel_slug}"
    kernel_dir = Path(f"/tmp/kernel_push_v{version}")
    kernel_dir.mkdir(exist_ok=True)

    (kernel_dir / "script.py").write_text(src)
    metadata = {
        "id": kernel_full_slug,
        "title": f"IMC2025 LGEU v{version}",
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [DATASET_SLUG, COMPETITION],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n[v{version}] Pushing kernel {'(SMOKE_TEST)' if smoke_only else '(FULL RUN)'} ...")
    result = _run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", str(kernel_dir)],
        check=False,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"  ✗ Kernel push FAILED (rc={result.returncode})")
        entry["steps"][step] = f"fail_rc{result.returncode}"
        return False

    entry["steps"][step] = "pass"
    entry["kernel_slug"] = kernel_full_slug
    print(f"  ✓ Kernel pushed: https://www.kaggle.com/code/{kernel_full_slug}")
    return True


# ---------------------------------------------------------------------------
# Step: wait for kernel to complete
# ---------------------------------------------------------------------------

def step_wait_kernel(version: int, entry: dict, timeout_min: int = 70) -> str:
    """Poll until kernel is COMPLETE or ERROR. Returns final status string."""
    kernel_slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    start = time.time()
    deadline = start + timeout_min * 60
    poll_interval = 30

    print(f"\n[v{version}] Waiting for kernel {kernel_slug} (timeout {timeout_min}m) ...")
    while time.time() < deadline:
        result = _run_kaggle(["kernels", "status", kernel_slug])
        status_line = (result.stdout + result.stderr).strip()
        print(f"  [{int((time.time()-start)/60):3d}m] {status_line}")
        if "COMPLETE" in status_line:
            return "COMPLETE"
        if "ERROR" in status_line or "CANCEL" in status_line:
            return "ERROR"
        time.sleep(poll_interval)

    return "TIMEOUT"


# ---------------------------------------------------------------------------
# Step: collect kernel outputs
# ---------------------------------------------------------------------------

def step_collect_outputs(version: int, entry: dict) -> bool:
    kernel_slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    out_dir = REPO_ROOT / "outputs" / f"full_run_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[v{version}] Collecting outputs from {kernel_slug} ...")
    result = _run(
        ["uv", "run", "kaggle", "kernels", "output", kernel_slug, "-p", str(out_dir)],
        check=False,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"  ✗ Output download FAILED")
        entry["steps"]["collect"] = "fail"
        return False

    # Parse diagnostics
    diag_path = out_dir / "submission_diagnostics.json"
    csv_path = out_dir / "submission.csv"
    summary = {}

    if csv_path.exists():
        with open(csv_path) as f:
            rows = sum(1 for _ in f) - 1  # subtract header
        summary["submission_rows"] = rows

    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        total_reg = 0
        total_imgs = 0
        dataset_errors = []
        for ds, info in diag.get("datasets", {}).items():
            if info.get("error"):
                dataset_errors.append(ds)
                continue
            for scene, sinfo in info.get("sfm_info", {}).items():
                total_reg += sinfo.get("registered", 0)
                total_imgs += sinfo.get("total", 0)
        summary["total_registered"] = total_reg
        summary["total_images_available"] = total_imgs
        summary["dataset_errors"] = dataset_errors
        summary["dataset_error_count"] = len(dataset_errors)

    entry["steps"]["collect"] = "pass"
    entry["results"] = summary
    print(f"  ✓ Outputs collected to {out_dir}")
    print(f"  Summary: {json.dumps(summary, indent=4)}")
    return True


# ---------------------------------------------------------------------------
# Step: check whether to submit + print instructions
# ---------------------------------------------------------------------------

def step_submission_check(version: int, entry: dict) -> None:
    """Check if a new submission is appropriate and print instructions."""
    print(f"\n[v{version}] Submission check ...")

    # Check pending submissions
    result = _run_kaggle(["competitions", "submissions", "-c", COMPETITION])
    lines = (result.stdout + result.stderr).strip().split("\n")
    pending = [l for l in lines if "PENDING" in l]
    complete = [l for l in lines if "COMPLETE" in l]
    print(f"  Existing: {len(pending)} PENDING, {len(complete)} COMPLETE")

    if pending:
        print(f"  ⚠ Active PENDING submission exists — wait for it to finish before submitting v{version}.")
        entry["steps"]["submission_check"] = "blocked_pending"
        return

    # CLI submission fails with 400 — provide manual procedure
    kernel_slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    kernel_url = f"https://www.kaggle.com/code/{kernel_slug}"
    print(f"""
  CLI submission returns HTTP 400 (competition API rejects late submissions).
  Manual submission procedure:
    1. Open {kernel_url}
    2. Click the three-dot menu (⋮) → "Submit to Competition"
       OR: Run the kernel via "Save Version" (already done), then from the
       Kaggle competition page go to "Submit Predictions" and select this kernel.
    3. Add description: "V{version} — exhaustive matching for video N<=100"
    4. Confirm submission.

  Alternatively try the CLI (may work if competition accepts submissions):
    uv run kaggle competitions submit {COMPETITION} \\
      -k {kernel_slug} -v <VERSION_NUMBER> \\
      -m "V{version} — exhaustive matching for video N<=100"

  Note: Replace <VERSION_NUMBER> with the kernel version number shown on Kaggle.
""")
    entry["steps"]["submission_check"] = "instructions_printed"


# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    version = args.version
    smoke_only = args.smoke_only

    log = _load_log()
    entry = _get_entry(log, version)
    _save_log(log)

    print(f"\n{'='*60}")
    print(f"Experiment controller: v{version} {'(SMOKE ONLY)' if smoke_only else '(FULL)'}")
    print(f"{'='*60}")

    # Step 1: local tests
    if not step_local_tests(version, entry):
        _save_log(log)
        sys.exit(1)
    _save_log(log)

    # Step 2: package src
    if not step_package_src(version, entry):
        _save_log(log)
        sys.exit(1)
    _save_log(log)

    # Step 3: upload dataset (skip if unchanged)
    if not args.skip_upload:
        if not step_upload_dataset(version, entry):
            _save_log(log)
            sys.exit(1)
        _save_log(log)

    # Step 4: push kernel
    if not step_push_kernel(version, entry, smoke_only=smoke_only):
        _save_log(log)
        sys.exit(1)
    _save_log(log)

    # Step 5: wait
    timeout = args.timeout if hasattr(args, "timeout") else (15 if smoke_only else 70)
    kernel_status = step_wait_kernel(version, entry, timeout_min=timeout)
    entry["kernel_status"] = kernel_status
    _save_log(log)

    if kernel_status == "ERROR":
        print(f"  ✗ Kernel FAILED — check https://www.kaggle.com/code/{KAGGLE_USER}/{KERNEL_PREFIX}{version}")
        sys.exit(1)
    elif kernel_status == "TIMEOUT":
        print(f"  ⚠ Kernel did not complete within {timeout}m — collect outputs later with:")
        print(f"    python scripts/experiment_controller.py collect --version {version}")
        sys.exit(0)

    # Step 6: collect
    step_collect_outputs(version, entry)
    _save_log(log)

    # Step 7: submission check (only for full runs)
    if not smoke_only:
        step_submission_check(version, entry)
        _save_log(log)

    print(f"\n{'='*60}")
    print(f"v{version} experiment complete. Results:")
    print(json.dumps(entry.get("results", {}), indent=2))
    print(f"{'='*60}")


def cmd_collect(args) -> None:
    log = _load_log()
    entry = _get_entry(log, args.version)
    step_collect_outputs(args.version, entry)
    _save_log(log)


def cmd_status(args) -> None:
    result = _run_kaggle(["competitions", "submissions", "-c", COMPETITION])
    print(result.stdout)
    print(result.stderr)


def cmd_history(args) -> None:
    log = _load_log()
    for key, entry in log.get("experiments", {}).items():
        print(f"\n{key}:")
        print(f"  steps  : {entry.get('steps', {})}")
        print(f"  results: {entry.get('results', {})}")
        print(f"  status : {entry.get('kernel_status', 'unknown')}")


def cmd_submit(args) -> None:
    """Attempt CLI submission, fall back to manual instructions."""
    version = args.version
    kernel_slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    v = args.kernel_version if hasattr(args, "kernel_version") else None
    v_flag = ["-v", str(v)] if v else []

    print(f"\nAttempting CLI submission of v{version} ...")
    result = _run(
        ["uv", "run", "kaggle", "competitions", "submit", COMPETITION,
         "-k", kernel_slug] + v_flag +
        ["-m", f"V{version} — exhaustive matching for video N<=100"],
        check=False,
        cwd=REPO_ROOT,
    )
    if result.returncode == 0:
        print("  ✓ Submission accepted")
    else:
        print(f"  ✗ CLI submission failed (rc={result.returncode})")
        print(f"  Manual procedure: open https://www.kaggle.com/code/{kernel_slug}")
        print("  and use the 'Submit to Competition' button.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IMC2025 LGEU experiment controller")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an experiment end-to-end")
    run_p.add_argument("--version", type=int, required=True)
    run_p.add_argument("--smoke-only", action="store_true", help="Run smoke test only")
    run_p.add_argument("--skip-upload", action="store_true", help="Skip dataset upload")
    run_p.add_argument("--timeout", type=int, default=70, help="Kernel wait timeout (minutes)")

    collect_p = sub.add_parser("collect", help="Download outputs for a version")
    collect_p.add_argument("--version", type=int, required=True)

    sub.add_parser("status", help="Show recent competition submissions")
    sub.add_parser("history", help="Show experiment log")

    submit_p = sub.add_parser("submit", help="Attempt submission (CLI or manual instructions)")
    submit_p.add_argument("--version", type=int, required=True)
    submit_p.add_argument("--kernel-version", type=int, help="Kaggle kernel version number")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "submit":
        cmd_submit(args)


if __name__ == "__main__":
    main()
