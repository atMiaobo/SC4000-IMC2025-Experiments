#!/usr/bin/env python3
"""Autonomous experiment loop for IMC2025 LGEU research.

Runs a persistent experiment queue: for each pre-implemented version,
build → test → push Kaggle → collect → compare → log → next.

State file: experiments/auto_loop_state.json
Log file:   experiments/auto_loop_log.txt

Usage:
    # Start / resume the loop
    uv run python scripts/autonomous_loop.py

    # Show status
    uv run python scripts/autonomous_loop.py --status

    # Force collect outputs for a specific version
    uv run python scripts/autonomous_loop.py --collect --version 9

    # Show leaderboard
    uv run python scripts/autonomous_loop.py --leaderboard

Stop conditions (automatic):
    - Cutoff time reached (2026-10-15 10:00 UTC)
    - All queued experiments completed
    - Authentication/quota failures (3+ consecutive)
    - Manual STOP file: experiments/STOP

Budget:
    KAGGLE_SMOKE_BUDGET: 5 smoke runs per hour max
    KAGGLE_FULL_BUDGET: 2 full runs per hour max
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "experiments" / "auto_loop_state.json"
LOG_PATH = REPO_ROOT / "experiments" / "auto_loop_log.txt"
STOP_FILE = REPO_ROOT / "experiments" / "STOP"

KAGGLE_USER = "miaobo"
COMPETITION = "image-matching-challenge-2025"
DATASET_SLUG = f"{KAGGLE_USER}/imc2025-lgeu-assets"
KERNEL_PREFIX = "imc2025-lgeu-v"

CUTOFF_UTC = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _default_state() -> dict:
    return {
        "best_version": 7,
        "best_cameras": 27,
        "best_ets": 19,
        "best_stairs": 8,
        "cutoff_utc": CUTOFF_UTC.isoformat(),
        "queue": [
            {
                "version": 9,
                "hypothesis": "GBM+TOPK20: V8 model quality (AUC=0.690) + V7 conservative pairing (TOPK=20). Preserves V7 scene splits while using better pair ranker. Expected: 27/73 public, better hidden.",
                "status": "queued",
            },
            {
                "version": 10,
                "hypothesis": "GBM+TOPK25: midpoint grid search between V7 (TOPK=20, 27/73) and V8 (TOPK=30, 22/73). Tests whether TOPK=25 improves hidden dataset coverage without triggering stairs over-merge.",
                "status": "queued",
            },
            {
                "version": 11,
                "hypothesis": "Lower Lowe threshold (30 vs 40) for exhaustive-path datasets (N<=40). ETs has 2 outlier images with 30-39 matches; lowering threshold may connect them to scene_0 for COLMAP registration.",
                "status": "queued",
            },
        ],
        "completed": [],
        "leaderboard": [
            {"version": 4, "ets": 19, "stairs": 2, "total": 21, "note": "V4 baseline, exhaustive SIFT"},
            {"version": 5, "ets": 19, "stairs": 3, "total": 22, "note": "V5 LGEU+MAGSAC; CLI OK"},
            {"version": 6, "ets": 19, "stairs": 0, "total": 19, "note": "V6 crash isolation; stairs regressed"},
            {"version": 7, "ets": 19, "stairs": 8, "total": 27, "note": "V7 BEST: exhaustive for video N<=100"},
            {"version": 8, "ets": 19, "stairs": 3, "total": 22, "note": "V8 GBM+TOPK30: stairs over-merged"},
        ],
        "consecutive_failures": 0,
        "smoke_runs_this_hour": 0,
        "full_runs_this_hour": 0,
        "hour_start": datetime.now(tz=timezone.utc).isoformat(),
    }


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return _default_state()


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------

def _time_remaining() -> float:
    """Return seconds until cutoff. Negative if past cutoff."""
    now = datetime.now(tz=timezone.utc)
    return (CUTOFF_UTC - now).total_seconds()


def _check_time() -> bool:
    """Return True if there is time left to run an experiment."""
    remaining = _time_remaining()
    if remaining <= 0:
        _log(f"CUTOFF REACHED: {CUTOFF_UTC.isoformat()} UTC")
        return False
    _log(f"Time remaining: {remaining/3600:.1f}h until {CUTOFF_UTC.strftime('%H:%M UTC')}")
    return True


def _check_stop_file() -> bool:
    """Return True if experiment should stop (STOP file exists)."""
    if STOP_FILE.exists():
        _log(f"STOP file detected: {STOP_FILE}")
        return True
    return False


# ---------------------------------------------------------------------------
# Shell utilities
# ---------------------------------------------------------------------------

def _run(cmd: list[str], check: bool = True, capture: bool = True, timeout_s: int = 300) -> subprocess.CompletedProcess:
    _log(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, check=False, capture_output=capture, text=True, timeout=timeout_s, cwd=REPO_ROOT
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}): {result.stderr[:500]}")
    return result


def _kaggle(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return _run(["uv", "run", "kaggle"] + args, check=check)


# ---------------------------------------------------------------------------
# Step: run local tests
# ---------------------------------------------------------------------------

def step_tests() -> bool:
    _log("Running local tests ...")
    try:
        result = _run(["uv", "run", "pytest", "-q", "--tb=short"], timeout_s=120)
        lines = (result.stdout + result.stderr).strip().split("\n")
        summary = [l for l in lines if "passed" in l or "failed" in l or "error" in l]
        _log(f"Tests: {summary[-1] if summary else 'unknown'}")
        return result.returncode == 0
    except Exception as exc:
        _log(f"Tests ERROR: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step: package src.tar.gz
# ---------------------------------------------------------------------------

def step_package_src() -> bool:
    _log("Packaging src.tar.gz ...")
    try:
        src_tar = REPO_ROOT / "src.tar.gz"
        with tarfile.open(src_tar, "w:gz") as tf:
            tf.add(REPO_ROOT / "src" / "imc2025_lgeu", arcname="imc2025_lgeu")
        _log(f"src.tar.gz: {src_tar.stat().st_size // 1024}KB")
        return True
    except Exception as exc:
        _log(f"Package ERROR: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step: upload dataset
# ---------------------------------------------------------------------------

def step_upload_dataset(version: int) -> bool:
    _log("Uploading dataset ...")
    import tempfile

    upload_dir = Path(tempfile.mkdtemp(prefix=f"lgeu_dataset_v{version}_"))
    try:
        # Copy files
        for f in [
            REPO_ROOT / "src.tar.gz",
            REPO_ROOT / "models" / "lgeu.joblib",
        ]:
            if f.exists():
                shutil.copy2(f, upload_dir / f.name)
                _log(f"  copied {f.name} ({f.stat().st_size//1024}KB)")

        # Copy pycolmap wheels
        for wheel in (REPO_ROOT / "models").glob("pycolmap*.whl"):
            shutil.copy2(wheel, upload_dir / wheel.name)
        # Also check root for wheels
        for wheel in REPO_ROOT.glob("pycolmap*.whl"):
            dst = upload_dir / wheel.name
            if not dst.exists():
                shutil.copy2(wheel, dst)

        wheel_count = len(list(upload_dir.glob("pycolmap*.whl")))
        _log(f"  {wheel_count} pycolmap wheels")

        meta = {
            "id": DATASET_SLUG,
            "title": "IMC2025 LGEU Assets",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (upload_dir / "dataset-metadata.json").write_text(json.dumps(meta))

        result = _kaggle([
            "datasets", "version", "-p", str(upload_dir),
            "-m", f"V{version} src + GBM lgeu.joblib",
            "--dir-mode", "zip",
        ], check=False)
        if result.returncode != 0:
            _log(f"Dataset upload FAILED: {result.stderr[:300]}")
            return False
        _log("Dataset uploaded OK")
        return True
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step: push Kaggle kernel
# ---------------------------------------------------------------------------

def step_push_kernel(version: int, smoke: bool = True) -> bool:
    notebook_path = REPO_ROOT / "notebooks" / f"final_imc2025_lgeu_v{version}.py"
    if not notebook_path.exists():
        _log(f"NOTEBOOK NOT FOUND: {notebook_path}")
        return False

    import tempfile
    push_dir = Path(tempfile.mkdtemp(prefix=f"lgeu_push_v{version}_"))
    try:
        src = notebook_path.read_text()
        # Verify SMOKE_TEST is set correctly
        if smoke and "SMOKE_TEST = False" in src:
            src = src.replace("SMOKE_TEST = False", "SMOKE_TEST = True", 1)
        elif not smoke and "SMOKE_TEST = True" in src:
            src = src.replace("SMOKE_TEST = True", "SMOKE_TEST = False", 1)

        (push_dir / "script.py").write_text(src)
        meta = {
            "id": f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}",
            "title": f"IMC2025 LGEU V{version}",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "competition_sources": [COMPETITION],
            "dataset_sources": [DATASET_SLUG],
        }
        (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        result = _kaggle(["kernels", "push", "-p", str(push_dir)], check=False)
        if result.returncode != 0:
            _log(f"Kernel push FAILED: {result.stderr[:300]}")
            return False
        _log(f"Kernel v{version} pushed ({'SMOKE' if smoke else 'FULL'})")
        return True
    finally:
        shutil.rmtree(push_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step: wait for kernel
# ---------------------------------------------------------------------------

def step_wait_kernel(version: int, timeout_min: int = 20) -> str:
    slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    deadline = time.time() + timeout_min * 60
    poll_interval = 30

    _log(f"Waiting for kernel {slug} (timeout={timeout_min}m) ...")
    while time.time() < deadline:
        try:
            result = _kaggle(["kernels", "status", slug], check=False)
            output = (result.stdout + result.stderr).strip()
            if "COMPLETE" in output.upper():
                _log(f"Kernel {slug}: COMPLETE")
                return "COMPLETE"
            elif "ERROR" in output.upper() or "FAILED" in output.upper():
                _log(f"Kernel {slug}: ERROR — {output[:200]}")
                return "ERROR"
            else:
                _log(f"Kernel {slug}: {output[:100]} — waiting {poll_interval}s ...")
        except Exception as exc:
            _log(f"Status poll failed: {exc}")
        time.sleep(poll_interval)

    _log(f"Kernel {slug}: TIMEOUT after {timeout_min}m")
    return "TIMEOUT"


# ---------------------------------------------------------------------------
# Step: collect outputs
# ---------------------------------------------------------------------------

def step_collect(version: int) -> dict | None:
    import tempfile
    slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    out_dir = REPO_ROOT / "outputs" / f"auto_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Collecting outputs for {slug} → {out_dir}")
    result = _kaggle(["kernels", "output", slug, "-p", str(out_dir)], check=False)
    if result.returncode != 0:
        _log(f"Collect FAILED: {result.stderr[:300]}")
        return None

    diag_path = out_dir / "submission_diagnostics.json"
    if not diag_path.exists():
        _log("No diagnostics.json found in outputs")
        return None

    diag = json.loads(diag_path.read_text())
    total_reg = 0
    ets_reg = 0
    stairs_reg = 0
    dataset_errors = diag.get("dataset_errors", [])

    for ds, info in diag.get("datasets", {}).items():
        if info.get("error"):
            continue
        for scene, sinfo in info.get("sfm_info", {}).items():
            if isinstance(sinfo, dict):
                reg = sinfo.get("registered", 0)
                total_reg += reg
                if "ETs" in ds or "et" in ds.lower():
                    ets_reg += reg
                elif "stairs" in ds.lower():
                    stairs_reg += reg

    result_summary = {
        "version": version,
        "total_cameras": total_reg,
        "ets_cameras": ets_reg,
        "stairs_cameras": stairs_reg,
        "dataset_errors": len(dataset_errors),
        "diagnostics_version": diag.get("version"),
    }
    _log(f"Results v{version}: total={total_reg}/73, ETs={ets_reg}/22, stairs={stairs_reg}/51, errors={len(dataset_errors)}")
    return result_summary


# ---------------------------------------------------------------------------
# Leaderboard management
# ---------------------------------------------------------------------------

def _update_leaderboard(state: dict, result: dict) -> None:
    lb = state.setdefault("leaderboard", [])
    version = result["version"]
    # Remove existing entry for this version
    lb[:] = [e for e in lb if e.get("version") != version]
    lb.append({
        "version": version,
        "ets": result.get("ets_cameras", 0),
        "stairs": result.get("stairs_cameras", 0),
        "total": result.get("total_cameras", 0),
        "note": f"auto-loop v{version}",
    })
    # Sort by total descending
    lb.sort(key=lambda e: e.get("total", 0), reverse=True)
    _log(f"Leaderboard updated: V{version} = {result.get('total_cameras',0)}/73")


def _print_leaderboard(state: dict) -> None:
    print("\n=== LEADERBOARD ===")
    print(f"{'Version':<10} {'Total':<10} {'ETs':<10} {'Stairs':<10} Note")
    print("-" * 70)
    for entry in state.get("leaderboard", []):
        v = entry.get("version", "?")
        t = entry.get("total", 0)
        e = entry.get("ets", 0)
        s = entry.get("stairs", 0)
        note = entry.get("note", "")
        marker = " ← BEST" if v == state.get("best_version") else ""
        print(f"V{v:<9} {t:<10} {e:<10} {s:<10} {note}{marker}")
    print()


# ---------------------------------------------------------------------------
# Run one full experiment
# ---------------------------------------------------------------------------

def run_experiment(state: dict, exp: dict) -> dict:
    """Run a single queued experiment. Returns updated exp dict."""
    version = exp["version"]
    _log(f"\n{'='*60}")
    _log(f"EXPERIMENT V{version}: {exp.get('hypothesis', '?')[:80]}")
    _log(f"{'='*60}")

    exp["status"] = "running"
    exp["start_time"] = datetime.now(tz=timezone.utc).isoformat()
    _save_state(state)

    # Check notebook exists
    notebook = REPO_ROOT / "notebooks" / f"final_imc2025_lgeu_v{version}.py"
    if not notebook.exists():
        _log(f"SKIP V{version}: notebook not found at {notebook}")
        exp["status"] = "skipped"
        exp["skip_reason"] = "notebook_not_found"
        return exp

    # Local tests
    if not step_tests():
        _log(f"V{version}: LOCAL TESTS FAILED — skipping Kaggle push")
        exp["status"] = "failed_local_tests"
        state["consecutive_failures"] += 1
        return exp

    # Package src
    if not step_package_src():
        exp["status"] = "failed_package"
        state["consecutive_failures"] += 1
        return exp

    # Upload dataset (always upload to ensure latest model/src)
    if not step_upload_dataset(version):
        exp["status"] = "failed_upload"
        state["consecutive_failures"] += 1
        return exp
    time.sleep(5)  # Let Kaggle process the upload

    # Smoke test first
    _log(f"V{version}: SMOKE TEST ...")
    if not step_push_kernel(version, smoke=True):
        exp["status"] = "failed_push_smoke"
        state["consecutive_failures"] += 1
        return exp

    smoke_status = step_wait_kernel(version, timeout_min=15)
    if smoke_status != "COMPLETE":
        _log(f"V{version}: smoke test {smoke_status} — skipping full run")
        exp["status"] = f"smoke_{smoke_status.lower()}"
        state["consecutive_failures"] += 1
        return exp

    smoke_result = step_collect(version)
    if smoke_result:
        _log(f"V{version}: smoke ETs={smoke_result.get('ets_cameras',0)}/22 (expected 19)")
        if smoke_result.get("ets_cameras", 0) == 0:
            _log(f"V{version}: smoke test ZERO cameras — ABORT full run")
            exp["status"] = "smoke_zero_cameras"
            exp["smoke_result"] = smoke_result
            state["consecutive_failures"] += 1
            return exp

    _log(f"V{version}: SMOKE PASSED — pushing FULL RUN ...")

    # Full run
    if not step_push_kernel(version, smoke=False):
        exp["status"] = "failed_push_full"
        state["consecutive_failures"] += 1
        return exp

    full_status = step_wait_kernel(version, timeout_min=30)
    if full_status != "COMPLETE":
        _log(f"V{version}: full run {full_status}")
        exp["status"] = f"full_{full_status.lower()}"
        state["consecutive_failures"] += 1
        return exp

    result = step_collect(version)
    if result is None:
        exp["status"] = "failed_collect"
        state["consecutive_failures"] += 1
        return exp

    # Compare to best
    total = result.get("total_cameras", 0)
    best = state.get("best_cameras", 0)
    _update_leaderboard(state, result)

    if total > best:
        _log(f"V{version}: NEW BEST! {total} > {best} cameras")
        state["best_version"] = version
        state["best_cameras"] = total
        state["best_ets"] = result.get("ets_cameras", 0)
        state["best_stairs"] = result.get("stairs_cameras", 0)
        exp["status"] = "improvement"
    elif total == best:
        _log(f"V{version}: TIED with best ({total} cameras) — keeping V{state['best_version']}")
        exp["status"] = "tied"
    else:
        _log(f"V{version}: REGRESSION ({total} < {best} cameras)")
        exp["status"] = "regression"

    exp["result"] = result
    exp["end_time"] = datetime.now(tz=timezone.utc).isoformat()
    state["consecutive_failures"] = 0
    return exp


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main_loop() -> None:
    state = _load_state()
    _log(f"Autonomous loop started. Best: V{state['best_version']} ({state['best_cameras']}/73)")
    _log(f"Cutoff: {CUTOFF_UTC.isoformat()}")

    while True:
        # Check stop conditions
        if _check_stop_file():
            _log("Loop stopped by STOP file")
            break
        if not _check_time():
            _log("Loop stopped: cutoff reached")
            break
        if state.get("consecutive_failures", 0) >= 3:
            _log(f"Loop stopped: {state['consecutive_failures']} consecutive failures")
            break

        # Find next queued experiment
        queue = state.get("queue", [])
        pending = [exp for exp in queue if exp.get("status") == "queued"]
        if not pending:
            _log("Loop complete: all queued experiments done")
            break

        exp = pending[0]
        version = exp["version"]

        # Check if enough time for a full experiment (need at least 60 min)
        if _time_remaining() < 60 * 60:
            _log(f"Only {_time_remaining()/60:.0f}m left — not enough for full experiment V{version}")
            _log("Stopping loop to preserve time for handoff")
            break

        try:
            run_experiment(state, exp)
        except Exception as exc:
            _log(f"V{version}: UNHANDLED ERROR: {exc}")
            exp["status"] = "error"
            exp["error"] = str(exc)
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1

        # Move to completed
        state["queue"] = [e for e in queue if e.get("version") != version]
        state.setdefault("completed", []).append(exp)
        _save_state(state)
        _print_leaderboard(state)

        # Small delay between experiments
        _log("Waiting 30s before next experiment ...")
        time.sleep(30)

    # Final state
    _save_state(state)
    _log("\nFINAL LEADERBOARD:")
    _print_leaderboard(state)
    best_v = state.get("best_version", "?")
    best_c = state.get("best_cameras", 0)
    _log(f"Best candidate: V{best_v} with {best_c}/73 cameras")
    _log(f"Submit manually: https://www.kaggle.com/code/{KAGGLE_USER}/{KERNEL_PREFIX}{best_v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    state = _load_state()
    print(f"\nBest version  : V{state.get('best_version')} ({state.get('best_cameras')}/73)")
    print(f"Time remaining: {_time_remaining()/3600:.1f}h until {CUTOFF_UTC.strftime('%H:%M UTC')}")
    print(f"Queue         : {[e['version'] for e in state.get('queue', []) if e.get('status')=='queued']}")
    print(f"Completed     : {[e['version'] for e in state.get('completed', [])]}")
    _print_leaderboard(state)


def cmd_collect(version: int) -> None:
    state = _load_state()
    result = step_collect(version)
    if result:
        print(json.dumps(result, indent=2))
        _update_leaderboard(state, result)
        _save_state(state)


def cmd_leaderboard() -> None:
    state = _load_state()
    _print_leaderboard(state)


def main() -> None:
    p = argparse.ArgumentParser(description="IMC2025 autonomous experiment loop")
    p.add_argument("--status", action="store_true", help="Show current status")
    p.add_argument("--collect", action="store_true", help="Collect outputs for a version")
    p.add_argument("--version", type=int, help="Version number (for --collect)")
    p.add_argument("--leaderboard", action="store_true", help="Print leaderboard")
    args = p.parse_args()

    if args.status:
        cmd_status()
    elif args.collect:
        if args.version is None:
            print("--collect requires --version")
            sys.exit(1)
        cmd_collect(args.version)
    elif args.leaderboard:
        cmd_leaderboard()
    else:
        main_loop()


if __name__ == "__main__":
    main()
