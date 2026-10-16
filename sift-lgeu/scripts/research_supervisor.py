#!/usr/bin/env python3
"""Genuine Claude-driven autonomous research supervisor for IMC2025 LGEU.

Each iteration:
  1. Builds context from leaderboard, diagnostics, and source code
  2. Invokes Claude Code non-interactively to analyse results and implement a new hypothesis
  3. Runs unit tests (with one Claude-driven fix attempt on failure)
  4. Packages code and runs a Kaggle smoke test → full run
  5. Updates the leaderboard and loops immediately

State:  experiments/research_state.json
Log:    experiments/research_log.txt
STOP:   experiments/STOP  (touch this to stop after current experiment)
Lock:   experiments/.supervisor.lock

Usage:
    # Start / resume loop
    uv run python scripts/research_supervisor.py

    # Status
    uv run python scripts/research_supervisor.py --status

    # Stop gracefully
    touch experiments/STOP

Stop conditions (automatic):
    - STOP file exists
    - Cutoff time reached (2026-10-15 10:00 UTC = 18:00 SGT)
    - Total Claude budget exhausted (default $25)
    - 3 consecutive complete failures (test AND Kaggle both fail)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
STATE_PATH = REPO_ROOT / "experiments" / "research_state.json"
LOG_PATH = REPO_ROOT / "experiments" / "research_log.txt"
STOP_PATH = REPO_ROOT / "experiments" / "STOP"
LOCK_PATH = REPO_ROOT / "experiments" / ".supervisor.lock"

CUTOFF_UTC = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)  # 18:00 Singapore

# Kaggle config
KAGGLE_USER = "miaobo"
COMPETITION = "image-matching-challenge-2025"
KERNEL_PREFIX = "imc2025-lgeu-v"
DATASET_SLUG = "miaobo/imc2025-lgeu-assets"

# Budget and failure limits
MAX_BUDGET_USD_PER_ITER = 2.50   # max $ per Claude planning invocation
MAX_BUDGET_USD_TOTAL = 25.00     # session-wide budget
MAX_CONSECUTIVE_FAILURES = 3     # stop after this many experiments that fail all stages
INTER_EXPERIMENT_SLEEP_S = 30    # pause between experiments

# Kaggle polling
SMOKE_TIMEOUT_MIN = 20
FULL_TIMEOUT_MIN = 40

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    line = f"[{datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _default_state() -> dict:
    """Initialise from existing auto_loop results if available."""
    base: dict = {
        "best_version": 7,
        "best_cameras": 27,
        "best_ets": 19,
        "best_stairs": 8,
        "next_version": 12,
        "experiments": [],
        "leaderboard": [
            {"version": 4,  "ets": 19, "stairs": 2,  "total": 21, "note": "V4 baseline"},
            {"version": 5,  "ets": 19, "stairs": 3,  "total": 22, "note": "V5 LGEU+MAGSAC"},
            {"version": 6,  "ets": 19, "stairs": 0,  "total": 19, "note": "V6 crash-isolation"},
            {"version": 7,  "ets": 19, "stairs": 8,  "total": 27, "note": "V7 BEST (LR+TOPK20)"},
            {"version": 8,  "ets": 19, "stairs": 3,  "total": 22, "note": "V8 GBM+TOPK30 regression"},
            {"version": 9,  "ets": 19, "stairs": 6,  "total": 25, "note": "V9 GBM+TOPK20"},
            {"version": 10, "ets": 19, "stairs": 3,  "total": 22, "note": "V10 GBM+TOPK25"},
            {"version": 11, "ets": 10, "stairs": 6,  "total": 16, "note": "V11 lower-Lowe-thr regression"},
        ],
        "budget": {"spent_usd": 0.0, "limit_usd": MAX_BUDGET_USD_TOTAL, "invocations": 0},
        "consecutive_failures": 0,
    }
    # Load completed experiments from auto_loop if present
    auto = REPO_ROOT / "experiments" / "auto_loop_state.json"
    if auto.exists():
        adata = json.loads(auto.read_text())
        for c in adata.get("completed", []):
            base["experiments"].append({
                "version": c["version"],
                "hypothesis": c.get("hypothesis", ""),
                "status": c.get("status", "unknown"),
                "result": c.get("result", {}),
                "cost_usd": 0.0,
                "source": "auto_loop",
            })
    return base


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return _default_state()


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))

# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str],
    check: bool = True,
    capture: bool = True,
    timeout_s: int = 300,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    _log(f"$ {' '.join(str(c) for c in cmd)}")
    kw: dict = dict(check=False, capture_output=capture, text=True, timeout=timeout_s,
                    cwd=REPO_ROOT)
    if env:
        kw["env"] = {**os.environ, **env}
    result = subprocess.run(cmd, **kw)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}): {(result.stderr or '')[:600]}"
        )
    return result


def _kaggle(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return _run(["uv", "run", "kaggle"] + args, check=check)

# ---------------------------------------------------------------------------
# Time / budget helpers
# ---------------------------------------------------------------------------

def _time_remaining_s() -> float:
    return max(0.0, (CUTOFF_UTC - datetime.now(tz=timezone.utc)).total_seconds())


def _budget_remaining(state: dict) -> float:
    return state["budget"]["limit_usd"] - state["budget"]["spent_usd"]


def _should_stop(state: dict) -> tuple[bool, str]:
    if STOP_PATH.exists():
        return True, "STOP file found"
    if _time_remaining_s() <= 0:
        return True, f"Cutoff reached ({CUTOFF_UTC.isoformat()})"
    if _budget_remaining(state) <= 0:
        return True, f"Budget exhausted (${state['budget']['spent_usd']:.2f} / ${state['budget']['limit_usd']:.2f})"
    if state.get("consecutive_failures", 0) >= MAX_CONSECUTIVE_FAILURES:
        return True, f"{MAX_CONSECUTIVE_FAILURES} consecutive failures"
    return False, ""

# ---------------------------------------------------------------------------
# Context building for Claude
# ---------------------------------------------------------------------------

def _build_context(state: dict) -> str:
    lines: list[str] = []

    # Leaderboard
    lines.append("## Experiment Leaderboard (sorted by total cameras)")
    lines.append(f"{'Version':<8} {'Total':<7} {'ETs/22':<8} {'Stairs/51':<11} Notes")
    lines.append("-" * 70)
    best_v = state.get("best_version", 7)
    for e in sorted(state["leaderboard"], key=lambda x: x.get("total", 0), reverse=True):
        marker = " ← BEST" if e["version"] == best_v else ""
        lines.append(
            f"V{e['version']:<7} {e['total']:<7} {e['ets']:<8} {e['stairs']:<11} {e.get('note','')}{marker}"
        )
    lines.append("")

    # Experiment details (most recent 6)
    lines.append("## Recent Experiments (full details)")
    for exp in state.get("experiments", [])[-6:]:
        lines.append(f"\n### V{exp['version']} — {exp.get('status','?')}")
        lines.append(f"Hypothesis: {exp.get('hypothesis','N/A')}")
        r = exp.get("result", {})
        if r:
            lines.append(f"Result: total={r.get('total_cameras','?')}/73, "
                         f"ETs={r.get('ets_cameras','?')}/22, "
                         f"stairs={r.get('stairs_cameras','?')}/51")
    lines.append("")

    # V9 diagnostics (the richest completed experiment with per-scene detail)
    diag_path = REPO_ROOT / "outputs" / "auto_v9" / "submission_diagnostics.json"
    if diag_path.exists():
        lines.append("## V9 Per-Scene Diagnostics (most informative)")
        try:
            diag = json.loads(diag_path.read_text())
            for ds, info in diag.get("datasets", {}).items():
                if info.get("images_available", 0) == 0:
                    continue
                lines.append(f"\nDataset: {ds} ({info['images_available']} images)")
                for sc, si in info.get("sfm_info", {}).items():
                    if not isinstance(si, dict):
                        continue
                    reg = si.get("registered", "?")
                    tot = si.get("total", "?")
                    err = si.get("error", "")
                    video = si.get("is_video", False)
                    mode = si.get("match_mode", "?")
                    lines.append(f"  {sc}: {reg}/{tot} cameras  video={video}  match={mode}  {err}")
        except Exception:
            pass
    lines.append("")

    # Key source code excerpts
    sc_path = REPO_ROOT / "src" / "imc2025_lgeu" / "scene_clustering.py"
    if sc_path.exists():
        src = sc_path.read_text()
        lines.append("## scene_clustering.py — key patterns")
        # Extract discover_scenes_v6 (the base function all others call)
        m = re.search(r"(def discover_scenes_v6\b.*?)(?=\ndef |\Z)", src, re.DOTALL)
        if m:
            lines.append("### discover_scenes_v6 (base function — all versions delegate here):")
            lines.append(m.group(1)[:2000])
        # Extract latest version
        all_versions = re.findall(r"def discover_scenes_v(\d+)\b", src)
        if all_versions:
            latest_v = max(all_versions, key=int)
            m2 = re.search(
                rf"(def discover_scenes_v{latest_v}\b.*?)(?=\ndef |\Z)", src, re.DOTALL
            )
            if m2:
                lines.append(f"\n### discover_scenes_v{latest_v} (most recent version):")
                lines.append(m2.group(1)[:1500])

    sfm_path = REPO_ROOT / "src" / "imc2025_lgeu" / "sfm.py"
    if sfm_path.exists():
        sfm_src = sfm_path.read_text()
        lines.append("\n## sfm.py — run_sfm_per_scene_v7 signature")
        m3 = re.search(r"(def run_sfm_per_scene_v7\b.*?)(?=\ndef |\Z)", sfm_src, re.DOTALL)
        if m3:
            lines.append(m3.group(1)[:2000])

    sub_path = REPO_ROOT / "src" / "imc2025_lgeu" / "submission.py"
    if sub_path.exists():
        sub_src = sub_path.read_text()
        lines.append("\n## submission.py — build_submission_v9 signature (template)")
        m4 = re.search(r"(def build_submission_v9\b.*?)(?=\ndef build_submission|\Z)",
                       sub_src, re.DOTALL)
        if m4:
            lines.append(m4.group(1)[:2000])

    # Dataset context
    lines.append("\n## Dataset Context")
    lines.append(textwrap.dedent("""
    Test data structure (at Kaggle evaluation time):
      ETs:   22 images, non-video, exhaustive SIFT matching.
             V9 diagnostics: scene_0 (10/11), scene_1 (9/9), outliers (0/2 — no_reconstruction).
             The 2 outlier images have < 40 Lowe matches with both main scenes.

      stairs: 51 images total, is_video=True for most scenes.
              Two simultaneous multi-camera recordings with OVERLAPPING timestamps.
              V9 diagnostics: scene_0 (3/15 video), scene_1 (0/2 no_recon),
                              outliers (0/14 video no_recon!), scene_2 (3/20 video).
              V7 (LR model, TOPK=20): 8/51 — scene_0 5/15, scene_2 3/20.
              V9 (GBM model, TOPK=20): 6/51 — GBM selects different pairs than LR.
              14 images end up in 'outliers' with no reconstruction at all.

    Other test datasets (hidden — not accessible locally):
      amy_gardens (200 imgs), fbk_vineyard (163), imc2023_haiper (54),
      imc2023_heritage (209), urban_country (113), ... etc.
      These are the primary source of the remaining 73 - 22 - 51 = 0 cameras
      (i.e., ALL public cameras come from ETs + stairs).
      Improvement must target ETs and/or stairs registration without breaking other datasets.
    """).strip())

    return "\n".join(lines)


def _build_implementation_prompt(version: int, context: str, best_version: int) -> str:
    return textwrap.dedent(f"""
    You are the research engineer for the IMC2025 image matching competition.
    Your task: implement ONE new pipeline version (V{version}) that may improve camera registration.

    ═══════════════════════════════════════════════════════
    CONTEXT
    ═══════════════════════════════════════════════════════
    {context}

    ═══════════════════════════════════════════════════════
    YOUR TASK: Implement V{version}
    ═══════════════════════════════════════════════════════

    1. **Analyse the bottleneck**: Look at the leaderboard and diagnostics.
       Identify the single most impactful change that might improve camera registration.
       Focus on SPECIFIC failures (e.g., why do 14 stairs images land in "outliers" with 0 cameras?
       Why does GBM select worse pairs than LR for stairs?).

    2. **Propose ONE hypothesis**: Write a concise, testable hypothesis.
       This must be a DIFFERENT approach from all previous versions.
       Do not repeat TOPK tuning (tried V8=30, V9=20, V10=25).
       Do not repeat Lowe threshold tuning (tried V11=30).

    3. **Implement V{version}** by editing these files:

       a) src/imc2025_lgeu/scene_clustering.py
          Add: `def discover_scenes_v{version}(image_paths, ...) -> tuple:`
          May call discover_scenes_v6() with new parameters, OR implement new logic.
          MUST NOT modify any existing function.

       b) src/imc2025_lgeu/sfm.py
          Add at end: `run_sfm_per_scene_v{version} = run_sfm_per_scene_v7`
          OR a real new function if you are changing SfM settings.

       c) src/imc2025_lgeu/submission.py
          Add import for the new discover/sfm functions (update the existing import block).
          Add: `def build_submission_v{version}(sample_path, image_root, output_path,
               lgeu_model_payload=None, diagnostics_path=None, verbose=False) -> pd.DataFrame:`
          Copy the pattern from build_submission_v9 — change version number and function names.

       d) notebooks/final_imc2025_lgeu_v{version}.py
          Copy from notebooks/final_imc2025_lgeu_v11.py.
          Update: docstring (describe hypothesis), version number, SMOKE_TEST=False,
          import build_submission_v{version}, call build_submission_v{version}(),
          result dict version="{version}".

       e) tests/test_v{version}.py
          Write ≥ 2 tests that verify the new behaviour.
          Import discover_scenes_v{version} from imc2025_lgeu.scene_clustering.
          Tests must be runnable with `uv run pytest tests/test_v{version}.py -q`.

       f) experiments/v{version}_hypothesis.txt
          Write ONE paragraph describing your hypothesis and why you expect it to help.

    4. **CRITICAL constraints**:
       - Do NOT modify V1–V{best_version} or any existing function.
       - Do NOT add SMOKE_TEST = True in the notebook (it must be False for full run).
       - Each scene_clustering function must return a tuple (scene_map, edge_data).
       - The notebook must import and call build_submission_v{version}.
       - Keep version={version} in the diagnostics dict inside the notebook.
       - Do NOT invent scores or results. Do NOT attempt Kaggle submissions.

    5. After implementation, output a SHORT summary (≤ 5 lines):
       - What you changed
       - What bottleneck you targeted
       - Why you expect improvement

    Begin by reading the source files (scene_clustering.py, sfm.py, submission.py,
    notebooks/final_imc2025_lgeu_v11.py) to understand the exact patterns, then implement.
    """).strip()


def _build_fix_prompt(version: int, test_output: str) -> str:
    return textwrap.dedent(f"""
    The unit tests for V{version} failed. Fix the implementation.

    Test output:
    ```
    {test_output[-3000:]}
    ```

    Instructions:
    - Read the failing test file and the source files it imports.
    - Fix the implementation so all tests pass.
    - Do NOT modify any function from V1 through V11.
    - Do NOT add new tests; fix the existing code.
    - Run `uv run pytest tests/test_v{version}.py -v` mentally to verify your fix is correct.

    After fixing, output "FIXED" and a one-line description of what was wrong.
    """).strip()

# ---------------------------------------------------------------------------
# Claude invocation
# ---------------------------------------------------------------------------

def _invoke_claude(prompt: str, budget_usd: float, label: str = "claude") -> tuple[bool, str, float]:
    """Run `claude -p prompt` and return (success, result_text, cost_usd)."""
    _log(f"Invoking Claude ({label}, budget=${budget_usd:.2f}) ...")
    try:
        result = _run(
            [
                "claude", "-p", prompt,
                "--allowedTools", "Read,Edit,Write",
                "--output-format", "json",
                "--max-budget-usd", str(budget_usd),
                "--model", "sonnet",
            ],
            check=False,
            timeout_s=int(budget_usd * 600 + 300),  # scale timeout with budget
        )
    except subprocess.TimeoutExpired:
        _log(f"Claude timed out ({label})")
        return False, "", 0.0

    # Parse JSON output
    raw = (result.stdout or "").strip()
    if not raw:
        _log(f"Claude produced no output ({label}). stderr: {(result.stderr or '')[:300]}")
        return False, "", 0.0

    try:
        # claude may emit non-JSON warnings before the JSON; find the last {...} blob
        json_start = raw.rfind('{"type"')
        if json_start == -1:
            json_start = raw.rfind("{")
        data = json.loads(raw[json_start:])
    except json.JSONDecodeError:
        _log(f"Could not parse Claude JSON output ({label}): {raw[:300]}")
        return False, "", 0.0

    cost = float(data.get("total_cost_usd", 0.0))
    is_error = data.get("is_error", False)
    result_text = data.get("result", "")
    subtype = data.get("subtype", "")

    _log(f"Claude done ({label}): subtype={subtype}, cost=${cost:.4f}, turns={data.get('num_turns',0)}")

    if is_error:
        _log(f"Claude error ({label}): {data.get('error', subtype)}")
        return False, result_text, cost

    return True, result_text, cost

# ---------------------------------------------------------------------------
# Implementation verification
# ---------------------------------------------------------------------------

def _verify_implementation(version: int) -> tuple[bool, str]:
    """Check that all required V{version} artifacts exist."""
    errors: list[str] = []

    sc_path = REPO_ROOT / "src" / "imc2025_lgeu" / "scene_clustering.py"
    if not re.search(rf"def discover_scenes_v{version}\b", sc_path.read_text()):
        errors.append(f"discover_scenes_v{version} missing from scene_clustering.py")

    sfm_path = REPO_ROOT / "src" / "imc2025_lgeu" / "sfm.py"
    if not re.search(rf"run_sfm_per_scene_v{version}\b", sfm_path.read_text()):
        errors.append(f"run_sfm_per_scene_v{version} missing from sfm.py")

    sub_path = REPO_ROOT / "src" / "imc2025_lgeu" / "submission.py"
    if not re.search(rf"def build_submission_v{version}\b", sub_path.read_text()):
        errors.append(f"build_submission_v{version} missing from submission.py")

    nb_path = REPO_ROOT / "notebooks" / f"final_imc2025_lgeu_v{version}.py"
    if not nb_path.exists():
        errors.append(f"notebook notebooks/final_imc2025_lgeu_v{version}.py missing")
    else:
        nb_src = nb_path.read_text()
        if f"build_submission_v{version}" not in nb_src:
            errors.append(f"notebook does not call build_submission_v{version}")
        if "SMOKE_TEST = True" in nb_src:
            # Patch it silently
            nb_path.write_text(nb_src.replace("SMOKE_TEST = True", "SMOKE_TEST = False", 1))

    test_path = REPO_ROOT / "tests" / f"test_v{version}.py"
    if not test_path.exists():
        errors.append(f"tests/test_v{version}.py missing")

    if errors:
        return False, "; ".join(errors)
    return True, "OK"

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests() -> tuple[bool, str]:
    try:
        r = _run(["uv", "run", "pytest", "-q", "--tb=short"], timeout_s=180, check=False)
        out = (r.stdout or "") + (r.stderr or "")
        lines = out.strip().split("\n")
        summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l), "")
        _log(f"Tests: {summary or 'unknown'}")
        return r.returncode == 0, out
    except Exception as exc:
        return False, str(exc)

# ---------------------------------------------------------------------------
# Kaggle pipeline steps (mirrors auto_loop.py)
# ---------------------------------------------------------------------------

def _step_package_src() -> bool:
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


def _step_upload_dataset(version: int) -> bool:
    import tempfile
    _log("Uploading dataset to Kaggle ...")
    upload_dir = Path(tempfile.mkdtemp(prefix=f"lgeu_ds_v{version}_"))
    try:
        for f in [REPO_ROOT / "src.tar.gz", REPO_ROOT / "models" / "lgeu.joblib"]:
            if f.exists():
                shutil.copy2(f, upload_dir / f.name)
                _log(f"  {f.name}: {f.stat().st_size // 1024}KB")
        # Include pycolmap wheel if present
        wheel_count = 0
        for wheel in list((REPO_ROOT / "models").glob("pycolmap*.whl")) + list(REPO_ROOT.glob("pycolmap*.whl")):
            dst = upload_dir / wheel.name
            if not dst.exists():
                shutil.copy2(wheel, dst)
                wheel_count += 1
        _log(f"  {wheel_count} pycolmap wheel(s)")
        meta = {
            "id": DATASET_SLUG,
            "title": "IMC2025 LGEU Assets",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (upload_dir / "dataset-metadata.json").write_text(json.dumps(meta))
        r = _kaggle([
            "datasets", "version", "-p", str(upload_dir),
            "-m", f"V{version} research-supervisor",
            "--dir-mode", "zip",
        ], check=False)
        if r.returncode != 0:
            _log(f"Dataset upload FAILED: {(r.stderr or '')[:300]}")
            return False
        _log("Dataset uploaded OK")
        return True
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


def _step_push_kernel(version: int, smoke: bool = True) -> bool:
    import tempfile
    nb_path = REPO_ROOT / "notebooks" / f"final_imc2025_lgeu_v{version}.py"
    if not nb_path.exists():
        _log(f"Notebook not found: {nb_path}")
        return False
    push_dir = Path(tempfile.mkdtemp(prefix=f"lgeu_push_v{version}_"))
    try:
        src = nb_path.read_text()
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
        r = _kaggle(["kernels", "push", "-p", str(push_dir)], check=False)
        if r.returncode != 0:
            _log(f"Kernel push FAILED: {(r.stderr or '')[:300]}")
            return False
        _log(f"Kernel v{version} pushed ({'SMOKE' if smoke else 'FULL'})")
        return True
    finally:
        shutil.rmtree(push_dir, ignore_errors=True)


def _step_wait_kernel(version: int, timeout_min: int = 20) -> str:
    slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    deadline = time.time() + timeout_min * 60
    _log(f"Waiting for {slug} (timeout={timeout_min}m) ...")
    while time.time() < deadline:
        try:
            r = _kaggle(["kernels", "status", slug], check=False)
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            upper = out.upper()
            if "COMPLETE" in upper:
                _log(f"{slug}: COMPLETE")
                return "COMPLETE"
            if "ERROR" in upper or "FAILED" in upper or "CANCELATION" in upper:
                _log(f"{slug}: ERROR — {out[:200]}")
                return "ERROR"
            _log(f"{slug}: {out[:100]} — waiting 30s ...")
        except Exception as exc:
            _log(f"Status poll error: {exc}")
        time.sleep(30)
    _log(f"{slug}: TIMEOUT after {timeout_min}m")
    return "TIMEOUT"


def _step_collect(version: int) -> dict | None:
    slug = f"{KAGGLE_USER}/{KERNEL_PREFIX}{version}"
    out_dir = REPO_ROOT / "outputs" / f"research_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Collecting outputs for {slug} ...")
    r = _kaggle(["kernels", "output", slug, "-p", str(out_dir)], check=False)
    if r.returncode != 0:
        _log(f"Collect FAILED: {(r.stderr or '')[:300]}")
        return None
    diag_path = out_dir / "submission_diagnostics.json"
    if not diag_path.exists():
        _log("No diagnostics.json in outputs")
        return None
    diag = json.loads(diag_path.read_text())
    total = ets = stairs = errors = 0
    for ds, info in diag.get("datasets", {}).items():
        if info.get("error"):
            errors += 1
        for sc, si in info.get("sfm_info", {}).items():
            if isinstance(si, dict):
                reg = si.get("registered", 0)
                total += reg
                if "et" in ds.lower() or "ETs" in ds:
                    ets += reg
                elif "stairs" in ds.lower():
                    stairs += reg
    summary = {
        "version": version,
        "total_cameras": total,
        "ets_cameras": ets,
        "stairs_cameras": stairs,
        "dataset_errors": errors,
        "diagnostics_version": diag.get("version"),
    }
    _log(f"Results v{version}: total={total}/73, ETs={ets}/22, stairs={stairs}/51, errors={errors}")
    return summary

# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def _update_leaderboard(state: dict, result: dict, note: str = "") -> None:
    lb = state.setdefault("leaderboard", [])
    v = result["version"]
    lb[:] = [e for e in lb if e.get("version") != v]
    lb.append({
        "version": v,
        "ets": result.get("ets_cameras", 0),
        "stairs": result.get("stairs_cameras", 0),
        "total": result.get("total_cameras", 0),
        "note": note or f"research-supervisor v{v}",
    })
    lb.sort(key=lambda e: e.get("total", 0), reverse=True)


def _print_leaderboard(state: dict) -> None:
    print("\n=== LEADERBOARD ===")
    print(f"{'Version':<8} {'Total':<7} {'ETs':<7} {'Stairs':<9} Note")
    print("-" * 65)
    best_v = state.get("best_version", 7)
    for e in state.get("leaderboard", []):
        marker = " ← BEST" if e["version"] == best_v else ""
        print(f"V{e['version']:<7} {e.get('total',0):<7} {e.get('ets',0):<7} {e.get('stairs',0):<9} {e.get('note','')}{marker}")
    spent = state["budget"]["spent_usd"]
    limit = state["budget"]["limit_usd"]
    print(f"\nBudget: ${spent:.3f} / ${limit:.2f} | Invocations: {state['budget']['invocations']}")
    print(f"Best:   V{state['best_version']} ({state['best_cameras']}/73)")

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_status_available() -> bool:
    """Check if git repo has at least one commit."""
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=False
    )
    return r.returncode == 0


def _git_commit(message: str) -> bool:
    """Stage all changes and commit."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, check=True,
                       capture_output=True)
        r = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=REPO_ROOT, check=False, capture_output=True, text=True
        )
        if r.returncode == 0:
            _log(f"Git commit: {message}")
            return True
        # Nothing to commit is not an error
        if "nothing to commit" in (r.stdout + r.stderr):
            return True
        _log(f"Git commit warning: {(r.stdout + r.stderr)[:200]}")
        return False
    except Exception as exc:
        _log(f"Git commit error: {exc}")
        return False

# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_one_experiment(state: dict) -> None:
    version = state["next_version"]
    best_v = state.get("best_version", 7)
    best_cams = state.get("best_cameras", 27)

    _log("")
    _log("=" * 60)
    _log(f"EXPERIMENT V{version} — best so far: V{best_v} ({best_cams}/73)")
    _log(f"Budget remaining: ${_budget_remaining(state):.2f} | Time: {_time_remaining_s()/3600:.1f}h")
    _log("=" * 60)

    exp: dict = {
        "version": version,
        "hypothesis": "pending",
        "status": "started",
        "result": None,
        "cost_usd": 0.0,
        "start_time": datetime.now(tz=timezone.utc).isoformat(),
        "source": "research_supervisor",
    }

    total_cost = 0.0

    # ------------------------------------------------------------------
    # Phase 1: Claude implements the hypothesis
    # ------------------------------------------------------------------
    context = _build_context(state)
    prompt = _build_implementation_prompt(version, context, best_v)
    success, result_text, cost = _invoke_claude(
        prompt, budget_usd=MAX_BUDGET_USD_PER_ITER, label=f"v{version}-plan"
    )
    total_cost += cost
    state["budget"]["spent_usd"] += cost
    state["budget"]["invocations"] += 1

    # Read hypothesis from file (Claude should have written it)
    hyp_path = REPO_ROOT / "experiments" / f"v{version}_hypothesis.txt"
    if hyp_path.exists():
        hypothesis = hyp_path.read_text().strip()
    elif result_text:
        hypothesis = result_text[:400]
    else:
        hypothesis = "Claude did not produce a hypothesis"
    exp["hypothesis"] = hypothesis
    _log(f"Hypothesis: {hypothesis[:200]}")

    # ------------------------------------------------------------------
    # Verify implementation artifacts exist
    # ------------------------------------------------------------------
    ok, verify_msg = _verify_implementation(version)
    if not ok:
        _log(f"V{version}: Implementation incomplete: {verify_msg}")
        exp["status"] = "implementation_incomplete"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        _save_state(state)
        return

    # ------------------------------------------------------------------
    # Phase 2: Run tests
    # ------------------------------------------------------------------
    tests_ok, test_out = _run_tests()

    if not tests_ok:
        _log(f"V{version}: Tests failed — asking Claude to fix ...")
        fix_prompt = _build_fix_prompt(version, test_out)
        fix_ok, _, fix_cost = _invoke_claude(
            fix_prompt, budget_usd=1.0, label=f"v{version}-fix"
        )
        total_cost += fix_cost
        state["budget"]["spent_usd"] += fix_cost
        state["budget"]["invocations"] += 1
        if fix_ok:
            tests_ok, test_out = _run_tests()

    if not tests_ok:
        _log(f"V{version}: Tests still failing — skipping Kaggle")
        exp["status"] = "failed_tests"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        _save_state(state)
        _git_commit(f"research: v{version} failed tests — hypothesis: {hypothesis[:80]}")
        return

    _log(f"V{version}: Tests PASSED")
    state["consecutive_failures"] = 0  # reset on test pass

    # ------------------------------------------------------------------
    # Phase 3: Kaggle evaluation
    # ------------------------------------------------------------------
    if not _step_package_src():
        exp["status"] = "package_failed"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        _save_state(state)
        return

    if not _step_upload_dataset(version):
        exp["status"] = "upload_failed"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        _save_state(state)
        return

    # Smoke test
    _log(f"V{version}: Pushing SMOKE kernel ...")
    if not _step_push_kernel(version, smoke=True):
        exp["status"] = "kernel_push_failed"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        _save_state(state)
        return

    smoke_status = _step_wait_kernel(version, timeout_min=SMOKE_TIMEOUT_MIN)
    if smoke_status != "COMPLETE":
        exp["status"] = f"smoke_{smoke_status.lower()}"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        state["consecutive_failures"] += 1
        _save_state(state)
        return

    smoke_result = _step_collect(version)
    if not smoke_result or smoke_result["total_cameras"] == 0:
        _log(f"V{version}: Smoke produced 0 cameras — aborting full run")
        exp["status"] = "smoke_zero_cameras"
        exp["result"] = smoke_result
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        state["consecutive_failures"] += 1
        _save_state(state)
        _update_leaderboard(state, smoke_result or {"version": version, "total_cameras": 0,
                                                     "ets_cameras": 0, "stairs_cameras": 0},
                            note=f"V{version} smoke-0")
        _save_state(state)
        return

    _log(f"V{version}: Smoke PASSED ({smoke_result['total_cameras']}/73 cameras in smoke)")

    # Full run
    _log(f"V{version}: Pushing FULL kernel ...")
    if not _step_push_kernel(version, smoke=False):
        exp["status"] = "full_push_failed"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        _save_state(state)
        return

    full_status = _step_wait_kernel(version, timeout_min=FULL_TIMEOUT_MIN)
    if full_status != "COMPLETE":
        exp["status"] = f"full_{full_status.lower()}"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        state["consecutive_failures"] += 1
        _save_state(state)
        return

    full_result = _step_collect(version)
    if not full_result:
        exp["status"] = "collect_failed"
        exp["cost_usd"] = total_cost
        state["experiments"].append(exp)
        state["next_version"] = version + 1
        _save_state(state)
        return

    # ------------------------------------------------------------------
    # Phase 4: Update leaderboard, decide improvement vs regression
    # ------------------------------------------------------------------
    total = full_result["total_cameras"]
    _update_leaderboard(state, full_result)

    if total > best_cams:
        _log(f"V{version}: IMPROVEMENT! {total}/73 > V{best_v} ({best_cams}/73)")
        state["best_version"] = version
        state["best_cameras"] = total
        state["best_ets"] = full_result.get("ets_cameras", 0)
        state["best_stairs"] = full_result.get("stairs_cameras", 0)
        exp["status"] = "improvement"
        state["consecutive_failures"] = 0
    elif total == best_cams:
        _log(f"V{version}: Tied best ({total}/73) — keeping V{best_v} as best")
        exp["status"] = "tied"
        state["consecutive_failures"] = 0
    else:
        _log(f"V{version}: Regression ({total}/73 < V{best_v}: {best_cams}/73)")
        exp["status"] = "regression"

    exp["result"] = full_result
    exp["cost_usd"] = total_cost
    exp["end_time"] = datetime.now(tz=timezone.utc).isoformat()
    state["experiments"].append(exp)
    state["next_version"] = version + 1
    _save_state(state)

    _git_commit(
        f"research: v{version} {exp['status']} ({total}/73)\n\n"
        f"Hypothesis: {hypothesis[:200]}"
    )

    _print_leaderboard(state)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main_loop() -> None:
    # Acquire lock to prevent concurrent supervisors
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another supervisor instance is running (lock held). Exiting.")
        sys.exit(1)

    try:
        state = _load_state()
        _save_state(state)

        _log(f"Research supervisor started. Best: V{state['best_version']} ({state['best_cameras']}/73)")
        _log(f"Cutoff: {CUTOFF_UTC.isoformat()} | Budget: ${state['budget']['limit_usd']:.2f}")
        _log(f"Next experiment: V{state['next_version']}")

        while True:
            stop, reason = _should_stop(state)
            if stop:
                _log(f"Stopping: {reason}")
                break

            try:
                run_one_experiment(state)
            except KeyboardInterrupt:
                _log("KeyboardInterrupt — stopping")
                break
            except Exception as exc:
                _log(f"Unexpected error in experiment: {exc}")
                traceback.print_exc()
                state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
                state["next_version"] = state.get("next_version", 12) + 1
                _save_state(state)

            stop, reason = _should_stop(state)
            if stop:
                _log(f"Stopping after experiment: {reason}")
                break

            _log(f"Sleeping {INTER_EXPERIMENT_SLEEP_S}s before next experiment ...")
            time.sleep(INTER_EXPERIMENT_SLEEP_S)

    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    _log("\nFINAL LEADERBOARD:")
    _print_leaderboard(state)
    _log(f"Best: V{state['best_version']} ({state['best_cameras']}/73) — submit manually via Kaggle web UI")

# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_status(state: dict) -> None:
    stop, reason = _should_stop(state)
    print(f"Status       : {'STOP — ' + reason if stop else 'running'}")
    print(f"Best version : V{state['best_version']} ({state['best_cameras']}/73)")
    print(f"Next version : V{state['next_version']}")
    print(f"Budget       : ${state['budget']['spent_usd']:.3f} / ${state['budget']['limit_usd']:.2f}")
    print(f"Invocations  : {state['budget']['invocations']}")
    print(f"Time remaining: {_time_remaining_s()/3600:.1f}h until {CUTOFF_UTC.strftime('%H:%M UTC')}")
    print(f"Experiments  : {len(state.get('experiments', []))} completed")
    _print_leaderboard(state)


def main() -> None:
    p = argparse.ArgumentParser(description="IMC2025 LGEU Research Supervisor")
    p.add_argument("--status", action="store_true")
    p.add_argument("--leaderboard", action="store_true")
    p.add_argument("--budget", type=float, help="Set total budget in USD")
    args = p.parse_args()

    state = _load_state()

    if args.budget:
        state["budget"]["limit_usd"] = args.budget
        _save_state(state)
        print(f"Budget set to ${args.budget:.2f}")
        return

    if args.status:
        cmd_status(state)
        return

    if args.leaderboard:
        _print_leaderboard(state)
        return

    main_loop()


if __name__ == "__main__":
    main()
