# A4R045 — optimization proposal for the A4R034 baseline (unvalidated)

> **This directory contains only the proposal.** The A4R034 implementation, notebooks, assets and
> private datasets are **not** included here. The single file under
> `overlay-files/scripts/reconstruct_baseline.py` is the overlay the patch modifies, included
> solely as the patch target. Full attribution in [../CONTRIBUTIONS.md](../CONTRIBUTIONS.md).

## What this is

A one-line, evidence-grounded parameter proposal on top of the A4R034 submission (Kaggle Public
38.16 / Private **42.25**). It is a candidate parameter change, not a system and not a validated
improvement.

## The change

`scripts/reconstruct_baseline.py`:

```diff
- options.mapper.abs_pose_min_num_inliers = 15
+ options.mapper.abs_pose_min_num_inliers = 12
```

Everything else (Top24 MASt3R, RDD/ALIKED Top8, angle8, fusion, verification, all other mapper
settings) is unchanged.

## Why (from the A4R034 ledger)

The A4R034 `abs_pose_min_num_inliers` sweep is **monotonically positive as the floor decreases**,
in both public and private:

| candidate | inlier floor | public | private |
|---|---:|---:|---:|
| a4r011 | 30 | 34.03 | 37.06 |
| a4r024 | 20 | 37.34 | 40.37 |
| **a4r034** | **15** | **38.16** | **42.25** |
| a4r035 | 25 (raised) | 35.35 | 38.60 (regressed) |

Three consecutive gains (30 → 20 → 15). The sweep stopped at 15 and never tested a lower value;
every other knob regressed or was neutral. A4R045 continues this one open, upward direction by
the smallest conservative step (15 → 12).

## Status: unvalidated — no score claimed

This was not run. The A4R034 asset datasets are private (403 to this account) and the notebook
needs the dual-GPU, sha-verified package, so it could not be evaluated even on one local dataset.
No claim is made that it beats 42.25.

**Recommendation:** run a4r045 (pose12) through the A4R034 pipeline and compare to a4r034's 42.25.
Falsification criterion:
- higher → adopt a4r045;
- lower/equal → keep a4r034 (15 is the optimum; the trend has turned, and pose10 should not be
  tried). Risk: too low a floor admits weakly-constrained P3P registrations (wrong poses) that
  can lower mAA — which is why it needs one real submission, not a blind claim.

## Files

| File | What |
|---|---|
| `OPTIMIZATION_REPORT.md` | the full analysis + recommendation |
| `a4r045-pose12.patch` | the one-line diff |
| `candidate-a4r045-pose12.json` | candidate spec (parent a4r034, new sha, evidence) |
| `overlay-files/scripts/reconstruct_baseline.py` | ready-to-apply overlay file (patch target) |
