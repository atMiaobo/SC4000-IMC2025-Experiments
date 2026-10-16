# A4R034 Optimization — report

An optimization of the A4R034 solution (Public 38.16 / Private **42.25**). This is a change on
top of that implementation, not a new system. The A4R034 architecture, assets and the
A4R001–A4R044 sweep are part of the team's A4R034 work (see Attribution); the contribution here
is the analysis plus one targeted change.

## Baseline (preserved, unchanged)
`original/SC4000-IMC2025-a4r034-top24-angle8-pose15/` — extracted read-only; the original overlay
`scripts/reconstruct_baseline.py` sha256 `7aef813b…` is untouched. Scored notebook = kernel
`yijisuk/sc4000-a4r034-top24-angle8-pose15` v1 (submission_ref 57156705), code deployed via the
sha256-verified asset+overlay mechanism (overlays/files/… applied over the frozen yijisuk asset
package).

## Analysis (no repeated experiments)
Read the full A4R034 ledger (`deliverables/Trial Catalogue.csv`, 44 candidates). The
`abs_pose_min_num_inliers` (minimum camera-registration inlier count) sweep is **monotonically
positive as it decreases**, in both public and private:

| candidate | abs_pose_min_num_inliers | public | private |
|---|---|---|---|
| a4r011 | 30 | 34.03 | 37.06 |
| a4r024 | 20 | 37.34 | 40.37 |
| **a4r034** | **15** | **38.16** | **42.25** |
| a4r035 | 25 (raised) | 35.35 | 38.60 (regressed) |

Lowering the threshold registers more cameras, and the metric rewards that — three consecutive
gains (30→20→15). The sweep stopped at 15 and never tested a lower value. Every other
reconstruction knob (init angle, tri-angle filters, reproj filters, BA window, init-pair inliers,
retrieval Top-K) regressed or was neutral — the inlier floor is the single open, upward direction.

## The change (minimal, one line)
`scripts/reconstruct_baseline.py`: `options.mapper.abs_pose_min_num_inliers = 15 → 12` (continue
the winning direction by the smallest conservative step). Patch: `optimized/a4r045-pose12.patch`;
modified overlay sha256 `d2ddd385…`. Everything else — Top24 MASt3R, RDD/ALIKED Top8, angle8,
fusion, verification, all other mapper settings — is unchanged.

## Deployment status
**Not deployed/validated.** The A4R034 asset datasets (`yijisuk/sc4000-imc2025-assets`,
`…-e16-rdd-assets`, `…-e20-aliked-assets`) are private and return 403 to this account, and the
notebook needs dual-GPU T4 + its frozen sha-verified package. It cannot be run even on one local
dataset (no weights locally). Per the stop condition (essential assets not reproducible), the
gigabyte asset package was not cloned/rebuilt.

The deliverable is therefore a **drop-in candidate (a4r045) for the A4R034 pipeline**, which
already holds the assets and auto-regenerates the overlay sha chain:
- `optimized/a4r045-pose12.patch` — the one-line diff
- `optimized/overlay-files/scripts/reconstruct_baseline.py` — ready overlay file
- `optimized/candidate-a4r045-pose12.json` — candidate spec (parent a4r034, new sha, evidence)

## Recommendation: next experiment — but unvalidated
The change is the best-grounded untested move (continues a 3-point monotonic positive trend that
is the A4R034 winning lever), minimal, and low-destabilisation. There is **no** hidden or local
score for it, so no claim is made that it beats 42.25. Run a4r045 (pose12) through the candidate
pipeline and compare to a4r034's 42.25:
- if higher → adopt a4r045;
- if lower/equal → keep a4r034 (the trend has turned; 15 is the optimum). A single further step
  (pose10) would then also be falsified and should not be tried.

Risk: too-low a floor admits weakly-constrained P3P registrations (wrong poses) that can lower
mAA — which is why this needs one real submission to confirm, not a blind claim.

## Attribution
- Yiji (`yijisuk`): the entire A4 architecture (DINOv2 + MASt3R + RDD + ALIKED fusion, COLMAP
  pipeline), all pretrained assets, and the A4R001–A4R044 empirical sweep including the pose20→15
  win.
- Public pretrained models: DINOv2, MASt3R (CC-BY-NC-SA), RDD, ALIKED, LightGlue, COLMAP.
- This proposal: the ledger analysis identifying the untested pose12 continuation; the one-line
  patch; candidate packaging; this report. No ownership of the A4R034 work is claimed.
