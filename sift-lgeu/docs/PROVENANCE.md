# Provenance

## Official Data

- Competition: Kaggle `image-matching-challenge-2025`
- Files accessed with Kaggle CLI:
  - `sample_submission.csv`
  - `train_labels.csv`
  - bounded official image subset under `train/` and `test/`
- Use: schema verification, training labels, local train/validation experiments,
  and visible-test submission serialization smoke tests.

## Third-party Libraries

- NumPy, pandas, scikit-learn, matplotlib, joblib: Python scientific stack.
- OpenCV Python headless: image loading, ORB descriptors, matching, essential
  matrix estimation, and relative pose recovery.
- Kaggle CLI: official competition file listing and downloads.

No teammate repositories, private Kaggle kernels, public notebook templates, or
derivative code were searched, loaded, copied, unzipped, mounted, imported, or
adapted. Pipeline orchestration, pair features, LGEU training, graph selection,
serialization, tests, and reports are original code written in this repository.

## Model Assets

- `models/lgeu.joblib` is produced locally by `training/train_lgeu.py` from
  official train images and `train_labels.csv`.
- No public pretrained weights are required by the current implementation.

## Clean-room Notes

The implementation uses a deliberately simple OpenCV/sklearn stack. It does not
copy COLMAP wrappers, public Kaggle notebooks, or teammate code. The local
minimal geometry path is not represented as a full SfM system; it is a real
feature/matching/essential-matrix pose-chain baseline with explicit fallbacks.
