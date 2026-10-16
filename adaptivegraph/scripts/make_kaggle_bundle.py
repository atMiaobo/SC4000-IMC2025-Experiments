#!/usr/bin/env python3
"""Create a small Kaggle input dataset containing this project, cached weights, and wheels.

Run after `python scripts/cache_models.py` and after bootstrap. It deliberately excludes
competition data, outputs, downloads and the virtualenv.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'kaggle_bundle')
    ap.add_argument('--kaggle-id', default=None, help='Optional USER/adaptive-graph-imc25')
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    project = out / 'project'
    project.mkdir(parents=True)

    include = ['src', 'configs', 'scripts', 'third_party', 'models', 'pyproject.toml', 'README.md', 'METHOD.md', 'THIRD_PARTY_NOTICES.md', 'FINAL_WORKFLOW.md']
    for name in include:
        src = ROOT / name
        if not src.exists():
            print(f'WARNING: missing {src}')
            continue
        dst = project / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        else:
            shutil.copy2(src, dst)

    # Download a portable pycolmap wheel matching the tested API. Other common deps
    # are already in Kaggle images; we include timm-related wheels as insurance.
    wheel_dir = out / 'wheels'
    wheel_dir.mkdir()
    pkgs = ['pycolmap==3.13.0', 'timm==1.0.20', 'huggingface-hub>=0.27,<1', 'safetensors>=0.4,<1']
    # IMPORTANT: Kaggle notebooks run Python 3.13, which may differ from the local
    # interpreter used to build this bundle. Download wheels for cp313 explicitly so
    # the offline `pip install` on Kaggle finds a compatible pycolmap build.
    platforms = ['manylinux_2_28_x86_64', 'manylinux2014_x86_64', 'manylinux_2_17_x86_64']
    cmd = [sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
           '--python-version', '313', '--implementation', 'cp']
    for plat in platforms:
        cmd += ['--platform', plat]
    cmd += ['-d', str(wheel_dir), *pkgs]
    print('Downloading portable wheels (target: CPython 3.13 / Kaggle)...')
    subprocess.run(cmd, check=True)

    if not (project / 'models').exists():
        raise SystemExit('models/ missing. Run: python scripts/cache_models.py')
    if not (project / 'third_party' / 'LightGlue').exists():
        raise SystemExit('third_party/LightGlue missing. Run: bash scripts/bootstrap.sh')

    meta = {
        'title': 'IMC2025 Adaptive Graph Bundle',
        'id': args.kaggle_id or 'REPLACE_ME/adaptive-graph-imc25-bundle',
        'licenses': [{'name': 'apache-2.0'}],
    }
    (out / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'\nBundle ready: {out}')
    print('Size:')
    subprocess.run(['du', '-sh', str(out)], check=False)
    if args.kaggle_id:
        print(f'\nUpload with:\n  kaggle datasets create -p {out} --dir-mode zip')
    else:
        print('\nRe-run with --kaggle-id YOUR_USERNAME/adaptive-graph-imc25-bundle')

if __name__ == '__main__':
    main()
