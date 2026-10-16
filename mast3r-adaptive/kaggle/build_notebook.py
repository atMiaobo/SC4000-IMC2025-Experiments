#!/usr/bin/env python3
"""Generate the MASt3R Kaggle notebook (.ipynb) for the ORIGINAL image-matching-challenge-2025."""
import json
from pathlib import Path

cells_src = []

cells_src.append(("markdown", """# IMC 2025 — MASt3R adaptive reconstruction (ORIGINAL competition)

Offline MASt3R pipeline (DINOv2 retrieval + MASt3R semi-dense matching + MAGSAC + COLMAP)
with reconstruction-aware adaptive matching (scene-preserving). Targets the **original**
`image-matching-challenge-2025`.

Attach: competition `image-matching-challenge-2025` + dataset `miaobo/mast3r-imc25-bundle`.
GPU, Internet OFF. Writes `/kaggle/working/submission.csv`.
"""))

cells_src.append(("code", """import os, sys, subprocess, shutil
from pathlib import Path
import torch
print('Python', sys.version.split()[0], '| Torch', torch.__version__, '| GPU',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
INPUT = Path('/kaggle/input'); assert INPUT.exists()
"""))

cells_src.append(("code", """# Locate the offline bundle (contains project/ + wheels/).
proj = None
for p in INPUT.rglob('scripts/infer_kaggle.py'):
    proj = p.parent.parent; break
assert proj is not None, 'Attach miaobo/mast3r-imc25-bundle'
PROJECT = proj
print('PROJECT =', PROJECT)
# Offline model caches (DINOv2 timm cache shipped in the bundle).
os.environ['HF_HOME'] = str(PROJECT/'models'/'hf')
os.environ['HF_HUB_CACHE'] = str(PROJECT/'models'/'hf'/'hub')
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = str(PROJECT/'models'/'torch')
sys.path.insert(0, str(PROJECT/'src'))
for sub in ('mast3r','mast3r/dust3r','mast3r/dust3r/croco'):
    sys.path.insert(0, str(PROJECT/'third_party'/sub))
"""))

cells_src.append(("code", """# Offline wheel install.
wheels = None
for w in INPUT.rglob('wheels'):
    if w.is_dir() and any(w.glob('pycolmap-*.whl')): wheels = w; break
assert wheels is not None, 'wheels/ not found in bundle'
def pip_install(pat):
    xs = sorted(wheels.glob(pat))
    if not xs: raise RuntimeError('missing wheel '+pat)
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
                    '--force-reinstall',str(xs[-1])], check=True)
pip_install('pycolmap-3.13.0-*.whl')
for pkg in ('roma','einops'):
    try: __import__(pkg)
    except Exception: pip_install(pkg+'-*.whl')
for pkg,imp in [('timm','timm'),('scipy','scipy')]:
    try: __import__(imp)
    except Exception: pip_install(pkg+'-*.whl')
import pycolmap, roma, einops
print('offline imports OK; pycolmap', pycolmap.__version__)
"""))

cells_src.append(("code", """# Find the ORIGINAL competition root (sample_submission.csv + test/).
roots = [s.parent for s in INPUT.rglob('sample_submission.csv') if (s.parent/'test').is_dir()]
# Prefer the non-ongoing original competition path.
roots.sort(key=lambda p: ('ongoing' in str(p).lower(), len(str(p))))
assert roots, 'Attach image-matching-challenge-2025'
DATA = roots[0]
print('DATA =', DATA, '| test imgs:', len(list((DATA/'test').rglob('*.png'))))
"""))

cells_src.append(("code", """# Run MASt3R adaptive inference -> /kaggle/working/submission.csv
cmd = [sys.executable, str(PROJECT/'scripts'/'infer_kaggle.py'),
       '--data-dir', str(DATA), '--project', str(PROJECT),
       '--adaptive-src', str(PROJECT/'src'), '--model-dir', str(PROJECT/'weights'/'mast3r_hf'),
       '--mode', 'adaptive', '--out', '/kaggle/working/submission.csv']
env = os.environ.copy()
env['PYTHONPATH'] = os.pathsep.join([str(PROJECT/'src'),
    str(PROJECT/'third_party'/'mast3r'), str(PROJECT/'third_party'/'mast3r'/'dust3r'),
    str(PROJECT/'third_party'/'mast3r'/'dust3r'/'croco'), env.get('PYTHONPATH','')])
print(' '.join(cmd)); subprocess.run(cmd, check=True, env=env)
"""))

cells_src.append(("code", """import pandas as pd
sample = pd.read_csv(DATA/'sample_submission.csv'); sub = pd.read_csv('/kaggle/working/submission.csv')
print('sample', len(sample), 'submission', len(sub))
assert len(sub)==len(sample)
assert not sub[['dataset','scene','image','rotation_matrix','translation_vector']].isna().any().any()
print(sub.head()); print('SUBMISSION READY')
"""))

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i,(typ,src) in enumerate(cells_src):
    cell = {"cell_type": typ, "metadata": {}, "source": src, "id": f"c{i}"}
    if typ == "code": cell["outputs"] = []; cell["execution_count"] = None
    nb["cells"].append(cell)

out = Path(__file__).parent / "MASt3R_Adaptive_Kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
