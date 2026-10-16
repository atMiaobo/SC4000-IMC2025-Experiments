# SIFT/LGEU setup (NixOS x86_64)

A clean-room environment for the SIFT/LGEU pipeline, independent of any other team repositories.

## First run

```bash
mkdir -p ~/projects/sc4000-cleanroom
# Unzip the project into that directory, then:
cd ~/projects/sc4000-cleanroom
git init
nix develop
bash scripts/setup.sh
kaggle auth login
bash scripts/preflight.sh
```

Run the official competition preflight (`scripts/preflight.sh`) before launching a full run. If
you use an existing repo with a different project name, use a fresh directory instead.

## Running

Within the `nix develop` shell, a long run can be launched under `tmux`:

```bash
tmux new -s sc4000
bash scripts/run_overnight.sh
```

Detach with Ctrl-B then D; reattach with `tmux attach -t sc4000`. Keep the machine powered and
awake; tmux does not prevent suspension by itself.

## Checking results

```bash
cat docs/REPORT.md
python -m pytest -q
```

Inspect the run logs under `logs/`, and verify any Kaggle run via the actual kernel
status/logs/score.

This is a development environment, not a pre-configured Kaggle GPU image. Public pretrained model
weights, training data, remote GPU quota and notebook metadata still require real checks.
