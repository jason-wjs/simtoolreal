#!/usr/bin/env bash
# Run from the repository root; uv must be on PATH.
set -euo pipefail
uv venv --python 3.11.15 .venv_isaacsim
uv pip install --python .venv_isaacsim/bin/python \
  'torch==2.7.0' 'numpy==1.26.0' 'isaaclab[isaacsim]==2.3.2.post1' \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install --python .venv_isaacsim/bin/python -e ./rl_games/ \
  'torch==2.7.0' 'numpy==1.26.0' omegaconf hydra-core 'gym==0.23.1' \
  scipy yourdfpy requests tqdm tyro 'imageio[ffmpeg]' wandb termcolor \
  'coacd==1.0.5' 'typing_extensions>=4.13'
uv pip install --python .venv_isaacsim/bin/python -e . --no-deps
# Root metadata targets the independent Gym runtime. Check only actual Lab
# dependencies; a full root metadata check would report intentionally omitted
# Gym-only dependencies (see upstream docs/isaacsim_installation.md).
uv pip freeze --python .venv_isaacsim/bin/python > outputs/lab-environment.txt
