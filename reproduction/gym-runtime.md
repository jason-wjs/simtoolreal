# Independent Gym runtime

Tele setup on 2026-10-12: Python 3.8.20, torch 2.0.1+cu118,
torchvision 0.15.2+cu118, NumPy **1.23.0**, Gym 0.23.1, original editable
`rl_games` 1.6.1 and Isaac Gym Preview 4. Keep `.venv-gym` separate from Lab
and the sibling Dex4D package environment. The standalone interpreter and SDK
may be shared; Python packages are installed independently.

From this repository root, with `uv` on PATH:

```bash
uv venv --python ../Dex4D-Simulation/.python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8 .venv-gym
uv --no-config pip install --python .venv-gym/bin/python \
  'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' 'numpy==1.23.0' \
  --index https://pypi.tuna.tsinghua.edu.cn/simple \
  --default-index https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
uv --no-config pip install --python .venv-gym/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'numpy==1.23.0' 'setuptools<76' wheel ninja fvcore iopath \
  'gym==0.23.1' hydra-core termcolor pyvirtualdisplay 'trimesh==3.23.5' \
  'coacd==1.0.5' scipy matplotlib 'wandb==0.12.21' tensorboard tensorboardx \
  'opencv-python<4.12' 'imageio[ffmpeg]' transforms3d yourdfpy viser \
  pytorch-kinematics tyro psutil setproctitle pyopenssl
uv --no-config pip install --python .venv-gym/bin/python --no-deps \
  -e . -e rl_games -e ../runtime/isaacgym/python
```

The tested install reused the already SHA-verified official torch wheel from
the sibling Dex4D download directory. Proxy environment variables are specific
to the cluster and are not credentials or package sources.

Only `pytorch3d.transforms` is imported by the original Gym task. Install the
official v0.7.4 source, which explicitly supports torch 2.0.1, using its existing
`PYTORCH3D_NO_EXTENSION` build option. This provides the original pure PyTorch
rotation functions; **compiled PyTorch3D ops/renderers are unavailable**. This
is sufficient for the inspected SimToolReal evaluation path, not a complete
PyTorch3D installation or validation of unrelated pipelines.

```bash
mkdir -p outputs/downloads
curl -fL https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/refs/tags/v0.7.4 \
  -o outputs/downloads/pytorch3d-0.7.4.tar.gz
echo 'b153b1a14807267b22e3f8efbdad153cb08b51e9015a43105bb4cec8ea15359e  outputs/downloads/pytorch3d-0.7.4.tar.gz' | sha256sum -c -
PYTORCH3D_NO_EXTENSION=1 uv --no-config pip install --python .venv-gym/bin/python \
  --no-build-isolation --no-deps outputs/downloads/pytorch3d-0.7.4.tar.gz
uv pip freeze --python .venv-gym/bin/python > outputs/gym-environment.txt
export PATH="$PWD/.venv-gym/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/../Dex4D-Simulation/.python/cpython-3.8.20-linux-x86_64-gnu/lib"
export MAX_JOBS=2
CUDA_VISIBLE_DEVICES='' .venv-gym/bin/python dextoolbench/eval_isaacgym.py --help
CUDA_VISIBLE_DEVICES='' .venv-gym/bin/python reproduction/check_gym_runtime.py
```

The CLI import and checkpoint loading passed. The second check records a finite
29-dimensional action from the original network and tests the actual rotation
functions. These setup checks do not establish native simulator behavior.
Root metadata also lists asset-generation and other tools that are not needed by
this evaluation path; they are not installed by this minimal runtime recipe.
