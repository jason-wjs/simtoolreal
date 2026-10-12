# Reproduction branch

Upstream reference: `tylerlum/simtoolreal` at
`313d5aea1f507c6cfe097b672b62945d7b0bbff5`.

This fork preserves the author's Gym/Lab implementations and SAPG learner.
The sibling `jason-wjs/dex-grasp-study` repository records protocols, source pins,
evaluation evidence and backend comparisons. Original Sharpa left + KUKA iiwa14
assets and task definitions remain the reference. A mjlab port follows successful
native baseline evaluation, not merely successful imports.

Use the upstream installation guide and separate Python 3.8 Gym / Python 3.11 Lab
environments. Local environments, downloaded weights and outputs stay untracked.
Keep patches needed for bounded reproducible execution here; do not fold the
previous custom grasp task into this codebase.
# Native Lab on a cluster without direct S3 access

The first native evaluation reached robot/tool creation, then failed opening
NVIDIA's `default_environment.usd`. Cache the exact upstream USD and textures:

```bash
bash reproduction/cache_ground.sh
```

Pass the SDK's existing asset-root override to `dextoolbench/eval_isaacsim.py`:

```bash
--kit_args "--/persistent/isaac/asset_root/cloud=$(pwd)/outputs/nvidia-assets"
```

The cache script verifies the USD SHA256 and records texture hashes. It does not
replace the ground geometry or modify the policy/task. Set `https_proxy` to the
cluster proxy when downloading. Runtime validation is recorded in the study repo.
