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
The tested independent Gym recipe and CPU checks are in [gym-runtime.md](gym-runtime.md).
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
--kit_args="--/persistent/isaac/asset_root/cloud=$(pwd)/outputs/nvidia-assets"
```

The cache script verifies the USD SHA256 and records texture hashes. It does not
replace the ground geometry or modify the policy/task. Set `https_proxy` to the
cluster proxy when downloading. Runtime validation is recorded in the study repo.

## Native physical audit

Add `--audit_physics --output_dir <run-directory>` to the original Lab evaluator.
The opt-in environment subclass records every physics substep (including the
initial zero-action reset tick) before automatic episode reset. It enables native
contact reporting on the manipulated object without changing solver settings,
geometry, policy or success calculations.

Outputs: `physics_trajectory.npz` contains joint position/velocity, object and goal
states (Isaac Lab wxyz quaternion convention), policy actions, and packed native
contact data. `contact_offsets` maps each recorded substep to its contact rows.
`physics_audit.json` reports signed PhysX contact separation, contact-weighted
penetration statistics, per-body breakdown, joint-limit deviation and finiteness.
Buffer saturation raises an error instead of silently declaring a physical pass.

Coverage is object-to-robot and object-to-table. Robot self-contact, robot-table
and ground contacts are not covered yet. A 5 mm exceedance count is a descriptive
diagnostic, not an agreed physical acceptance threshold. No contacts is not a pass.

Verified first fixture: `claw_hammer / swing_down`, original checkpoint, one
episode: 37/37 goals and 534 policy steps both with and without recording. Audit
recorded 1,070 substeps and 29,859 contacts, max penetration 2.95 mm, contact-weighted
p95 0.59 mm. See the study repository for further tasks and limitations.

## Read-only native parameter snapshots

Both original evaluators now accept `--audit_state` (Gym also accepts the hyphen
spelling `--audit-state`) with an output directory. At the first policy action
of the first episode, `native_state.json` records native joint gains, limits,
armature/friction, body masses/COM/inertia, shape counts, joint states/targets,
object/goal states, the actual policy observation and action. Joint arrays carry
names so Gym/Lab ordering can be aligned. Object/goal state quaternions are xyzw;
raw inertia is retained without claiming matching reference frames.

The Lab snapshot reads both deprecated friction coefficients and the current
static/dynamic/viscous friction properties. Equal numeric coefficients across
PhysX versions do not by themselves establish equal friction models. Recording
is opt-in and does not change resets, actions, rewards or simulator settings.
The full original Lab short-screwdriver outcome is unchanged with recording
(25/38 goals, 1,386 steps); Gym shows rollout variation even without recording,
so bitwise trajectory invariance is not claimed.

Study `experiments/simtoolreal/compare_native_state.py` compares these snapshots.
The study also preserves explicit diagnostic patches for startup ticks, reset
height and legacy friction. Those patches are separate from the author's default
evaluator behavior and are not silently enabled by the audit flag.
