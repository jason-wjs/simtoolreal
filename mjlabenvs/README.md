# SimToolReal mjlab port

Reference: the native **Isaac Lab teacher** recipe in this fork. Keep its
Sharpa left hand + iiwa14, generated tools, control timing, observations,
rewards, curriculum and SAPG learner. Gym/Lab differences remain explicit.
This is a concrete task port, separate from the archived custom grasp pilot.

## Implemented and checked

`assets.object_spec` builds a MuJoCo spec directly from the original generated
single-link tool URDF. It preserves each box and replaces collision cylinders
with capsules, matching the native Lab importer. USD capsule height is the
cylindrical spine length; its hemispheres extend beyond the source cylinder.
Pose, explicit mass, center of mass and full inertia tensor are preserved. It does not
replace a handle/head composite by its convex hull or infer mass from the
collision geometry. The author's cylinder/capsule inertia approximation is
preserved as supplied; compilation agreement does not validate that modeling
choice against a real tool.

`reproduction/prepare_mjlab_tools.py --output <new-directory>` invokes the
original pure Lab generator with seed42: twelve distributions, 100 samples
each, six tool families, original shuffled pool and normalized scales.
It uses no Isaac runtime and no CUDA. Each of the 1200 compiled tools is
checked against its source URDF for mass, COM, inertia, collision count,
primitive type, dimensions and local transform. The output manifest records
generator hashes, source commit, ordered URDF hashes and per-tool checks.

On 2026-10-13, source `ba6c563` passed all 1200 CPU MuJoCo checks in the existing
Tele mjlab environment. Maximum absolute inertia reconstruction error:
`1.734723475976807e-18`. Evidence: `outputs/mjlab-native-tools-20261013/manifest.json`.

The initial `ba6c563` check below compared URDF cylinders; it did not establish
equivalence with native Lab's imported collision shapes. Source `99a0689` fixes
this omission. A CPU comparison of all1200 actual baked Lab USDs now passes:
600capsules and1700boxes, maximum dimension error7.39934e-9m and full-inertia
error1.14979e-10. Local transforms, mass and COM pass as well. Evidence:
`outputs/mjlab-native-tools-20261013/lab_geometry_audit.json`; the associated
native USD export is in the study repository's ignored outputs.

`robot.py` uses29 canonical joints and native PD/effort/armature settings.
`scene.py` preserves the original env-to-tool assignment using three native
topology groups, with robot, table, goal and tool state writes. A1536-world,
240-policy-step PD-hold/freefall check ran successfully before the capsule fix,
but found deep impact contacts and has not passed physical review. The corrected full-scene check is finite but still flags deep contacts. Isolated box-tool/table impacts agree closely
between CPU MuJoCo and native mjlab; this does not validate hand interactions or
variant scene dynamics.

At source `136d7a7`, the scene also corrects a native variant limitation:
`VariantEntityCfg` scatters mesh geometry fields but leaves primitive dimensions
and local transforms at the first variant. Original per-world geometry errors
reached0.122m despite correct varying body inertias. The task now writes the five
primitive geometry fields into already allocated per-world arrays and updates
native defaults in place. The actual GPU geometry and inertial values for all1536
worlds pass exact float32 checks against each source-compiled tool. This fixes a
modeling error; it does not resolve all full-scene contact-quality findings.

`control.py` preserves native action delay, smoothing and clamping in canonical
joint order. Its 100-step CPU comparison against the frozen native implementation
passes exact float32 targets and queues, including permuted hand order, goal hit,
partial reset and restored control state.

`progress.py` preserves the teacher reward, success/termination counters and
tolerance curriculum. Four 100-step CPU cases match the frozen native code
exactly for all reward terms, counters and terminations, covering consecutive
and accumulated success, evaluation tolerance, partial reset and restored state.
The native order clears per-goal trackers before reward computation; it is
preserved. This check uses supplied geometric distances and does not validate
quaternion encoding or scene geometry. Evidence: `action_parity.json` and
`progress_parity.json` in `outputs/mjlab-native-tools-20261013`.

Matched full-robot single-tool CPU/GPU controls settle below0.2mm. In the full
batch, only3/1536 worlds had final-half-second contact maxima above5mm; exact
final-pose CPU reconstruction instead finds0.17–0.34mm ground contacts. Those
window maxima describe late impacts, not persistent final overlap. The diagnostic
0.9m drop is not the original task reset distribution, and all three tools ended
below the original fall-termination height. Impact peaks remain audit evidence;
native-task contact acceptance is still pending.

`observations.py` supplies native teacher actor/critic fields from canonical
scene tensors. Four60-step CPU comparisons against frozen native0eb3edc and
independently loaded installed Isaac Lab math pass within2e-6 absolute error
(observed maximum9.54e-7). Native dimensions are140actor/162critic. Nonidentity
rotations, nonzero origins, canonical joint permutations, raw fingertip versus
pad offset, both keypoint geometries, identical RNG advancement, delays, partial
reset and separate-object restore are covered. Evidence: `observation_parity.json`.
This does not yet establish live scene velocity-frame correctness.

`randomization.py` preserves baseline pose and goal sampling plus persistent
world-frame random wrench state. Three reset cases (native, table XY/yaw,
fixed poses) and four100-step wrench cases pass exact float32 and RNG-state
comparisons to frozen native8e18764. Partial resets and restored wrench state
are covered. The reset fixture uses canonical joint order and zero origins;
actual Lab parser order consumes random samples in its own joint order, so
cross-engine identical seeded reset vectors are not claimed. The native
fixed-trajectory ablation is outside this teacher port and explicitly rejected.
Evidence: `randomization_parity.json`.

Installed API source confirms native Lab `body_state_w` combines link pose with
COM velocity, matching this scene's fields. mjlab external wrench writes use
the world frame, matching native `is_global=True`. Live numerical verification
is still required; these are interface-source checks.

`task.py` connects the components using the installed DirectRLEnv order:
pre-physics actions/wrench, decimated physics, counters, dones and goal resampling,
rewards, full episode resets, then observations. It uses the exact configured
ceiling for episode length. Original-reset and fixed-clear-reset GPU checks have run but failed physical
gates. A labeled CPU variant with clear resets,480Hz physics and5ms contact/hand
limit settings completes64 policy steps and44 real resets; residual physical
errors remain under review. This does not establish GPU candidate acceptance. `check_mjlab_task.py` bounds the first full-pool lifecycle check to64steps,
records substep depth/limit violations and stops on persistent deep contact or
large joint violations. A completed diagnostic does not imply physical acceptance.

`rlgames.py` implements the original single-device SAPG interface. CPU parity
against the installed Lab wrapper passes action detach/clone/clipping, actor and
critic routing/clipping, done union, finite/infinite-horizon timeout handling,
spaces and checkpoint delegation. Evidence: `rlgames_adapter_parity.json`.
The original vendored Runner imports successfully in the existing mjlab venv
with `reproduction/mjlab-learner-requirements.txt`; PyTorch2.7.0, NumPy1.26.0,
MuJoCo3.12.0, mujoco-warp3.14.0 and mjlab1.6.0+gyrofix remain unchanged.

Task checkpoints save component buffers, counters, RNG and scene integration
fields including warmstart, applied forces, controls and mocap poses. They now
also record the scene update timestep/decimation and actual per-group Warp
timestep, contact `geom_solref` and joint-limit `jnt_solref`. Restore validates
these alongside assets/config before changing task components or any group.
It checks the requested physics recipe; it does not silently change the running
model to another recipe. Older mjlab checkpoints lacking this identity are
explicitly rejected; capture a new checkpoint with the intended configuration.

A full1536-world/all1200-tool CPU test saves after four real task steps and
compares eight more steps against an independent-process restoration. Outputs,
physical state, component histories and RNG match exactly at every step.
Five negative controls (three runtime fields in the last group, decimation,
and missing identity) reject without changing task state. This uses recorded
trained-actor actions, not learner/optimizer restoration or new closed-loop
inference; GPU and formal-training acceptance remain open. Reproduce with the
study script `experiments/simtoolreal/check_candidate_task_restore_cpu.py` in
separate `save` and `restore` processes under finite process limits.

`train.py` loads the original env+agent YAML with resolved Hydra references,
constructs the concrete task, and uses the vendored Runner with six SAPG cohorts.
It shares the unchanged native metric observer with Lab. The CLI requires an
explicit final epoch and output directory; run it only under a finite GPU guard.

A CPU fixture exercises this same runner/adapter path with full native networks,
140actor/162critic inputs, six cohorts, six synthetic environments, small batches
and AMP disabled. Actual rollout/update/save/restore/update advances epoch2→3
and frames192→288. All13 checked learner/task groups restore exactly, actor
weights change, both optimizers advance and all tensors stay finite. Evidence:
`outputs/mjlab-native-learner-cpu-20261013/result.json`. This verifies learner
integration, not native physics or separate-process simulator recovery.

## Remaining

Live task rollout/checkpoint validation, GPU learner updates, separate-process
full continuation and native evaluation remain to verify. Full native contact acceptance,
training/save/restore and evaluation remain to verify.
No working training environment, physical contact acceptance or reproduction
result is claimed by the tool-asset check.
