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

## Remaining

Original action delay/smoothing, observation/reward/reset/DR semantics and SAPG
environment adapter remain to implement. Full native contact acceptance,
training/save/restore and evaluation remain to verify.
No working training environment, physical contact acceptance or reproduction
result is claimed by the tool-asset check.
