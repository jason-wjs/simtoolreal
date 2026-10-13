# SimToolReal mjlab port

Reference: the native **Isaac Lab teacher** recipe in this fork. Keep its
Sharpa left hand + iiwa14, generated tools, control timing, observations,
rewards, curriculum and SAPG learner. Gym/Lab differences remain explicit.
This is a concrete task port, separate from the archived custom grasp pilot.

## Implemented and checked

`assets.object_spec` builds a MuJoCo spec directly from the original generated
single-link tool URDF. It preserves each box/cylinder collision primitive,
pose, explicit mass, center of mass and full inertia tensor. It does not
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

## Remaining

Robot/table conversion and native mjlab scene, original actuator/control and
observation/reward/reset/DR semantics, SAPG environment adapter, native contact
audits, training/save/restore and evaluation are still to implement and verify.
No working training environment, physical contact acceptance or reproduction
result is claimed by the tool-asset check.
