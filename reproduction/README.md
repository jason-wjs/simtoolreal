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
