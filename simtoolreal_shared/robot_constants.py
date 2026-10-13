"""Canonical Sharpa + iiwa14 task constants shared by the Lab and mjlab ports."""
from __future__ import annotations

ARM_JOINT_REGEX = "iiwa14_joint_.*"
HAND_JOINT_REGEX = "left_.*"

# Legacy policy order; Isaac Lab tensors are permuted at action/obs boundaries.
JOINT_NAMES_CANONICAL: tuple[str, ...] = (
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)
assert len(JOINT_NAMES_CANONICAL) == 29

PALM_BODY_NAME = "iiwa14_link_7"
# Merged fingertip bodies land on the DP links in both sims.
FINGERTIP_BODY_REGEX = "left_(index|middle|ring|thumb|pinky)_DP"
FINGERTIP_LINK_NAMES: tuple[str, ...] = (
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
)


# Per-joint PD gains and dynamics (verified with pretrained checkpoint).
ARM_JOINT_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}
ARM_JOINT_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}

HAND_JOINT_STIFFNESS: dict[str, float] = {
    "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
    "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
    "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9, "left_index_DIP": 0.9,
    "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
    "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
    "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
    "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
}
HAND_JOINT_DAMPING: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
    "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
    "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
    "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}
HAND_JOINT_ARMATURE: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
    "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
    "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
    "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
    "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
}
HAND_JOINT_FRICTION: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.132, "left_thumb_CMC_AA": 0.132,
    "left_thumb_MCP_FE": 0.07456, "left_thumb_MCP_AA": 0.07456, "left_thumb_IP": 0.01276,
    "left_2_index_MCP_FE": 0.07456, "left_index_MCP_AA": 0.07456,
    "left_index_PIP": 0.01276, "left_index_DIP": 0.00378738,
    "left_3_middle_MCP_FE": 0.07456, "left_middle_MCP_AA": 0.07456,
    "left_middle_PIP": 0.01276, "left_middle_DIP": 0.00378738,
    "left_4_ring_MCP_FE": 0.07456, "left_ring_MCP_AA": 0.07456,
    "left_ring_PIP": 0.01276, "left_ring_DIP": 0.00378738,
    "left_5_pinky_CMC": 0.012, "left_pinky_MCP_FE": 0.07456,
    "left_pinky_MCP_AA": 0.07456, "left_pinky_PIP": 0.01276, "left_pinky_DIP": 0.00378738,
}

assert len(ARM_JOINT_STIFFNESS) == 7 and len(ARM_JOINT_DAMPING) == 7
assert len(HAND_JOINT_STIFFNESS) == 22 and len(HAND_JOINT_DAMPING) == 22
assert len(HAND_JOINT_ARMATURE) == 22 and len(HAND_JOINT_FRICTION) == 22

# Proven-working default arm pose (isaacsim_conversion/isaacsim_env.py:101-109).
ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}

