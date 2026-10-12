"""CPU checks for the independent original Gym runtime and official checkpoint."""

# The SDK requires this import before torch.
from isaacgym import gymapi, gymtorch  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pytorch3d
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix
import torch

from deployment.rl_player import RlPlayer


rotation = axis_angle_to_matrix(torch.tensor([[0.1, -0.2, 0.3]]))
assert torch.allclose(rotation, quaternion_to_matrix(matrix_to_quaternion(rotation)), atol=1e-6)
player = RlPlayer(140, 29, "pretrained_policy/config.yaml", "pretrained_policy/model.pth", "cpu", 1)
action = player.get_normalized_action(torch.zeros((1, 140)))
assert action.shape == (1, 29) and torch.isfinite(action).all()
result = {"numpy": np.__version__, "torch": torch.__version__, "pytorch3d": pytorch3d.__version__,
          "action_shape": list(action.shape), "finite_action": True,
          "scope": "CPU transform roundtrip and original checkpoint forward; no simulation or success-rate claim"}
Path("outputs").mkdir(exist_ok=True)
Path("outputs/native-gym-runtime-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
