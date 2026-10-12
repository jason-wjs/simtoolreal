"""Native contact/state recording for the one-environment DexToolBench audit.

Records each physics substep before automatic resets. This is deliberately
separate from paper success metrics and does not change task calculations.
"""

import json
from pathlib import Path

import numpy as np
import torch
from isaaclab.sim import activate_contact_sensors
from isaacsim.core.simulation_manager import SimulationManager

from ..simtoolreal_env import SimToolRealEnv


class AuditedSimToolRealEnv(SimToolRealEnv):
    def _setup_scene(self):
        super()._setup_scene()
        activate_contact_sensors("/World/envs/env_0/Object", threshold=0.0)

    def __init__(self, cfg):
        assert cfg.scene.num_envs == 1, "This recorder audits one DexToolBench episode at a time"
        super().__init__(cfg)
        self.audit_substep = 0
        self.audit_control_step = -1
        self.audit_rows = []
        self.audit_contacts = []
        self.audit_contact_offsets = [0]
        self.audit_capacity = 4096
        self.audit_object_path = self.object.root_physx_view.prim_paths[0]
        self.audit_filter_paths = list(self.robot.root_physx_view.link_paths[0])
        self.audit_filter_paths += list(self.table.root_physx_view.prim_paths)
        self.audit_view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
            self.audit_object_path,
            self.audit_filter_paths,
            max_contact_data_count=self.audit_capacity,
        )
        assert self.audit_view.sensor_count == 1
        assert self.audit_view.filter_count == len(self.audit_filter_paths)
        self.audit_joint_limits = self.robot.root_physx_view.get_dof_limits()[0].cpu().numpy().copy()

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.audit_control_step += 1
        self.audit_substep = 0
        self.audit_action = actions[0].detach().cpu().numpy().copy()

    def _apply_action(self):
        if self.audit_substep:
            self._capture_audit_substep()
        super()._apply_action()
        self.audit_substep += 1

    def _get_dones(self):
        # Called after the last physics substep and before DirectRLEnv resets.
        result = super()._get_dones()
        self._capture_audit_substep()
        return result

    def _capture_audit_substep(self):
        force, point, normal, separation, counts, starts = self.audit_view.get_contact_data(self.physics_dt)
        counts = counts.cpu().numpy().reshape(-1).astype(np.int64)
        starts = starts.cpu().numpy().reshape(-1).astype(np.int64)
        total = int(counts.sum())
        if total >= self.audit_capacity:
            raise RuntimeError("Native contact buffer capacity reached; audit would be incomplete")
        packed = []
        for body, (count, start) in enumerate(zip(counts, starts)):
            if not count:
                continue
            if start + count > self.audit_capacity:
                raise RuntimeError("Native contact indices exceed the audit buffer")
            section = slice(start, start + count)
            data = torch.cat((force[section], point[section], normal[section], separation[section]), dim=1)
            packed.append(np.column_stack((np.full(count, body), data.cpu().numpy())))
        contacts = np.concatenate(packed) if packed else np.empty((0, 9))
        self.audit_contacts.append(contacts)
        self.audit_contact_offsets.append(self.audit_contact_offsets[-1] + total)
        self.audit_rows.append({
            "control_step": self.audit_control_step,
            "substep": self.audit_substep,
            "joint_pos": self.robot.data.joint_pos[0].cpu().numpy().copy(),
            "joint_vel": self.robot.data.joint_vel[0].cpu().numpy().copy(),
            "object_state_w": self.object.data.root_state_w[0].cpu().numpy().copy(),
            "goal_state_w": self.goal_viz.data.root_state_w[0].cpu().numpy().copy(),
            "action": self.audit_action,
        })

    def write_audit(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray([row[key] for row in self.audit_rows]) for key in self.audit_rows[0]}
        contacts = np.concatenate(self.audit_contacts)
        arrays["contacts"] = contacts
        arrays["contact_offsets"] = np.asarray(self.audit_contact_offsets)
        arrays["joint_limits"] = self.audit_joint_limits
        np.savez_compressed(output_dir / "physics_trajectory.npz", **arrays)
        penetration = np.maximum(-contacts[:, -1], 0.0)
        q = arrays["joint_pos"]
        violation = np.maximum(np.maximum(self.audit_joint_limits[:, 0] - q, q - self.audit_joint_limits[:, 1]), 0.0)
        by_body = []
        for body, path in enumerate(self.audit_filter_paths):
            body_contacts = contacts[contacts[:, 0] == body]
            if not len(body_contacts):
                continue
            depth = np.maximum(-body_contacts[:, -1], 0.0)
            by_body.append({
                "body": path, "contact_samples": len(body_contacts),
                "max_penetration_m": float(depth.max()),
                "penetration_p95_m": float(np.quantile(depth, 0.95)),
                "max_normal_force_N": float(np.abs(body_contacts[:, 1]).max()),
            })
        summary = {
            "physics_dt": self.physics_dt,
            "decimation": self.cfg.decimation,
            "physics_samples": len(self.audit_rows),
            "contact_samples": len(contacts),
            "contact_capacity": self.audit_capacity,
            "object_body": self.audit_object_path,
            "filter_bodies": self.audit_filter_paths,
            "contact_columns": ["filter_body_index", "normal_force_N", "point_x", "point_y", "point_z", "normal_x", "normal_y", "normal_z", "separation_m"],
            "max_penetration_m": float(penetration.max()) if len(penetration) else None,
            "penetration_p95_m": float(np.quantile(penetration, 0.95)) if len(penetration) else None,
            "contacts_deeper_than_5mm": int((penetration > 0.005).sum()),
            "max_joint_limit_violation_rad": float(violation.max()),
            "max_abs_joint_velocity_rad_s": float(np.abs(arrays["joint_vel"]).max()),
            "max_object_linear_speed_m_s": float(np.linalg.norm(arrays["object_state_w"][:, 7:10], axis=1).max()),
            "contacts_by_body": by_body,
            "finite_states_actions_contacts": all(bool(np.isfinite(value).all()) for key, value in arrays.items() if key != "joint_limits"),
            "scope": "Native object-to-robot and object-to-table contacts at every physics substep, before automatic resets. Excludes robot self-contact, robot-table and ground contacts. No geometry reconstruction. Empty contact data is not a physical pass.",
        }
        (output_dir / "physics_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
