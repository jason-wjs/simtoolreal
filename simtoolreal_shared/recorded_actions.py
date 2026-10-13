"""Recorded-control validation for the finite native Lab physics replay."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_recorded_actions(path, action_dim):
    with np.load(path, allow_pickle=False) as data:
        steps = data['control_step']
        substeps = data['substep']
        decimation = int(substeps.max())
        count = int(steps[-1]) + 1
        np.testing.assert_array_equal(steps, np.repeat(np.arange(count), decimation))
        np.testing.assert_array_equal(substeps, np.tile(np.arange(1, decimation + 1), count))
        actions = data['action'][::decimation].copy()
        assert actions.shape == (count, action_dim) and np.isfinite(actions).all()
        np.testing.assert_array_equal(data['action'], np.repeat(actions, decimation, axis=0))
        np.testing.assert_array_equal(actions[0], np.zeros(action_dim))
        return dict(actions=actions, source_decimation=decimation,
                    initial_joint_pos=data['joint_pos'][decimation - 1].copy(),
                    initial_object_pose=data['object_state_w'][decimation - 1, :7].copy())


def write_replay_check(source, output, replay, decimation, policy_dt, done):
    with np.load(Path(output) / 'physics_trajectory.npz', allow_pickle=False) as data:
        controls = data['action'][::decimation]
        np.testing.assert_array_equal(controls, replay['actions'][:len(controls)])
        np.testing.assert_array_equal(data['action'], np.repeat(controls, decimation, axis=0))
        report = dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            source_control_steps=len(replay['actions']), executed_control_steps=len(controls),
            source_decimation=replay['source_decimation'], decimation=decimation,
            policy_dt=policy_dt, environment_done=bool(done), actions_exact=True,
            reset_tick_joint_max_abs_error=float(np.max(np.abs(
                data['joint_pos'][decimation - 1] - replay['initial_joint_pos']))),
            reset_tick_object_pose_max_abs_error=float(np.max(np.abs(
                data['object_state_w'][decimation - 1, :7] - replay['initial_object_pose']))),
            scope='Recorded actions at fixed control interval; native resets/dones retained. '
                  'Changed physics can change termination time. No policy inference or training.')
        audit = json.loads((Path(output) / 'physics_audit.json').read_text())
        table_id, = [i for i, name in enumerate(audit['filter_bodies']) if '/Table/' in name]
        contacts = data['contacts']
        selected = np.flatnonzero(contacts[:, 0] == table_id)
        assert len(selected), 'Replay has no recorded table contacts'
        peak = int(selected[np.argmin(contacts[selected, -1])])
        sample = int(np.searchsorted(data['contact_offsets'], peak, side='right') - 1)
        report.update(table_peak_depth_m=max(0., -float(contacts[peak, -1])),
                      table_peak_control_step=int(data['control_step'][sample]))
    (Path(output) / 'replay_check.json').write_text(json.dumps(report, indent=2) + '\n')
