"""Profile native Lab continuation, rejecting mismatched restored state."""
import json
from pathlib import Path
import runpy

import check_lab_resume as audit
import profile_lab_training  # Installs the synchronized epoch recorder.
from rl_games.algos_torch.a2c_continuous import A2CAgent


def restore(self, weights, set_epoch=True):
    audit.set_state(self, weights, set_epoch=set_epoch)
    report = json.loads((Path(self.config['train_dir']) / 'resume_probe.json').read_text())
    assert all(report['task_state_exact'].values()), 'Task restore mismatch'
    assert all(report['learner_state_exact'].values()), 'Learner restore mismatch'
    assert all(v['max_abs_difference'] <= 1e-6 for v in report['restored_scene_vs_physx'].values())


A2CAgent.set_full_state_weights = restore

if __name__ == '__main__':
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'isaacsimenvs/train.py'), run_name='__main__')
