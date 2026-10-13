"""CPU parity against the pre-extraction native Lab action implementation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import yaml
    from mjlabenvs.control import SimToolRealControl
    from simtoolreal_shared.robot_constants import JOINT_NAMES_CANONICAL, ARM_DEFAULT_JOINT_POS

    repo = Path(__file__).resolve().parents[1]
    reference = subprocess.check_output(['git','show',
        '3c606d6:isaacsimenvs/tasks/simtoolreal/utils/action_utils.py'],cwd=repo,text=True)
    native = {}
    exec(compile(reference,'native-action-reference','exec'),native)
    data = yaml.safe_load(args.config.read_text())['env']
    cfg = SimpleNamespace(action=SimpleNamespace(**data['action']),
                          domain_randomization=SimpleNamespace(**data['domain_randomization']))
    urdf = ET.parse(repo/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf').getroot()
    limits = {j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
              for j in urdf.findall('joint') if j.get('type')=='revolute'}
    limits = torch.tensor([limits[name] for name in JOINT_NAMES_CANONICAL])
    default = torch.tensor([ARM_DEFAULT_JOINT_POS.get(name,0.) for name in JOINT_NAMES_CANONICAL])
    torch.manual_seed(42)
    step_dt = data['sim']['dt'] * data['decimation']
    port = SimToolRealControl(cfg,limits,default,8,'cpu',step_dt)
    # A nontrivial hand traversal tests the actual ordering boundary; arm stays
    # first as in the native parser. No assumption of identical hand ordering.
    perm = torch.cat([torch.arange(7),torch.randperm(22)+7])
    inv = torch.argsort(perm)
    old = copy.deepcopy(port)
    old._perm_canon_to_lab = perm
    old._cur_targets = old._cur_targets[:,perm].clone()
    old._prev_targets = old._prev_targets[:,perm].clone()
    old._hand_lower,old._hand_upper = limits[perm][7:].unbind(-1)
    lengths = torch.zeros(8,dtype=torch.long)
    successes = torch.zeros_like(lengths)
    for step in range(100):
        if step == 20:
            lengths[0],successes[0] = 0,1  # Goal hit is not a full episode reset.
        if step == 40:
            ids = torch.tensor([1,3])
            q = default.expand(2,-1).clone()
            port.reset(ids,q)
            old.reset(ids,q[:,perm])
            lengths[ids],successes[ids] = 0,0
        if step == 50:
            state = port.get_state()
            port = SimToolRealControl(cfg,limits,default,8,'cpu',step_dt)
            port.set_state(state)
        actions = torch.empty(8,29).uniform_(-1.5,1.5)
        rng = torch.get_rng_state()
        old.episode_length_buf,old._successes = lengths,successes
        native['apply_action_pipeline'](old,actions)
        torch.set_rng_state(rng)
        actual = port.targets(actions,lengths,successes)
        torch.testing.assert_close(actual,old._cur_targets[:,inv],atol=0,rtol=0)
        torch.testing.assert_close(port._action_queue,old._action_queue[:,:,inv],atol=0,rtol=0)
        lengths += 1
    report = {'passed':True,'steps':100,'environments':8,'exact_float32_targets_and_delay_queues':True,
              'reference_commit':'3c606d6','config':str(args.config),'action_delay_max':cfg.domain_randomization.action_delay_max,
              'checks':['nontrivial hand order','goal-hit delay retention','partial full reset','separate control object restore','clamping beyond action range']}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
