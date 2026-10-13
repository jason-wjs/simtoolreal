"""CPU adapter contract parity against the installed native Lab wrapper."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

repo=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(repo/'rl_games'),str(repo)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lab-wrapper',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from mjlabenvs.rlgames import SimToolRealVecEnv
    source=args.lab_wrapper.read_text()
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='RlGamesVecEnvWrapper')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'step','reset','_process_obs'}]
    tree=ast.parse('from __future__ import annotations\n')
    tree.body.extend(methods)
    reference={'torch':torch}
    exec(compile(tree,'native-wrapper-reference','exec'),reference)

    class Fixture:
        device='cpu'
        def __init__(self,finite):
            self.cfg=NS(is_finite_horizon=finite,obs=NS(obs_list=['joint_pos'],state_list=['joint_pos','joint_vel']))
            self.counter=0
        def reset(self):
            self.counter=0
            return dict(policy=torch.full((4,29),4.),critic=torch.full((4,58),-4.)),{}
        def step(self,actions):
            self.last_action=actions.clone()
            self.counter+=1
            return (dict(policy=actions*4,critic=torch.cat([actions,-actions],-1)*4),
                actions.sum(-1),torch.tensor([True,False,False,False]),
                torch.tensor([False,True,False,False]),{'successes':torch.arange(4).float()})
        def get_state(self):return {'counter':self.counter}
        def set_state(self,state):self.counter=state['counter']

    cases=[]
    for finite in (False,True):
        task=Fixture(finite)
        port=SimToolRealVecEnv(task,2.,1.)
        native_task=Fixture(finite)
        native=NS(env=native_task,unwrapped=native_task,_sim_device='cpu',_rl_device='cpu',
                  _clip_obs=2.,_clip_actions=1.,_obs_groups={'obs':['policy'],'states':['critic']},
                  _concate_obs_groups=True,_obs_concat_fn=lambda values:torch.cat(values,-1),
                  _states_concat_fn=lambda values:torch.cat(values,-1))
        native._process_obs=lambda obs:reference['_process_obs'](native,obs)
        for k,v in reference['reset'](native).items():
            torch.testing.assert_close(port.reset()[k],v,atol=0,rtol=0)
        for step in range(10):
            actions=torch.linspace(-3,3,116).reshape(4,29).requires_grad_()
            original=actions.detach().clone()
            expected=reference['step'](native,actions)
            actual=port.step(actions)
            for k in expected[0]:torch.testing.assert_close(actual[0][k],expected[0][k],atol=0,rtol=0)
            for a,b in zip(actual[1:3],expected[1:3]):torch.testing.assert_close(a,b,atol=0,rtol=0)
            assert actual[3].keys()==expected[3].keys()
            for k in expected[3]:torch.testing.assert_close(actual[3][k],expected[3][k],atol=0,rtol=0)
            torch.testing.assert_close(task.last_action,native_task.last_action,atol=0,rtol=0)
            torch.testing.assert_close(actions,original,atol=0,rtol=0)
            assert not task.last_action.requires_grad
        info=port.get_env_info()
        assert info['observation_space'].shape==(29,) and info['state_space'].shape==(58,)
        assert info['action_space'].shape==(29,) and info['action_space'].low.min()==-1
        assert info['observation_space'].high.max()==2
        saved=copy.deepcopy(port.get_env_state());task.counter=999;port.set_env_state(saved)
        assert task.counter==10
        cases.append(dict(is_finite_horizon=finite,passed=True))
    report=dict(passed=True,cases=cases,lab_wrapper_sha256=hashlib.sha256(args.lab_wrapper.read_bytes()).hexdigest(),
        checks=['actor/critic routing and clip','action detach/clone/clip','done union','finite/infinite timeout semantics','spaces','checkpoint delegation'],
        scope='CPU adapter fixture; does not establish physical state recovery or learner update')
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
