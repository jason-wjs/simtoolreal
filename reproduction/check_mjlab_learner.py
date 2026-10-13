"""CPU full-network SAPG/adapter/update/recovery check with synthetic dynamics."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

repo=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(repo/'rl_games'),str(repo)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    from mjlabenvs.train import prepare_runner
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    data=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    cfg=data['agent']['params']['config']
    cfg.update(device='cpu',device_name='cpu',mixed_precision=False,num_actors=6,
               expl_coef_block_size=1,minibatch_size=96,full_experiment_name='0_mjlab_cpu_interface')
    cfg['central_value_config']['minibatch_size']=96

    class Fixture:
        device='cpu'
        num_envs=6
        def __init__(self):
            self.cfg=NS(is_finite_horizon=False,obs=NS(**data['env']['obs']))
            self.counter=0
            self.last=torch.zeros(6,29)
        def obs(self):
            actor=torch.zeros(6,140)
            actor[:,:29]=self.last
            actor[:,29]=self.counter/100
            return dict(policy=actor,critic=torch.cat([actor,torch.ones(6,22)*.5],-1))
        def reset(self):
            self.counter=0;self.last.zero_()
            return self.obs(),{}
        def step(self,actions):
            self.last=actions.clone();self.counter+=1
            reward=1.-actions.square().mean(-1)
            timeout=torch.arange(6)==self.counter%6
            return self.obs(),reward,torch.zeros(6,dtype=torch.bool),timeout,{'successes':torch.zeros(6)}
        def get_state(self):return dict(counter=self.counter,last=self.last.clone(),rng=torch.get_rng_state())
        def set_state(self,state):
            self.counter=state['counter'];self.last=state['last'].clone();torch.set_rng_state(state['rng'])

    def agent(path,epochs):
        task=Fixture();runner=prepare_runner(data,task,path,epochs)
        return runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    def finite(x):
        if isinstance(x,torch.Tensor):return bool(torch.isfinite(x).all())
        if isinstance(x,dict):return all(finite(v) for v in x.values())
        if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
        return True
    first=agent(args.output/'fresh',2)
    initial={k:v.clone() for k,v in first.model.state_dict().items()}
    first.train()
    assert first.epoch_num==2 and first.frame==192
    assert any(not torch.equal(initial[k],v) for k,v in first.model.state_dict().items())
    first.save(str(args.output/'full_state'))
    expected=torch.load(args.output/'full_state.pth',map_location='cpu',weights_only=False)[0]
    assert finite(expected)
    assert expected['optimizer']['state'] and expected['central_value_training']['optimizer']['state']
    second=agent(args.output/'resumed',3)
    second.restore(str(args.output/'full_state.pth'))
    restored=second.get_full_state_weights()
    fields=['model','optimizer','assymetric_vf_nets','central_value_training','epoch','frame',
            'rnn_states','dones','obs','current_rewards','current_shaped_rewards','current_lengths','env_state']
    matches={key:equal(expected[key],restored[key]) for key in fields}
    assert all(matches.values()),matches
    second.train()
    final=second.get_full_state_weights()
    assert final['epoch']==3 and final['frame']==288 and finite(final)
    for before,after in [(expected['optimizer'],final['optimizer']),
                         (expected['central_value_training']['optimizer'],final['central_value_training']['optimizer'])]:
        assert all(float(after['state'][key]['step'])>float(value['step']) for key,value in before['state'].items())
    report=dict(passed=True,environments=6,cohorts=6,actor_dim=140,critic_dim=162,
                epochs=[2,3],frames=[192,288],restore_exact=matches,
                actor_and_critic_optimizers_advanced=True,finite=True,
                scope='Real full native networks/SAPG/adapter on CPU synthetic dynamics, AMP off and reduced batches. Not physical rollout or separate-process simulator recovery.')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
