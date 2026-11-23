"""CPU-only parallel fitting of independent scalar-context ensemble members."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing

def validate_workers(workers,members,device,context_mode,test_fraction):
    if not isinstance(workers,int) or isinstance(workers,bool) or workers<1 or members<1:
        raise ValueError('Worker/member counts must be positive integers')
    if device!='cpu' or context_mode!='scalar' or test_fraction!=0.:
        raise ValueError('Parallel CPU members require scalar CPU training without a test split')
    return min(workers,members)

def fit_member(payload):
    import torch
    from crqsf_core import ConditionalRQSplineFlow,FlowConfig
    from crqsf_data import make_loader
    from train_crqsf import train_model_to_best
    torch.set_num_threads(1)
    torch.manual_seed(payload['seed'])
    model=ConditionalRQSplineFlow(FlowConfig(**payload['flow_config'])).cpu()
    x=payload['x'];y=payload['y'];train=payload['train'];val=payload['val'];weights=payload['weights'];settings=payload['settings']
    train_loader=make_loader(x[train],y[train],batch_size=settings['batch_size'],sample_weights=weights[train] if weights is not None else None,shuffle=True)
    val_loader=make_loader(x[val],y[val],batch_size=settings['batch_size'],sample_weights=weights[val] if weights is not None else None,shuffle=False)
    options={k:v for k,v in settings.items() if k!='batch_size'}
    history,test_nll=train_model_to_best(model,train_loader,val_loader,None,device=torch.device('cpu'),member_label=f"[CPU member {payload['index']+1} seed={payload['seed']}]",**options)
    return dict(index=payload['index'],seed=payload['seed'],state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()},history=history,test_nll=test_nll,rng_state=torch.get_rng_state().numpy().copy(),training=model.training)

def train_members(args,flow_cfg,x,y,folds,seeds,weights,device):
    import os
    workers=validate_workers(int(os.environ['SAGE_CPU_ENSEMBLE_WORKERS']),len(seeds),str(device),args.context_mode,args.test_fraction)
    settings={k:getattr(args,k) for k in ['batch_size','lr','weight_decay','grad_clip','epochs','early_stopping','early_stopping_patience','early_stopping_min_delta']}
    payloads=[dict(index=i,seed=int(seed),flow_config=asdict(flow_cfg),x=x,y=y,train=folds[i][0],val=folds[i][1],weights=weights,settings=settings) for i,seed in enumerate(seeds)]
    print(f'CPU ensemble: {workers} worker processes; one thread each; {len(seeds)} members',flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        results=list(executor.map(fit_member,payloads))
    if [r['index'] for r in results]!=list(range(len(seeds))) or [r['seed'] for r in results]!=list(seeds):raise ValueError('CPU member ordering mismatch')
    return results
