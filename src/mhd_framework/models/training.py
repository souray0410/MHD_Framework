"""Finite distributed training of complete classification models."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, Subset

from . import create_model
from .artifacts import (atomic_json, digest, file_sha256, export_bundle,
                        publish_bundle, validate_request, verify_runtime)


class ArrayDataset(Dataset):
    """Read a verified manifest of V,C,spatial uint8 arrays."""
    def __init__(self, path, preprocessing, expected_sha):
        self.path=Path(path)
        if file_sha256(self.path)!=expected_sha: raise ValueError('Dataset manifest changed')
        self.manifest=json.loads(self.path.read_text())
        if self.manifest['role'] not in ('train','development'): raise ValueError('Training only accepts train/development')
        self.rows=self.manifest['samples']; self.preprocessing=preprocessing
        for row in self.rows:
            file=(self.path.parent/row['file']).resolve()
            if not file.is_relative_to(self.path.parent.resolve()) or file_sha256(file)!=row['sha256']:
                raise ValueError('Dataset array changed or escapes its root')
        self.ids=[r['id'] for r in self.rows]
        if len(self.ids)!=len(set(self.ids)): raise ValueError('Duplicate sample identity')
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];path=(self.path.parent/row['file']).resolve()
        if not path.is_relative_to(self.path.parent.resolve()): raise ValueError('Array path outside dataset')
        value=np.load(path,allow_pickle=False)
        if list(value.shape)!=self.preprocessing['shape'] or value.dtype!=np.uint8:
            raise ValueError('Unexpected input shape/dtype')
        x=torch.from_numpy(value.copy()).float()/255
        shape=(1,len(self.preprocessing['mean']))+(1,)*(x.ndim-2)
        x=(x-torch.tensor(self.preprocessing['mean']).reshape(shape))/torch.tensor(self.preprocessing['std']).reshape(shape)
        return x,int(row['label']),index


def macro_f1(labels, probabilities):
    pred=probabilities.argmax(1);scores=[]
    for c in range(probabilities.shape[1]):
        tp=((pred==c)&(labels==c)).sum();fp=((pred==c)&(labels!=c)).sum();fn=((pred!=c)&(labels==c)).sum()
        scores.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return sum(scores)/len(scores)


def epoch_indices(length,rank,world,seed,epoch):
    if length%world: raise ValueError('Training participants must shard exactly; no padding/duplicates')
    return torch.randperm(length,generator=torch.Generator().manual_seed(seed+epoch)).tolist()[rank::world]


def optimizer_for(model, training):
    native=model._native_reference
    head=next(getattr(native,key) for key in ('fc','classifier','head') if hasattr(native,key))
    head_ids={id(p) for p in head.parameters()}
    backbone=[p for p in model.parameters() if id(p) not in head_ids]
    groups=[{'params':backbone,'lr':training['backbone_lr']},
            {'params':list(head.parameters()),'lr':training['head_lr']}]
    if sum(len(g['params']) for g in groups)!=len({id(p) for g in groups for p in g['params']}):raise ValueError('Repeated optimizer parameters')
    return torch.optim.AdamW(groups,weight_decay=training['weight_decay'])


def evaluate(model,data,rank,world,batch,device):
    model.eval()
    for b in model.buffers(): dist.broadcast(b,src=0)
    indices=list(range(rank,len(data),world));out=[]
    with torch.no_grad():
        for x,y,i in DataLoader(Subset(data,indices),batch_size=batch,num_workers=0):
            p=model(x.to(device)).softmax(1).cpu()
            out.extend(zip(i.tolist(),y.tolist(),p.tolist()))
    all_rows=[None]*world;dist.all_gather_object(all_rows,out)
    rows=sorted(row for part in all_rows for row in part)
    if [r[0] for r in rows]!=list(range(len(data))):raise ValueError('Evaluation coverage/identity mismatch')
    y=np.asarray([r[1] for r in rows]);p=np.asarray([r[2] for r in rows])
    return macro_f1(y,p),y,p


def save_state(path,state):
    tmp=Path(str(path)+'.partial');torch.save(state,tmp);os.replace(tmp,path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}


def run(job_path,output,mode):
    job=json.loads(Path(job_path).read_text());request=job['request'];validate_request(request)
    verify_runtime(request['framework'])
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=request['training']['world_size']:raise ValueError('Declared and actual world size differ')
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(job['allocator_limit_gib']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def status(stage,**kwargs):
        atomic_json(out/f'rank{rank}.json',dict(stage=stage,pid=os.getpid(),rank=rank,world_size=world,
          seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(device),**kwargs))
    status('initializing')
    try:
        cfg=request['training'];seed=cfg['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        pre=request['data']['preprocessing'];train=ArrayDataset(job['train_manifest'],pre,request['data']['train_manifest_sha256'])
        dev=ArrayDataset(job['development_manifest'],pre,request['data']['development_manifest_sha256'])
        if set(train.ids)&set(dev.ids):raise ValueError('Train/development overlap')
        if len(train)!=request['data']['train_count'] or len(dev)!=request['data']['development_count']:raise ValueError('Unexpected cohort size')
        if cfg['loss']!='cross_entropy_unweighted' or cfg['optimizer']!='AdamW' or cfg['augmentation']!='none' or cfg['batchnorm']!='ordinary_per_rank':
            raise ValueError('Unsupported training recipe; implement it explicitly')
        torch.hub.set_dir(job['torch_home'])
        init=request['initialization']
        if file_sha256(job['initialization_file'])!=init['sha256']:raise ValueError('Initialization file changed')
        model=create_model(request['model'],weights=init.get('weights'),device=device)
        if init['kind']=='retfound':
            receipt=model.load_pretrained_encoder(job['initialization_file'],sha256=init['sha256'],modality=init['modality'],source=init['source'])
            if rank==0:atomic_json(out/'initialization_receipt.json',receipt)
        torch.manual_seed(seed+rank);np.random.seed(seed+rank);random.seed(seed+rank)
        model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True)
        opt=optimizer_for(model.module,cfg)
        micro=cfg['microbatch_per_rank'];effective=cfg['effective_batch'];accum=effective//(world*micro)
        if accum<1 or effective!=world*micro*accum or len(train)%effective:
            raise ValueError('Effective batch must divide the participant count exactly')
        clip=cfg['gradient_clip'];loss_fn=nn.CrossEntropyLoss()
        def train_epoch(epoch,limit=None):
            model.train();order=epoch_indices(len(train),rank,world,seed,epoch)
            loader=DataLoader(Subset(train,order),batch_size=micro,shuffle=False,num_workers=0)
            opt.zero_grad(set_to_none=True);loss_sum=0.;seen=0;updates=0
            for step,(x,y,_) in enumerate(loader):
                sync=(step+1)%accum==0
                with nullcontext() if sync else model.no_sync():
                    logits=model(x.to(device));loss=loss_fn(logits,y.to(device))/accum
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                loss_sum+=loss.item()*accum*len(y);seen+=len(y)
                if sync:
                    for group in opt.param_groups:nn.utils.clip_grad_norm_(group['params'],clip,error_if_nonfinite=True)
                    opt.step();opt.zero_grad(set_to_none=True);updates+=1
                    if updates%10==0:status('training',epoch=epoch,optimizer_step=updates)
                    if limit is not None and updates>=limit:break
            values=torch.tensor([loss_sum,seen],device=device);dist.all_reduce(values)
            return float(values[0]/values[1]),updates
        if mode=='preflight':
            loss,updates=train_epoch(1,limit=2)
            for b in model.module.buffers():dist.broadcast(b,src=0)
            model.eval();x,_,_=next(iter(DataLoader(train,batch_size=micro)))
            with torch.no_grad():prediction=model.module(x.to(device)).cpu()
            # Check rank synchronization on a shared input after two real updates.
            gathered=[torch.empty_like(prediction.to(device)) for _ in range(world)]
            dist.all_gather(gathered,prediction.to(device))
            for p in gathered:torch.testing.assert_close(p.cpu(),prediction,rtol=1e-4,atol=1e-5)
            if rank==0:save_state(out/'preflight.pt',{'model':model.module.state_dict(),'optimizer':opt.state_dict()})
            dist.barrier();saved=torch.load(out/'preflight.pt',map_location=device,weights_only=True)
            model.module.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);del saved
            with torch.no_grad():torch.testing.assert_close(model.module(x.to(device)).cpu(),prediction,rtol=0,atol=0)
            dist.barrier()
            if rank==0:(out/'preflight.pt').unlink()
            status('preflight_passed',optimizer_updates=updates,real_training_data=True)
            return
        best,y,p=evaluate(model.module,dev,rank,world,micro,device)
        best_epoch=0;material_best=best;stall=0;history=[]
        if rank==0:
            atomic_json(out/'request.json',request)
            (out/'README.md').write_text('# Native task model training\n\nThis attempt is defined by request.json. Architecture, initialization, task/data, loss and optimization are recorded separately. No test evaluation.\n\nSelected complete model: best.pt. Full stopping state: last.pt. An accepted immutable bundle is published only after plateau and development replay.\n\n```json\n'+json.dumps(request,indent=2)+'\n```\n')
            save_state(out/'best.pt',{'model':model.module.state_dict(),'epoch':0})
            np.savez(out/'development_predictions.npz',indices=np.arange(len(dev)),labels=y,probabilities=p)
        final_rng=None
        for epoch in range(1,cfg['max_epochs']+1):
            loss,updates=train_epoch(epoch)
            score,y,p=evaluate(model.module,dev,rank,world,micro,device)
            if score>best:
                best=score;best_epoch=epoch
                if rank==0:
                    save_state(out/'best.pt',{'model':model.module.state_dict(),'epoch':epoch})
                    np.savez(out/'development_predictions.npz',indices=np.arange(len(dev)),labels=y,probabilities=p)
            if score>material_best+cfg['min_delta']:material_best=score;stall=0
            else:stall+=1
            if stall and stall%cfg['lr_patience']==0:
                for g in opt.param_groups:g['lr']*=cfg['lr_factor']
            state=rng_state();states=[None]*world;dist.all_gather_object(states,state);final_rng=states
            row={'epoch':epoch,'loss':loss,'development_macro_f1':score,'best_macro_f1':best,'best_epoch':best_epoch,'stall':stall,'learning_rates':[g['lr'] for g in opt.param_groups]}
            history.append(row)
            if rank==0:
                save_state(out/'last.pt',{'model':model.module.state_dict(),'optimizer':opt.state_dict(),
                    'scheduler':{'stall':stall,'material_best':material_best,'best':best,'best_epoch':best_epoch,'learning_rates':row['learning_rates']},
                    'epoch':epoch,'rng':states,'sampler':{'seed':seed,'next_epoch':epoch+1,'world_size':world,'order_rule':'randperm(seed+epoch)[rank::world]'}})
                atomic_json(out/'history.json',history)
            status('epoch_complete',**row)
            if epoch>=cfg['min_epochs'] and stall>=cfg['patience']:break
        plateau=epoch>=cfg['min_epochs'] and stall>=cfg['patience']
        if not plateau:
            if rank==0:atomic_json(out/'summary.json',{'status':'needs_attention','reason':'max_epochs_without_plateau','test_used':False})
            status('needs_attention',reason='max_epochs_without_plateau');return
        selected=torch.load(out/'best.pt',map_location=device,weights_only=True)
        model.module.load_state_dict(selected['model'],strict=True);del selected
        score,y,p=evaluate(model.module,dev,rank,world,micro,device)
        if rank==0:
            saved=np.load(out/'development_predictions.npz',allow_pickle=False)
            np.testing.assert_allclose(saved['probabilities'],p,rtol=1e-5,atol=1e-6)
            if abs(score-best)>1e-12:raise ValueError('Selected checkpoint replay metric mismatch')
            training={'status':'plateau','stop_epoch':epoch,'best_epoch':best_epoch,'macro_f1':best,'test_used':False,'seconds':time.monotonic()-started}
            replay={'status':'passed','development_participants':len(dev),'macro_f1':score,'prediction_replay':True}
            atomic_json(out/'training_receipt.json',training);atomic_json(out/'replay_receipt.json',replay)
            acceptance={'status':'accepted','training_receipt_sha256':file_sha256(out/'training_receipt.json'),'replay_receipt_sha256':file_sha256(out/'replay_receipt.json')}
            model.module.cpu()
            # This is our own complete continuation checkpoint, including NumPy RNG state.
            resume=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            bundle=out/'bundle';export_bundle(model.module,bundle,request,acceptance=acceptance,resume_state=resume)
            publish_bundle(job['store'],bundle,request)
            atomic_json(out/'summary.json',dict(training,status='accepted',request_id=digest(request),bundle=str(bundle)))
        dist.barrier();status('accepted',best_epoch=best_epoch,stop_epoch=epoch)
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=('preflight','train'),required=True)
    args=parser.parse_args();run(args.job,args.output,args.mode)
