"""Run explicitly with torchrun --standalone --nproc-per-node=2.

Synthetic correctness check, not model pretraining or benchmark performance.
"""
import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mhd_framework.models import MHDResNet, ResNetConfig

def main():
    torch.set_num_threads(2)
    cuda = os.environ.get('MHD_MODELS_TEST_DEVICE') == 'cuda'
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda',rank) if cuda else torch.device('cpu')
    if cuda: torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if cuda else 'gloo')
    try:
        torch.manual_seed(123)
        model = MHDResNet(ResNetConfig(name='resnet18'),device=device)
        ddp = DistributedDataParallel(model,device_ids=[rank] if cuda else None)
        opt=torch.optim.SGD(ddp.parameters(),lr=.001)
        torch.manual_seed(100+rank)
        for _ in range(2):
            opt.zero_grad()
            loss=ddp(torch.randn(2,3,32,32,device=device)).square().mean()
            loss.backward();opt.step()
        for p in model.parameters():
            values=[torch.empty_like(p) for _ in range(dist.get_world_size())]
            dist.all_gather(values,p.detach())
            for other in values[1:]: torch.testing.assert_close(other,values[0],rtol=0,atol=0)
        if dist.get_rank()==0: print(json.dumps({'status':'passed','world_size':dist.get_world_size(),
            'device':str(device),'steps':2,'scope':'synthetic DDP parameter synchronization; BN local per rank'}))
    finally: dist.destroy_process_group()

if __name__=='__main__': main()
