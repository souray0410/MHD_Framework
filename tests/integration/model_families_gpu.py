"""Finite single-GPU family acceptance; no training data or checkpoint download."""
import json
import torch
from mhd_framework.models import create_model

torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)
torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)

def reference(model,x):
    native=model._native_reference
    if model.config.name.startswith('densenet'):
        return native(x)
    x=native.patch_embed(x)
    x=torch.cat((native.cls_token.expand(x.shape[0],-1,-1),x),1)+native.pos_embed
    for block in native.blocks:
        x=block(x)
    return native.head(native.fc_norm(x[:,1:].mean(1)))

receipts=[]
for name,size in [('densenet121',64),('retfound_mae_vit_large_patch16',224)]:
    torch.manual_seed(42)
    model=create_model({'name':name},device='cuda').train()
    x=torch.randn(2 if size==64 else 1,3,size,size,device='cuda',requires_grad=True)
    target=torch.zeros(x.shape[0],dtype=torch.long,device='cuda')
    before={k:v.clone() for k,v in model._native_reference.named_buffers()}
    actual=model(x)
    torch.nn.functional.cross_entropy(actual,target).backward()
    actual_y=actual.detach().cpu();actual_g=x.grad.detach().cpu()
    grads={k:p.grad.detach().cpu() for k,p in model._native_reference.named_parameters()}
    model.zero_grad(set_to_none=True);x.grad=None
    # Restore BN between independent forward paths.
    with torch.no_grad():
        for k,b in model._native_reference.named_buffers(): b.copy_(before[k])
    expected=reference(model,x)
    torch.nn.functional.cross_entropy(expected,target).backward()
    torch.testing.assert_close(actual_y,expected.detach().cpu(),rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(actual_g,x.grad.detach().cpu(),rtol=5e-4,atol=5e-5)
    for k,p in model._native_reference.named_parameters():
        torch.testing.assert_close(grads[k],p.grad.detach().cpu(),rtol=5e-4,atol=5e-5)
    torch.optim.SGD(model.parameters(),lr=1e-4).step()
    model.zero_grad(set_to_none=True)
    y=model(x.detach());torch.nn.functional.cross_entropy(y,target).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        saved_output=model(x.detach()).cpu()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path=directory+'/model.pt'
        torch.save({'config':model.configuration(),'state_dict':model.state_dict()},path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        restored=create_model(payload['config'],device='cuda').eval()
        restored.load_state_dict(payload['state_dict'],strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(x.detach()).cpu(),saved_output,rtol=0,atol=0)
        del restored,payload
    receipts.append({'name':name,'input_shape':list(x.shape),'parameters':sum(p.numel() for p in model.parameters()),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'output_gradient_and_update':'passed',
        'strict_checkpoint_replay':'passed','official_weights_loaded':False,'world_size':1})
    print(json.dumps(receipts[-1]),flush=True)
    del model,x,actual,expected,y,grads,before
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
print(json.dumps({'status':'passed','device':torch.cuda.get_device_name(0),'tests':receipts}))
