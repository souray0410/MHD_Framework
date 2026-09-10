import copy
import os
import pytest
import torch
from mhd_framework.models import create_model, ViTConfig, SwinConfig
from mhd_framework.models.training import optimizer_for

torch.set_num_threads(2)


@pytest.mark.parametrize('name', ['vit_b_16', 'swin_t'])
def test_transformer_native_output_gradient_update_and_restore(name):
    device = os.environ.get('MHD_MODEL_TEST_DEVICE', 'cpu')
    torch.manual_seed(194)
    graph = create_model({'name':name}, device=device)
    native = copy.deepcopy(graph._native_reference).to(device)
    optimizer = optimizer_for(graph, {'backbone_lr':1e-4,'head_lr':1e-3,'weight_decay':.01})
    head = native.heads if name.startswith('vit') else native.head
    ids = {id(p) for p in head.parameters()}
    reference_optimizer = torch.optim.AdamW([
        {'params':[p for p in native.parameters() if id(p) not in ids], 'lr':1e-4},
        {'params':list(head.parameters()), 'lr':1e-3}], weight_decay=.01)
    assert len(list(graph.parameters())) == len(list(native.parameters()))
    x = torch.randn(1,3,224,224,device=device)
    labels = torch.tensor([1],device=device)
    for step in range(2):
        a=x.clone().requires_grad_();b=x.clone().requires_grad_()
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None
        y=graph(a)
        torch.set_rng_state(cpu)
        if gpu is not None: torch.cuda.set_rng_state_all(gpu)
        z=native(b)
        torch.testing.assert_close(y,z,rtol=1e-5,atol=1e-6)
        torch.nn.functional.cross_entropy(y,labels).backward()
        torch.nn.functional.cross_entropy(z,labels).backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)
        for (ka,pa),(kb,pb) in zip(graph._native_reference.named_parameters(),native.named_parameters()):
            assert ka==kb
            torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-4,atol=1e-6)
        if step==1: assert a.grad.abs().sum()>0
        optimizer.step();reference_optimizer.step()
        optimizer.zero_grad(set_to_none=True);reference_optimizer.zero_grad(set_to_none=True)
        for k,v in graph.native_state_dict().items():
            torch.testing.assert_close(v,native.state_dict()[k],rtol=1e-4,atol=1e-6)
    del native,reference_optimizer
    graph.eval()
    with torch.no_grad():
        expected=graph(x)
        endpoint='block06' if name.startswith('vit') else 'stage2'
        features=graph.forward_until(endpoint,x)
        torch.testing.assert_close(graph.forward_from(endpoint,features),expected,rtol=0,atol=0)
    restored=create_model(graph.configuration(),device=device).eval()
    restored.load_state_dict(graph.state_dict(),strict=True)
    with torch.no_grad(): torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)


def test_dimensions_are_explicit():
    with pytest.raises(ValueError): ViTConfig(spatial_dims=3)
    with pytest.raises(ValueError): SwinConfig(spatial_dims=3)
    from mhd_framework.models.vision_transformer import PatchGrid
    p=PatchGrid(torch.nn.Conv2d(3,4,16,16),224)
    with pytest.raises(ValueError): p(torch.zeros(1,3,224,225))
