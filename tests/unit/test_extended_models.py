"""Explicit-port parity, including nonlinear/stochastic and volumetric paths."""
import copy,os
import pytest
import torch
from mhd_framework.models import create_model
from mhd_framework.models.training import optimizer_for

NAMES=['convnextv2_base','convnextv2_large','deit3_base_patch16_224','deit3_large_patch16_224',
       'mae_vit_base_patch16','mae_vit_large_patch16','beit_base_patch16_224','beit_large_patch16_224',
       'openclip_vit_b16','openclip_vit_l14','medicalnet_resnet50_3d','medicalnet_resnet101_3d',
       'monai_densenet121_3d','swin_unetr_encoder_3d']

@pytest.mark.parametrize('name',NAMES)
def test_native_graph_outputs_gradients_updates_and_resume(name):
    if name.startswith('openclip_'):pytest.importorskip('open_clip')
    if name.startswith(('monai_','swin_unetr')):pytest.importorskip('monai')
    torch.set_num_threads(2);torch.manual_seed(3416)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=os.environ.get('MHD_MODEL_TEST_DEVICE','cpu')
    graph=create_model({'name':name},device=device)
    # Identity view boundaries change FP32 gradient summation order in GRN.
    # Adam amplifies near-zero differences. Use higher precision for strict
    # native optimizer parity, and independently check FP32 outputs/gradients.
    dtype=torch.float64 if name.startswith('convnextv2_') else torch.float32
    graph.to(dtype=dtype);native=copy.deepcopy(graph._native_reference).to(device)
    ids={id(p) for p in native.head.parameters()}
    opt=optimizer_for(graph,dict(backbone_lr=1e-4,head_lr=1e-3,weight_decay=.01))
    ref=torch.optim.AdamW([{'params':[p for p in native.parameters() if id(p) not in ids],'lr':1e-4},
                          {'params':list(native.head.parameters()),'lr':1e-3}],weight_decay=.01)
    assert len(list(graph.parameters()))==len(list(native.parameters()))
    x=torch.randn((2,1,32,64,64) if name.endswith('_3d') else (1,3,224,224),device=device,dtype=dtype)
    y=torch.arange(len(x),device=device)%2
    for _ in range(2):
        a=x.detach().clone().requires_grad_();b=x.detach().clone().requires_grad_()
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None
        actual=graph(a);torch.set_rng_state(cpu)
        if gpu is not None:torch.cuda.set_rng_state_all(gpu)
        expected=native(b);torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        torch.nn.functional.cross_entropy(actual,y).backward();torch.nn.functional.cross_entropy(expected,y).backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)
        for (k,p),(key,q) in zip(graph._native_reference.named_parameters(),native.named_parameters()):
            assert k==key
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)
        opt.step();ref.step();opt.zero_grad(set_to_none=True);ref.zero_grad(set_to_none=True)
        for key,val in graph.native_state_dict().items():
            try:torch.testing.assert_close(val,native.state_dict()[key],rtol=1e-4,atol=1e-6)
            except AssertionError as error:raise AssertionError(key+' '+str(error)) from error
    del native,ref,opt
    graph.eval()
    with torch.no_grad():
        expected=graph(x);features=graph.forward_until('features',x)
        torch.testing.assert_close(graph.forward_from('features',features),expected,rtol=0,atol=0)
    restored=create_model(graph.configuration(),device=device).to(dtype=dtype).eval();restored.load_state_dict(graph.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)

@pytest.mark.parametrize('name',['convnextv2_base','convnextv2_large'])
def test_convnext_v2_fp32_native_gradients(name):
    torch.set_num_threads(2);torch.manual_seed(3416)
    device=os.environ.get('MHD_MODEL_TEST_DEVICE','cpu')
    graph=create_model({'name':name},device=device);native=copy.deepcopy(graph._native_reference)
    x=torch.randn(1,3,224,224,device=device,requires_grad=True);ref=x.detach().clone().requires_grad_()
    y=graph(x);z=native(ref)
    torch.testing.assert_close(y,z,rtol=1e-5,atol=1e-6)
    y.square().mean().backward();z.square().mean().backward()
    torch.testing.assert_close(x.grad,ref.grad,rtol=1e-4,atol=1e-6)
    for p,q in zip(graph._native_reference.parameters(),native.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)

@pytest.mark.parametrize('pool',['max','average'])
@pytest.mark.parametrize('ties',[False,True])
def test_deterministic_volume_pool_matches_original_geometry(pool,ties):
    from mhd_framework.models.volume_encoders import DeterministicMaxPool3d,DeterministicAvgPool3d
    torch.manual_seed(73)
    original=torch.nn.MaxPool3d(3,2,1) if pool=='max' else torch.nn.AvgPool3d(2,2)
    replacement=DeterministicMaxPool3d() if pool=='max' else DeterministicAvgPool3d()
    x=torch.randn(2,3,9,11,13,dtype=torch.float64)
    if ties:x=x.round()
    a=x.requires_grad_();b=x.detach().clone().requires_grad_()
    y=original(a);z=replacement(b)
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    weights=torch.randn_like(y);(y*weights).sum().backward();(z*weights).sum().backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=1e-14,atol=1e-14)
    if os.environ.get('MHD_MODEL_TEST_DEVICE','cpu').startswith('cuda'):
        torch.use_deterministic_algorithms(True)
        c=x.detach().cuda().requires_grad_();actual=replacement(c)
        (actual*weights.cuda()).sum().backward()
        torch.testing.assert_close(actual.cpu(),y,rtol=0,atol=0)
        torch.testing.assert_close(c.grad.cpu(),a.grad,rtol=1e-14,atol=1e-14)
