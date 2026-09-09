import copy
import pytest
import torch
from torch import nn
pytest.importorskip('torchvision')
from mhd_framework.models import create_model, DenseNetConfig, RETFoundConfig
from mhd_framework.models.graph import NamedModelGraph
from mhd_framework.models.retfound import retfound_operations, _load_encoder

torch.set_num_threads(2)

@pytest.mark.parametrize('name',['densenet121','densenet161','densenet169','densenet201'])
def test_densenet_complete_native_parity(name):
    torch.manual_seed(71)
    model=create_model(dict(name=name,views=2))
    native=copy.deepcopy(model._native_reference)
    x=torch.randn(2,2,3,32,32,requires_grad=True)
    xr=x.detach().clone().requires_grad_()
    opt=torch.optim.SGD(model.parameters(),lr=1e-4)
    refopt=torch.optim.SGD(native.parameters(),lr=1e-4)
    y=model(x)
    f=torch.relu(native.features(xr.flatten(0,1)))
    expected=native.classifier(nn.functional.adaptive_avg_pool2d(f,1).flatten(1).reshape(2,2,-1).mean(1))
    torch.testing.assert_close(y,expected)
    y.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(x.grad,xr.grad)
    for (_,p),(_,q) in zip(model._native_reference.named_parameters(),native.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad)
    opt.step();refopt.step()
    for k,v in model.native_state_dict().items():
        torch.testing.assert_close(v,native.state_dict()[k])
    assert 'denseblock4' in model.endpoint_nodes and 'transition3' in model.endpoint_nodes
    assert len(tuple(model.parameters())) == len(tuple(native.parameters()))
    model.eval()
    with torch.no_grad():
        full=model(x.detach())
        features=model.forward_until('denseblock2',x.detach())
        torch.testing.assert_close(full,model.forward_from('denseblock2',features))
    restored=create_model(model.configuration()).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(x.detach()),full,rtol=0,atol=0)


def fixture_vit():
    pytest.importorskip('timm')
    from timm.models.vision_transformer import VisionTransformer
    from functools import partial
    native=VisionTransformer(img_size=224,patch_size=16,embed_dim=32,depth=2,
        num_heads=4,mlp_ratio=4,qkv_bias=True,num_classes=2,global_pool='avg',fc_norm=True,
        norm_layer=partial(nn.LayerNorm,eps=1e-6))
    for block in native.blocks: block.attn.fused_attn=False
    return native


def original_formula(native,x):
    # Independent explicit global-pool formula from the original RETFound definition.
    x=native.patch_embed(x)
    x=torch.cat((native.cls_token.expand(x.shape[0],-1,-1),x),1)+native.pos_embed
    for block in native.blocks:
        a=block.norm1(x)
        b,n,c=a.shape
        q,k,v=block.attn.qkv(a).reshape(b,n,3,block.attn.num_heads,c//block.attn.num_heads).permute(2,0,3,1,4).unbind(0)
        attention=((q*block.attn.scale)@k.transpose(-2,-1)).softmax(-1)
        x=x+block.attn.proj((attention@v).transpose(1,2).reshape(b,n,c))
        a=block.norm2(x)
        x=x+block.mlp.fc2(block.mlp.act(block.mlp.fc1(a)))
    return native.head(native.fc_norm(x[:,1:].mean(1)))


def test_retfound_partition_formula_and_encoder_contract():
    torch.manual_seed(19)
    native=fixture_vit();ref=copy.deepcopy(native)
    model=NamedModelGraph(RETFoundConfig(),native,retfound_operations(native,RETFoundConfig()),{})
    x=torch.randn(2,3,224,224,requires_grad=True);xr=x.detach().clone().requires_grad_()
    y=model(x);expected=original_formula(ref,xr)
    torch.testing.assert_close(y,expected,rtol=1e-4,atol=1e-6)
    y.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(x.grad,xr.grad,rtol=1e-4,atol=1e-6)
    for (_,p),(_,q) in zip(native.named_parameters(),ref.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)
    encoder={k:v.clone() for k,v in ref.state_dict().items() if not k.startswith(('head.','fc_norm.'))}
    encoder['norm.weight']=torch.ones(32)
    receipt=_load_encoder(native,encoder)
    assert receipt['ignored_pretraining_keys']==['norm.weight']
    before={k:v.clone() for k,v in native.state_dict().items()}
    del encoder['pos_embed']
    with pytest.raises(ValueError,match='Encoder mismatch'): _load_encoder(native,encoder)
    for k,v in native.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_retfound_full_architecture():
    pytest.importorskip('timm')
    from mhd_framework.models.retfound import native_retfound
    model=native_retfound(RETFoundConfig())
    assert len(model.blocks)==24 and model.embed_dim==1024
    assert model.patch_embed.patch_size==(16,16)
    assert model.blocks[0].attn.num_heads==16
    assert model.fc_norm.eps==1e-6


def test_dimensionality_is_not_silently_converted():
    with pytest.raises(ValueError): DenseNetConfig(spatial_dims=3)
    with pytest.raises(ValueError): RETFoundConfig(spatial_dims=3)
