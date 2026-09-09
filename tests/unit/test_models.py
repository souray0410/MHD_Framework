import copy
import json
import os
import pytest
import torch
pytest.importorskip("torchvision")
from mhd_framework.models import MHDResNet, ResNetConfig
from mhd_framework.models.artifacts import (digest, export_bundle, load_bundle,
    materialize_bundle, publish_bundle, resolve_or_request, verify_bundle, runtime_source_sha256)

torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
DEVICE = os.environ.get('MHD_MODELS_TEST_DEVICE', 'cpu')

def reference_forward(model, x):
    batch, views = x.shape[:2]
    x = model.maxpool(model.relu(model.bn1(model.conv1(x.flatten(0,1)))))
    for layer in (model.layer1,model.layer2,model.layer3,model.layer4):
        x=layer(x)
    x=model.avgpool(x).flatten(1).reshape(batch,views,-1).mean(1)
    return model.fc(x)

@pytest.mark.parametrize('name,dims,channels', [('resnet18',2,3),('resnet50',2,3),('resnet18',3,1),('resnet50',3,1)])
def test_complete_model_matches_native_outputs_gradients_bn_updates(name,dims,channels):
    torch.manual_seed(13)
    model = MHDResNet(ResNetConfig(name=name, spatial_dims=dims, in_channels=channels, views=2, implementation='inflated_3d' if dims == 3 else 'torchvision_2d'),device=DEVICE)
    reference = copy.deepcopy(model._native_reference).to(DEVICE)
    actual_opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=.9)
    expected_opt = torch.optim.SGD(reference.parameters(), lr=0.0001, momentum=.9)
    shape = (2, 2, channels) + ((8,16,16) if dims == 3 else (32,32))
    target = torch.tensor([0,1],device=DEVICE)
    for _ in range(3):
        actual_opt.zero_grad(); expected_opt.zero_grad()
        x = torch.randn(shape,device=DEVICE,requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        actual = model(x)
        expected = reference_forward(reference,x_ref)
        torch.testing.assert_close(actual, expected, rtol=2e-4,atol=2e-5)
        torch.nn.functional.cross_entropy(actual,target).backward()
        torch.nn.functional.cross_entropy(expected,target).backward()
        torch.testing.assert_close(x.grad,x_ref.grad,rtol=5e-4,atol=5e-5)
        for (n,p),(rn,rp) in zip(model._native_reference.named_parameters(),reference.named_parameters()):
            assert n == rn
            torch.testing.assert_close(p.grad,rp.grad,rtol=5e-4,atol=5e-5)
        actual_opt.step(); expected_opt.step()
        for key,value in model.native_state_dict().items():
            torch.testing.assert_close(value,reference.state_dict()[key],rtol=5e-4,atol=5e-5)
    model.eval(); reference.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x.detach()),reference_forward(reference,x_ref),rtol=5e-4,atol=5e-5)
    assert len(tuple(model.parameters())) == len(tuple(reference.parameters()))


def make_request(model):
    return {'model':model.configuration(),'framework':{'api':__import__('mhd_framework').__api_version__,'commit':'a'*40,'source_sha256':runtime_source_sha256()},
        'data':{'split_sha256':'b'*64,'preprocessing':{'input':'already_preprocessed'},'preprocessing_sha256':digest({'input':'already_preprocessed'}),
        'label_schema':{'0':'negative','1':'positive'},'label_schema_sha256':digest({'0':'negative','1':'positive'})},
        'initialization':{'source':'synthetic_fixture'},'training':{'seed':13,'protocol':'test_only'}}


def test_full_artifact_lifecycle_and_fail_closed(tmp_path):
    model = MHDResNet(ResNetConfig(name='resnet18')).eval()
    request = make_request(model)
    store = tmp_path/'models'
    missing = resolve_or_request(store,request)
    assert missing['status']=='pending'
    assert resolve_or_request(store,request)==missing
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    bundle = store/'artifacts'/'run1'
    export_bundle(model,bundle,request,acceptance={'status':'accepted','training_receipt_sha256':'e'*64,
        'replay_receipt_sha256':'f'*64},resume_state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':{},'epoch':0,'rng':{'torch_cpu':torch.get_rng_state()},'sampler':{'epoch':0}})
    publish_bundle(store,bundle,request)
    ready=resolve_or_request(store,request)
    assert ready['status']=='ready'
    copied=materialize_bundle(bundle,tmp_path/'project/artifacts/native',request)
    restored,_=load_bundle(copied,request)
    restored.eval()
    x=torch.randn(2,3,32,32)
    with torch.no_grad(): torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
    assert not copied.is_symlink()
    changed=copy.deepcopy(request);changed['data']['split_sha256']='0'*64
    with pytest.raises(ValueError): load_bundle(copied,changed)
    assert resolve_or_request(store,changed)['status']=='pending'
    with (copied/'selected.pt').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError): verify_bundle(copied)
    verify_bundle(bundle)


def test_endpoint_handoff_preserves_outputs_and_input_gradients():
    model=MHDResNet(ResNetConfig(name='resnet18')).eval()
    x=torch.randn(2,3,32,32,requires_grad=True)
    original=model(x)
    expected=torch.autograd.grad(original.sum(),x)[0]
    features=model.forward_until('stage3',x)
    actual=model.forward_from('stage3',features)
    torch.testing.assert_close(actual,original,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),x)[0],expected,rtol=0,atol=0)
    assert model.endpoint_nodes['stage3'] in {n['id'] for n in model.describe_nodes()}
    assert any(n['name']=='stage1.block0.skip' for n in model.describe_nodes())
    with pytest.raises(ValueError): model.forward_from('stage1.block0.main', features)


def test_grid_identity_and_exact_misses(tmp_path):
    from mhd_framework.models.recipes import expand_grid, prepare_grid
    model=MHDResNet(ResNetConfig(name='resnet18'))
    base=make_request(model);base['training']['backbone_lr']=3e-5
    grid={'training.seed':[1,2,2], 'training.backbone_lr':[1e-5,3e-5]}
    assert len(list(expand_grid(base,grid)))==4
    result=prepare_grid(tmp_path,base,grid)
    assert len({r['request_id'] for r in result})==4
    assert all(r['status']=='pending' for r in result)
    with pytest.raises(ValueError): list(expand_grid(base,{'training.lrr':[1]}))


def test_slice_pool_is_the_same_depth_one_operator():
    from mhd_framework.models.resnet import SliceMaxPool3d
    x=torch.randn(2,4,5,12,12,requires_grad=True)
    expected=torch.nn.functional.max_pool3d(x,(1,3,3),(1,2,2),(0,1,1))
    actual=SliceMaxPool3d()(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    g=torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual,x,g)[0],torch.autograd.grad(expected,x,g)[0],rtol=0,atol=0)
