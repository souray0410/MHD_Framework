import random
import numpy as np
import pytest
import torch
from mhd_framework.utils import _pipeline_examples


def test_pipeline_rejects_unsynchronized_cross_stage_parameter_sharing():
    from mhd_framework.utils import MHD_ParallelConfig, _build_auto_pipeline_stages
    from tests.unit.test_unified import make
    module = torch.nn.Linear(2, 2)
    levels = [[(0, [0], [1])], [(1, [1], [2])],
              [(1, [2], [1])], [(0, [1], [0])]]
    graph = make([module, module], levels, count=3)
    config = MHD_ParallelConfig(pipeline_size=2, pipeline_stages={'e0': 0, 'e1': 1})
    with pytest.raises(ValueError, match='共享参数'):
        _build_auto_pipeline_stages(graph, config, ['n0'], ['n2'], [0, 1], [2, 3])


class Stage(torch.nn.Module):
    def __init__(self, source, destination, module):
        super().__init__()
        self.input_names=(source,)
        self.output_names=(destination,)
        self.module=module

    def forward(self,x):
        return self.module(x)


@pytest.mark.parametrize('precision,dtype',[('fp32',torch.float32),('bf16',torch.bfloat16)])
def test_boundary_metadata_matches_microbatch_autocast_without_consuming_training_state(precision,dtype):
    first=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.BatchNorm1d(3),torch.nn.Dropout(.5))
    modules=[Stage('x','h',first),Stage('h','y',torch.nn.Linear(3,2))]
    values={'x':torch.randn(8,4)}
    buffers={name:value.clone() for name,value in first.named_buffers()}
    rng=torch.get_rng_state().clone()
    py=random.getstate();np_state=np.random.get_state()
    samples=_pipeline_examples(modules,values,4,torch.device('cpu'),precision)
    assert samples[0][0][0].shape==(2,4)
    assert samples[0][1].shape==samples[1][0][0].shape==(2,3)
    assert samples[0][1].dtype==samples[1][0][0].dtype==dtype
    assert samples[1][1].shape==(2,2)
    assert samples[1][0][0].requires_grad
    assert all(output.grad_fn is None for _,output in samples)
    assert not values['x'].requires_grad
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert random.getstate()==py
    np.testing.assert_equal(np.random.get_state(),np_state)
    for name,value in first.named_buffers():
        torch.testing.assert_close(value,buffers[name],rtol=0,atol=0)


def test_invalid_microbatch_shape_rejected_before_module_execution():
    class Forbidden(torch.nn.Module):
        def forward(self,x):pytest.fail('shape must be rejected before execution')
    for inputs in [{'x':torch.zeros(3,2)},{'x':torch.zeros(0,2)},
                   {'x':torch.zeros(4,2),'target':torch.zeros(8)}]:
        with pytest.raises(ValueError,match='batch'):
            _pipeline_examples([Stage('x','y',Forbidden())],inputs,2,torch.device('cpu'),'fp32')


def test_metadata_failure_restores_buffers_and_rng():
    class Fail(torch.nn.Module):
        def forward(self,x):
            random.random();np.random.rand();torch.rand(1)
            raise ValueError('example failure')
    bn=torch.nn.BatchNorm1d(2)
    modules=[Stage('x','h',bn),Stage('h','y',Fail())]
    values={'x':torch.randn(4,2)}
    buffers={name:value.clone() for name,value in bn.named_buffers()}
    rng=torch.get_rng_state().clone();py=random.getstate();np_state=np.random.get_state()
    with pytest.raises(ValueError,match='example failure'):
        _pipeline_examples(modules,values,2,torch.device('cpu'),'fp32')
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert random.getstate()==py
    np.testing.assert_equal(np.random.get_state(),np_state)
    for name,value in bn.named_buffers():
        torch.testing.assert_close(value,buffers[name],rtol=0,atol=0)
