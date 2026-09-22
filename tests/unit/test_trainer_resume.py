import copy
import random
import numpy as np
import pytest
import torch
from mhd_framework.utils import MHD_Trainer, MHD_Monitor, MHD_DistributedContext
from tests.unit.test_unified import make, CHAIN


class RandomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4),
                                        torch.nn.Dropout(.3),torch.nn.Linear(4,2))
        self.unused=torch.nn.Parameter(torch.ones(1))

    def forward(self,x):
        return self.layers(x)*(0.5+random.random()+float(np.random.rand()))


def make_trainer(path,precision='fp32',accumulation=3):
    module=RandomModel()
    graph=make([module,lambda x:x.square().mean()],CHAIN)
    optimizer=torch.optim.AdamW(graph.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    trainer=MHD_Trainer(graph,optimizer,MHD_Monitor(['n2']),[0,1],[2,3],
        criteria=lambda g:g.get_node_by_id(2).feature_message.current_state,
        save_dir=str(path),input_nodes=['n0'],output_nodes=['n2'],grad_accum_steps=accumulation,
        precision=precision,lr_scheduler=scheduler,
        distributed_context=MHD_DistributedContext(0,0,1,torch.device('cpu'),'gloo'))
    return trainer,module


def assert_tree(actual,expected):
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:assert_tree(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):assert_tree(a,b)
    else:assert actual==expected


@pytest.mark.parametrize('precision',['fp32','bf16'])
@pytest.mark.parametrize('save_after',[0,1,3,4])
def test_completed_step_resume_restores_pending_gradients_rng_buffers_optimizer_and_cursor(tmp_path,precision,save_after):
    torch.manual_seed(491);random.seed(491);np.random.seed(491)
    batches=[{'n0':torch.randn(4,3)} for _ in range(9)]
    original,module=make_trainer(tmp_path,precision)
    def advance(trainer,items):
        for batch in items:
            before=trainer._optimizer_steps
            trainer.train_step(batch)
            if trainer._optimizer_steps>before:trainer.lr_scheduler.step()
    advance(original,batches[:save_after])
    pending={name:None if p.grad is None else p.grad.clone() for name,p in module.named_parameters()}
    optimizer_before=copy.deepcopy(original.optimizer.state_dict())
    cursor={'epoch':2,'next_batch':save_after,'sampler_seed':491}
    original.save_checkpoint(2,data_cursor=cursor)
    assert_tree(original.optimizer.state_dict(),optimizer_before)
    for name,p in module.named_parameters():assert_tree(p.grad,pending[name])
    advance(original,batches[save_after:])
    expected=copy.deepcopy(module.state_dict())
    expected_opt=copy.deepcopy(original.optimizer.state_dict())
    expected_scheduler=copy.deepcopy(original.lr_scheduler.state_dict())
    expected_random=(random.random(),float(np.random.rand()),torch.rand(4))
    restored,restored_module=make_trainer(tmp_path,precision)
    assert restored.load_checkpoint(epoch=2)==2
    assert restored._accumulation_step==save_after%3
    assert restored.data_cursor==cursor
    for name,p in restored_module.named_parameters():assert_tree(p.grad,pending[name])
    assert restored_module.unused.grad is None
    advance(restored,batches[save_after:])
    assert_tree(restored_module.state_dict(),expected)
    assert_tree(restored.optimizer.state_dict(),expected_opt)
    assert_tree(restored.lr_scheduler.state_dict(),expected_scheduler)
    assert restored._optimizer_steps==original._optimizer_steps==3
    assert restored._micro_step==9 and restored._accumulation_step==0
    assert_tree((random.random(),float(np.random.rand()),torch.rand(4)),expected_random)


def test_partial_or_failed_step_cannot_be_serialized(tmp_path):
    trainer,_=make_trainer(tmp_path)
    trainer._checkpoint_boundary=False
    with pytest.raises(RuntimeError,match='边界'):trainer.save_checkpoint(1)
    assert not (tmp_path/'epoch_1').exists()


def test_checkpoint_without_complete_v5_state_requires_explicit_migration(tmp_path):
    from torch.distributed import checkpoint as dcp
    trainer,_=make_trainer(tmp_path)
    state=trainer._checkpoint_state(1)
    state.pop('runtime');state.pop('format')
    dcp.save(state,checkpoint_id=str(tmp_path/'epoch_1'),no_dist=True)
    with pytest.raises(ValueError,match='迁移工具'):trainer.load_checkpoint(epoch=1)


def test_contract_mismatch_rejected_before_mutating_live_training_state(tmp_path):
    source,_=make_trainer(tmp_path,accumulation=3)
    source.train_step({'n0':torch.randn(4,3)})
    source.save_checkpoint(1)
    target,module=make_trainer(tmp_path,accumulation=2)
    target.train_step({'n0':torch.randn(4,3)})
    parameters=copy.deepcopy(module.state_dict())
    optimizer=copy.deepcopy(target.optimizer.state_dict())
    gradients={name:None if p.grad is None else p.grad.clone() for name,p in module.named_parameters()}
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='配置不匹配'):target.load_checkpoint(epoch=1)
    assert_tree(module.state_dict(),parameters)
    assert_tree(target.optimizer.state_dict(),optimizer)
    for name,p in module.named_parameters():assert_tree(p.grad,gradients[name])
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert target._accumulation_step==1 and target._micro_step==1


def test_load_into_an_existing_pending_window(tmp_path):
    torch.manual_seed(39);random.seed(39);np.random.seed(39)
    source,module=make_trainer(tmp_path)
    batches=[{'n0':torch.randn(4,3)} for _ in range(6)]
    for batch in batches[:3]:source.train_step(batch)
    source.save_checkpoint(1,data_cursor=3)
    for batch in batches[3:]:source.train_step(batch)
    expected=copy.deepcopy(module.state_dict())
    expected_optimizer=copy.deepcopy(source.optimizer.state_dict())
    target,target_module=make_trainer(tmp_path)
    target.train_step(batches[0])
    target.load_checkpoint(epoch=1)
    for batch in batches[3:]:target.train_step(batch)
    assert_tree(target_module.state_dict(),expected)
    assert_tree(target.optimizer.state_dict(),expected_optimizer)


def test_current_checkpoint_inference_uses_selected_weights_and_eval_mode(tmp_path):
    from mhd_framework.utils import MHD_Inferencer
    def build_graph(batch_size,device):
        module=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4),
                                   torch.nn.Dropout(.5),torch.nn.Linear(4,2))
        return make([module,lambda x:x.square().mean()],CHAIN,device=device)
    graph=build_graph(4,'cpu')
    trainer=MHD_Trainer(graph,lambda p:torch.optim.AdamW(p,lr=.001),MHD_Monitor(['n2']),[0,1],[2,3],
                        criteria=lambda g:g.get_node_by_id(2).feature_message.current_state,
                        save_dir=str(tmp_path),input_nodes=['n0'],output_nodes=['n2'],
                        distributed_context=MHD_DistributedContext(0,0,1,torch.device('cpu'),'gloo'))
    values=torch.randn(4,3)
    trainer.train_step({'n0':values})
    trainer._save_distributed_checkpoint('best',1)
    graph.eval()
    expected=graph.get_edge_by_id(0).edge_operations[0].function(values).detach()
    inference=MHD_Inferencer(build_graph,str(tmp_path),device='cpu',input_nodes=['n0'],output_nodes=['n1'],levels=[0])
    torch.testing.assert_close(inference.run({'n0':values})['n1'],expected,rtol=1e-5,atol=1e-6)
    assert not inference.model.training
    assert not any(m.training for m in inference.graph.modules())
