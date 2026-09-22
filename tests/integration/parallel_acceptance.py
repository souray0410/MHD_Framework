"""Explicit CUDA acceptance; run with torchrun --nproc_per_node=2.

Uses only synthetic inputs. Checks native global-batch gradients/updates and a
completed-step checkpoint, including pending rank-local gradients outside PP.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from mhd_framework.utils import (
    MHD_DistributedContext, MHD_Monitor, MHD_ParallelConfig, MHD_Trainer, MHD_Inferencer,
    initialize_mhd_distributed, destroy_mhd_distributed,
)
from tests.unit.test_unified import make


class SquaredLoss(torch.nn.Module):
    def forward(self, value):
        return value.float().square().mean()


LEVELS = [[(0, [0], [1])], [(1, [1], [2])], [(2, [2], [3])],
          [(2, [3], [2])], [(1, [2], [1])], [(0, [1], [0])]]


def materialize(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def run(args):
    context = initialize_mhd_distributed()
    assert context.world_size == 2 and context.device.type == 'cuda'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    reference = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2)).to(context.device)
    initial = copy.deepcopy(reference.state_dict())
    generator = torch.Generator().manual_seed(891)
    batches = [torch.randn(8, 4, generator=generator).to(context.device) for _ in range(6)]
    pp = args.mode in {'gpipe', '1f1b'}
    accum = 1 if pp else 3
    options = ({'data_parallel': args.mode} if args.mode in {'ddp', 'fsdp2'} else
               {'tensor_parallel_size': 2, 'tensor_parallel_plan': {'e0:0': 'colwise', 'e1:0': 'rowwise'}}
               if args.mode == 'tp' else
               {'pipeline_size': 2, 'pipeline_stages': {'e0': 0, 'e1': 1, 'e2': 1},
                'pipeline_microbatches': 4, 'pipeline_schedule': args.mode})
    config = MHD_ParallelConfig(**options)
    def build_graph(batch_size,device):
        modules = [torch.nn.Linear(4, 8), torch.nn.Linear(8, 2)]
        for i, module in enumerate(modules):
            module.load_state_dict({key.split('.', 1)[1]: value for key, value in initial.items() if key.startswith(f'{i}.')})
        graph = make([*modules, SquaredLoss()], LEVELS, count=4,
                     initial={0: torch.zeros(batch_size, 4)}, device=device)
        return graph
    def create():
        graph=build_graph(8,context.device)
        modules=[graph.get_edge_by_id(i).edge_operations[0].function for i in range(2)]
        trainer = MHD_Trainer(graph, lambda params: torch.optim.SGD(params, lr=.01, momentum=.9),
                              MHD_Monitor(['n3']), [0, 1, 2], [3, 4, 5],
                              criteria=lambda g: g.get_node_by_id(3).feature_message.current_state,
                              save_dir=args.output, input_nodes=['n0'], output_nodes=['n3'],
                              parallel=config, distributed_context=context, precision=args.precision,
                              grad_accum_steps=accum, monitor_interval_steps=100)
        return trainer, modules
    trainer, modules = create()
    reference_opt = torch.optim.SGD(reference.parameters(), lr=.01, momentum=.9)
    dtype = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    tolerances = dict(rtol=1e-5, atol=1e-6) if dtype is None else dict(rtol=2e-2, atol=2e-3)

    def compare_native():
        local_ids = {id(p) for p in trainer.model.parameters()}
        for index, module in enumerate(modules):
            for name, parameter in module.named_parameters():
                if id(parameter) not in local_ids:
                    continue
                expected = dict(reference[index].named_parameters())[name]
                torch.testing.assert_close(materialize(parameter), expected, **tolerances)
                assert parameter.grad is not None and expected.grad is not None
                torch.testing.assert_close(materialize(parameter.grad), expected.grad, **tolerances)

    def advance_native(index):
        if index % accum == 0:
            reference_opt.zero_grad(set_to_none=True)
        inputs = batches[index].detach().clone().requires_grad_()
        with torch.autocast('cuda', enabled=dtype is not None, dtype=dtype):
            loss = reference(inputs).float().square().mean()
        (loss / accum).backward()
        if (index + 1) % accum == 0:
            reference_opt.step()
        return inputs.grad

    def advance(trainer, index):
        inputs = batches[index].chunk(2)[context.rank] if args.mode in {'ddp', 'fsdp2'} else batches[index]
        return trainer.train_step({'n0': inputs.detach().clone().requires_grad_()})

    for index in range(args.save_after):
        input_gradient = advance_native(index); advance(trainer, index)
        if pp and context.rank == 0:
            torch.testing.assert_close(trainer.mhd_graph.get_node_by_id(0).gradient_message.current_state,
                                       input_gradient, **tolerances)
        if (index + 1) % accum == 0:
            compare_native()
    # DDP pending grads are rank-local before synchronization; compare after the
    # complete accumulation window, then independently replay the pending window.
    trainer.save_checkpoint(0, data_cursor={'next_batch': args.save_after, 'mode': args.mode})
    for index in range(args.save_after, 6):
        input_gradient = advance_native(index); advance(trainer, index)
        if pp and context.rank == 0:
            torch.testing.assert_close(trainer.mhd_graph.get_node_by_id(0).gradient_message.current_state,
                                       input_gradient, **tolerances)
        if (index + 1) % accum == 0:
            compare_native()
    expected_parameters = {name: materialize(p).detach().clone() for name, p in trainer.model.named_parameters()}
    trainer._save_distributed_checkpoint('best',1)
    expected_steps = trainer._optimizer_steps
    restored, _ = create()
    assert restored.load_checkpoint(epoch=0) == 0
    assert restored.data_cursor == {'next_batch': args.save_after, 'mode': args.mode}
    assert restored._accumulation_step == args.save_after % accum
    for index in range(args.save_after, 6):
        advance(restored, index)
    assert restored._optimizer_steps == expected_steps == 6 // accum
    for name, parameter in restored.model.named_parameters():
        torch.testing.assert_close(materialize(parameter), expected_parameters[name], **tolerances)
    if pp:
        with torch.no_grad(), torch.autocast('cuda', enabled=dtype is not None, dtype=dtype):
            expected_eval = reference(batches[0]).float().square().mean()
        actual_eval = restored.eval_step({'n0': batches[0]})['n3']
        torch.testing.assert_close(torch.tensor(actual_eval,device=context.device),expected_eval,**tolerances)
        if dtype == torch.float16:
            # Overflow on just one stage must skip the complete logical update.
            before = {name: p.detach().clone() for name,p in restored.model.named_parameters()}
            steps, scale = restored._optimizer_steps, restored.grad_scaler.get_scale()
            hook = next(restored.model.parameters()).register_hook(lambda grad: torch.full_like(grad,float('inf'))) if context.rank == 0 else None
            advance(restored,0)
            if hook is not None:hook.remove()
            assert restored._optimizer_steps == steps
            assert restored.grad_scaler.get_scale() == scale/2
            for name,p in restored.model.named_parameters():
                torch.testing.assert_close(p,before[name],rtol=0,atol=0)
            restored.save_checkpoint(1,data_cursor={'overflow_skipped':True})
            reloaded,_=create();reloaded.load_checkpoint(epoch=1)
            assert reloaded.grad_scaler.state_dict()==restored.grad_scaler.state_dict()
    dist.barrier()
    destroy_mhd_distributed()
    if context.is_main:
        inference=MHD_Inferencer(build_graph,args.output,device='cpu',input_nodes=['n0'],output_nodes=['n2'],levels=[0,1])
        with torch.no_grad():
            expected=reference(batches[0]).cpu()
        torch.testing.assert_close(inference.run({'n0':batches[0].cpu()})['n2'],expected,**tolerances)
        evidence = {'mode': args.mode, 'precision': args.precision, 'steps': 6,
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'device': torch.cuda.get_device_name(), 'world_size': 2,
                    'save_after': args.save_after, 'native_parity': True, 'checkpoint_resume': True, 'single_cpu_inference':True, **tolerances}
        Path(args.output, 'accepted.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ddp', 'fsdp2', 'tp', 'gpipe', '1f1b'], required=True)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    parser.add_argument('--output', required=True)
    parser.add_argument('--save-after', type=int, choices=[1,4], default=4)
    run(parser.parse_args())
