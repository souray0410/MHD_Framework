# V4 to V5

Message remains a Tensor. Imports and the four core concepts are unchanged:
`MHD_Node`, `MHD_Edge`, `MHD_Topo`, `MHD_Graph`. Explicit forward/backward levels
remain the execution interface. This release does not add dictionary states,
dynamic routing, higher-order derivatives or a new optimizer.

## Graph code

A single-producer V4 node using `aggregation="replace"` becomes
`aggregation="sum", memory=False`. For multiple producers, choose the intended
aggregate explicitly; replacing the last producer and summing producers are
different mathematics. `memory` only includes or excludes the previous state.

V5 backward computes the native first-order vector-Jacobian product over the
selected forward occurrences. Only a real, single-element terminal defaults
to cotangent one. For vector or complex terminals, assign the intended Tensor
to `node.gradient_message.initial_state` before forward/backward. Explicit
zero is preserved as a configured input; it is different from leaving the
generated default unset. Do not infer a training objective from mean-value logs.

Custom PyTorch training loops remain supported. Optional `MHD_Trainer` requires
a criteria callable and explicit forward/backward levels. Its accumulation
window closes at each optimizer update, including an uneven epoch tail.

## Step checkpoints

After any successfully completed `train_step`, including a pending accumulation
window, use `trainer.save_checkpoint(epoch, data_cursor=cursor)`. Reconstruct
the same graph, optimizer, scheduler and Trainer configuration, then call
`trainer.load_checkpoint(epoch=epoch)`. The caller consumes
`trainer.data_cursor` to restore its iterator/sampler/augmentation state.
The cursor is supplied explicitly: Trainer cannot infer an external iterator.

The current `mhd_trainer_v5_step_1` contract saves pending gradients, accumulation
position and divisor, optimizer/scheduler/scaler, node values and explicit
cotangent provenance, Python/NumPy/Torch RNG and rank-local CUDA RNG. It preserves
`grad=None`. It never serializes a live autograd graph or an unfinished step.
All ranks participate in distributed save/load. Resume requires the same parallel
layout and precision; changing world size or hardware needs explicit migration
and acceptance. `MHD_Inferencer` can consolidate the canonical saved model onto
one device and sets evaluation mode. Checkpoints are trusted execution artifacts.

There is no V4 or preview Trainer fallback in the current loader. Separate
one-time migration tools must inventory and convert source formats. Missing
optimizer, RNG or data state prevents a claim of exact continuation. Selected
weights are not full training checkpoints. Preserve source artifacts, training
history and checksums; record new serialized hashes even if tensors are unchanged.

## Parallel boundaries

Use one family at a time: DDP, FSDP2, explicit module TP, or PP. PP prepares
communication shape and dtype by executing the configured autocast path on a
microbatch while restoring buffers and RNG. Input batches must be nonempty,
equal in size and divisible by the configured microbatch count. BatchNorm,
cross-sample losses and stochastic layers still require a matched microbatch
reference: equal effective batch does not prove equivalence.

GPipe and 1F1B use `pipeline_loss_fn` to define the scalar objective. Their loss
normalization applies to both parameter and input gradients; FP16 overflow
decisions are synchronized across stages. PP saves only after a full schedule,
requires `grad_accum_steps=1`, and rejects retained graphs or explicit terminal
cotangents. Shared parameters must stay within one stage: cross-stage tied
weights are rejected because native stages do not synchronize their gradients.

## Models and acceptance

Install `.[models]` or `.[models_extended]`. `mhd_framework.models.create_model`
preserves the ported V4 architectures, node identities, feature endpoints and
parameter sharing. Current loaders require accepted V5 artifacts. Retain custom
research loops and dataset recipes in their existing owning projects.

Validate same-input outputs, input/parameter gradients, buffers and consecutive
updates, then strict reload and continued training. FP32 defaults are
`rtol=1e-5, atol=1e-6`; mixed precision uses `rtol=2e-2, atol=2e-3`.
A passing architecture fixture does not accept every production asset or run.
