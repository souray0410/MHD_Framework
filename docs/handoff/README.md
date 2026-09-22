# V5 candidate handoff — 2026-09-22

[status.json](status.json) separates implementation, isolated validation and
production acceptance. Message is Tensor. The candidate repairs incomplete
epoch accumulation, complete train-step checkpoint/recovery, native PP AMP
metadata and scaling, and canonical distributed-checkpoint inference. Optional
architectures are ported from the actual V4 consumer revision.

CPU baseline: 171 passed, 34 explicitly skipped. Isolated single Ada GPU full
extended model suite: 201 passed, 3 unsupported CPU FP16 skips. Two Ada GPUs:
15 parallel-family/precision fixtures passed, including optimizer state, pending
gradients, continued updates and consolidated CPU inference. GPU source manifests
and [receipts](../acceptance/20260922/gpu_receipts.json) identify the snapshots;
subsequent changes still require exact-commit acceptance. NCCL P2P is disabled
in this validation environment because default device peer initialization hangs;
shared-memory transport is recorded rather than hidden.

[English migration](../migration-v5.md) / [中文迁移](../migration-v5.zh-CN.md).
V5 remains a preview. Production artifacts and consumers have not been switched;
Ibex incidents and cross-platform recovery remain rollout gates. V4 is untouched.
Do not replace the tag or announce completion from these synthetic checks alone.
