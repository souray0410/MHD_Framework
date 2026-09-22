## Exact current implementation acceptance

Commit `c78cfbdc9ec01e4dc1dc8ee04d04b2e6a8446808` passed CPU 172/34 skips, single Ada 203/3 skips, all 15 two-Ada mode/precision cases and a new-environment wheel installation with both examples. [Exact evidence](../acceptance/20260922/exact_commit_c78cfbd.json). Subsequent changes here only publish evidence and repair the shared status contract; they do not change runtime source. V5 remains a candidate pending consumer, cross-platform and release-download gates.

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

## Exact candidate commit accepted

Commit `a9e6b41cdf646a0775185386f5260e95ee0b9083` passed GitHub CI
35730436334, all 15 two-GPU fixtures, and the full single-GPU suite
(202 passed, 3 unsupported CPU FP16 skips). The built wheel was installed in a
new environment outside the repository and both examples passed.
[Exact evidence](../acceptance/20260922/exact_commit_a9e6b41.json) includes source
archive and package SHA256 plus environment pins. This closes the candidate
framework fixture gates, not production migration or publication.
