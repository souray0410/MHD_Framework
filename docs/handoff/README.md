## Formal V5 package release — 2026-09-23

The annotated `V5` tag peels to commit `1287681c08846e11364c81653048435482e772a7`; GitHub release [V5](https://github.com/souray0410/MHD_Framework/releases/tag/V5) is published as a non-preview release. The original preview tag and assets were archived before replacement; `V4` is unchanged. The wheel and sdist were downloaded back from GitHub, checked against the published SHA256SUMS, installed in separate clean environments outside the checkout, and each ran `basic.py` and `model.py`. The exact release commit passed PyTorch 2.8 CPU 172/34 skips and GitHub CI 35828618583. No runtime or package metadata changed from the GPU-validated implementation below. Application asset conversion, full-data training, Ibex incidents and cross-platform resume retain separate gates.

## Exact implementation acceptance

Commit `c78cfbdc9ec01e4dc1dc8ee04d04b2e6a8446808` passed CPU 172/34 skips, single Ada 203/3 skips, all 15 two-Ada mode/precision cases and a new-environment wheel installation with both examples. [Exact evidence](../acceptance/20260922/exact_commit_c78cfbd.json). Subsequent changes only update documentation and shared status, not runtime source. Formal package publication is separate from consumer migration, cross-platform continued training and medical acceptance.

# V5 implementation history — 2026-09-22

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
The original V5 preview was archived before the formal release update. Production
artifacts and consumers still need their own cutover evidence; Ibex incidents and
cross-platform recovery remain project rollout gates. V4 is untouched. Synthetic
framework checks alone do not certify medical application results.

## Exact candidate commit accepted

Commit `a9e6b41cdf646a0775185386f5260e95ee0b9083` passed GitHub CI
35730436334, all 15 two-GPU fixtures, and the full single-GPU suite
(202 passed, 3 unsupported CPU FP16 skips). The built wheel was installed in a
new environment outside the repository and both examples passed.
[Exact evidence](../acceptance/20260922/exact_commit_a9e6b41.json) includes source
archive and package SHA256 plus environment pins. This closes the candidate
framework fixture gates, not production migration or publication.
