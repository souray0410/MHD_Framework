# Model zoo and training evidence

MHD Framework's model zoo has two independent parts: reusable neural architecture adapters and immutable records of trained weights. Core framework releases remain independently versioned. A V5 implementation, a larger cohort or a new task adds an artifact; it never replaces a V4/smaller-cohort record.

## Catalog dimensions

Identify each artifact by architecture/adapter revision, spatial dimensions, framework API and exact commit, dataset/cohort/split and exposure ledger, label/task definition, preprocessing/input geometry, initialization source/conversion, training protocol and seed, checkpoint SHA and evidence. `resnet50` or `V4` alone is not a weight identifier.

The initial standard ResNet family is 18/34/50/101/152. Torchvision's bottleneck implementation is the V1.5 stride-placement variant; record that reference explicitly. A volumetric inflation adaptation has its own architecture ID, depth/stride specification and channel conversion. It does not inherit certification from a 2D model. Later DenseNet, U-Net and transformer families enter through the same acceptance process.

The catalog may index V4 and V5 artifacts simultaneously. This does not put multiple framework implementations into one installed package. Users install the pinned release required by an artifact. Cross-release compatibility is supported only by explicit output, gradient and checkpoint evidence; loading without an exception is insufficient. Existing V4/V5 tags remain immutable.

## Acceptance lifecycle

Architecture: planned -> implemented -> reference-equivalent on a declared device/precision -> accepted feature interface.

Training run: prepared -> training/paused -> converged -> strict reload and provenance accepted -> reusable. Failures and interrupted attempts remain visible; a checkpoint's existence does not establish convergence. Test evaluation is an independent recorded event, not a prerequisite to fitting a model or a selector of the best checkpoint.

Reference comparisons use equal weights, inputs, batches and losses, and cover meaningful intermediate blocks, skip paths, logits, gradients, multiple updates and state restoration. Native reference modules may remain PyTorch operations within MHD edges; wrapping an entire model in a single edge does not demonstrate useful internal intervention sites. Report numerical tolerances and precision explicitly.

A study may initialize accepted adapters with external pretrained weights and fine-tune on its own data. This is architecture reproduction and dataset-specific training, not reproduction of the original paper's benchmark score. Random initialization, generic pretraining, domain pretraining and dataset-specific fine-tuning remain distinguishable. Classification labels cannot establish segmentation training performance for a U-Net; segmentation requires the corresponding target annotations.

## Evidence and artifact storage

Each record includes its model card, protocol, source/environment locks, initialization digest, input/label/split manifests, acceptance checks, all attempts and logs, selected and stopping/resume checkpoints, optimizer/scheduler/scaler/RNG/sampler/BN state, learning curves, metrics, hardware/memory/throughput measurements and feature-interface contract. Reference conversion code is retained alongside the evidence. Incomplete checkpoints are never advertised as reusable. Retention is explicit; no automatic deletion of prior accepted weights or evidence.

Public source and aggregate model cards are separate from restricted artifacts. Data-derived weights are not automatically redistributable; access/licensing is recorded per artifact. Keep participant predictions, features and identifiers in authorized storage. Consumer projects take verified project-owned copies and retain the source artifact ID; subsequent training creates a new artifact rather than mutating the source.

Framework source, architecture adapter revision, data version and trained artifact version are separate axes. Keep generic registry/adapters and tests in this toolbox; institution-specific dataset loaders, clinical cohort decisions and cluster scheduling stay in external study code. The registry format is portable across institutions and datasets.

## Current state

The initial [catalog](../benchmarks/model_zoo/catalog.json) is a design registry, not a completed model collection. Broad ResNet/3D adapters, native dataset training and a priority dispatcher still need implementation and acceptance. The framework's existing basic model and distributed tests do not constitute that full collection.

Reference: [Torchvision ResNet definitions](https://docs.pytorch.org/vision/stable/models/resnet.html).

## Resolution and migration contract

Callers select an `architecture_id` independently from a `weights_id`. A human-readable alias may resolve to an immutable artifact ID for exploration, but a launched training run must save the resolved ID, checkpoint SHA, framework commit, adapter commit and preprocessing/task configuration. Never resume a run through a floating `latest` alias. The intended convenience API must fail clearly when a builder, required release, checkpoint or acceptance is unavailable; no silent fallback to random weights or another input dimension. The registry design does not yet implement this loading API.

Use semantic feature endpoints (`stem`, `stage1`... or named transformer blocks) mapped to exact original MHD Node IDs. Each endpoint declares channels, batch/eye axes, grid or token order, class-token handling, spatial stride and supported inputs. Classifier identity, prediction unit, label order and pooling also form part of the contract. Backbone-only transfer with a new head is a new initialization/transfer event, never a strict full-model reproduction. Consumers keep bridge-specific code outside the native adapter.

Separate two migration operations: (1) replay/convert the same trained model under a new framework/adapter while preserving input, task and state, and (2) train/fine-tune on a new dataset or task. The first requires a key-conversion map, complete reference receipts, source and destination SHAs and numerical tolerances; the second creates a new training artifact with a parent link. A successful load alone certifies neither. A change of preprocessing or label definition may invalidate transfer-as-baseline comparability even when tensor dimensions fit. Never overwrite the parent checkpoint or move a published release tag.

Retain compact machine-readable `model_card.json`, `environment.lock.json`, `framework.lock.json`, `preprocessing.json`, `task.json`, `feature_interface.json`, `acceptance.json` and an artifact manifest with checkpoint digests alongside logs, predictions and resume files in authorized storage. Public metadata contains no participant identifiers. The same artifact resolver must work with explicit deployment roots on ws02, Ibex or another host; source code never embeds the original absolute training path.
