# Models

`mhd_framework.models` supplies complete task models, graph discovery, and verified
artifacts. Each `models/<architecture_id>/` directory contains its README and architecture
metadata; `models/` also contains the shared catalog and bundle schema. Model
source, trained artifacts and downstream research methods have independent lives.
No import submits a job, reads a dataset or downloads weights.

## Explicit model implementations

```python
from mhd_framework.models import create_model
model = create_model({
    "name": "resnet50", "implementation": "torchvision_2d",
    "spatial_dims": 2, "in_channels": 3, "num_classes": 2,
    "views": 1, "granularity": "block", "schema": "mhd_resnet_task_v1",
})
logits = model(images)
```

The first family supports ResNet18/34/50/101/152. Each architecture remains
independent of its task, initialization, training recipe and dataset. 3D must be
selected explicitly as `implementation="inflated_3d"`, `spatial_dims=3`; this is
one named adaptation, **not** a native medical 3D architecture or a default rule
for converting all 2D networks. Other 3D architectures require separate builders
and acceptance evidence. Segmentation builders are not yet implemented. RETFound-MAE has its own
independent builder described below. Multiple views pool features before the task head;
this policy is part of the model configuration.

Torchvision's bottleneck stride placement follows ResNet V1.5, which differs
from the original paper's placement. Explicit weight releases must match that
implementation. The 3D adaptation repeats 2D kernels along depth with division
by kernel depth, uses a depth-preserving stem stride/pool and sums RGB stem
weights for one-channel input. Its depth-one pool is executed as independent
2D slices, avoiding nondeterministic CUDA MaxPool3d backward. None of this is
presented as official pretrained OCT weights.

## Graph structure and endpoints

`granularity="block"` is the default: stem, each residual block's main branch,
shortcut and merge, spatial/view pooling, and task head are explicit graph nodes.
`granularity="stage"` is an explicit coarser alternative. Both must preserve
outputs and gradients. All **constructed** nodes are returned by
`model.describe_nodes()`; convolution and normalization inside a main-branch
hyperedge are not falsely advertised as separate graph nodes.

Stable aliases `stem`, `stage1`–`stage4`, `features` and `logits` point to real
nodes. Internal numeric IDs are local to the configuration; projects must remap
IDs when combining graphs. Runtime shape metadata is marked as unavailable until
that node has actually executed. Node access is not the same as a valid graph
cut: resuming from a residual main branch alone is rejected, because the shortcut
input is missing. Stale values from a previous forward are never substituted.

```python
features = model.forward_until("stage3", images)
# A project may apply its own explicitly validated shape-compatible operation.
logits = model.forward_from("stage3", features)
```

These segment calls retain PyTorch autograd. They do not accumulate separate
MHD execution traces into one trace: a project using explicit `graph.backward`
over a combined topology must construct that topology and its complete paths.
Ordinary `loss.backward()` traverses the composed differentiable operations.
Models can also contribute their nodes, edges and endpoint contracts to a
project-owned combined MHD graph; the project owns cross-model dimension mapping.

## Complete, immutable model bundles

A bundle contains:

- `selected.pt`: complete task-model state, including the task head, and explicit
  construction configuration; no pickled Python model instance.
- `resume.pt`: the stopping model, optimizer, scheduler, epoch, RNG and sampler
  state. It must not pair an earlier selected model with a later optimizer.
- `manifest.json`: exact request, data/preprocessing/label definitions and hashes,
  initialization, framework commit and implementation hashes, acceptance receipts,
  endpoints, and checksums of both checkpoints.

`request_id` hashes the complete model/framework/data/initialization/training
request. `artifact_id` is `model_` plus the canonical manifest hash excluding its
own artifact ID. A run's acceptance receipt distinguishes repeated realizations.
There were no trained artifacts under the earlier draft `weights_` schema; the
complete-bundle schema replaces that draft, rather than introducing a second
runtime registry. File integrity and structure checks are not evidence that a
scientific training or replay criterion was satisfied; receipts come from the
study's independently audited acceptance procedure.

`resolve_or_request` returns an exact accepted match or writes a pending request.
`ensure_trained(..., trainer=callback)` can execute an explicitly supplied
approved training callback; concurrent workers serialize the same request.
The callback owns data, training, checkpointing and scientific acceptance. Failed
attempts are retained. A cache miss never returns random weights as a trained
model. `materialize_bundle` creates an independent verified project copy;
`load_bundle` checks configuration, hashes, installed API and implementation bytes
before strict state loading. The generic code does not select a best run by cache
order or silently change hyperparameters. Stores currently use POSIX file locks.

Public source and architecture metadata can live in Git. Trained weights and
participant-derived artifacts require their own access/licensing review and
verified durable storage. A server cache can be retired only after another
verified recoverable copy exists. The loader currently accepts verified local
bundles; automatic remote download is a storage-layer responsibility, not an
implicit network action on model import.

## Validation

Run `pip install -e '.[dev,models]'`, `python scripts/manage.py test`, and
`python -m unittest discover -s models`. Optional models tests require torchvision.
For explicit distributed acceptance:

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 tests/integration/models_ddp.py
```

CPU and CUDA, actual model names, dtype, input size, BN policy and world size must
be reported separately. Synthetic acceptance does not certify a real training
batch, large-volume memory budget, pretrained conversion, or downstream benefit.

## Family-specific structure and initialization

| Family | Implementation | Exposed structural boundaries | Pretraining source |
|---|---|---|---|
| ResNet18/34/50/101/152 | torchvision 2D | stem, residual main/shortcut/merge, stages, features, head | explicit ImageNet release |
| ResNet18/34/50/101/152 | explicitly inflated 3D | actual 3D operators and residual structure | 2D kernel inflation is initialization, not a claim of native OCT pretraining |
| DenseNet121/161/169/201 | torchvision 2D | stem, four dense blocks, three transitions, final norm, features, head | explicit ImageNet release |
| RETFound-MAE ViT-L/16 | 2D, timm 0.9.2 adaptation | patch embedding, CLS/position addition, 24 individual transformer blocks, token pooling/norm, head | CFP and OCT B-scan checkpoints are distinct initializations |

DenseNet dense connections remain inside their published dense blocks; individual
dense layers are not separately exposed by this implementation. RETFound is a
24-block transformer, not a four-stage convolutional model. A project must declare
its choice of block(s); the toolbox does not invent four paper-defined ViT stages.
`describe_nodes()` lists the actual graph structure.

RETFound-MAE's architecture is one shared 2D definition; CFP versus OCT describes
its pretraining source, not a change into a 3D volume encoder. Other RETFound
releases/backbones must get their own architecture identities. OCT B-scans require
explicit RGB preprocessing. This release does not implement a 3D DenseNet or a
3D RETFound. Processing slices independently and aggregating them is a separate
task adaptation and cannot be labelled native 3D RETFound.

```python
from mhd_framework.models import create_model
model = create_model({"name": "densenet121", "num_classes": 2})
retinal_model = create_model({"name": "retfound_mae_vit_large_patch16", "num_classes": 2})
# No checkpoint is downloaded or loaded by these calls.
```

For RETFound initialization, `load_pretrained_encoder(path, sha256=..., modality=
"cfp" or "oct_bscan", source=...)` validates every encoder tensor before writing.
It returns the initialization receipt for the training request. Decoder tensors
and pretraining normalization/head tensors are reported as unused; classification
head and post-pooling normalization remain new. A mismatched or partial encoder
fails instead of using random replacement tensors. Full trained task checkpoints
use strict `load_native_state_dict` or `load_bundle`, preserving their trained heads.
The same loader never silently interpolates positions or substitutes a modality.

Architecture, input dimension, views/aggregation, initialization, task labels,
preprocessing, training settings, seed and exact framework revision are distinct
parts of a request. Matching the family name alone never permits weight reuse.
This release locks the optional implementation dependencies to torchvision 0.23.0
and timm 0.9.2; receipts also record the installed numerical environment.

References: [DenseNet builders](https://docs.pytorch.org/vision/stable/models/densenet.html),
[RETFound-MAE encoder definition](https://github.com/RViMLab/RETFound_MAE/blob/main/models_vit.py).
The original RETFound environment used timm 0.3.2. This is a declared modern-runtime
adaptation with formula/gradient checks, not a claim of reproducing the original
training environment or already verifying all official pretrained checkpoints.

## Dataset-specific classifier training

`python -m mhd_framework.models.training --job job.json --output attempt_directory
--mode preflight` executes two real distributed updates and checkpoint replay.
`--mode train` runs the explicitly configured classification protocol. Launch it
with `torchrun`; world size, effective batch and per-rank microbatch are recorded
separately. A manifest gives sample IDs, labels, relative NumPy array paths and
SHA256 values. Train and development manifests must be disjoint; this entry point
rejects other data roles. It performs no test evaluation or data discovery.

Model definitions contain the prediction graph. This trainer owns cross-entropy,
AdamW, gradient accumulation, clipping, development selection and plateau rules.
The loss is ordinary unweighted cross-entropy, with no smoothing. All parameters
train; ordinary BN uses per-rank microbatches. Rank-0 buffers define the selected
checkpoint and are broadcast before sharded evaluation. Accumulation does not
make different BN microbatches equivalent. Preflight changes are recorded before
training; training never silently changes the declared batch or protocol.

Each attempt retains request, progress, curves, best full-model state, last full
continuation state including per-rank RNG, development predictions and acceptance
receipts. Only a plateau-terminated, strictly replayed model enters the shared
store. Hitting the maximum epoch count is `needs_attention`. Selected weights and
last optimizer states remain separate. Interruption leaves the attempt intact;
a controller may restart the same initialization/seed in a new attempt. The CLI
does not claim transparent mid-epoch resume. The saved continuation state is
available for explicitly implemented restoration procedures.

Input exports and study recipes belong outside the toolbox. A new task/dataset
creates a new request and README, even when the architecture is identical. No
clinical labels, participant lists or research project logic belong in this package.

## ViT and Swin

`create_model({"name":"vit_b_16"})` and `create_model({"name":"swin_b"})`
construct complete random-initialized task graphs. ViT-B/16 and ViT-L/16, and
Swin-T/S/B are separate named2D implementations using torchvision0.23.0. Their
model directories document block/stage endpoints, tensor layout and initialization.
ViT uses normalized CLS features; RETFound-MAE uses mean non-CLS tokens followed
by normalization. Equal encoder widths do not make those architectures identical.

The optional models package contains architecture code; its core graph API is
unchanged. Projects own token-to-grid conversion, spatial dimensionality mapping,
losses, data selection and training protocols. Input modalities and pretrained
weight identities must be explicit. No import downloads or starts training.
