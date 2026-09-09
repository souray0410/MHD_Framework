# RETFound-MAE ViT-L/16 (2D)

## Definition

| Field | Value |
|---|---|
| Architecture ID | `retfound_mae_vit_large_patch16_2d` |
| Implementation | `retfound_mae_timm_0_9_2` |
| Framework | V4; each trained artifact pins the exact commit |
| Single-view input | `[B, 3, 224, 224]` |
| Task output | `[B, num_classes]` logits |
| Configuration metadata | [config.json](config.json) |
| Implementation source | [retfound.py](../../src/mhd_framework/models/retfound.py) |

Patch embedding → CLS token and position embedding → 24 Transformer blocks → non-CLS token mean → LayerNorm → task head.

Hidden width 1024, 16 attention heads, patch size 16, 24 blocks; post-pooling LayerNorm uses epsilon 1e−6. Each Transformer block is exposed. This encoder does not define four convolutional stages.

## Graph endpoints

`patch_embed`, `tokens`, `block01`–`block24`, `features`, `logits`.

Use `describe_nodes()` to enumerate constructed nodes. Numeric node IDs are local
to the model configuration. Spatial and token features keep their original form;
cross-model mapping belongs in the consuming project. Node discovery does not
imply that every internal node is a valid standalone continuation boundary.

## Construction and input contract

```python
from mhd_framework.models import create_model

model = create_model({
    "name": "retfound_mae_vit_large_patch16",
    "spatial_dims": 2,
    "in_channels": 3,
    "num_classes": 2,
    "views": 1
})
```

The example defines an untrained two-class model, not a dataset or approved
training recipe. `config.json` is architecture metadata; it is not itself a
complete training request to pass to `create_model`.

For `views > 1`, input is `[B, V, C, ...]` and `V` must equal the declared number
of views. Each view is encoded independently; features are averaged before the
task head. This is an explicit task adaptation. Image normalization, crop rules,
spatial sampling and label definitions are recorded in the training request.

## Initialization and checkpoint loading

CFP and OCT B-scan pretrained checkpoints are separate initialization sources for this same **2D** architecture. Neither is a native 3D volume model. The task manifest records `modality`, source and actual checkpoint SHA256. No weights are bundled or downloaded by construction.

Call `load_pretrained_encoder(path, sha256=..., modality="cfp" or "oct_bscan", source=...)` explicitly. Every encoder tensor must match; pretraining decoder tensors and old norm/head tensors are listed as unused. Task head and post-pooling normalization stay newly initialized. There is no implicit positional interpolation. Use strict loading of complete task models after fine-tuning.

Architecture, initialization, task/labels, preprocessing, training recipe, seed
and framework revision are independently recorded. Matching the architecture name
alone does not establish that trained weights are interchangeable.

## Validation and current scope

The full 24-block model passed synthetic single-GPU forward/gradient/update and strict checkpoint replay at batch 1, 224×224. A smaller Transformer additionally checked explicit attention and pooling formulas. Official pretrained checkpoints and downstream training have not yet been accepted.

Historical numerical receipts are recorded at source revision `0c2bcf9` in
[acceptance](../acceptance/). Synthetic acceptance does not establish downstream
accuracy, an approved training batch or multi-GPU acceptance for every model.

This is the RETFound-MAE encoder adaptation using timm 0.9.2. The original code used timm 0.3.2. Other RETFound releases need independent definitions. RGB conversion of OCT B-scans belongs in explicit preprocessing. Slice aggregation is a separate task adaptation, not native 3D RETFound.

## References and shared interfaces

[RETFound-MAE encoder](https://github.com/RViMLab/RETFound_MAE/blob/main/models_vit.py). See [complete-model APIs](../../docs/models.md) for verified bundles,
exact-match training requests and project-owned copies, or return to [Models](../README.md).
