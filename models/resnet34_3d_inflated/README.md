# ResNet-34 (3D inflation adaptation)

## Definition

| Field | Value |
|---|---|
| Architecture ID | `resnet34_3d_inflated` |
| Implementation | `inflated_3d` |
| Framework | V4; each trained artifact pins the exact commit |
| Single-view input | `[B, 1, D, H, W]` |
| Task output | `[B, num_classes]` logits |
| Configuration metadata | [config.json](config.json) |
| Implementation source | [resnet.py](../../src/mhd_framework/models/resnet.py) |

Stem → residual stage 1 → stage 2 → stage 3 → stage 4 → spatial/view pooling → task head.

Block: `BasicBlock`; blocks per stage: `[3, 4, 6, 3]`; stage output channels: `[64, 128, 256, 512]`. Default graph granularity is `block`; explicit `stage` granularity is also supported.

## Graph endpoints

`stem`, `stage1`–`stage4`, individual residual `main`/`skip`/merge nodes, `features`, `logits`.

Use `describe_nodes()` to enumerate constructed nodes. Numeric node IDs are local
to the model configuration. Spatial and token features keep their original form;
cross-model mapping belongs in the consuming project. Node discovery does not
imply that every internal node is a valid standalone continuation boundary.

## Construction and input contract

```python
from mhd_framework.models import create_model

model = create_model({
    "name": "resnet34",
    "spatial_dims": 3,
    "in_channels": 1,
    "num_classes": 2,
    "views": 1,
    "implementation": "inflated_3d"
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

Optional initialization uses an explicit torchvision ImageNet weight release; `weights=None` is random initialization. The 2D convolution kernels are repeated along depth and divided by kernel depth; RGB stem weights are summed for one input channel. Subsequent computation uses Conv3d and BatchNorm3d. **2D initialization does not make this a 2D network.**

Use an explicit weight release for initialization, or strict native-state/bundle loading for an already trained full task model. A source ImageNet classification head is replaced when class counts differ; a trained task head is preserved by strict full-model loading.

Architecture, initialization, task/labels, preprocessing, training recipe, seed
and framework revision are independently recorded. Matching the architecture name
alone does not establish that trained weights are interchangeable.

## Validation and current scope

Builder implemented; dedicated numerical acceptance for this depth remains pending. Representative 18/50 tests do not certify this variant. No trained task weights are registered.

Historical numerical receipts are recorded at source revision `0c2bcf9` in
[acceptance](../acceptance/). Synthetic acceptance does not establish downstream
accuracy, an approved training batch or multi-GPU acceptance for every model.

Torchvision bottleneck variants use the ResNet V1.5 stride placement. This distinction is part of the implementation identity. The stem has depth stride 1; depth-one max pooling is computed slice-wise with the equivalent 2D operator. This is a named 3D inflation adaptation, not an official OCT-pretrained or video-ResNet checkpoint.

## References and shared interfaces

[Torchvision ResNet](https://docs.pytorch.org/vision/stable/models/resnet.html). See [complete-model APIs](../../docs/models.md) for verified bundles,
exact-match training requests and project-owned copies, or return to [Models](../README.md).
