# DenseNet-201 (2D)

## Definition

| Field | Value |
|---|---|
| Architecture ID | `densenet201_2d` |
| Implementation | `torchvision_2d` |
| Framework | V4; each trained artifact pins the exact commit |
| Single-view input | `[B, 3, H, W]` |
| Task output | `[B, num_classes]` logits |
| Configuration metadata | [config.json](config.json) |
| Implementation source | [densenet.py](../../src/mhd_framework/models/densenet.py) |

Stem → Dense block 1 → Transition 1 → Dense block 2 → Transition 2 → Dense block 3 → Transition 3 → Dense block 4 → final normalization/ReLU → pooling → task head.

Dense layers per block: `[6, 12, 48, 32]`. Dense connections are preserved inside each dense block; this implementation exposes dense blocks and transitions, not each internal dense layer.

## Graph endpoints

`stem`, `denseblock1`–`denseblock4`, `transition1`–`transition3`, `norm`, `features`, `logits`.

Use `describe_nodes()` to enumerate constructed nodes. Numeric node IDs are local
to the model configuration. Spatial and token features keep their original form;
cross-model mapping belongs in the consuming project. Node discovery does not
imply that every internal node is a valid standalone continuation boundary.

## Construction and input contract

```python
from mhd_framework.models import create_model

model = create_model({
    "name": "densenet201",
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

Optional torchvision ImageNet initialization must use an explicit weight release. Construction without `weights` starts from random initialization. A target classification head replaces the source head when the number of classes differs.

Pass an explicit torchvision weights enum/string when intentionally loading reference initialization. Complete task models use `load_native_state_dict` or the verified bundle loader. Their task head is retained during strict restoration.

Architecture, initialization, task/labels, preprocessing, training recipe, seed
and framework revision are independently recorded. Matching the architecture name
alone does not establish that trained weights are interchangeable.

## Validation and current scope

CPU native-reference output, input/parameter gradient, update, BN-state and strict replay checks passed for this variant. This variant has no separate GPU acceptance receipt yet. No trained task weights are registered.

Historical numerical receipts are recorded at source revision `0c2bcf9` in
[acceptance](../acceptance/). Synthetic acceptance does not establish downstream
accuracy, an approved training batch or multi-GPU acceptance for every model.

This builder implements 2D DenseNet only. A future 3D DenseNet needs its own architecture definition and acceptance. Three-channel preprocessing is explicit. These are architectural conventions, not fixed clinical tasks.

## References and shared interfaces

[Torchvision DenseNet](https://docs.pytorch.org/vision/stable/models/densenet.html). See [complete-model APIs](../../docs/models.md) for verified bundles,
exact-match training requests and project-owned copies, or return to [Models](../README.md).
