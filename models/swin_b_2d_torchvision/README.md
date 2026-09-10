# swin_b

This is a complete 2D classification model, implemented as an MHD V4 graph using
Torchvision0.23.0. Architecture metadata is in config.json; task, dataset, seed,
initialization and optimizer belong to independently versioned training recipes.

Patch embedding, each published stage/block, three patch merges, normalization,
NHWC-to-NCHW conversion, pooled features and classifier are explicit graph nodes.
Stage/block tensors use NHWC. Stage aliases identify the last block of each stage.
Stochastic depth follows the selected Torchvision variant; its RNG state is part
of exact training recovery. Input preprocessing belongs to the task recipe.

Random construction is supported. Explicit ImageNet weights are a separate
initialization; they do not turn this graph into an eye-specific foundation model.
No 3D conversion, task-trained weights or training workflow is implied.

[Reference](https://docs.pytorch.org/vision/0.23/models/swin_transformer.html). See [reproduction contract](../REPRODUCIBILITY.md) and
[complete-model interfaces](../../docs/models.md). Numerical architecture acceptance
and completed task training are distinct.
