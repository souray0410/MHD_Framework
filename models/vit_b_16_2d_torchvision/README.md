# vit_b_16

This is a complete 2D classification model, implemented as an MHD V4 graph using
Torchvision0.23.0. Architecture metadata is in config.json; task, dataset, seed,
initialization and optimizer belong to independently versioned training recipes.

Patch grid, class-token assembly, positional embedding, each Transformer block,
normalization, CLS features and classification head are explicit graph nodes.
Tokens use NLC with CLS first; patch_grid uses NCHW. Input is224x224 RGB.
The native Torchvision task head is zero initialized: on the first update only
the head receives a nonzero gradient. Subsequent backbone gradients must be tested.

Random construction is supported. Explicit ImageNet weights are a separate
initialization; they do not turn this graph into an eye-specific foundation model.
No 3D conversion, task-trained weights or training workflow is implied.

[Reference](https://docs.pytorch.org/vision/0.23/models/vision_transformer.html). See [reproduction contract](../REPRODUCIBILITY.md) and
[complete-model interfaces](../../docs/models.md). Numerical architecture acceptance
and completed task training are distinct.
