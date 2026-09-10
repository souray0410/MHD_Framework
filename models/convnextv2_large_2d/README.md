# convnextv2_large

GRN convolutional stages. Every residual block and downsampling operation is a graph edge. Task features include global pooling and native head normalization.

## Interface and task boundary

`create_model(config)` returns a complete V4 MHD task graph. For views=1 the input is `[N,3,224,224]`. Multiple views use an explicit leading view dimension and mean features before the task classifier. Research participant/eye aggregation, preprocessing, loss and training live in the separate workflow.

`features` and `logits` are complete graph cuts; named blocks/stages are enumerated by `describe_nodes()`. `forward_until` and `forward_from` use original graph node IDs.

Install `mhd-framework[models_extended]` from the pinned V4 revision. Checkpoint downloads are never implicit. Default weights are random; architecture availability is not official-weight acceptance or a trained model. Source-specific image normalization and any task adaptation belong in the explicit recipe.

ConvNeXt V2: FP32 native output/input/parameter-gradient tolerance is tested separately from strict native Adam parity in float64. Identity-view boundaries in MHD change floating-point gradient accumulation order; near-zero gradient discrepancies can be amplified by Adam. Do not claim bit-identical native FP32 optimization. Production FP32 checkpoint replay is a separate exact acceptance gate.

## Acceptance scope

Tests cover native outputs and gradients, two optimizer updates, strict state loading and feature-cut continuation. Three-dimensional tests use bounded volumes and do not certify full-resolution fit. Every training input/precision/batch class needs complete resource and recovery acceptance before scheduling. No clinical performance is implied.

[Original implementation](https://github.com/facebookresearch/ConvNeXt-V2)
