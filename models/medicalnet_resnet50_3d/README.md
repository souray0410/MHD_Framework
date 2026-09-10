# medicalnet_resnet50_3d

MedicalNet encoder pinned to 20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd, shortcut B. Stages 3/4 preserve stride one and dilation 2/4. The original segmentation decoder is replaced by global pooling and a two-class head; this is not the earlier inflated ResNet. MIT source and provenance are packaged under models/licenses.

## Interface and task boundary

`create_model(config)` returns a complete V4 MHD task graph. For views=1 the input is `[N,1,D,H,W], with valid positive sizes accepted by the native encoder`. Multiple views use an explicit leading view dimension and mean features before the task classifier. Research participant/eye aggregation, preprocessing, loss and training live in the separate workflow.

`features` and `logits` are complete graph cuts; named blocks/stages are enumerated by `describe_nodes()`. `forward_until` and `forward_from` use original graph node IDs.

Install `mhd-framework[models_extended]` from the pinned V4 revision. Checkpoint downloads are never implicit. Default weights are random; architecture availability is not official-weight acceptance or a trained model. Source-specific image normalization and any task adaptation belong in the explicit recipe.

The deterministic pooling adapter preserves 3x3x3 max-pool stride 2/padding 1 and 2x2x2 nonoverlapping average-pool geometry. It uses deterministic 2D max operations or explicit ordered sums. CPU/GPU values and gradients, including tied maxima, are tested against native 3D pooling. Core MHD semantics are unchanged.

## Acceptance scope

Tests cover native outputs and gradients, two optimizer updates, strict state loading and feature-cut continuation. Three-dimensional tests use bounded volumes and do not certify full-resolution fit. Every training input/precision/batch class needs complete resource and recovery acceptance before scheduling. No clinical performance is implied.

[Original implementation](https://github.com/Tencent/MedicalNet)
