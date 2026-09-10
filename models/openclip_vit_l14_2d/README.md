# openclip_vit_l14

Visual-only OpenAI-style encoder, QuickGELU and native visual projection, followed by a new classifier. No text encoder, contrastive training or zero-shot prediction is implied.

## Interface and task boundary

`create_model(config)` returns a complete V4 MHD task graph. For views=1 the input is `[N,3,224,224]`. Multiple views use an explicit leading view dimension and mean features before the task classifier. Research participant/eye aggregation, preprocessing, loss and training live in the separate workflow.

`features` and `logits` are complete graph cuts; named blocks/stages are enumerated by `describe_nodes()`. `forward_until` and `forward_from` use original graph node IDs.

Install `mhd-framework[models_extended]` from the pinned V4 revision. Checkpoint downloads are never implicit. Default weights are random; architecture availability is not official-weight acceptance or a trained model. Source-specific image normalization and any task adaptation belong in the explicit recipe.

## Acceptance scope

Tests cover native outputs and gradients, two optimizer updates, strict state loading and feature-cut continuation. Three-dimensional tests use bounded volumes and do not certify full-resolution fit. Every training input/precision/batch class needs complete resource and recovery acceptance before scheduling. No clinical performance is implied.

[Original implementation](https://github.com/mlfoundations/open_clip/tree/v2.24.0)
