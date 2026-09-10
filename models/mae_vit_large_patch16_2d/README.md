# mae_vit_large_patch16

Classification adapter with patch-token mean and fc_norm. The L encoder constructor is shared with RETFound-MAE; pretraining sources remain independent. Random initialization does not reproduce MAE pretraining.

## Interface and task boundary

`create_model(config)` returns a complete V4 MHD task graph. For views=1 the input is `[N,3,224,224]`. Multiple views use an explicit leading view dimension and mean features before the task classifier. Research participant/eye aggregation, preprocessing, loss and training live in the separate workflow.

`features` and `logits` are complete graph cuts; named blocks/stages are enumerated by `describe_nodes()`. `forward_until` and `forward_from` use original graph node IDs.

Install `mhd-framework[models_extended]` from the pinned V4 revision. Checkpoint downloads are never implicit. Default weights are random; architecture availability is not official-weight acceptance or a trained model. Source-specific image normalization and any task adaptation belong in the explicit recipe.

## Acceptance scope

Tests cover native outputs and gradients, two optimizer updates, strict state loading and feature-cut continuation. Three-dimensional tests use bounded volumes and do not certify full-resolution fit. Every training input/precision/batch class needs complete resource and recovery acceptance before scheduling. No clinical performance is implied.

[Original implementation](https://github.com/facebookresearch/mae)
