# vit_large_patch14_dinov2

Explicit 2D MHD V4 graph using `dinov2_timm_0_9_2_fixed224`. Architecture definitions are independent of datasets, training workflows and initialization checkpoints.

## Interface

`mhd_framework.models.create_model(config)` returns the complete task graph. RGB inputs; `views=1` is `[N,3,H,W]`. For DINOv2 use exactly224x224. OCT B-scans require an explicitly declared grayscale-to-RGB adapter; none of these definitions is a3D OCT encoder.

Configuration defaults describe a randomly initialized two-class task model, not pretrained or accepted research weights. Native checkpoint and initialization provenance are separate. The task head remains independently initialized when loading only an encoder.

## MHD endpoints

`patch_embed`, individual `blockNN`, `norm`, `features` (CLS token) and `logits`. LayerScale and native attention/MLP operations are retained. Uses timm0.9.2 explicit attention. The fixed224 task port is distinct from official DINOv2 dynamic-resolution inference; any source518-to224 positional conversion must be explicitly recorded by the training workflow.

## Validation and reference

Family representative tests compare native outputs, input/parameter gradients, two optimizer updates and strict restoration. Every deployment variant still requires its own full resource and resume acceptance; a family test is not proof of clinical utility or all-size GPU fit.

[Reference](https://github.com/facebookresearch/dinov2)
