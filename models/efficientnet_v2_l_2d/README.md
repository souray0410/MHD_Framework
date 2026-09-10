# efficientnet_v2_l

Explicit 2D MHD V4 graph using `torchvision_efficientnet_v2_0_23`. Architecture definitions are independent of datasets, training workflows and initialization checkpoints.

## Interface

`mhd_framework.models.create_model(config)` returns the complete task graph. RGB inputs; `views=1` is `[N,3,H,W]`. For DINOv2 use exactly224x224. OCT B-scans require an explicitly declared grayscale-to-RGB adapter; none of these definitions is a3D OCT encoder.

Configuration defaults describe a randomly initialized two-class task model, not pretrained or accepted research weights. Native checkpoint and initialization provenance are separate. The task head remains independently initialized when loading only an encoder.

## MHD endpoints

`stem`, individual `stageN.blockNN`, stage aliases, `features`, `logits` and explicit downsampling operations. Native stochastic depth, normalization and dropout are retained.

## Validation and reference

Family representative tests compare native outputs, input/parameter gradients, two optimizer updates and strict restoration. Every deployment variant still requires its own full resource and resume acceptance; a family test is not proof of clinical utility or all-size GPU fit.

[Reference](https://docs.pytorch.org/vision/0.23/models/efficientnetv2.html)
