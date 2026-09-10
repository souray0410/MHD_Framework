# Models

Each architecture has its own directory with a README and configuration metadata.
Model implementations remain under `src/mhd_framework/models/`; related variants
share their implementation without duplicating code.

```text
models/
  <architecture_id>/
    README.md
    config.json
  catalog.json
  acceptance/
src/mhd_framework/models/
  resnet.py
  densenet.py
  retfound.py
```

Each README documents structure, dimensions, endpoints, initialization, input and
task conventions, loading, references and the precise scope of validation.
Configuration metadata is machine-readable; detailed explanation belongs in the
README. Model code contains computation, concise API docstrings, necessary
implementation comments and executable validation.

| Model documentation | Configuration metadata |
|---|---|
| [ResNet-18 (2D)](resnet18_2d_torchvision/README.md) | `resnet18_2d_torchvision/config.json` |
| [ResNet-34 (2D)](resnet34_2d_torchvision/README.md) | `resnet34_2d_torchvision/config.json` |
| [ResNet-50 (2D)](resnet50_2d_torchvision/README.md) | `resnet50_2d_torchvision/config.json` |
| [ResNet-101 (2D)](resnet101_2d_torchvision/README.md) | `resnet101_2d_torchvision/config.json` |
| [ResNet-152 (2D)](resnet152_2d_torchvision/README.md) | `resnet152_2d_torchvision/config.json` |
| [ResNet-18 (3D inflation adaptation)](resnet18_3d_inflated/README.md) | `resnet18_3d_inflated/config.json` |
| [ResNet-34 (3D inflation adaptation)](resnet34_3d_inflated/README.md) | `resnet34_3d_inflated/config.json` |
| [ResNet-50 (3D inflation adaptation)](resnet50_3d_inflated/README.md) | `resnet50_3d_inflated/config.json` |
| [ResNet-101 (3D inflation adaptation)](resnet101_3d_inflated/README.md) | `resnet101_3d_inflated/config.json` |
| [ResNet-152 (3D inflation adaptation)](resnet152_3d_inflated/README.md) | `resnet152_3d_inflated/config.json` |
| [DenseNet-121 (2D)](densenet121_2d/README.md) | `densenet121_2d/config.json` |
| [DenseNet-161 (2D)](densenet161_2d/README.md) | `densenet161_2d/config.json` |
| [DenseNet-169 (2D)](densenet169_2d/README.md) | `densenet169_2d/config.json` |
| [DenseNet-201 (2D)](densenet201_2d/README.md) | `densenet201_2d/config.json` |
| [RETFound-MAE ViT-L/16 (2D)](retfound_mae_vit_large_patch16_2d/README.md) | `retfound_mae_vit_large_patch16_2d/config.json` |

No trained artifacts are registered yet. Architecture implementation, numerical
acceptance, official weight acceptance and downstream training are distinct states.

See [shared APIs](../docs/models.md) and [bundle schema](artifact.schema.json).
Validate the catalog with `python models/validate.py`; identity checks use
`python -m unittest discover -s models`. Neither command reads research data.

See the [model companion and reproducibility contract](REPRODUCIBILITY.md) for data-processing source, training recipes, environments, weights and replay requirements.
