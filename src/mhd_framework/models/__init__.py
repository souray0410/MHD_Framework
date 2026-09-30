"""Named graph and ResNet adapters included in this public package."""
from .graph import NamedModelGraph
from .resnet import MHDResNet, ResNetConfig


def create_model(config, *, weights=None, device="cpu"):
    """Build an explicit supported ResNet adapter; default weights are random."""
    if not isinstance(config, dict):
        raise TypeError("Model configuration must be a dictionary")
    name = config.get("name", "resnet50")
    if name not in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        raise ValueError("This package includes ResNet adapters only")
    return MHDResNet(ResNetConfig(**config), weights=weights, device=device)


__all__ = ["NamedModelGraph", "MHDResNet", "ResNetConfig", "create_model"]
