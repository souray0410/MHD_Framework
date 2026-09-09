"""Reusable complete task models and immutable, verified model artifacts."""
from .resnet import MHDResNet, ResNetConfig
from .densenet import MHDDenseNet, DenseNetConfig
from .retfound import MHDRETFound, RETFoundConfig


def create_model(config, *, weights=None, device='cpu'):
    """Build explicitly; imports never download weights or dispatch training."""
    name = config.get('name', 'resnet50')
    if name.startswith('densenet'):
        return MHDDenseNet(DenseNetConfig(**config), weights=weights, device=device)
    if name.startswith('retfound'):
        return MHDRETFound(RETFoundConfig(**config), weights=weights, device=device)
    return MHDResNet(ResNetConfig(**config), weights=weights, device=device)


__all__ = ['MHDResNet', 'ResNetConfig', 'create_model', 'MHDDenseNet', 'DenseNetConfig', 'MHDRETFound', 'RETFoundConfig']
