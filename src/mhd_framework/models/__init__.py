"""Reusable complete task models and immutable, verified model artifacts."""
from .resnet import MHDResNet, ResNetConfig
from .densenet import MHDDenseNet, DenseNetConfig
from .retfound import MHDRETFound, RETFoundConfig
from .vision_transformer import MHDViT, ViTConfig, MHDSwin, SwinConfig

from .modern_cnn import MHDConvNeXt, ConvNeXtConfig, MHDEfficientNetV2, EfficientNetV2Config
from .dinov2 import MHDDINOv2, DINOv2Config

from .timm_encoders import MHDTimmEncoder, TimmEncoderConfig, NAMES as TIMM_ENCODER_NAMES

from .clip_visual import MHDCLIPVisual, CLIPVisualConfig
from .volume_encoders import MHDVolumeEncoder, VolumeEncoderConfig, VOLUME_NAMES

def create_model(config, *, weights=None, device='cpu'):
    """Build explicitly; imports never download weights or dispatch training."""
    name = config.get('name', 'resnet50')
    if name in VOLUME_NAMES:
        return MHDVolumeEncoder(VolumeEncoderConfig(**config),weights=weights,device=device)
    if name.startswith('openclip_'):
        return MHDCLIPVisual(CLIPVisualConfig(**config),weights=weights,device=device)
    if name in TIMM_ENCODER_NAMES:
        return MHDTimmEncoder(TimmEncoderConfig(**config),weights=weights,device=device)
    if name.endswith('_dinov2'):
        return MHDDINOv2(DINOv2Config(**config), weights=weights, device=device)
    if name.startswith('convnext_'):
        return MHDConvNeXt(ConvNeXtConfig(**config), weights=weights, device=device)
    if name.startswith('efficientnet_v2_'):
        return MHDEfficientNetV2(EfficientNetV2Config(**config), weights=weights, device=device)
    if name.startswith('vit_'):
        return MHDViT(ViTConfig(**config), weights=weights, device=device)
    if name.startswith('swin_'):
        return MHDSwin(SwinConfig(**config), weights=weights, device=device)
    if name.startswith('densenet'):
        return MHDDenseNet(DenseNetConfig(**config), weights=weights, device=device)
    if name.startswith('retfound'):
        return MHDRETFound(RETFoundConfig(**config), weights=weights, device=device)
    return MHDResNet(ResNetConfig(**config), weights=weights, device=device)


__all__ = ['MHDResNet', 'ResNetConfig', 'create_model', 'MHDDenseNet', 'DenseNetConfig', 'MHDRETFound', 'RETFoundConfig', 'MHDViT', 'ViTConfig', 'MHDSwin', 'SwinConfig']

__all__ += ['MHDConvNeXt','ConvNeXtConfig','MHDEfficientNetV2','EfficientNetV2Config','MHDDINOv2','DINOv2Config']

__all__ += ['MHDTimmEncoder','TimmEncoderConfig']

__all__ += ['MHDCLIPVisual','CLIPVisualConfig','MHDVolumeEncoder','VolumeEncoderConfig']
