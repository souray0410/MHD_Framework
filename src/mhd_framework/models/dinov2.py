"""DINOv2 B/L task adaptation using the explicitly pinned timm 0.9.2 port."""
from dataclasses import dataclass
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews
from .retfound import TokenPosition
from .vision_transformer import ClassFeatures


@dataclass(frozen=True)
class DINOv2Config:
    name: str = 'vit_base_patch14_dinov2'
    spatial_dims: int = 2
    in_channels: int = 3
    image_size: int = 224
    num_classes: int = 2
    views: int = 1
    implementation: str = 'dinov2_timm_0_9_2_fixed224'
    schema: str = 'mhd_dinov2_task_v1'
    def __post_init__(self):
        if self.name not in ('vit_base_patch14_dinov2','vit_large_patch14_dinov2'):
            raise ValueError('Explicit DINOv2 B/14 or L/14 required')
        if self.spatial_dims!=2 or self.in_channels!=3 or self.image_size!=224 or self.views<1 or self.num_classes<2:
            raise ValueError('This DINOv2 task adapter uses RGB 224x224, not volumes')
        if self.implementation!='dinov2_timm_0_9_2_fixed224' or self.schema!='mhd_dinov2_task_v1':
            raise ValueError('Unknown DINOv2 implementation')


class MHDDINOv2(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        if weights is not None:raise ValueError('Load a separately verified DINOv2 encoder checkpoint')
        import timm
        if timm.__version__!='0.9.2':raise RuntimeError('This DINOv2 port requires timm==0.9.2')
        native=timm.create_model(config.name,pretrained=False,img_size=config.image_size,num_classes=config.num_classes)
        for block in native.blocks:block.attn.fused_attn=False
        ops=[('views',FlattenViews(config)),('patch_embed',native.patch_embed),('positions',TokenPosition(native)),
             ('patch_drop',native.patch_drop),('norm_pre',native.norm_pre)]
        channels={'patch_embed':native.num_features}
        for i,block in enumerate(native.blocks,1):
            name=f'block{i:02}';ops.append((name,block));channels[name]=native.num_features
        ops += [('norm',native.norm),('features',nn.Sequential(ClassFeatures(config.views),native.fc_norm,native.head_drop)),('logits',native.head)]
        channels['features']=native.num_features
        super().__init__(config,native,ops,channels,device=device)
