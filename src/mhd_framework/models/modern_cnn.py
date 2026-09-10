"""Explicit 2D ConvNeXt and EfficientNetV2 graphs from torchvision 0.23."""
from dataclasses import dataclass
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews, PoolViews
from .vision_transformer import torchvision_builder


@dataclass(frozen=True)
class ConvNeXtConfig:
    name: str = 'convnext_base'
    spatial_dims: int = 2
    in_channels: int = 3
    num_classes: int = 2
    views: int = 1
    implementation: str = 'torchvision_convnext_0_23'
    schema: str = 'mhd_convnext_task_v1'
    def __post_init__(self):
        if self.name not in ('convnext_tiny','convnext_small','convnext_base','convnext_large'):
            raise ValueError('Unsupported ConvNeXt variant')
        if self.spatial_dims!=2 or self.in_channels!=3 or self.views<1 or self.num_classes<2:
            raise ValueError('ConvNeXt requires explicit RGB 2D task inputs')
        if self.implementation!='torchvision_convnext_0_23' or self.schema!='mhd_convnext_task_v1':
            raise ValueError('Unknown ConvNeXt implementation')


@dataclass(frozen=True)
class EfficientNetV2Config:
    name: str = 'efficientnet_v2_s'
    spatial_dims: int = 2
    in_channels: int = 3
    num_classes: int = 2
    views: int = 1
    implementation: str = 'torchvision_efficientnet_v2_0_23'
    schema: str = 'mhd_efficientnet_v2_task_v1'
    def __post_init__(self):
        if self.name not in ('efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l'):
            raise ValueError('Unsupported EfficientNetV2 variant')
        if self.spatial_dims!=2 or self.in_channels!=3 or self.views<1 or self.num_classes<2:
            raise ValueError('EfficientNetV2 requires explicit RGB 2D task inputs')
        if self.implementation!='torchvision_efficientnet_v2_0_23' or self.schema!='mhd_efficientnet_v2_task_v1':
            raise ValueError('Unknown EfficientNetV2 implementation')


class ConvNeXtFeatures(nn.Module):
    def __init__(self, native, views):
        super().__init__(); self.pool=native.avgpool;self.norm=native.classifier[0]
        self.flatten=native.classifier[1];self.views=views
    def forward(self,x):
        x=self.flatten(self.norm(self.pool(x)))
        return x if self.views==1 else x.reshape(-1,self.views,x.shape[-1]).mean(1)


class MHDConvNeXt(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        native=torchvision_builder(config.name,weights,num_classes=1000 if weights is not None else config.num_classes)
        if native.classifier[-1].out_features!=config.num_classes:
            native.classifier[-1]=nn.Linear(native.classifier[-1].in_features,config.num_classes)
            nn.init.trunc_normal_(native.classifier[-1].weight,std=.02);nn.init.zeros_(native.classifier[-1].bias)
        ops=[('views',FlattenViews(config)),('stem',native.features[0])];channels={'stem':native.features[0][0].out_channels}
        for stage in range(1,5):
            for i,block in enumerate(native.features[2*stage-1],1):
                name=f'stage{stage}.block{i:02}';ops.append((name,block));channels[name]=block.block[0].out_channels
            if stage<4:
                name=f'downsample{stage}';module=native.features[2*stage];ops.append((name,module));channels[name]=module[-1].out_channels
        ops += [('features',ConvNeXtFeatures(native,config.views)),('logits',native.classifier[-1])]
        channels['features']=native.classifier[-1].in_features
        super().__init__(config,native,ops,channels,device=device)
        for stage in range(1,5):
            name=f'stage{stage}.block{len(native.features[2*stage-1]):02}'
            self.endpoint_nodes[f'stage{stage}']=self.endpoint_nodes[name];self.feature_channels[f'stage{stage}']=channels[name]


class MHDEfficientNetV2(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        native=torchvision_builder(config.name,weights,num_classes=1000 if weights is not None else config.num_classes)
        if native.classifier[-1].out_features!=config.num_classes:
            native.classifier[-1]=nn.Linear(native.classifier[-1].in_features,config.num_classes)
        ops=[('views',FlattenViews(config)),('stem',native.features[0])];channels={'stem':native.features[0][0].out_channels}
        for stage,group in enumerate(native.features[1:-1],1):
            for i,block in enumerate(group,1):
                name=f'stage{stage}.block{i:02}';ops.append((name,block))
                channels[name]=next(m.out_channels for m in reversed(list(block.modules())) if isinstance(m,nn.Conv2d))
        ops += [('final_conv',native.features[-1]),('features',PoolViews(native.avgpool,config.views)),('logits',native.classifier)]
        channels['features']=native.classifier[-1].in_features
        super().__init__(config,native,ops,channels,device=device)
        for stage,group in enumerate(native.features[1:-1],1):
            name=f'stage{stage}.block{len(group):02}'
            self.endpoint_nodes[f'stage{stage}']=self.endpoint_nodes[name];self.feature_channels[f'stage{stage}']=channels[name]
