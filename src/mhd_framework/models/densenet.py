"""DenseNet classifiers partitioned at published dense blocks and transitions."""
from dataclasses import dataclass
from collections import OrderedDict
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews, PoolViews

@dataclass(frozen=True)
class DenseNetConfig:
    name: str = 'densenet121'
    spatial_dims: int = 2
    in_channels: int = 3
    num_classes: int = 2
    views: int = 1
    implementation: str = 'torchvision_2d'
    schema: str = 'mhd_densenet_task_v1'

    def __post_init__(self):
        if self.name not in ('densenet121','densenet161','densenet169','densenet201'):
            raise ValueError('Unsupported DenseNet architecture')
        if self.spatial_dims != 2 or self.implementation != 'torchvision_2d':
            raise ValueError('This builder is explicitly 2D; 3D needs a separate implementation')
        if self.schema != 'mhd_densenet_task_v1' or self.in_channels != 3 or self.views < 1 or self.num_classes < 2:
            raise ValueError('Invalid DenseNet task configuration; explicit RGB preprocessing required')

class MHDDenseNet(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        from torchvision import models
        if weights == 'DEFAULT':
            raise ValueError('Choose an explicit pretrained weight version')
        native = getattr(models,config.name)(weights=weights)
        if native.classifier.out_features != config.num_classes:
            native.classifier = nn.Linear(native.classifier.in_features,config.num_classes)
        f = native.features
        ops = [('views',FlattenViews(config)),('stem',nn.Sequential(OrderedDict(
            (n,getattr(f,n)) for n in ('conv0','norm0','relu0','pool0'))))]
        channels = {}
        for i in range(1,5):
            block = getattr(f,f'denseblock{i}')
            ops.append((f'denseblock{i}',block))
            last = list(block.children())[-1]
            channels[f'denseblock{i}'] = last.norm1.num_features + last.conv2.out_channels
            if i < 4:
                trans = getattr(f,f'transition{i}')
                ops.append((f'transition{i}',trans))
                channels[f'transition{i}'] = trans.conv.out_channels
        ops += [('norm',nn.Sequential(f.norm5,nn.ReLU(inplace=False))),
                ('features',PoolViews(nn.AdaptiveAvgPool2d(1),config.views)),
                ('logits',native.classifier)]
        super().__init__(config,native,ops,channels,device=device)
