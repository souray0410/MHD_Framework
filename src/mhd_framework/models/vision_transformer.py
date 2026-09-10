"""Explicit Torchvision ViT and Swin task graphs; initialization is optional."""
from dataclasses import dataclass
import torch
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews, PoolViews


@dataclass(frozen=True)
class ViTConfig:
    name: str = 'vit_b_16'
    spatial_dims: int = 2
    in_channels: int = 3
    image_size: int = 224
    num_classes: int = 2
    views: int = 1
    dropout: float = 0.0
    attention_dropout: float = 0.0
    implementation: str = 'torchvision_vit_0_23'
    schema: str = 'mhd_vit_task_v1'

    def __post_init__(self):
        if self.name not in ('vit_b_16', 'vit_l_16') or self.implementation != 'torchvision_vit_0_23':
            raise ValueError('Select an explicit supported ViT implementation')
        if self.spatial_dims != 2 or self.in_channels != 3 or self.image_size != 224:
            raise ValueError('This ViT is RGB 224x224, not a volume model')
        if self.views < 1 or self.num_classes < 2 or self.schema != 'mhd_vit_task_v1':
            raise ValueError('Invalid ViT task configuration')
        if not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1:
            raise ValueError('Invalid dropout')


@dataclass(frozen=True)
class SwinConfig:
    name: str = 'swin_b'
    spatial_dims: int = 2
    in_channels: int = 3
    num_classes: int = 2
    views: int = 1
    implementation: str = 'torchvision_swin_0_23'
    schema: str = 'mhd_swin_task_v1'

    def __post_init__(self):
        if self.name not in ('swin_t', 'swin_s', 'swin_b') or self.implementation != 'torchvision_swin_0_23':
            raise ValueError('Select an explicit supported Swin implementation')
        if self.spatial_dims != 2 or self.in_channels != 3:
            raise ValueError('This Swin is a 2D RGB model')
        if self.views < 1 or self.num_classes < 2 or self.schema != 'mhd_swin_task_v1':
            raise ValueError('Invalid Swin task configuration')


class PatchGrid(nn.Module):
    def __init__(self, projection, image_size):
        super().__init__(); self.projection = projection; self.image_size = image_size
    def forward(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError('ViT input height and width must match its configured image size')
        return self.projection(images)


class ClassTokens(nn.Module):
    def __init__(self, token):
        super().__init__(); self.class_token = token
    def forward(self, grid):
        patches = grid.flatten(2).transpose(1, 2)
        return torch.cat((self.class_token.expand(grid.shape[0], -1, -1), patches), dim=1)


class AddPositions(nn.Module):
    def __init__(self, positions, dropout):
        super().__init__(); self.positions = positions; self.dropout = dropout
    def forward(self, tokens):
        return self.dropout(tokens + self.positions)


class ClassFeatures(nn.Module):
    def __init__(self, views):
        super().__init__(); self.views = views
    def forward(self, tokens):
        features = tokens[:, 0]
        return features if self.views == 1 else features.reshape(-1, self.views, features.shape[-1]).mean(1)


def torchvision_builder(name, weights, **kwargs):
    import torchvision
    if torchvision.__version__.split('+')[0] != '0.23.0':
        raise RuntimeError('This implementation requires torchvision==0.23.0')
    if weights == 'DEFAULT': raise ValueError('Use an explicit weight version')
    return getattr(torchvision.models, name)(weights=weights, **kwargs)


class MHDViT(NamedModelGraph):
    def __init__(self, config, *, weights=None, device='cpu'):
        native = torchvision_builder(config.name, weights,
                 num_classes=1000 if weights is not None else config.num_classes,
                 image_size=config.image_size, dropout=config.dropout,
                 attention_dropout=config.attention_dropout)
        if native.heads.head.out_features != config.num_classes:
            native.heads.head = nn.Linear(native.hidden_dim, config.num_classes)
            nn.init.zeros_(native.heads.head.weight); nn.init.zeros_(native.heads.head.bias)
        ops = [('views', FlattenViews(config)), ('patch_grid', PatchGrid(native.conv_proj, config.image_size)),
               ('tokens', ClassTokens(native.class_token)),
               ('positions', AddPositions(native.encoder.pos_embedding, native.encoder.dropout))]
        channels = {name:native.hidden_dim for name in ('patch_grid', 'tokens', 'positions', 'features')}
        for i, block in enumerate(native.encoder.layers, 1):
            name = f'block{i:02}'; ops.append((name, block)); channels[name] = native.hidden_dim
        ops += [('norm', native.encoder.ln), ('features', ClassFeatures(config.views)), ('logits', native.heads)]
        super().__init__(config, native, ops, channels, device=device)


class MHDSwin(NamedModelGraph):
    def __init__(self, config, *, weights=None, device='cpu'):
        native = torchvision_builder(config.name, weights,
                  num_classes=1000 if weights is not None else config.num_classes)
        if native.head.out_features != config.num_classes:
            native.head = nn.Linear(native.head.in_features, config.num_classes)
            nn.init.trunc_normal_(native.head.weight, std=.02); nn.init.zeros_(native.head.bias)
        ops = [('views', FlattenViews(config)), ('patch_embed', native.features[0])]
        channels = {'patch_embed':native.features[0][0].out_channels}
        for stage in range(1, 5):
            for i, block in enumerate(native.features[2*stage-1], 1):
                name = f'stage{stage}.block{i:02}'; ops.append((name, block))
                channels[name] = block.norm1.normalized_shape[0]
            if stage < 4:
                merge = native.features[2*stage]; name = f'merge{stage}'
                ops.append((name, merge)); channels[name] = merge.reduction.out_features
        ops += [('norm', native.norm), ('channels_first', native.permute),
                ('features', PoolViews(native.avgpool, config.views)), ('logits', native.head)]
        channels['features'] = native.head.in_features
        super().__init__(config, native, ops, channels, device=device)
        # Stage aliases identify complete cuts; individual block nodes remain visible.
        for stage in range(1, 5):
            last = len(native.features[2*stage-1])
            self.endpoint_nodes[f'stage{stage}'] = self.endpoint_nodes[f'stage{stage}.block{last:02}']
            self.feature_channels[f'stage{stage}'] = channels[f'stage{stage}.block{last:02}']
