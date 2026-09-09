"""MHD ResNet task models."""
from dataclasses import asdict, dataclass
import copy

import torch
from torch import nn

from mhd_framework.core import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo


@dataclass(frozen=True)
class ResNetConfig:
    name: str = 'resnet50'
    spatial_dims: int = 2
    in_channels: int = 3
    num_classes: int = 2
    views: int = 1
    implementation: str = 'torchvision_2d'
    granularity: str = 'block'
    schema: str = 'mhd_resnet_task_v1'

    def __post_init__(self):
        if self.name not in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'):
            raise ValueError('Unsupported ResNet architecture')
        if self.schema != 'mhd_resnet_task_v1' or self.spatial_dims not in (2, 3):
            raise ValueError('Unsupported configuration schema or dimensionality')
        if (self.spatial_dims, self.implementation) not in ((2, 'torchvision_2d'), (3, 'inflated_3d')):
            raise ValueError('Choose an explicit independent implementation; 3D is never inferred from 2D')
        if self.granularity not in ('stage', 'block'):
            raise ValueError('Supported graph granularities are stage and block')
        if self.in_channels not in (1, 3) or self.num_classes < 2 or self.views < 1:
            raise ValueError('Invalid channels, classification task or view count')


class FlattenViews(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        c = self.config
        expected = c.spatial_dims + (3 if c.views > 1 else 2)
        if x.ndim != expected or x.shape[-c.spatial_dims-1] != c.in_channels:
            raise ValueError('Input rank/channels disagree with model configuration')
        if c.views > 1:
            if x.shape[1] != c.views:
                raise ValueError('Input view count differs from model configuration')
            return x.flatten(0, 1)
        return x


class PoolViews(nn.Module):
    def __init__(self, pool, views):
        super().__init__()
        self.pool, self.views = pool, views

    def forward(self, x):
        x = self.pool(x).flatten(1)
        if self.views > 1:
            x = x.reshape(-1, self.views, x.shape[-1]).mean(1)
        return x


class SliceMaxPool3d(nn.Module):
    """Depth-one max pooling via independent 2D slices (deterministic CUDA backward)."""
    def forward(self, x):
        b, c, d, h, w = x.shape
        y = nn.functional.max_pool2d(x.permute(0,2,1,3,4).reshape(b*d,c,h,w),3,2,1)
        return y.reshape(b,d,c,*y.shape[-2:]).permute(0,2,1,3,4).contiguous()


def inflate_3d(module):
    """Repeat a 2D kernel along depth and divide by depth; preserve its DC scale."""
    if isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        result = nn.Conv3d(module.in_channels, module.out_channels, (kh, kh, kw),
            stride=(module.stride[0], *module.stride),
            padding=(module.padding[0], *module.padding),
            dilation=(module.dilation[0], *module.dilation), groups=module.groups,
            bias=module.bias is not None, padding_mode=module.padding_mode)
        with torch.no_grad():
            result.weight.copy_(module.weight.unsqueeze(2).repeat(1, 1, kh, 1, 1) / kh)
            if module.bias is not None:
                result.bias.copy_(module.bias)
        return result
    if isinstance(module, nn.BatchNorm2d):
        result = nn.BatchNorm3d(module.num_features, module.eps, module.momentum,
                                module.affine, module.track_running_stats)
        result.load_state_dict(module.state_dict())
        return result
    if isinstance(module, nn.MaxPool2d):
        return SliceMaxPool3d()
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return nn.AdaptiveAvgPool3d(1)
    result = copy.deepcopy(module)
    for name, child in module.named_children():
        setattr(result, name, inflate_3d(child))
    return result


def native_resnet(config, *, weights=None):
    """Build an independent native reference; weights are explicit, never 'DEFAULT'."""
    from torchvision import models
    if weights == 'DEFAULT':
        raise ValueError('Use an explicit torchvision weight version, not DEFAULT')
    reference = getattr(models, config.name)(weights=weights)
    if config.num_classes != reference.fc.out_features:
        reference.fc = nn.Linear(reference.fc.in_features, config.num_classes)
    if config.spatial_dims == 3:
        reference = inflate_3d(reference)
        reference.conv1.stride = (1, 2, 2)
    if config.in_channels == 1:
        old = reference.conv1
        cls = nn.Conv2d if config.spatial_dims == 2 else nn.Conv3d
        new = cls(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        reference.conv1 = new
    return reference


class ResidualAddReLU(nn.Module):
    def forward(self, main, skip):
        return torch.relu(main + skip)


class MHDResNet(MHD_Graph):
    """ResNet classifier with named graph endpoints."""
    def __init__(self, config: ResNetConfig, *, weights=None, device='cpu'):
        reference = native_resnet(config, weights=weights)
        names, operations, definitions = ['input'], [], []
        aliases = {'input': 0}
        def add(name, module, inputs):
            node_id, edge_id = len(names), len(operations)
            names.append(name); operations.append((name, module))
            definitions.append((edge_id, tuple(inputs), node_id))
            aliases[name] = node_id
            return node_id
        current = add('eyes', FlattenViews(config), [0])
        current = add('stem', nn.Sequential(reference.conv1, reference.bn1,
                            reference.relu, reference.maxpool), [current])
        for stage_index, layer in enumerate((reference.layer1,reference.layer2,
                                             reference.layer3,reference.layer4), start=1):
            stage_name = f'stage{stage_index}'
            if config.granularity == 'stage':
                current = add(stage_name, layer, [current])
            else:
                for block_index, block in enumerate(layer):
                    prefix = f'{stage_name}.block{block_index}'
                    source = current
                    main = [block.conv1, block.bn1, nn.ReLU(inplace=False), block.conv2, block.bn2]
                    if hasattr(block, 'conv3'):
                        main += [nn.ReLU(inplace=False), block.conv3, block.bn3]
                    main_id = add(prefix+'.main', nn.Sequential(*main), [source])
                    skip_id = add(prefix+'.skip', block.downsample if block.downsample is not None else nn.Identity(), [source])
                    current = add(prefix, ResidualAddReLU(), [main_id, skip_id])
                aliases[stage_name] = current
        current = add('features', PoolViews(reference.avgpool, config.views), [current])
        add('logits', reference.fc, [current])
        nodes = [MHD_Node(i, name, MHD_Node.Message(torch.zeros(1)), aggregation='replace')
                 for i, name in enumerate(names)]
        edges = [MHD_Edge(i, name, [MHD_Edge.Operation(module)])
                 for i, (name, module) in enumerate(operations)]
        roles, sorts = [], []
        for edge_id, inputs, output in definitions:
            r = torch.zeros(len(edges), len(nodes), dtype=torch.long)
            order = torch.zeros_like(r)
            for j, input_id in enumerate(inputs):
                r[edge_id,input_id] = -1; order[edge_id,input_id] = j
            r[edge_id,output] = 1; order[edge_id,output] = len(inputs)
            roles.append(r); sorts.append(order)
        topo = MHD_Topo(roles + [-r for r in reversed(roles)], sorts + list(reversed(sorts)))
        super().__init__(set(nodes), set(edges), {topo}, device=torch.device(device))
        self.config = config
        self.endpoint_nodes = aliases
        self._producer_levels = {output:edge_id for edge_id,_,output in definitions}
        self._definitions = definitions
        expansion = 1 if config.name in ('resnet18', 'resnet34') else 4
        self.feature_channels = dict(zip(('stage1', 'stage2', 'stage3', 'stage4'),
                                        (x*expansion for x in (64, 128, 256, 512))))
        self.forward_levels = tuple(range(len(edges)))
        self._executed_nodes = set()
        # A reference view of the SAME modules for portable native state_dict I/O.
        # Do not register it as a second module owner.
        object.__setattr__(self, '_native_reference', reference)

    def describe_nodes(self):
        """Enumerate every real graph node; aliases do not hide internal nodes."""
        return [{'id': node.id, 'name': node.name,
                 'aliases': [k for k,v in self.endpoint_nodes.items() if v == node.id],
                 'shape': list(node.feature_message.current_state.shape),
                 'shape_is_runtime': node.id in self._executed_nodes}
                for node in self._nodes_in_id_order]

    def forward_from(self, endpoint, value, *, return_features=False):
        if endpoint not in self.endpoint_nodes or endpoint == 'logits':
            raise ValueError('Unknown or terminal feature endpoint')
        index = self.endpoint_nodes[endpoint]
        node = self.get_node_by_id(index)
        node.feature_message.current_state = value
        start = self._producer_levels.get(index, -1) + 1
        # A main/skip branch is not a sufficient cut: continuation would require
        # additional tensors. Never silently reuse stale values from a prior call.
        available = {index}
        for _, inputs, output in self._definitions[start:]:
            if not set(inputs).issubset(available):
                raise ValueError('Endpoint is not a complete graph cut; supply the other branch through an explicit project topology')
            available.add(output)
        MHD_Graph.forward(self, levels=self.forward_levels[start:])
        self._executed_nodes = available
        if return_features:
            return {name: self.get_node_by_id(i).feature_message.current_state
                    for name, i in self.endpoint_nodes.items() if i >= index}
        return self.get_node_by_id(self.endpoint_nodes['logits']).feature_message.current_state

    def forward_until(self, endpoint, x):
        """Execute through a named endpoint exactly once, for external graph composition."""
        if endpoint not in self.endpoint_nodes or endpoint == "input":
            raise ValueError("Unknown or initial endpoint")
        self.get_node_by_name("input").feature_message.current_state = x
        stop = self._producer_levels[self.endpoint_nodes[endpoint]] + 1
        MHD_Graph.forward(self, levels=self.forward_levels[:stop])
        self._executed_nodes = {0} | {out for _,_,out in self._definitions[:stop]}
        return self.get_node_by_id(self.endpoint_nodes[endpoint]).feature_message.current_state

    def forward(self, x, *, return_features=False):
        return self.forward_from('input', x, return_features=return_features)

    def native_state_dict(self):
        return self._native_reference.state_dict()

    def load_native_state_dict(self, state):
        return self._native_reference.load_state_dict(state, strict=True)

    def configuration(self):
        return asdict(self.config)
