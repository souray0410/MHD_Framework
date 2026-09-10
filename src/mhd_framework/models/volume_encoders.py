"""Independent MedicalNet and MONAI volumetric classification task graphs."""
from dataclasses import dataclass
import torch
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews,PoolViews

class DeterministicMaxPool3d(nn.Module):
    """3x3x3/stride2/pad1 max pooling via two deterministic CUDA 2D pools.

    Spatial maximization precedes depth maximization, preserving the native
    depth-major first-maximum tie rule. No learned parameters or geometry change.
    """
    def forward(self,x):
        b,c,d,h,w=x.shape
        y=torch.nn.functional.max_pool2d(x.permute(0,2,1,3,4).reshape(b*d,c,h,w),3,2,1)
        oh,ow=y.shape[-2:]
        y=y.reshape(b,d,c,oh,ow).permute(0,2,3,4,1).reshape(b*c*oh*ow,1,d,1)
        y=torch.nn.functional.max_pool2d(y,(3,1),(2,1),(1,0))
        return y.reshape(b,c,oh,ow,-1).permute(0,1,4,2,3).contiguous()

class DeterministicAvgPool3d(nn.Module):
    """Nonoverlapping 2x2x2 average with native depth/row/column sum order."""
    def forward(self,x):
        d,h,w=(v//2*2 for v in x.shape[-3:])
        values=[x[:,:,a:d:2,b:h:2,c:w:2] for a in range(2) for b in range(2) for c in range(2)]
        y=values[0]
        for value in values[1:]:y=y+value
        return y/8

def deterministic_pools(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.MaxPool3d):
            triple=lambda v:(v,v,v) if isinstance(v,int) else tuple(v)
            if (triple(child.kernel_size),triple(child.stride),triple(child.padding),triple(child.dilation),child.ceil_mode,child.return_indices)!=((3,3,3),(2,2,2),(1,1,1),(1,1,1),False,False):
                raise ValueError('Unsupported deterministic max-pool geometry')
            setattr(module,name,DeterministicMaxPool3d())
        elif isinstance(child,nn.AvgPool3d):
            triple=lambda v:(v,v,v) if isinstance(v,int) else tuple(v)
            if (triple(child.kernel_size),triple(child.stride),triple(child.padding),child.ceil_mode,child.divisor_override)!=((2,2,2),(2,2,2),(0,0,0),False,None):
                raise ValueError('Unsupported deterministic average-pool geometry')
            setattr(module,name,DeterministicAvgPool3d())
        else:deterministic_pools(child)

VOLUME_NAMES=('medicalnet_resnet50_3d','medicalnet_resnet101_3d','monai_densenet121_3d','swin_unetr_encoder_3d')
@dataclass(frozen=True)
class VolumeEncoderConfig:
    name: str='medicalnet_resnet50_3d'
    spatial_dims: int=3
    in_channels: int=1
    num_classes: int=2
    views: int=1
    pooling: str='deterministic_equivalent_v1'
    implementation: str='medicalnet20f76aa_monai1_5_1_explicit'
    schema: str='mhd_volume_encoder_task_v1'
    def __post_init__(self):
        if self.name not in VOLUME_NAMES or self.spatial_dims!=3 or self.in_channels!=1:
            raise ValueError('Explicit single-channel3D model required')
        if self.num_classes<2 or self.views<1 or self.implementation!='medicalnet20f76aa_monai1_5_1_explicit' or self.schema!='mhd_volume_encoder_task_v1':
            raise ValueError('Invalid volumetric task configuration')
        if self.pooling!='deterministic_equivalent_v1':raise ValueError('Unknown pooling implementation')

class MedicalNetTask(nn.Module):
    def __init__(self,config):
        super().__init__()
        from . import _medicalnet
        builder=_medicalnet.resnet50 if config.name=='medicalnet_resnet50_3d' else _medicalnet.resnet101
        self.backbone=builder(sample_input_D=128,sample_input_H=224,sample_input_W=224,num_seg_classes=config.num_classes,shortcut_type='B',no_cuda=True)
        # Replace the original segmentation decoder by a declared classification
        # adapter. All original encoder strides/dilations and names are retained.
        del self.backbone.conv_seg
        deterministic_pools(self.backbone)
        self.pool=nn.AdaptiveAvgPool3d(1);self.head=nn.Linear(2048,config.num_classes)
    def forward(self,x):
        b=self.backbone;x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        for layer in (b.layer1,b.layer2,b.layer3,b.layer4):x=layer(x)
        return self.head(self.pool(x).flatten(1))

class SwinVolumeTask(nn.Module):
    def __init__(self,config):
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer
        self.backbone=SwinTransformer(in_chans=1,embed_dim=48,window_size=(7,7,7),patch_size=(2,2,2),
            depths=(2,2,2,2),num_heads=(3,6,12,24),drop_path_rate=0,spatial_dims=3,use_checkpoint=False,downsample='merging',use_v2=False)
        self.head=nn.Linear(768,config.num_classes)
    def forward(self,x):return self.head(self.backbone(x,normalize=True)[-1].mean((2,3,4)))

class StageContiguous(nn.Module):
    def __init__(self,stage):super().__init__();self.stage=stage
    def forward(self,x):return self.stage(x.contiguous())

class SwinVolumeFeatures(nn.Module):
    def __init__(self,views):super().__init__();self.views=views
    def forward(self,x):
        x=torch.nn.functional.layer_norm(x.permute(0,2,3,4,1),[x.shape[1]]).permute(0,4,1,2,3)
        x=x.mean((2,3,4))
        return x if self.views==1 else x.reshape(-1,self.views,x.shape[-1]).mean(1)

class DenseVolumeTask(nn.Module):
    def __init__(self,config):
        super().__init__()
        from monai.networks.nets import DenseNet121
        self.encoder=DenseNet121(spatial_dims=3,in_channels=1,out_channels=config.num_classes)
        deterministic_pools(self.encoder)
        self.head=self.encoder.class_layers.out
        # Native registration contains the head once; the reference forward uses it.
        del self.encoder.class_layers.out
    def forward(self,x):return self.head(self.encoder(x))

class MHDVolumeEncoder(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        if weights is not None:raise ValueError('Random architecture and accepted pretrained encoder are distinct')
        if not config.name.startswith('medicalnet_'):
            import monai
            if monai.__version__!='1.5.1':raise RuntimeError('Requires monai==1.5.1')
        channels={};aliases={};ops=[('views',FlattenViews(config))]
        if config.name.startswith('medicalnet_'):
            native=MedicalNetTask(config);b=native.backbone
            ops += [('stem',nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool))]
            for stage in range(1,5):
                for i,block in enumerate(getattr(b,f'layer{stage}'),1):
                    name=f'stage{stage}.block{i:02}';ops.append((name,block));channels[name]=block.conv3.out_channels
                aliases[f'stage{stage}']=name
            ops += [('features',PoolViews(native.pool,config.views)),('logits',native.head)]
        elif config.name=='monai_densenet121_3d':
            native=DenseVolumeTask(config);b=native.encoder
            for name,module in b.features.named_children():
                if name.startswith('denseblock'):
                    for key,layer in module.named_children():ops.append((name+'.'+key,layer))
                    aliases[name]=name+'.'+key
                else:ops.append((name,module))
            # Native class_layers now contains only relu/pool/flatten.
            ops += [('prepool_relu',b.class_layers.relu),('features',PoolViews(b.class_layers.pool,config.views)),('logits',native.head)]
        else:
            native=SwinVolumeTask(config);b=native.backbone
            ops += [('patch_embed',b.patch_embed),('positions_dropout',b.pos_drop)]
            for stage in range(1,5):
                ops.append((f'stage{stage}',StageContiguous(getattr(b,f'layers{stage}')[0])))
                channels[f'stage{stage}']=48*2**stage
            ops += [('features',SwinVolumeFeatures(config.views)),('logits',native.head)]
        channels['features']=native.head.in_features
        super().__init__(config,native,ops,channels,device=device)
        for alias,name in aliases.items():
            self.endpoint_nodes[alias]=self.endpoint_nodes[name]
            if name in channels:self.feature_channels[alias]=channels[name]
