"""OpenCLIP2.24 visual Transformers with an independently supervised task head."""
from dataclasses import dataclass
from importlib.metadata import version
import torch
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews

@dataclass(frozen=True)
class CLIPVisualConfig:
    name: str='openclip_vit_b16'
    spatial_dims: int=2
    in_channels: int=3
    image_size: int=224
    num_classes: int=2
    views: int=1
    activation: str='quick_gelu'
    implementation: str='open_clip_2_24_visual'
    schema: str='mhd_clip_visual_task_v1'
    def __post_init__(self):
        if self.name not in ('openclip_vit_b16','openclip_vit_l14') or self.spatial_dims!=2 or self.in_channels!=3 or self.image_size!=224:
            raise ValueError('Explicit RGB224 CLIP visual variant required')
        if self.num_classes<2 or self.views<1 or self.activation!='quick_gelu' or self.implementation!='open_clip_2_24_visual' or self.schema!='mhd_clip_visual_task_v1':
            raise ValueError('Invalid CLIP visual task configuration')

class CLIPTask(nn.Module):
    def __init__(self,config):
        super().__init__()
        if version('open_clip_torch')!='2.24.0':raise RuntimeError('Requires open_clip_torch==2.24.0')
        from open_clip.transformer import VisionTransformer,QuickGELU
        width,depth,heads,patch,out=(768,12,12,16,512) if config.name=='openclip_vit_b16' else (1024,24,16,14,768)
        self.visual=VisionTransformer(image_size=224,patch_size=patch,width=width,layers=depth,heads=heads,mlp_ratio=4,output_dim=out,act_layer=QuickGELU)
        self.head=nn.Linear(out,config.num_classes)
    def forward(self,x):return self.head(self.visual(x))

class CLIPPositions(nn.Module):
    def __init__(self,visual):
        super().__init__();self.class_embedding=visual.class_embedding;self.positional_embedding=visual.positional_embedding
    def forward(self,x):
        x=x.flatten(2).permute(0,2,1)
        token=self.class_embedding.reshape(1,1,-1).expand(x.shape[0],1,-1).to(x.dtype)
        return torch.cat((token,x),dim=1)+self.positional_embedding.to(x.dtype)

class PermuteTokens(nn.Module):
    def forward(self,x):return x.permute(1,0,2)

class CLIPFeatures(nn.Module):
    def __init__(self,visual,views):
        super().__init__();self.norm=visual.ln_post;self.proj=visual.proj;self.views=views
    def forward(self,x):
        x=self.norm(x)[:,0]@self.proj
        return x if self.views==1 else x.reshape(-1,self.views,x.shape[-1]).mean(1)

class MHDCLIPVisual(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        if weights is not None:raise ValueError('Load an independently verified visual encoder, not a text task')
        native=CLIPTask(config);v=native.visual
        ops=[('views',FlattenViews(config)),('patch_embed',v.conv1),('positions',CLIPPositions(v)),('patch_dropout',v.patch_dropout),('norm_pre',v.ln_pre),('sequence_first',PermuteTokens())]
        channels={}
        for i,block in enumerate(v.transformer.resblocks,1):
            name=f'block{i:02}';ops.append((name,block));channels[name]=v.conv1.out_channels
        ops += [('batch_first',PermuteTokens()),('features',CLIPFeatures(v,config.views)),('logits',native.head)]
        channels['features']=native.head.in_features
        super().__init__(config,native,ops,channels,device=device)
