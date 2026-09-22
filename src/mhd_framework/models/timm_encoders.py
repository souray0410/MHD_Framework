"""Explicit timm0.9.2 task graphs for ConvNeXtV2, DeiTIII, MAE and BEiT."""
from dataclasses import dataclass
from functools import partial
import torch
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews
from .retfound import native_retfound, RETFoundConfig, retfound_operations

NAMES=('convnextv2_base','convnextv2_large','deit3_base_patch16_224','deit3_large_patch16_224',
       'mae_vit_base_patch16','mae_vit_large_patch16','beit_base_patch16_224','beit_large_patch16_224')

@dataclass(frozen=True)
class TimmEncoderConfig:
    name: str='deit3_base_patch16_224'
    spatial_dims: int=2
    in_channels: int=3
    image_size: int=224
    num_classes: int=2
    views: int=1
    implementation: str='timm_0_9_2_explicit_task'
    schema: str='mhd_timm_encoder_task_v1'
    def __post_init__(self):
        if self.name not in NAMES or self.spatial_dims!=2 or self.in_channels!=3 or self.image_size!=224:
            raise ValueError('Select a declared RGB224 2D encoder')
        if self.num_classes<2 or self.views<1 or self.implementation!='timm_0_9_2_explicit_task' or self.schema!='mhd_timm_encoder_task_v1':
            raise ValueError('Invalid timm task configuration')

class PositionTokens(nn.Module):
    def __init__(self,native):
        super().__init__();self.cls_token=native.cls_token;self.pos_embed=native.pos_embed
        self.drop=native.pos_drop;self.no_embed_class=getattr(native,'no_embed_class',False)
    def forward(self,x):
        if self.no_embed_class:x=x+self.pos_embed
        x=torch.cat((self.cls_token.expand(x.shape[0],-1,-1),x),dim=1)
        if not self.no_embed_class and self.pos_embed is not None:x=x+self.pos_embed
        return self.drop(x)

class TokenFeatures(nn.Module):
    def __init__(self,native,views):
        super().__init__();self.pool=native.global_pool;self.prefix=native.num_prefix_tokens
        self.norm=native.fc_norm;self.drop=native.head_drop;self.views=views
    def forward(self,x):
        x=x[:,self.prefix:].mean(1) if self.pool=='avg' else x[:,0]
        x=self.drop(self.norm(x))
        return x if self.views==1 else x.reshape(-1,self.views,x.shape[-1]).mean(1)

class ConvFeatures(nn.Module):
    def __init__(self,native,views):
        super().__init__();h=native.head
        self.layers=nn.Sequential(h.global_pool,h.norm,h.flatten,h.pre_logits,h.drop);self.views=views
    def forward(self,x):
        x=self.layers(x)
        return x if self.views==1 else x.reshape(-1,self.views,x.shape[-1]).mean(1)

class MHDTimmEncoder(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        if weights is not None:raise ValueError('Load separately verified encoder weights')
        import timm
        if timm.__version__!='0.9.2':raise RuntimeError('Requires timm==0.9.2')
        if config.name=='mae_vit_large_patch16':
            # Same complete constructor and operations as the RETFound-MAE L encoder;
            # a different pretraining source does not create another architecture.
            native=native_retfound(RETFoundConfig(num_classes=config.num_classes,views=config.views))
        elif config.name=='mae_vit_base_patch16':
            from timm.models.vision_transformer import VisionTransformer
            native=VisionTransformer(img_size=224,patch_size=16,embed_dim=768,depth=12,num_heads=12,
                mlp_ratio=4,qkv_bias=True,num_classes=config.num_classes,global_pool='avg',fc_norm=True,
                norm_layer=partial(nn.LayerNorm,eps=1e-6))
        else:native=timm.create_model(config.name,pretrained=False,num_classes=config.num_classes)
        channels={};aliases={}
        if config.name.startswith('convnextv2_'):
            ops=[('views',FlattenViews(config)),('stem',native.stem)]
            for stage,group in enumerate(native.stages,1):
                ops.append((f'downsample{stage}',group.downsample))
                for i,block in enumerate(group.blocks,1):
                    name=f'stage{stage}.block{i:02}';ops.append((name,block));channels[name]=block.conv_dw.out_channels
                aliases[f'stage{stage}']=name
            ops += [('norm_pre',native.norm_pre),('features',ConvFeatures(native,config.views)),('logits',native.head.fc)]
        else:
            for block in native.blocks:
                if hasattr(block.attn,'fused_attn'):block.attn.fused_attn=False
            if config.name.startswith('mae_'):ops=retfound_operations(native,config)
            else:
                if getattr(native,'rel_pos_bias',None) is not None:
                    raise ValueError('Shared relative bias requires an explicitly declared graph; not this BEiT variant')
                ops=[('views',FlattenViews(config)),('patch_embed',native.patch_embed),('positions',PositionTokens(native))]
                for name in ('patch_drop','norm_pre'):
                    if hasattr(native,name):ops.append((name,getattr(native,name)))
                ops += [(f'block{i:02}',block) for i,block in enumerate(native.blocks,1)]
                ops += [('norm',native.norm),('features',TokenFeatures(native,config.views)),('logits',native.head)]
            channels.update({f'block{i:02}':native.num_features for i in range(1,len(native.blocks)+1)})
        channels['features']=native.num_features
        super().__init__(config,native,ops,channels,device=device)
        for alias,name in aliases.items():self.endpoint_nodes[alias]=self.endpoint_nodes[name];self.feature_channels[alias]=channels[name]
