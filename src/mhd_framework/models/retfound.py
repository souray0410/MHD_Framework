"""RETFound-MAE ViT-L/16 classification encoder, explicit 2D implementation.

Architecture/formula reference: RViMLab/RETFound_MAE models_vit.py.
Uses timm 0.9.2 building blocks; no upstream source or weights are vendored.
"""
from dataclasses import dataclass
from functools import partial
import torch
from torch import nn
from .graph import NamedModelGraph
from .resnet import FlattenViews

@dataclass(frozen=True)
class RETFoundConfig:
    name: str = 'retfound_mae_vit_large_patch16'
    spatial_dims: int = 2
    in_channels: int = 3
    image_size: int = 224
    num_classes: int = 2
    views: int = 1
    drop_path_rate: float = 0.0
    implementation: str = 'retfound_mae_timm_0_9_2'
    schema: str = 'mhd_retfound_task_v1'

    def __post_init__(self):
        if self.name != 'retfound_mae_vit_large_patch16' or self.implementation != 'retfound_mae_timm_0_9_2':
            raise ValueError('Select the explicit RETFound-MAE implementation')
        if self.spatial_dims != 2 or self.in_channels != 3 or self.image_size != 224:
            raise ValueError('RETFound requires RGB 224x224 images/B-scans, not a 3D volume')
        if self.num_classes < 2 or self.views < 1 or not 0 <= self.drop_path_rate < 1 or self.schema != 'mhd_retfound_task_v1':
            raise ValueError('Invalid RETFound task configuration')


def native_retfound(config):
    import timm
    if timm.__version__ != '0.9.2':
        raise RuntimeError('This implementation is validated against timm==0.9.2')
    from timm.models.vision_transformer import VisionTransformer
    model = VisionTransformer(img_size=config.image_size,patch_size=16,
        embed_dim=1024,depth=24,num_heads=16,mlp_ratio=4,qkv_bias=True,
        num_classes=config.num_classes,global_pool='avg',fc_norm=True,
        norm_layer=partial(nn.LayerNorm,eps=1e-6),drop_path_rate=config.drop_path_rate)
    # Match the original explicit attention formula, not backend-selected SDPA.
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


class TokenPosition(nn.Module):
    def __init__(self,native):
        super().__init__()
        self.cls_token = native.cls_token
        self.pos_embed = native.pos_embed
        self.drop = native.pos_drop
    def forward(self,x):
        token = self.cls_token.expand(x.shape[0],-1,-1)
        return self.drop(torch.cat((token,x),dim=1)+self.pos_embed)

class TokenPool(nn.Module):
    def __init__(self,norm,views):
        super().__init__(); self.norm=norm; self.views=views
    def forward(self,x):
        # Original global pooling excludes CLS and applies normalization AFTER averaging.
        x=self.norm(x[:,1:].mean(dim=1))
        if self.views > 1:
            x=x.reshape(-1,self.views,x.shape[-1]).mean(dim=1)
        return x


def retfound_operations(native,config):
    return ([('views',FlattenViews(config)),('patch_embed',native.patch_embed),
            ('tokens',TokenPosition(native))]
        + [(f'block{i+1:02}',b) for i,b in enumerate(native.blocks)]
        + [('features',TokenPool(native.fc_norm,config.views)),('logits',native.head)])


class MHDRETFound(NamedModelGraph):
    def __init__(self,config,*,weights=None,device='cpu'):
        if weights is not None:
            raise ValueError('Use load_pretrained_encoder with a local file, modality and SHA256')
        native=native_retfound(config)
        channels={f'block{i+1:02}':1024 for i in range(24)}
        channels.update(patch_embed=1024,tokens=1024,features=1024)
        super().__init__(config,native,retfound_operations(native,config),channels,device=device)

    def load_pretrained_encoder(self,path,*,sha256,modality,source):
        """Explicit pretraining transfer; task head/fc_norm remain newly initialized.

        Complete fine-tuned task models instead use strict load_native_state_dict
        or verified load_bundle. SHA and source must describe the actual local file.
        No positional interpolation or partial encoder fallback is performed.
        """
        from .artifacts import file_sha256
        if modality not in ('cfp','oct_bscan') or not source:
            raise ValueError('Explicit CFP/OCT-B-scan initialization provenance required')
        if file_sha256(path) != sha256:
            raise ValueError('Pretrained checkpoint SHA256 mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        state=payload.get('model',payload)
        receipt=_load_encoder(self._native_reference,state)
        return dict(receipt,source=source,modality=modality,sha256=sha256)


def _load_encoder(native,state):
    expected=native.state_dict()
    required={k for k in expected if not k.startswith(('head.','fc_norm.'))}
    allowed_extras=('decoder_','mask_token','norm.','head.','fc_norm.')
    missing=required-set(state)
    unexpected={k for k in state if k not in required and not k.startswith(allowed_extras)}
    bad={k for k in required & set(state) if not isinstance(state[k],torch.Tensor) or state[k].shape != expected[k].shape}
    if missing or unexpected or bad:
        raise ValueError(f'Encoder mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shape={sorted(bad)}')
    # Validate the complete encoder before mutating any tensor.
    merged=dict(expected); merged.update({k:state[k] for k in required})
    native.load_state_dict(merged,strict=True)
    return {'loaded_encoder_keys':len(required),'new_task_parameters':['head','fc_norm'],
            'ignored_pretraining_keys':sorted(set(state)-required)}
