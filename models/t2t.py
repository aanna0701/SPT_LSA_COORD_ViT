# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn
from torch import einsum

from timm.models.layers import trunc_normal_
import numpy as np
from .SPT import PatchShifting
from .Coord import CoordLinear
from einops import rearrange

"""
Take the standard Transformer as T2T Transformer
"""
import torch.nn as nn
from timm.models.layers import DropPath

"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""

class Mlp(nn.Module):
    def __init__(self, num_tokens, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., is_SCL=False, exist_cls_token=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) if not is_SCL else CoordLinear(in_features, hidden_features, exist_cls_token=exist_cls_token)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features) if not is_SCL else CoordLinear(hidden_features, out_features, exist_cls_token=exist_cls_token)
        self.drop = nn.Dropout(drop)
        self.exist_cls_token = exist_cls_token
        self.num_tokens = num_tokens
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.is_SCL = is_SCL

    def forward(self, x, coords=None):
        x = self.fc1(x) if not self.is_SCL else self.fc1(x, coords)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) if not self.is_SCL else self.fc2(x, coords)
        x = self.drop(x)
        return x
    
    def flops(self):
        flops = 0
        if self.exist_cls_token:
            flops += self.in_features * self.hidden_features
            flops += self.hidden_features * self.out_features
        
        if self.is_SCL:
            flops += (self.in_features+2) * self.hidden_features * self.num_tokens
            flops += self.out_features * (self.hidden_features+2) * self.num_tokens
        else:
            flops += self.in_features * self.hidden_features * self.num_tokens
            flops += self.out_features * self.hidden_features * self.num_tokens
            
            
        return flops
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=0, is_SCL=False, is_last=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches,is_SCL=is_SCL)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(num_tokens=num_patches, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, is_SCL=is_SCL if not is_last else False)
        self.num_tokens = num_patches
        self.dim = dim

    def forward(self, x, coords):
        x = x + self.drop_path(self.attn(self.norm1(x), coords))
        x = x + self.drop_path(self.mlp(self.norm2(x), coords))
        return x

    def flops(self):
        flops = 0
        
        flops += self.dim * self.num_tokens * 2
        flops += self.attn.flops()
        flops += self.mlp.flops()
        
        return flops 


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = 256, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 num_patches=0, is_LSA=False, exist_cls_token=True, is_SCL=False):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim = dim
        self.num_patches = num_patches
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5        
        self.is_SCL = is_SCL
        self.exist_cls_token = exist_cls_token
        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias) if not is_SCL else CoordLinear(dim, in_dim*3, bias=qkv_bias, exist_cls_token=exist_cls_token)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim) if not is_SCL else CoordLinear(in_dim, in_dim, exist_cls_token=exist_cls_token)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if self.is_SCL:
            self.mask = torch.eye(num_patches, num_patches)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
            self.inf = float('-inf')
            self.scale = nn.Parameter(self.scale*torch.ones(num_heads))

    def forward(self, x, coords=None):
        B, N, C = x.shape
        residual = x

        if not self.is_SCL:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x, coords).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.is_SCL:
            attn = (q * self.scale) @ k.transpose(-2, -1)
        
        else:
            scale = self.scale
            attn = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((B, self.num_heads, 1, 1)))
        
            attn[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
        
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x) if not self.is_SCL else self.proj(x, coords)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x if self.num_heads == 1 else x + residual   # because the original x has different size with current x, use v to do skip connection

        return x
    
    def flops(self):
        flops = 0
        
        if not self.exist_cls_token:
            if not self.is_SCL:
                flops += self.dim * self.in_dim * 3 * self.num_patches
            else:    
                flops += (self.dim+2) *self.in_dim * 3 * self.num_patches
                
            flops += self.in_dim * (self.num_patches**2)
            flops += self.in_dim * (self.num_patches**2)
            flops += self.in_dim * self.dim * self.num_patches
        
        else:
            if not self.is_SCL:
                flops += self.dim * self.in_dim * 3 * (self.num_patches+1)
            else:
                flops += self.dim * self.in_dim * 3
                flops += (self.dim+2) * self.in_dim * 3 * self.num_patches
                
            flops += self.in_dim * (self.num_patches**2)
            flops += self.in_dim * (self.num_patches**2)
            flops += self.in_dim * self.dim * (self.num_patches+1)     
        
        return flops


class Token_transformer(nn.Module):

    def __init__(self, num_tokens, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=0, is_SCL=False):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.in_dim = in_dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            num_patches=num_patches, is_SCL=is_SCL, exist_cls_token=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(num_tokens=num_tokens, in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop, exist_cls_token=False, is_SCL=is_SCL)

    def forward(self, x, coords=None):
        x = self.attn(self.norm1(x), coords=coords)
        x = x + self.drop_path(self.mlp(self.norm2(x), coords=coords))
        return x
    
    def flops(self):
        flops = 0
        flops += self.num_tokens * self.dim
        flops += self.attn.flops()
        flops += self.num_tokens * self.in_dim
        flops += self.mlp.flops()
        
        return flops    

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, token_dim=64, is_SCL=False):
        super().__init__()
        
        print('adopt transformer encoder for tokens-to-token')
        self.is_SCL = is_SCL
        in_chans = in_chans*5 if is_SCL else in_chans
        self.img_size = img_size
        if img_size == 64:
            
            self.soft_split0 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            
            self.num_patches = (img_size // (2)) * (img_size // (2))
            self.attention1 = Token_transformer(self.num_patches, dim=in_chans * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0, num_patches=self.num_patches, is_SCL=is_SCL)
            self.num_patches = (img_size // (2 * 2)) * (img_size // (2 * 2))
            self.attention2 = Token_transformer(self.num_patches, dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0, num_patches=self.num_patches, is_SCL=is_SCL)            
            self.num_patches = (img_size // (2 * 2 * 2)) * (img_size // (2 * 2 * 2))

        else:
            self.soft_split0 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            
            self.num_patches = (img_size // (2)) * (img_size // (2))
            self.attention1 = Token_transformer(self.num_patches, dim=in_chans * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0, num_patches=self.num_patches, is_SCL=is_SCL)
            self.num_patches = (img_size // (2 * 2)) * (img_size // (2 * 2))
            self.attention2 = None    
        
        if is_SCL:
            self.spt = PatchShifting(2)
        self.project = nn.Linear(token_dim * 3 * 3, embed_dim) if not is_SCL else CoordLinear(token_dim * 3 * 3, embed_dim, exist_cls_token=False)
        self.proj_flops = token_dim * 3 * 3 * embed_dim * self.num_patches if not is_SCL else (token_dim * 3 * 3 + 2) * embed_dim * self.num_patches
          # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        if self.is_SCL:
            x = self.spt(x)
        
        B = x.size(0)
        
        x = self.soft_split0(x).transpose(1, 2)
        
        coords = self.addcoords(B, self.img_size//2, self.img_size//2) if self.is_SCL else None
        
        # iteration1: re-structurization/reconstruction
        x = self.attention1(x, coords)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        
        
        
        x = self.soft_split1(x).transpose(1, 2)
        coords = self.addcoords(B, self.img_size//4, self.img_size//4) if self.is_SCL else None
        
        if self.attention2 is None:
            x = self.project(x, coords) if self.is_SCL else self.project(x)
            return x, coords
        
        # iteration2: re-structurization/reconstruction
        x = self.attention2(x, coords)  
        
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)
        coords = self.addcoords(B, self.img_size//8, self.img_size//8) if self.is_SCL else None
        # final tokens
        x = self.project(x, coords) if self.is_SCL else self.project(x)
        
        return x, coords
    
    
    def addcoords(self, B, H, W):
        xx_channel = torch.arange(H).repeat(1, W, 1)
        yy_channel = torch.arange(W).repeat(1, H, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (H - 1)
        yy_channel = yy_channel.float() / (W - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(B, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(B, 1, 1, 1).transpose(2, 3)

        xy_channel = torch.cat([
			        xx_channel, yy_channel],
			        dim=1)
        xy_channel = rearrange(xy_channel, 'b d h w -> b (h w) d')
    
        return xy_channel
    
    def flops(self):
        flops = 0
        flops += self.attention1.flops()
        if self.attention2 is not None:
            flops += self.attention2.flops() 
        
        flops += self.proj_flops
        
        return flops    

class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dim=256, depth=12,
                 num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64, is_SCL=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim  
        self.is_SCL = is_SCL
        self.tokens_to_token = T2T_module(
                img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim, is_SCL=is_SCL)
        num_patches = self.tokens_to_token.num_patches
        self.num_patches = num_patches
        self.num_classes = num_classes

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if not is_SCL:
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_patches=num_patches+1, is_SCL=is_SCL, is_last= False if not i==depth-1 else True)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, coords = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if not self.is_SCL:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, coords)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.tokens_to_token.flops()
        
        for blk in self.blocks:
            flops += blk.flops()
        
        flops += self.embed_dim * self.num_patches
        flops += self.embed_dim * self.num_classes
        
        return flops    
