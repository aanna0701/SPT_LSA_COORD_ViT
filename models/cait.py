from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import Module
from .SPT import ShiftedPatchTokenization
from .Coord import CoordLinear

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# helpers
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, coords=None, **kwargs):
        return self.fn(x, coords=coords, **kwargs) * self.scale
    def flops(self):
        flops = 0
        
        flops += self.fn.flops()
        
        return flops

class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.dim = dim
        self.num_tokens = num_tokens
        
    def forward(self, x, coords=None, **kwargs):
        return self.fn(self.norm(x), coords=coords, **kwargs)
    def flops(self):
        flops = 0
        
        flops += self.fn.flops()
        flops += self.dim * self.num_tokens
        
        return flops

class FeedForward(nn.Module):
    def __init__(self, num_tokens, dim, hidden_dim, dropout = 0., is_Coord=False, if_patch_attn=True):
        super().__init__()
        self.is_Coord = is_Coord
        self.if_patch_attn = if_patch_attn
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        
        if not if_patch_attn:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        
        else:
            if not is_Coord:
                self.net = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, dim),
                    nn.Dropout(dropout)
                )
            else:
                self.net = nn.Sequential(
                    CoordLinear(dim, hidden_dim, exist_cls_token=False),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    CoordLinear(hidden_dim, dim, exist_cls_token=False),
                    nn.Dropout(dropout)
                )
            
    def forward(self, x, coords=None):
        if not self.is_Coord or not self.if_patch_attn:
            out = self.net(x)
        else:
            out = self.net[0](x, coords)
            out = self.net[1:3](out)
            out = self.net[3](out, coords)
            out = self.net[-1](out)
        
        return out

    def flops(self):
        flops = 0
        if not self.if_patch_attn:
            flops += self.dim * self.hidden_dim
            flops += self.dim * self.hidden_dim
        else:
            if self.is_Coord:
                flops += (self.dim+2) * self.hidden_dim * self.num_tokens
                flops += self.dim * (self.hidden_dim+2) * self.num_tokens
            else:
                flops += self.dim * self.hidden_dim * self.num_tokens
                flops += self.dim * self.hidden_dim * self.num_tokens
            
        return flops

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., if_patch_attn=False, is_LSA=False, is_Coord=False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.is_Coord = is_Coord
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_patches = num_patches
        if not is_Coord:
            self.to_q = nn.Linear(dim, inner_dim, bias = False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        else:
            self.to_q = CoordLinear(dim, inner_dim, bias = False, exist_cls_token=False)
            self.to_kv = CoordLinear(dim, inner_dim * 2, bias = False, exist_cls_token=False)

        self.attend = nn.Softmax(dim = -1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        if not is_Coord:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Sequential(
                CoordLinear(inner_dim, dim, exist_cls_token=False),
                nn.Dropout(dropout)
            )
        
        self.is_LSA = is_LSA
        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(num_patches, num_patches)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        self.if_patch_attn = if_patch_attn

    def forward(self, x, context = None, coords=None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else context

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1)) if not self.is_Coord else (self.to_q(x, coords), *self.to_kv(context, coords).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if not self.is_LSA:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale        
        else:
            """ LSA """
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))            
            if self.if_patch_attn:
                dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -1e-9
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)        
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if not self.is_Coord:
            out = self.to_out(out)
        else:
            out = self.to_out[0](out, coords)
            out = self.to_out[1](out)
        
        return out
    
    def flops(self):
        flops = 0
        
        if self.if_patch_attn:
            if not self.is_Coord:
                flops += self.dim * self.inner_dim * 3 * self.num_patches
            else:    
                flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches
                
            flops += self.inner_dim * (self.num_patches**2)
            flops += self.inner_dim * (self.num_patches**2)
            flops += self.inner_dim * self.dim * self.num_patches
        
        else:
            if not self.is_Coord:
                flops += self.dim * self.inner_dim 
                flops += self.dim * self.inner_dim * 2 * (self.num_patches+1)
            else:
                flops += self.dim * self.inner_dim 
                flops += (self.dim+2) * self.inner_dim * 2 * (self.num_patches+1)
                
            flops += self.inner_dim * self.num_patches
            flops += self.inner_dim * self.num_patches
            flops += self.inner_dim * self.dim      
        
        return flops

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., stochastic_depth=0., 
                 if_patch_attn=False, is_LSA=False, is_Coord=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, if_patch_attn=if_patch_attn, is_LSA=is_LSA, is_Coord=is_Coord)), depth = ind + 1),
                LayerScale(dim, PreNorm(num_patches, dim, FeedForward(num_patches, dim, mlp_dim, dropout = dropout, is_Coord=is_Coord, if_patch_attn=if_patch_attn)), depth = ind + 1)
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x, context = None, coords=None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            
            x = self.drop_path(attn(x, context = context, coords=coords)) + x
            x = self.drop_path(ff(x, coords=coords)) + x
        return x
    
    def flops(self):
        flops = 0
        
        for (attn, ff) in self.layers:       
            flops += attn.flops()
            flops += ff.flops()
        
        return flops
    
class CaiT(nn.Module):
    def __init__(
        self,
        *,
        img_size,
        patch_size,
        num_classes,
        dim=192,
        depth=24,
        cls_depth=2,
        heads=4,
        mlp_dim=384,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.,
        stochastic_depth = 0.,
        is_LSA=False,
        is_SPT=False,
        is_Coord=False
    ):
        super().__init__()
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        if not is_SPT:
            patch_dim = 3 * patch_size ** 2
            
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, dim),
            )
            self.pe_flops = patch_dim * dim * num_patches
        
        else:
            self.to_patch_embedding = ShiftedPatchTokenization(img_size**2, 3, dim, patch_size, is_pe=True, is_Coord=is_Coord)

        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        
        self.dim = dim
        self.num_classes = num_classes
        self.is_SPT = is_SPT
        self.is_Coord = is_Coord
        if not is_Coord:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth, if_patch_attn=True, is_LSA=is_LSA, is_Coord=is_Coord)
        self.cls_transformer = Transformer(dim, num_patches, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth, is_LSA=is_LSA)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        if not self.is_Coord:
            x = self.to_patch_embedding(img)
            coords = None
        else:
            x, coords = self.to_patch_embedding(img)
            
        b, n, _ = x.shape

        if not self.is_Coord:
            x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x, coords=coords)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])

    def flops(self):
        flops = 0
        
        flops_pe = self.pe_flops if not self.is_SPT else self.to_patch_embedding.flops()
        flops += flops_pe
        
        flops += self.patch_transformer.flops()   
        flops += self.cls_transformer.flops()   
        
        flops += self.dim               # layer norm
        flops += self.dim * self.num_classes    # linear
        
        return flops


    