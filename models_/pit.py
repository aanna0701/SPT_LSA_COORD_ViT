from math import sqrt
from utils.drop_path import DropPath
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from .SPT import ShiftedPatchTokenization
from .Coord import CoordLinear

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# helpers

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, coords, **kwargs):
        
        return self.fn(self.norm(x), coords=coords, **kwargs)
    
    def flops(self):
        flops = 0
        
        flops += self.fn.flops()
        flops += self.dim * self.num_tokens
        
        return flops
    

class FeedForward(nn.Module):
    def __init__(self, num_tokens, dim, hidden_dim, dropout = 0., is_SCL=False):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.is_SCL = is_SCL
        
        if not is_SCL:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            self.net = nn.Sequential(
                CoordLinear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                CoordLinear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
  
    def forward(self, x, coords):
        if not self.is_SCL:
            out = self.net(x)
        else:
            out = self.net[0](x, coords)
            out = self.net[1:3](out)
            out = self.net[3](out, coords)
            out = self.net[-1](out)
        
        return out
  
    def flops(self):
        flops = 0
        if not self.is_SCL:
            flops += self.dim * self.hidden_dim * self.num_tokens
            flops += self.dim * self.hidden_dim * self.num_tokens
        else:
            flops += (self.dim+2) * self.hidden_dim * self.num_tokens
            flops += self.dim * self.hidden_dim
            flops += self.dim * (self.hidden_dim+2) * self.num_tokens
            flops += self.dim * self.hidden_dim
        
        return flops
      
class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_SCL=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_patches = num_patches
        self.is_SCL = is_SCL     
        if self.is_SCL:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(num_patches, num_patches)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
            self.inf = float('-inf')

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) if not is_SCL else CoordLinear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)        
 
        if not is_SCL:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                CoordLinear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        

    def forward(self, x, coords):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1) if not self.is_SCL else self.to_qkv(x, coords).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
      
        if not self.is_SCL:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))           
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        if not self.is_SCL:
            out = self.to_out(out)
        else:
            out = self.to_out[0](out, coords)
            out = self.to_out[1](out)
        
        return out

    def flops(self):
        flops = 0
        if not self.is_SCL:
            flops += self.dim * self.inner_dim * 3 * self.num_patches
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3 
        flops += self.inner_dim * (self.num_patches**2)
        flops += self.inner_dim * (self.num_patches**2)
        if not self.is_SCL:
            flops += self.inner_dim * self.dim * self.num_patches
        else:
            flops += (self.inner_dim+2) * self.dim * self.num_patches
            flops += self.inner_dim * self.dim
        
        return flops

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_SCL=False, is_last=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        num_patches = num_patches**2 + 1
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_SCL=is_SCL)),
                PreNorm(num_patches, dim, FeedForward(num_patches, dim, mlp_dim_ratio, dropout = dropout, is_SCL=is_SCL if not (i == depth-1 and is_last) else False))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x, coords=None):
        for (attn, ff) in self.layers:       
            x = self.drop_path(attn(x, coords)) + x
            x = self.drop_path(ff(x, coords)) + x
            
        return x

    def flops(self):
        flops = 0
        
        for (attn, ff) in self.layers:       
            flops += attn.flops()
            flops += ff.flops()
        
        return flops
    
# depthwise convolution, for pooling

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class DepthWiseConv2d(nn.Module):
    def __init__(self, img_size, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.out_size = img_size // 2
        
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_out, dim_out, kernel_size = 1, bias = bias)
        ) 
                
    def forward(self, x):
        return self.net(x)
    
    def flops(self):
        flops = 0
             
        flops += self.dim_out * self.kernel_size * self.kernel_size * self.out_size * self.out_size
        flops += self.dim_out * self.dim_out * self.out_size * self.out_size
        
        return flops
    
    
# pooling layer

class Pool(nn.Module):
    def __init__(self, img_size, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(img_size, dim, dim * 2, kernel_size = 3, stride = 2, padding = 1)
        self.cls_ff = nn.Linear(dim, dim * 2)
        self.dim = dim
        

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h = int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, tokens), dim = 1)
    
    def flops(self):
        flops = 0
             
        flops += self.downsample.flops()
        flops += self.dim * self.dim * 2
        
        return flops
    
# main class

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PiT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, dim_head = 64, dropout = 0., 
                 emb_dropout = 0., stochastic_depth=0., 
                 is_SPT=False, is_LSA=False, is_Coord=False):
        super(PiT, self).__init__()
        heads = cast_tuple(heads, len(depth))
        self.num_classes = num_classes
        self.is_SCL = is_SCL
        output_size = img_size // patch_size
        num_patches = output_size ** 2  
        self.dim = dim
        self.num_classes = num_classes

        if not is_SCL:
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(3, dim, patch_size, patch_size),
                Rearrange('b c h w -> b (h w) c')
            )
            self.pe_flops = 3 * dim * patch_size*2 * patch_size*2 * num_patches
            
        else:
            self.to_patch_embedding = ShiftedPatchTokenization(img_size**2, 3, dim, patch_size, is_pe=True)
            
        
        if not is_SCL:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            self.layers.append(Transformer(dim, output_size, layer_depth, layer_heads,
                                           dim_head, dim*mlp_dim_ratio, dropout, stochastic_depth, is_SCL=is_SCL, is_last = True if ind == len(depth)-1 else False))

            if not_last:
                if not is_SCL:
                    self.layers.append(Pool(output_size, dim))
                    output_size = conv_output_size(output_size, 3, 2, 1)
                else:
                    self.layers.append(ShiftedPatchTokenization(output_size**2, dim, dim*2, 2, exist_class_t=True)) 
                    output_size //= 2
                
                dim *= 2
       
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        if not self.is_SCL:
            x = self.to_patch_embedding(img)     
            coords = None
        else:
            x, coords = self.to_patch_embedding(img)
                 
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        if not self.is_SCL:
            x += self.pos_embedding
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):  
            if i % 2 == 0:
                x = layer(x, coords)
            else:
                if self.is_SCL:
                    x, coords = layer(x)
                else:
                    x = layer(x)

        return self.mlp_head(x[:, 0])
    
    def flops(self):
        flops = 0
        
        flops_pe = self.pe_flops if not self.is_SCL else self.to_patch_embedding.flops()
        flops += flops_pe
        
        for layer in self.layers:            
            flops += layer.flops()  
        
        flops += self.dim               # layer norm
        flops += self.dim * self.num_classes    # linear
        
        return flops
    