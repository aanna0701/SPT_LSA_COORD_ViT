import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from .Coord import CoordLinear
import math

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, input_size, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False, is_Coord=True):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.exist_class_t = exist_class_t
        self.num_patches = input_size // (merging_size**2)
        self.patch_shifting = PatchShifting(merging_size)
        self.is_Coord = is_Coord
        
        patch_dim = (in_dim*5) * (merging_size**2) 
        self.patch_dim = patch_dim
        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, dim)

        self.is_pe = is_pe
    
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim) if not is_Coord else CoordLinear(patch_dim, dim, exist_cls_token=False)
        )

    def forward(self, x):
        input_size = int(math.sqrt(self.num_patches))
        coords = self.addcoords(x.size(0), input_size, input_size)
        
        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out_visual = self.patch_shifting(reshaped)
            if not self.is_Coord:
                out_visual = self.merging(out_visual)
            else:
                out_visual = self.merging[:2](out_visual)
                out_visual = self.merging[-1](out_visual, coords)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)
        
        else:
            out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out = self.patch_shifting(out)
            if not self.is_Coord:
                out = self.merging(out)
            else:
                out = self.merging[:2](out)
                out = self.merging[-1](out, coords)    
        
        return out, coords
    
    def flops(self):
        flops = 0
        
        flops += self.num_patches * self.patch_dim
        
        if self.exist_class_t:
            flops += self.in_dim * self.dim
   
        flops += self.num_patches * self.patch_dim * self.dim
        
        
        return flops
    
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
        

        
        
class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1/2))
        
    def forward(self, x):
     
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)
        
        """ 4 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1) 
        #############################
        
        """ 4 diagonal directions """
        # #############################
        x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1) 
        # #############################
        
        """ 8 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1) 
        #############################
        
        # out = self.out(x_cat)
        out = x_cat
        
        return out
