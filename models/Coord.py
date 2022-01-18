import torch
import torch.nn as nn
from math import sqrt
from einops import rearrange
    
    
class AddCoords1D(nn.Module):
    
    def __init__(self, input_size, batch_size):
        super().__init__()
        self.xx_channel = torch.arange(input_size).repeat(1, input_size, 1)
        self.yy_channel = self.xx_channel.clone().transpose(1, 2)

        self.xx_channel = self.xx_channel.float() / (input_size - 1)
        self.yy_channel = self.yy_channel.float() / (input_size - 1)

        self.xx_channel = self.xx_channel * 2 - 1
        self.yy_channel = self.yy_channel * 2 - 1
        
        self.xx_channel = self.xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        self.yy_channel = self.yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        
        self.input_size = input_size
        
    def forward(self, input_tensor):
        
        input_tensor = rearrange(input_tensor, 'b (h w) d -> b d h w', h = self.input_size)     

        ret = torch.cat([
            input_tensor,
            self.xx_channel.type_as(input_tensor),
            self.yy_channel.type_as(input_tensor)], dim=1)

        ret = rearrange(ret, 'b d h w -> b (h w) d')

        return ret

# class AddCoords1D(nn.Module):
    
#     def __init__(self, with_r=False):
#         super().__init__()
#         self.with_r = with_r
        
#     def forward(self, input_tensor):
#         """
#         Args:
#             input_tensor: shape(batch, channel, x_dim, y_dim)
#         """

#         batch_size, n, _ = input_tensor.size()
#         t_dim = int(sqrt(n)) 
#         self.xx_channel = torch.arange(t_dim).repeat(1, t_dim, 1)
#         self.yy_channel = self.xx_channel.clone().transpose(1, 2)

#         self.xx_channel = self.xx_channel.float() / (t_dim - 1)
#         self.yy_channel = self.yy_channel.float() / (t_dim - 1)

#         self.xx_channel = self.xx_channel * 2 - 1
#         self.yy_channel = self.yy_channel * 2 - 1

#         self.xx_channel = self.xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
#         self.yy_channel = self.yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        
#         input_tensor = rearrange(input_tensor, 'b (h w) d -> b d h w', h = t_dim)     
        

#         ret = torch.cat([
#             input_tensor,
#             self.xx_channel.type_as(input_tensor),
#             self.yy_channel.type_as(input_tensor)], dim=1)
      
#         if self.with_r:
#             rr = torch.sqrt(torch.pow(self.xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(self.yy_channel.type_as(input_tensor) - 0.5, 2))
#             ret = torch.cat([ret, rr], dim=1)

#         ret = rearrange(ret, 'b d h w -> b (h w) d')

#         return ret


class CoordLinear(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, with_r=False, exist_cls_token=True, addcoords=None):
        super().__init__()
        self.addcoords = addcoords
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.linear = nn.Linear(in_size, out_channels, bias=bias)
        
        self.exist_cls_token = exist_cls_token
        if exist_cls_token:
            self.cls_linear = nn.Linear(in_channels, out_channels, bias=bias) 


    def forward(self, x):
        if self.exist_cls_token:
            cls_token = self.cls_linear(x[:, (0,)]) # (b, 1, d')
            ret = self.addcoords(x[:, 1:])  # (b, n, d+2) or (b, n, d+3)
            ret = self.linear(ret)          # (b, n, d')
            out = torch.cat([cls_token, ret], dim=1)    # (b, n+1, d')
        else:
            ret = self.addcoords(x) # (b, n, d+2) or (b, n, d+3)
            ret = self.linear(ret)  # (b, n, d')
            out = ret   
        
        return out
    