from email.mime import image
import torch
import torch.nn as nn

from .SPT import PatchShifting, ShiftedPatchTokenization
from .Coord import CoordLinear
from einops import rearrange
from einops.layers.torch import Rearrange

POOL = False

def conv_3x3_bn(inp, oup, image_size, downsample=False, is_Coord=False, is_SPT=False):
    # stride = 1 if downsample == False else 2
    return nn.Sequential(
        PatchShifting(2) if (is_SPT and downsample) else nn.Identity(),
        # nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.Conv2d(inp if not (is_SPT and downsample) else inp*5, oup, 3, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm, num_patches):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn
        self.num_patches = num_patches
        self.dim = dim

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
    def flops(self):
        flops = 0
        
        flops += self.num_patches * self.dim
        flops += self.fn.flops()
        
        return flops


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, image_size, dim, hidden_dim, dropout=0., is_Coord=False):
        super().__init__()
        self.ih, self.iw = image_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.is_Coord = is_Coord
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim) if not is_Coord else CoordLinear(dim, hidden_dim, exist_cls_token=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)if not is_Coord else CoordLinear(hidden_dim, dim, exist_cls_token=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

    def flops(self):
        flops = 0
        if not self.is_Coord:
            flops += self.dim * self.hidden_dim * self.ih * self.iw
            flops += self.dim * self.hidden_dim * self.ih * self.iw
        else:
            flops += (self.dim+2) * self.hidden_dim * self.ih * self.iw
            flops += self.dim * (self.hidden_dim+2) * self.ih * self.iw
        
        return flops


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=2, is_Coord=False, is_SPT=False):
        super().__init__()
        global POOL
        self.downsample = downsample
        self.ih, self.iw = image_size
        # if image_size[0] > 32 and self.downsample:
        #     POOL = True
        # stride = 1 if self.downsample == False else 2
        stride = 2 if downsample and POOL else 1
        self.inp = inp if not is_SPT else inp*5
        hidden_dim = int(inp * expansion)
        self.oup = oup
        self.hidden_dim = hidden_dim

        if self.downsample:
            self.SPT = PatchShifting(2) if is_SPT else nn.Identity()
            self.pool = nn.MaxPool2d(3, stride, 1)
            self.proj = nn.Conv2d(self.inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d, self.ih*self.iw)

    def forward(self, x):
        if self.downsample:
            x = self.SPT(x)
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)
        
    def flops(self):
        flops = 0
        if self.downsample:
            flops += self.ih * self.iw * (self.inp * self.oup)
            flops += self.ih * self.iw * (self.inp * self.oup)
                
        else:
            flops += self.ih * self.iw * (self.inp * self.hidden_dim)
            flops += self.ih * self.iw * self.hidden_dim
            flops += self.ih * self.iw * self.hidden_dim * self.hidden_dim * 3**2
            flops += self.hidden_dim * self.inp * 0.25
            flops += self.hidden_dim * self.inp * 0.25
            flops += self.ih * self.iw * self.hidden_dim
            flops += self.ih * self.iw * self.hidden_dim * self.oup
            flops += self.ih * self.iw * self.oup
                
        
        return flops


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0., is_LSA=False, is_Coord=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size
        self.inp = inp
        self.oup = oup
        self.inner_dim = inner_dim

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False) if not is_Coord else CoordLinear(inp, inner_dim * 3, bias=False, exist_cls_token=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup) if not is_Coord else CoordLinear(inner_dim, oup, exist_cls_token=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.is_LSA = is_LSA
        if self.is_LSA:
            self.mask = torch.eye(self.ih**2, self.ih**2)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
            self.inf = float('-inf')
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
        self.is_Coord = is_Coord

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if not self.is_LSA:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:                
            scale = self.scale
            dots = torch.matmul(q, k.transpose(-1, -2))
            dots = torch.mul(dots, scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))
             

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
    def flops(self):
        flops = 0
        if not self.is_Coord:
            flops += self.inp * self.inner_dim * 3 * self.ih * self.iw
        else:
            flops += (self.inp+2) * self.inner_dim * 3 * self.ih * self.iw  
            
        flops += self.inner_dim * ((self.ih * self.iw)**2)
        flops += self.inner_dim * ((self.ih * self.iw)**2)
        
        if not self.is_Coord:
            flops += self.inner_dim * self.oup * self.ih * self.iw
        else:
            flops += (self.inner_dim+2) * self.oup * self.ih * self.iw
        
        return flops


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0., 
                 is_LSA=False, is_Coord=False, is_last=False, is_SPT=False):
        super().__init__()
        hidden_dim = int(inp * 2)
        
        self.ih, self.iw = image_size
        self.downsample = downsample
        self.is_SPT = is_SPT
        self.inp = inp
        self.oup = oup
        
        if self.downsample:
            if not is_SPT:
                self.pool1 = nn.MaxPool2d(3, 2, 1)
                self.pool2 = nn.MaxPool2d(3, 2, 1)
                self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)     
            else:
                inp *= 5
                self.SPT = PatchShifting(2)
                self.pool1 = nn.MaxPool2d(3, 2, 1)
                self.pool2 = nn.MaxPool2d(3, 2, 1)
                self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)     

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout, is_LSA=is_LSA, is_Coord=is_Coord)
        self.ff = FeedForward(image_size, oup, hidden_dim, dropout, is_Coord=is_Coord if not is_last else False)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm, self.ih*self.iw),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm, self.ih*self.iw),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            if not self.is_SPT:
                x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
            else:
                x = self.SPT(x)
                x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x
    
    def flops(self):
        flops = 0
        if self.downsample:
            if not self.is_SPT:
                flops += self.ih * self.iw * (self.inp * self.oup)
                flops += self.attn[1].flops()
            else:
                flops += self.ih * self.iw * (self.inp * 5 * self.oup)
                flops += self.attn[1].flops()
                
        else:
            flops += self.attn[1].flops()
            flops += self.ff[1].flops()
        
        return flops


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=100, block_types=['C', 'C', 'T', 'T'],
                 is_LSA=False, is_SPT=False, is_Coord=False):
        super().__init__()
        global POOL
        self.channels = channels
        self.in_channels = in_channels
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}
        self.image_size = ih
        self.is_LSA = is_LSA
        self.is_Coord = is_Coord
        self.channel = channels[-1]
        self.n_classes = num_classes
        if ih == 32:
            self.s0 = self._make_layer(
                conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih, iw), is_SPT=is_SPT)
            self.s1 = self._make_layer(
                block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih, iw), is_SPT=is_SPT)
            POOL = True
            ih//=2
            iw//=2
            self.s2 = self._make_layer(
                block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih, iw), is_SPT=is_SPT)
            ih//=2
            iw//=2
            self.s3 = self._make_layer(
                block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih, iw), is_transformer=True, is_Coord=is_Coord, is_SPT=is_SPT)
            ih//=2
            iw//=2
            self.s4 = self._make_layer(
                block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih, iw), is_transformer=True, is_last=True, is_Coord=is_Coord, is_SPT=is_SPT)
        else:
            self.s0 = self._make_layer(
                conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih, iw), is_SPT=is_SPT)
            POOL = True
            ih//=2
            iw//=2
            self.s1 = self._make_layer(
                block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih, iw), is_SPT=is_SPT)
            ih//=2
            iw//=2
            self.s2 = self._make_layer(
                block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih, iw), is_SPT=is_SPT)
            ih//=2
            iw//=2
            self.s3 = self._make_layer(
                block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih, iw), is_transformer=True, is_Coord=is_Coord, is_SPT=is_SPT)
            ih//=2
            iw//=2
            self.s4 = self._make_layer(
                block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih, iw), is_transformer=True, is_last=True, is_Coord=is_Coord, is_SPT=is_SPT)

        self.pool = nn.AvgPool2d(ih, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size, is_transformer=False, is_last=False, is_Coord=False, is_SPT=False):
        layers = nn.ModuleList([])
        if not is_transformer:
            for i in range(depth):
                if i == 0:
                    layers.append(block(inp, oup, image_size, downsample=True, is_Coord=is_Coord, is_SPT=is_SPT))
                else:
                    layers.append(block(oup, oup, image_size))
        else:
            for i in range(depth):
                if i == 0:
                    layers.append(block(inp, oup, image_size, downsample=True, is_Coord=is_Coord, is_SPT=is_SPT))
                else:
                    layers.append(block(oup, oup, image_size, is_LSA=self.is_LSA, is_Coord=self.is_Coord, is_last = False if not (i == depth-1 and is_last) else True))
        # return layers
        return nn.Sequential(*layers)

    def flops(self):
        flops = 0
        if not self.is_Coord:
            flops += self.in_channels * self.channels[0] * 3**2 * self.image_size * self.image_size
            flops += self.channels[0] * self.image_size * self.image_size
        else:
            flops += self.in_channels * 5 * self.channels[0] * 3**2 * self.image_size * self.image_size
            flops += self.channels[0] * self.image_size * self.image_size
        
        for s1 in self.s1:
            flops += s1.flops()
        for s2 in self.s2:
            flops += s2.flops()
        for s3 in self.s3:
            flops += s3.flops()
        for s4 in self.s4:
            flops += s4.flops()
            
        flops += self.channel * self.n_classes
        
        return flops

def coatnet_0(img_size, n_classes, is_LSA=False, is_SPT=False, is_Coord=False):
    # if img_size > 32:
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((img_size, img_size), 3, num_blocks, channels, num_classes=n_classes, is_LSA=is_LSA, is_SPT=is_SPT, is_Coord=is_Coord)


def coatnet_1(img_size, n_classes, is_LSA=False, is_SPT=False, is_Coord=False):
    # if img_size > 32:
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    # else:
    #     num_blocks = [2, 6, 14, 2]           # L
    #     channels = [64, 192, 384, 768]      # D
    return CoAtNet((img_size, img_size), 3, num_blocks, channels, num_classes=n_classes, is_LSA=is_LSA, is_SPT=is_SPT, is_Coord=is_Coord)


def coatnet_2(img_size, n_classes, is_LSA=False, is_SPT=False, is_Coord=False):
    if img_size > 32:
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
    else:
        num_blocks = [2, 6, 14, 4]           # L
        channels = [128, 256, 512, 1026]   # D
    return CoAtNet((img_size, img_size), 3, num_blocks, channels, num_classes=n_classes, is_LSA=is_LSA, is_SPT=is_SPT, is_Coord=is_Coord)


def coatnet_3(img_size, n_classes, is_LSA=False, is_SPT=False, is_Coord=False):
    if img_size > 32:
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [192, 192, 384, 768, 1536]   # D
    else:
        num_blocks = [2, 6, 14, 2]           # L
        channels = [192, 384, 768, 1536]   # D
    return CoAtNet((img_size, img_size), 3, num_blocks, channels, num_classes=n_classes, is_LSA=is_LSA, is_SPT=is_SPT, is_Coord=is_Coord)


def coatnet_4(img_size, n_classes, is_LSA=False, is_SPT=False, is_Coord=False):
    if img_size > 32:
        num_blocks = [2, 2, 12, 28, 2]          # L
        channels = [192, 192, 384, 768, 1536]   # D
    else:
        num_blocks = [2, 12, 28, 2]          # L
        channels = [192, 384, 768, 1536]   # D
    return CoAtNet((img_size, img_size), 3, num_blocks, channels, num_classes=n_classes, is_LSA=is_LSA, is_SPT=is_SPT, is_Coord=is_Coord)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))