from .vit import ViT
from .cait import CaiT
from .pit import PiT
from .swin import SwinTransformer
from .t2t import T2T_ViT
from .regnet import *
from .effiv2 import *
from .resnet import resnet56, resnet110
from .efficientnet import EfficientNetB0
from .coatnet import *
from .coatnet_2 import coatnet2_0, coatnet2_1
from .coatnet_3 import coatnet3_0
import timm

def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        if not img_size == 224:
            patch_size = 4 if img_size == 32 else 8
            model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                        mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                        stochastic_depth=0.1, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        else:
            model = ViT(img_size=224, patch_size = 16,
                          num_classes=1000, dim=192, 
                    mlp_dim_ratio=4, depth=12, heads=3, dim_head=192//3,
                    stochastic_depth=0.1, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)

    if args.model == 'vit_s':
        if not img_size == 224:
            patch_size = 4 if img_size == 32 else 8
            model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=384, 
                        mlp_dim_ratio=2, depth=12, heads=6, dim_head=384//6,
                        stochastic_depth=0.1, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        else:
            model = ViT(img_size=224, patch_size = 16,
                          num_classes=1000, dim=384, 
                    mlp_dim_ratio=4, depth=12, heads=6, dim_head=384//6,
                    stochastic_depth=0.1, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)



    elif args.model == 'cait_xxs24':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=0.1, 
                     is_LSA=args.is_LSA, is_SPT=args.is_SPT, is_Coord=args.is_Coord)

    elif args.model == 'cait_xs24':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=0.1, dim=288,
                     is_LSA=args.is_LSA, is_SPT=args.is_SPT, is_Coord=args.is_Coord)

    elif args.model == 'cait_s24':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=0.1, dim=384,
                     is_LSA=args.is_LSA, is_SPT=args.is_SPT, is_Coord=args.is_Coord)

    elif args.model == 'cait_xxs36':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=0.1, depth=36,
                     is_LSA=args.is_LSA, is_SPT=args.is_SPT, is_Coord=args.is_Coord)
        
    elif args.model == 'pit':
        patch_size = 2 if img_size == 32 else 4    
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]
        
        model = PiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, 
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, 
                    stochastic_depth=0.1, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)

    elif args.model =='t2t':
        model = T2T_ViT(img_size=img_size, num_classes=n_classes, drop_path_rate=0.1, 
                        is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
    
        
    elif args.model =='swin_t':
        if not img_size == 224:
            depths = [2, 6, 4] if img_size == 32 else [2, 2, 6, 2]
            num_heads = [3, 6, 12] if img_size == 32 else [3, 6, 12, 24]
            mlp_ratio = 2
            window_size = 4
            embed_dim = 96
            patch_size = 2
                
            model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0.1, embed_dim=embed_dim,
                                    patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                    is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        else:
            model = SwinTransformer(is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        
    elif args.model =='coatnet_0':
        model = coatnet_0(img_size=img_size, n_classes=n_classes, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
    elif args.model =='coatnet_1':
        model = coatnet_1(img_size=img_size, n_classes=n_classes, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
    elif args.model =='coatnet_2':
        model = coatnet_2(img_size=img_size, n_classes=n_classes, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
    elif args.model =='coatnet_3':
        model = coatnet_3(img_size=img_size, n_classes=n_classes, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
    
    elif args.model =='coatnet2_0':
        model = coatnet2_0(img_size=img_size, n_classes=n_classes, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
    elif args.model =='coatnet2_1':
        model = coatnet2_1(img_size=img_size, n_classes=n_classes, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
    elif args.model =='coatnet3_0':
        model = coatnet3_0(img_size=img_size, n_classes=n_classes, is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        
    elif args.model =='swin_s':
        depths = [2, 18, 4] if img_size == 32 else [2, 2, 18, 2]
        num_heads = [3, 6, 12] if img_size == 32 else [3, 6, 12, 24]
        mlp_ratio = 2
        window_size = 4
        embed_dim = 96
        patch_size = 2
            
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0.1, embed_dim=embed_dim,
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        
    elif args.model =='swin_b':
        depths = [2, 18, 4] if img_size == 32 else [2, 2, 18, 2]
        num_heads = [4, 8, 16] if img_size == 32 else [4, 8, 16, 32]
        mlp_ratio = 2
        window_size = 4
        embed_dim = 128
        patch_size = 2
            
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0.1, embed_dim=embed_dim,
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        
        
    elif args.model =='swin_l':
        depths = [2, 18, 4] if img_size == 32 else [2, 2, 18, 2]
        num_heads = [6, 12, 24] if img_size == 32 else [6, 12, 24, 48]
        mlp_ratio = 2
        window_size = 4
        embed_dim = 192
        patch_size = 2
            
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=0.1, embed_dim=embed_dim,
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                is_SPT=args.is_SPT, is_LSA=args.is_LSA, is_Coord=args.is_Coord)
        
    elif args.model =='regnetY_400m':
        model = RegNetY_400MF(n_classes)
        
    elif args.model =='regnetY_4G':
        model = timm.create_model('regnety_040')
    
    elif args.model =='regnetY_8G':
        model = timm.create_model('regnety_080')

    elif args.model =='res56':
        model = resnet56(n_classes)

    elif args.model =='res110':
        model = resnet110(n_classes)

    elif args.model =='effib0':
        model = EfficientNetB0(n_classes)
        
    elif args.model =='effiv2':        
        model = effnetv2_s(n_classes)
        
    elif args.model =='regnetX_200m':
        model = RegNetX_200MF(n_classes)

    elif args.model =='regnetY_200m':
        model = RegNetY_200MF(n_classes)
        
    elif args.model =='regnetX_400m':
        model = RegNetX_400MF(n_classes)
        
    elif args.model =='effiv2_m':
        model = effnetv2_m(n_classes)
        
    return model
