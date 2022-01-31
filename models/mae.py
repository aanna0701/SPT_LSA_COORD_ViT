import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from .vit import Transformer
from .Coord import CoordLinear

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        mlp_ratio = 4,
        is_SPT=False, is_LSA=False, is_Coord=False
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.is_Coord = is_Coord
        self.is_SPT = is_SPT
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        if not is_Coord:
            num_patches, encoder_dim = encoder.pos_embedding.shape[-2:] # (B, N^2+1, d)
        else:
            num_patches = encoder.num_patches
            encoder_dim = encoder.dim
        
        if not is_SPT:
            self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
            pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        else:
            self.to_patch = encoder.to_patch_embedding.merging[0]
            self.patch_to_emb = encoder.to_patch_embedding
            pixel_values_per_patch = encoder.dim

        # decoder parameters
        if not is_Coord:
            self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        else:
            self.enc_to_dec = CoordLinear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim_ratio = mlp_ratio
                                   ,is_LSA=is_LSA, is_Coord=is_Coord, num_patches=1)
        if not is_Coord:
            self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch) if not is_Coord else CoordLinear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches
        if not self.is_SPT:
            patches = self.to_patch(img)
            batch, num_patches, *_ = patches.shape

            # patch to encoder tokens and add positions

            tokens = self.patch_to_emb(patches)

        else:
            patches = self.to_patch(img)
            batch, num_patches, *_ = patches.shape
            tokens = self.patch_to_emb(img)
        
        if not self.is_Coord:
            tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices) if not self.is_Coord else decoder_tokens

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices) if not self.is_Coord else mask_tokens

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss