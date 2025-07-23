# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_


from util.pos_embed import get_2d_sincos_pos_embed

def get_ortho(dim: int, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:

    A = torch.randn(dim, dim, device=device)
    U, S, Vh = torch.linalg.svd(A)
    
    return U, Vh
# def get_ortho(dim: int, device: torch.device = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Generates two random orthogonal matrices of size dim x dim using PyTorch.
#     """
#     # Generate the first random matrix and find its orthogonal part via QR decomposition
#     random_matrix1 = torch.randn(dim, dim, device=device)
#     q1, r1 = torch.linalg.qr(random_matrix1)
#     # Enforce a unique Q by making the diagonal of R positive
#     q1 = q1 @ torch.diag(torch.sign(torch.diag(r1)))

#     # Generate the second random matrix and find its orthogonal part
#     random_matrix2 = torch.randn(dim, dim, device=device)
#     q2, r2 = torch.linalg.qr(random_matrix2)
#     # Enforce a unique Q by making the diagonal of R positive
#     q2 = q2 @ torch.diag(torch.sign(torch.diag(r2)))

#     return q1, q2

def get_ortho_like(dim: int,
                   alpha: float,
                   beta: float,
                   dist: str = 'uniform',
                   device: torch.device = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    if dist == 'normal':
        std = alpha / (dim ** 0.5)
        A = torch.normal(mean=0.0, std=std, size=(dim, dim), device=device) \
            + beta * torch.eye(dim, device=device)
    elif dist == 'uniform':
        low = -((3 ** 0.5) / (dim ** 0.5))
        high = (3 ** 0.5) / (dim ** 0.5)
        A = alpha * (low + (high - low) * torch.rand(dim, dim, device=device)) \
            + beta * torch.eye(dim, device=device)
    else:
        raise NotImplementedError(f"Distribution '{dist}' not implemented.")

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    sqrt_S = torch.diag(torch.sqrt(S))
    L = U @ sqrt_S
    R = sqrt_S @ Vh

    return L, R
def suo_initialize(m: int, k: int, device: torch.device = 'cuda') -> torch.Tensor:
    """
    Generates an m x k matrix using SUO initialization with PyTorch.
    """
    if m <= k:
        X = torch.randn(m, k, device=device)
        A = X @ X.T # Create the m x m matrix XX^T
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        inv_sqrt_eigenvalues = eigenvalues.clamp(min=1e-8).rsqrt()
        A_inv_sqrt = eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        
        W = A_inv_sqrt @ X

    else: # m > k
        X = torch.randn(k, m, device=device)
        A = X @ X.T 
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        inv_sqrt_eigenvalues = eigenvalues.clamp(min=1e-8).rsqrt()
        A_inv_sqrt = eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        W_intermediate = A_inv_sqrt @ X
        W = W_intermediate.T
        
    return W

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,skipless=False):
        super().__init__()
        self.skipless = skipless
        if skipless:
            print("Skipless training activated")
        else:
            print("Skipless training NOT activated")
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.skipless:
            # Skipless training: no residual connection
            x = self.norm1(x)
            x = self.attn(x)
            x = self.norm2(x)
            x = self.mlp(x)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,skipless=False, mimetic=None,
                 unet_style=False,encoder_skiplists=[9,7,5,3,1],decoder_skiplists=[1,2,3,4,5],W_v=1.0,W_p=1.0):
        super().__init__()

        self.mimetic = mimetic
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.W_v = W_v
        self.W_p = W_p


        self.unet_style = unet_style # enable UNet-style encoder-decoder connections
        self.encoder_skiplists = encoder_skiplists
        self.decoder_skiplists = decoder_skiplists

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,skipless=skipless)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        if self.unet_style:
            print("✅ Initializing with U-Net style skip connections.")
            assert len(self.encoder_skiplists) == len(self.decoder_skiplists), "Encoder and decoder skip lists must have the same length."
            
            self.skip_connections = nn.ModuleList([
                nn.Linear(embed_dim, decoder_embed_dim, bias=True)
                for _ in range(len(self.encoder_skiplists))
            ])
            # Create a dictionary to map decoder layers to their corresponding skip connection module and encoder layer index
            self.skip_map = {
                dec_idx: (enc_idx, proj_layer)
                for (dec_idx, enc_idx, proj_layer) in zip(self.decoder_skiplists, self.encoder_skiplists, self.skip_connections)
            }
            for enc_idx, dec_idx in zip(self.encoder_skiplists, self.decoder_skiplists):
                print(f"  - Connecting encoder layer {enc_idx} to decoder layer {dec_idx}")

        # --------------------------------------------------------------------------
        

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        if self.mimetic is not None:
            alpha1, beta1 = self.mimetic[0], self.mimetic[1]
            alpha1 = float(alpha1)
            beta1 = float(beta1)

            head_dim_encoder = self.embed_dim // self.num_heads
            for i in range(self.depth):
                print(f"Initializing encoder block {i} with mimetic init (alpha1={alpha1}, beta1={beta1}), W_v={self.W_v}, W_p={self.W_p}")
                for h in range(self.num_heads):
                    Q, K = get_ortho_like(self.embed_dim, alpha1, beta1, dist='uniform')
                    Q = Q[:, :head_dim_encoder]
                    K = K.T[:, :head_dim_encoder]
                    self.blocks[i].attn.qkv.weight.data[(h * head_dim_encoder):((h + 1) * head_dim_encoder)] = Q.T.detach().clone().float()
                    self.blocks[i].attn.qkv.weight.data[self.embed_dim + (h * head_dim_encoder):self.embed_dim + ((h + 1) * head_dim_encoder)] = K.T.detach().clone().float()
                
                V, Proj = get_ortho(self.embed_dim)
                self.blocks[i].attn.qkv.weight.data[self.embed_dim * 2:] = self.W_v*V.T.detach().clone().float() # V is stored transposed in qkv
                self.blocks[i].attn.proj.weight.data = self.W_p*Proj.detach().clone().float()

            # print("Initializing ENCODER MLP layers with SUO initialization.")
            # Use a weight from the model to get the correct device
            # device = self.cls_token.device
            # for i in range(self.depth):
            #     # Get the mlp layers from the i-th encoder block
            #     fc1 = self.blocks[i].mlp.fc1
            #     fc2 = self.blocks[i].mlp.fc2

            #     # Initialize fc1
            #     m_fc1, k_fc1 = fc1.weight.shape
            #     fc1.weight.data.copy_(suo_initialize(m_fc1, k_fc1, device=device))

            #     # Initialize fc2
            #     m_fc2, k_fc2 = fc2.weight.shape
            #     fc2.weight.data.copy_(suo_initialize(m_fc2, k_fc2, device=device))

            # head_dim_decoder = self.decoder_embed_dim // self.decoder_num_heads
            # for i in range(self.decoder_depth):
            #     print(f"Initializing decoder block {i} with mimetic init (alpha1={alpha1}, beta1={beta1})")
            #     for h in range(self.decoder_num_heads):
            #         Q, K = get_ortho_like(self.decoder_embed_dim, alpha1, beta1, dist='uniform')
            #         Q = Q[:, :head_dim_decoder]
            #         K = K.T[:, :head_dim_decoder]
            #         self.decoder_blocks[i].attn.qkv.weight.data[(h * head_dim_decoder):((h + 1) * head_dim_decoder)] = Q.T.detach().clone().float()
            #         self.decoder_blocks[i].attn.qkv.weight.data[self.decoder_embed_dim + (h * head_dim_decoder):self.decoder_embed_dim + ((h + 1) * head_dim_decoder)] = K.T.detach().clone().float()

            #     V, Proj = get_ortho(self.decoder_embed_dim)
            #     self.decoder_blocks[i].attn.qkv.weight.data[self.decoder_embed_dim * 2:] = V.T.detach().clone().float() # V is stored transposed in qkv
            #     self.decoder_blocks[i].attn.proj.weight.data = Proj.detach().clone().float()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)

        # Store outputs from specified layers for skip connections
        encoder_skips = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.unet_style and i in self.encoder_skiplists:
                encoder_skips[i] = x
        
        

        x = self.norm(x)

        return x, mask, ids_restore,encoder_skips

    def forward_decoder(self, x, ids_restore,encoder_skips=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x)
        for i, blk in enumerate(self.decoder_blocks):
            if self.unet_style and i in self.skip_map:
                # Get the corresponding encoder layer and projection module
                enc_idx, proj_layer = self.skip_map[i]
                skip_connection = encoder_skips[enc_idx]

                # Project the skip connection to the decoder's dimension
                skip_proj = proj_layer(skip_connection)

                # "Un-shuffle" the skip connection to match the decoder's full sequence
                skip_mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - skip_proj.shape[1], 1)
                skip_ = torch.cat([skip_proj[:, 1:, :], skip_mask_tokens], dim=1)
                skip_ = torch.gather(skip_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                skip_proj_unshuffled = torch.cat([skip_proj[:, :1, :], skip_], dim=1)
                
                # Add the skip connection to the decoder input
                x = x + skip_proj_unshuffled
            
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore , encoder_skips= self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, encoder_skips)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_small_patch16_dec256d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_small_patch16_dec256d8b  # decoder: 
