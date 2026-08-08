# Scenario 1: GhostBlock ở stage 1, 2 và Transformer ở stage 3

from functools import partial
from typing import Callable, Sequence
import torch
import torch.nn as nn

from vision_transformer import PatchEmbed, Block
from blocks_ghost import GhostV2BlockMasked
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderConvViT_AllGhost(nn.Module):
    """Scenario 1: GhostBlock at stages 1, 2 + Transformer at stage 3"""
    
    def __init__(self, img_size: Sequence[int] = (224, 56, 28), patch_size: Sequence[int] = (4, 2, 2), in_chans: int = 3,
                 embed_dim: Sequence[int] = (256, 384, 768), depth: Sequence[int] = (2, 2, 11), num_heads: int = 12,
                 decoder_embed_dim: int = 512, decoder_depth: int = 8, decoder_num_heads: int = 16,
                 mlp_ratio: Sequence[float] = (4.0, 4.0, 4.0), norm_layer: Callable[..., nn.Module] = nn.LayerNorm, norm_pix_loss: bool = False):
        super().__init__()
        self.patch_embed1 = PatchEmbed(
                img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.patch_embed = self.patch_embed3
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        
        # Stages 1, 2: GhostV2BlockMasked
        self.blocks1 = nn.ModuleList([GhostV2BlockMasked(dim=embed_dim[0]) for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([GhostV2BlockMasked(dim=embed_dim[1]) for i in range(depth[1])])
        
        # Stage 3: Full Transformer (same as S0)
        self.blocks3 = nn.ModuleList([
            Block(dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],
                qkv_bias=True, qk_scale=None, norm_layer=norm_layer)  # type: ignore[arg-type]
            for i in range(depth[2])])
        
        self.norm = norm_layer(embed_dim[-1])

        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, qk_scale=None, norm_layer=norm_layer)  # type: ignore[arg-type]
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[2])**2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed1.patch_size[0] * self.patch_embed2.patch_size[0] * self.patch_embed3.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N = x.shape[0]
        L = self.patch_embed3.num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)
        
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
        # Stage 3 with Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def convmae_allghost(**kwargs):
    model = MaskedAutoencoderConvViT_AllGhost(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convmae_ghost12_transformer3(**kwargs):
    return convmae_allghost(**kwargs)
