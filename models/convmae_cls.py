"""One classification wrapper for all four arms.

The four arms differ only in which blocks their pretrained MAE encoder holds (CBlock vs Ghost at
Stages 1-2; Transformer vs Mamba at Stage 3). The classifier forward is otherwise identical, so a
single ``ConvMAECls`` reuses any arm's encoder — its patch embeds, Stage-1/2 blocks, Stage-3 blocks,
multi-scale fusion and final norm — and adds a linear head. This replaces the four near-duplicate
``model_convmae_cls_*`` files.

Stage-3 blocks are called with ``(x, H, W)`` when they advertise ``supports_s1_s2`` (the Mamba
blocks, for the S1/S2 refinements) and with ``(x)`` otherwise (Transformer) — so the same forward
covers every arm. Linear probing wraps ``.head`` as ``Sequential(BatchNorm1d(affine=False), Linear)``
externally; that is left to the probe launcher, not baked in here.
"""
import torch.nn as nn

from .model_convmae_baseline import convmae_baseline
from .model_convmae_allghost import convmae_allghost


class ConvMAECls(nn.Module):
    """Frozen-or-finetuned ConvMAE encoder + linear classification head."""

    def __init__(self, mae, num_classes: int = 1000):
        super().__init__()
        # reuse the encoder pieces of a built MAE model (decoder/MAE-only params are dropped)
        self.patch_embed1 = mae.patch_embed1
        self.patch_embed2 = mae.patch_embed2
        self.patch_embed3 = mae.patch_embed3
        self.patch_embed4 = mae.patch_embed4
        self.stage1_output_decode = mae.stage1_output_decode
        self.stage2_output_decode = mae.stage2_output_decode
        self.pos_embed = mae.pos_embed
        self.blocks1 = mae.blocks1
        self.blocks2 = mae.blocks2
        self.blocks3 = mae.blocks3
        self.norm = mae.norm
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, None)               # Stage-1 block (CBlock / Ghost), dense path
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, None)               # Stage-2 block
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x)
        H3, W3 = x.shape[-2], x.shape[-1]
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed
        for blk in self.blocks3:
            if getattr(blk, "supports_s1_s2", False):
                x = blk(x, H3, W3)         # Mamba block with S1/S2 refinements
            else:
                x = blk(x)                 # Transformer block
        x = x + stage1_embed + stage2_embed

        x = self.norm(x)
        x = x.mean(dim=1)                  # global pooling
        return self.head(x)


def _build_mae(arm: str, **s1_s2_kwargs):
    """Build the arm's MAE model. S1/S2 kwargs apply to the Mamba arms only; they are ignored for
    the convolution/Transformer arms (which take no such flags)."""
    if arm == "baseline":
        return convmae_baseline()
    if arm == "ghost":
        return convmae_allghost()
    if arm == "bimamba":
        from .model_convmae_bimamba import convmae_bimamba          # CUDA-only (mamba_ssm), lazy
        return convmae_bimamba(**s1_s2_kwargs)
    if arm == "forwardmamba":
        from .model_convmae_forwardmamba import convmae_forwardmamba
        return convmae_forwardmamba(**s1_s2_kwargs)
    raise ValueError(f"Unknown arm: {arm!r} (expected baseline|ghost|bimamba|forwardmamba)")


def convmae_cls(arm: str, num_classes: int = 1000, **s1_s2_kwargs):
    """Classifier for one arm: ``convmae_cls("ghost")`` → ConvMAECls over the Ghost encoder."""
    return ConvMAECls(_build_mae(arm, **s1_s2_kwargs), num_classes=num_classes)
