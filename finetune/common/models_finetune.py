"""Finetune encoders for the fair baseline-vs-Ghost ConvMAE comparison.

Both encoders are the ConvMAE *encoder* (the MAE decoder + multi-scale fusion used only
during pretraining are dropped at finetune time, exactly as in the original ConvMAE recipe).
They are byte-for-byte identical except for the block type used in stages 1 & 2:

    baseline : stages 1/2 = CBlock              (masked depthwise conv, original ConvMAE)
    ghost    : stages 1/2 = GhostV2BlockMasked  (GhostNetV2 block, run in dense/mask=None mode)

Keeping everything else identical (patch embeds, stage-3 Transformer, pos_embed, head,
global-pool + fc_norm) is what makes the comparison fair: the ONLY independent variable is
the stage-1/2 operator, which is also the only thing that differs between the two pretrained
checkpoints in ../models/.

This module lives outside the legacy `models_convvit` so we can depend on a modern timm
(0.9.x) — the torch 2.x / Blackwell (sm_120) stack the real 6xRTX5090 box runs on cannot use
the pinned timm 0.3.2 that `main_finetune.py` asserts.
"""
from functools import partial

import torch
import torch.nn as nn

# Reuse the exact building blocks from the repo so state_dict keys match the checkpoints.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import models_convvit  # noqa: E402  (baseline ConvViT with global-pool + fc_norm)
from models.blocks_ghost import GhostV2BlockMasked  # noqa: E402  (same class used by the pretrain ghost model)


# --------------------------------------------------------------------------------------
# Baseline: reuse the repo's finetune encoder unchanged.
# --------------------------------------------------------------------------------------
def convvit_base_patch16(**kwargs):
    """Baseline ConvMAE finetune encoder (CBlock stages 1/2, Transformer stage 3)."""
    return models_convvit.convvit_base_patch16(**kwargs)


# --------------------------------------------------------------------------------------
# Ghost: same ConvViT, but stages 1/2 use GhostV2BlockMasked (dense inference path).
# --------------------------------------------------------------------------------------
class ConvViTGhost(models_convvit.ConvViT):
    """ConvViT whose stage-1/2 CBlocks are swapped for GhostV2BlockMasked.

    We subclass the baseline ConvViT (so forward_features, global-pool, fc_norm, head and
    all patch embeds are identical) and only rebuild blocks1/blocks2. GhostV2BlockMasked
    called as ``blk(x)`` (mask=None) takes its dense path, matching how the pretrained
    ghost encoder is used at inference/finetune time.
    """

    def __init__(self, embed_dim, depth, **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, **kwargs)
        # Replace the CBlock stacks built by the parent with Ghost blocks. The submodule
        # names inside GhostV2BlockMasked (primary_conv, cheap_conv, input_proj,
        # horizontal_fc, vertical_fc, se_reduce, se_expand, shortcut) match the ghost
        # pretrain checkpoint keys blocks1.*/blocks2.* exactly.
        self.blocks1 = nn.ModuleList([GhostV2BlockMasked(dim=embed_dim[0]) for _ in range(depth[0])])
        self.blocks2 = nn.ModuleList([GhostV2BlockMasked(dim=embed_dim[1]) for _ in range(depth[1])])
        # Ghost blocks were freshly initialised by their own __init__; the pretrained weights
        # are loaded afterwards by load_pretrained_encoder().


def convvit_ghost_base_patch16(**kwargs):
    """Ghost ConvMAE finetune encoder (GhostV2BlockMasked stages 1/2, Transformer stage 3)."""
    model = ConvViTGhost(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
        depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# --------------------------------------------------------------------------------------
# Mamba arms (bimamba / forwardmamba): stages 1/2 = Ghost, stage 3 adds BiMamba/forward-Mamba
# blocks with local-scan + ConvFFN. The finetune model is assembled here (same place as the
# baseline/ghost builders) from the repo-root MAE factory `convmae_{bimamba,forwardmamba}`, which
# depends on mamba_ssm + causal_conv1d (CUDA-only). The factory is imported LAZILY inside each
# builder so importing this module never pulls mamba on the CPU baseline/ghost path. The pretrain
# checkpoints were trained with the "S1-S2" config (configs/ablations/s1_s2.yaml), so the model
# MUST be built with the same flags or the state_dict keys won't match the checkpoint.
# --------------------------------------------------------------------------------------
MAMBA_S1S2 = dict(
    use_local_scan=True,
    local_scan_window_size=4,
    scan_direction="horizontal",
    use_convffn=True,
    convffn_expand_ratio=4.0,
    convffn_dw_kernel=3,
)


class ConvMAEMambaCls(nn.Module):
    """Finetune head shared by the bimamba/forwardmamba arms.

    Wraps a pretrained ConvMAE-Mamba MAE (`mae`): Ghost stages 1/2 + Mamba stage 3, with stage-1/2
    multi-scale fusion added to the stage-3 output and norm-then-mean pooling before a fresh linear
    head. This is these arms' native finetune head (no global-pool `fc_norm`, unlike the ConvViT
    encoders) — a documented fairness caveat. Only mamba stage-3 blocks with `supports_s1_s2` take
    the (x, H, W) signature; plain Transformer blocks are called as blk(x).
    """

    def __init__(self, mae, num_classes=1000):
        super().__init__()
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
        self.head = nn.Linear(768, num_classes)   # embed_dim[-1] = 768 (ConvViT-Base)

    def forward_features(self, x):
        """Pooled 768-d embedding (before the head), mirroring ConvViT.forward_features.

        This is the representation the head classifies AND the embedding the verification task
        (LFW) compares by cosine similarity — the engine's `_evaluate_verification` calls
        `forward_features`, so the mamba arms must expose it exactly like the baseline/ghost arms.
        """
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, None)                                   # Ghost block (dense path)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, None)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed3(x)
        H3, W3 = x.shape[-2], x.shape[-1]
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed
        for blk in self.blocks3:
            x = blk(x, H3, W3) if getattr(blk, "supports_s1_s2", False) else blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))

    @torch.jit.ignore
    def no_weight_decay(self):
        # pos_embed is not weight-decayed (matches the ConvViT encoders, fairness)
        return {"pos_embed"}


def _mamba_cfg(kwargs):
    cfg = {**MAMBA_S1S2}
    cfg.update({k: v for k, v in kwargs.items() if k in MAMBA_S1S2})
    return cfg


def convmae_bimamba_base_patch16(**kwargs):
    """BiMamba ConvMAE finetune model (Ghost stages 1/2 + BiMamba stage 3, S1-S2 config).

    drop_path_rate/global_pool are accepted for a uniform interface but not consumed by this arm
    (its head is fixed — see ConvMAEMambaCls). Only the S1-S2 flags + num_classes matter.
    """
    from models.model_convmae_bimamba import convmae_bimamba  # noqa: E402  (lazy, CUDA-only deps)
    mae = convmae_bimamba(**_mamba_cfg(kwargs))
    return ConvMAEMambaCls(mae, num_classes=kwargs.get("num_classes", 1000))


def convmae_forwardmamba_base_patch16(**kwargs):
    """ForwardMamba ConvMAE finetune model (Ghost stages 1/2 + forward Mamba stage 3, S1-S2)."""
    from models.model_convmae_forwardmamba import convmae_forwardmamba  # noqa: E402  (lazy, CUDA-only)
    mae = convmae_forwardmamba(**_mamba_cfg(kwargs))
    return ConvMAEMambaCls(mae, num_classes=kwargs.get("num_classes", 1000))


# Registry so configs/CLI can select by string, mirroring models_convvit.__dict__ lookup.
MODELS = {
    "convvit_base_patch16": convvit_base_patch16,                  # baseline
    "convvit_ghost_base_patch16": convvit_ghost_base_patch16,      # ghost
    "convmae_bimamba_base_patch16": convmae_bimamba_base_patch16,  # ghost + BiMamba (S1-S2)
    "convmae_forwardmamba_base_patch16": convmae_forwardmamba_base_patch16,  # ghost + fwd Mamba
}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise KeyError(f"Unknown finetune model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](**kwargs)
