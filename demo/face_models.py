"""Self-contained ConvMAE-Base / Ghost+ConvMAE finetune classifiers for the demo.

Mirrors finetune/common/models_finetune.py but without the mamba/finetune-engine imports,
so the demo package runs on a laptop (CPU or a small CUDA GPU like an RTX 1650) with only
torch + timm installed. The face fine-tuned checkpoints load into these builders directly.
"""
import os
import sys
from functools import partial

import torch
import torch.nn as nn

# The bundled model files (vision_transformer.py, models_convvit.py, blocks_ghost.py) use
# flat top-level imports of each other, so put their directory on sys.path and import flat.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
import models_convvit  # noqa: E402
from blocks_ghost import GhostV2BlockMasked  # noqa: E402


def convvit_base(num_classes):
    """Baseline ConvViT-Base finetune encoder (CBlock stages 1/2, global-pool + fc_norm + head)."""
    m = models_convvit.ConvViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
        depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=True)
    m.head = nn.Linear(768, num_classes)
    return m


class ConvViTGhost(models_convvit.ConvViT):
    """ConvViT whose stage-1/2 CBlocks are swapped for GhostV2BlockMasked (dense path)."""

    def __init__(self, embed_dim, depth, **kwargs):
        super().__init__(embed_dim=embed_dim, depth=depth, **kwargs)
        self.blocks1 = nn.ModuleList([GhostV2BlockMasked(dim=embed_dim[0]) for _ in range(depth[0])])
        self.blocks2 = nn.ModuleList([GhostV2BlockMasked(dim=embed_dim[1]) for _ in range(depth[1])])


def convvit_ghost(num_classes):
    """Ghost+ConvMAE-Base finetune encoder (GhostV2 stages 1/2, global-pool + fc_norm + head)."""
    m = ConvViTGhost(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
        depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=True)
    m.head = nn.Linear(768, num_classes)
    return m


BUILDERS = {"baseline": convvit_base, "ghost": convvit_ghost}


def build_and_load(arm, num_classes, ckpt_path, map_location="cpu"):
    """Build the arm's model and load a fine-tuned checkpoint. Asserts a clean load."""
    model = BUILDERS[arm](num_classes)
    sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing]
    assert not missing, f"{arm}: missing keys on load: {missing[:8]}"
    model.eval()
    return model


@torch.no_grad()
def embed(model, x):
    """Pooled 768-d embedding (pre-head), used for LFW verification."""
    return model.forward_features(x)
